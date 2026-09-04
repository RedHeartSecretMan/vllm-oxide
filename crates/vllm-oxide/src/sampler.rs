//! Token sampler — `SamplingParams` + `Sampler` (ADR-0004 micro-decision M1).
//!
//! Pipeline: upcast logits to F32 → apply penalties (presence / frequency /
//! repetition) → scale by per-sequence temperature (greedy short-circuits to
//! argmax when `temperature == 0` OR `top_k == Some(1)`) → top-k mask →
//! top-p nucleus → Gumbel-max sample. Each row of the `[batch, vocab]` logits
//! tensor is sampled independently; batch-mates with different
//! `SamplingParams.temperature` (or `top_k` / `top_p`) compose correctly in
//! one `Sampler::forward` call. CPU tensors use the reference host adapter;
//! CUDA tensors dispatch fail-closed to the device adapter, whose reusable
//! workspace keeps logits, filtering, and token choice on the GPU. Normal
//! generation transfers only the resulting `[batch]` U32 token tensor.
//!
//! # Divergence from nano-vllm
//!
//! nano-vllm's `Sampler` (`nanovllm/layers/sampler.py`) only supports
//! temperature sampling and asserts `temperature > 1e-10`. v0.2.0 explicitly
//! supports greedy decoding (`temperature == 0`) and adds top-k, top-p, and
//! the three penalties (per the issue #17 spec). Sampler correctness rests
//! on T8 property tests — goldens validate the model forward pass
//! (pre-sampling logits), NOT sampling.

use std::fmt;

use candle_core::{DType, Error as CandleError, Result, Tensor};
#[cfg(feature = "cuda")]
use rand::RngCore;
use rand::SeedableRng;
use rand_distr::{Distribution, Gumbel};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(test)]
use std::cell::RefCell;

/// Host-transfer categories that are observable in sampler tests. The
/// instrumentation sits at the transfer seam rather than inspecting Candle
/// internals, so CUDA regression tests can distinguish the forbidden
/// full-vocabulary path from the permitted selected-token copy.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostTransfer {
    FullLogits { elements: usize },
    SelectedTokens { elements: usize },
}

#[cfg(test)]
thread_local! {
    static HOST_TRANSFERS: RefCell<Vec<HostTransfer>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn record_host_transfer(transfer: HostTransfer) {
    HOST_TRANSFERS.with(|transfers| transfers.borrow_mut().push(transfer));
}

#[cfg(not(test))]
fn record_full_logits_host_transfer(_elements: usize) {}

#[cfg(test)]
fn record_full_logits_host_transfer(elements: usize) {
    record_host_transfer(HostTransfer::FullLogits { elements });
}

fn full_logits_to_host(logits: &Tensor) -> Result<Vec<f32>> {
    record_full_logits_host_transfer(logits.elem_count());
    logits.to_vec1::<f32>()
}

#[cfg(not(test))]
fn record_selected_tokens_host_transfer(_elements: usize) {}

#[cfg(test)]
fn record_selected_tokens_host_transfer(elements: usize) {
    record_host_transfer(HostTransfer::SelectedTokens { elements });
}

/// The one normal-generation device-to-host seam: a compact `[batch]` tensor
/// of selected token identifiers. Keeping this operation named and
/// instrumented makes accidental full-logit transfers regression-testable.
pub(crate) fn selected_token_ids_to_host(token_ids: &Tensor) -> Result<Vec<u32>> {
    record_selected_tokens_host_transfer(token_ids.elem_count());
    token_ids.to_vec1::<u32>()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SamplingExecution {
    Host,
    Device,
}

fn sampling_execution(is_cuda: bool) -> SamplingExecution {
    if is_cuda {
        SamplingExecution::Device
    } else {
        SamplingExecution::Host
    }
}

/// Per-prompt sampling configuration. `Default` is **greedy** — `temperature
/// = 0` and `top_k = None`, which is the deterministic path used by golden
/// fixtures and tests (user story 6: "deterministic, reproducible output for
/// tests and golden fixtures").
///
/// `max_tokens` and `ignore_eos` are honoured by the engine loop (#21), not
/// by the internal `Sampler` itself — sampling is single-step (one token per call)
/// and stateless across steps. They live here, not in `engine/sequence.rs`,
/// because nano-vllm's `sampling_params.py` is top-level and `LLM::generate`
/// accepts them user-facing (ADR-0004 M1).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature. `0.0` short-circuits to greedy argmax. NaN and
    /// negative values are rejected; positive infinity is the uniform
    /// pre-filter corner case. nano-vllm asserts `> 1e-10`; v0.2.0 supports 0.
    pub temperature: f32,

    /// Top-k truncation: keep only the `k` highest-logit tokens, mask the
    /// rest to probability 0. `Some(1)` is equivalent to greedy (forces
    /// argmax). Values must be at least one; values at least as large as the
    /// vocabulary are a no-op. `None` disables truncation.
    pub top_k: Option<usize>,

    /// Top-p (nucleus) truncation: keep the smallest token set whose
    /// cumulative probability ≥ `p`. `None` disables. Typical: `0.9`.
    /// Must be finite and in `(0.0, 1.0]` when `Some`; `Some(1.0)` is a no-op.
    pub top_p: Option<f32>,

    /// Maximum completion tokens to generate. Must be at least one. Enforced
    /// by the engine loop (#21), not by the internal `Sampler`.
    pub max_tokens: usize,

    /// If `true`, do not stop generation when the sampler returns a resolved
    /// EOS token. The `max_tokens` limit still applies. Enforced by the engine
    /// loop (#21).
    pub ignore_eos: bool,

    /// Additive presence penalty: subtract this from the logit of any token
    /// that has appeared at least once in the sequence. `0.0` = no-op
    /// (default). Must be finite in `[-2, 2]`; positive discourages repetition.
    pub presence_penalty: f32,

    /// Additive frequency penalty: subtract `frequency_penalty * count` from
    /// the logit of a token that has appeared `count` times. `0.0` = no-op
    /// (default). Must be finite in `[-2, 2]`; positive discourages repetition.
    pub frequency_penalty: f32,

    /// Multiplicative repetition penalty: divide the logit of any previously
    /// seen token by this value. `0.0` = no-op (default, treated as skip);
    /// positive values retain the accepted sampler meaning. Must be finite and
    /// non-negative. This matches Issue #17's `0 = no-op` convention.
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            max_tokens: 16,
            ignore_eos: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 0.0,
        }
    }
}

impl SamplingParams {
    /// Validate the complete public sampling contract before request admission.
    pub(crate) fn validate(&self) -> std::result::Result<(), SamplingParamsValidationError> {
        if self.temperature.is_nan() {
            return Err(SamplingParamsValidationError::new(
                "temperature",
                self.temperature,
                "must not be NaN",
            ));
        }
        if self.temperature < 0.0 {
            return Err(SamplingParamsValidationError::new(
                "temperature",
                self.temperature,
                "must be greater than or equal to 0",
            ));
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err(SamplingParamsValidationError::new(
                    "top_k",
                    top_k,
                    "must be at least 1 when set",
                ));
            }
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() {
                return Err(SamplingParamsValidationError::new(
                    "top_p",
                    top_p,
                    "must be finite",
                ));
            }
            if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
                return Err(SamplingParamsValidationError::new(
                    "top_p",
                    top_p,
                    "must be in (0, 1]",
                ));
            }
        }
        validate_additive_penalty("presence_penalty", self.presence_penalty)?;
        validate_additive_penalty("frequency_penalty", self.frequency_penalty)?;
        if !self.repetition_penalty.is_finite() {
            return Err(SamplingParamsValidationError::new(
                "repetition_penalty",
                self.repetition_penalty,
                "must be finite",
            ));
        }
        if self.repetition_penalty < 0.0 {
            return Err(SamplingParamsValidationError::new(
                "repetition_penalty",
                self.repetition_penalty,
                "must be greater than or equal to 0",
            ));
        }
        if self.max_tokens == 0 {
            return Err(SamplingParamsValidationError::new(
                "max_tokens",
                self.max_tokens,
                "must be at least 1",
            ));
        }
        Ok(())
    }

    /// Whether this params config triggers the greedy argmax short-circuit:
    /// `temperature == 0` OR `top_k == Some(1)`. Both are equivalent paths to
    /// deterministic argmax — listed separately in the issue #17 acceptance
    /// criteria so the property test suite can verify each independently.
    fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == Some(1)
    }

    /// Whether any of the three penalties is active (non-default). Skips the
    /// penalty pass when all are at no-op values — the typical greedy case
    /// greedy runs.
    fn has_penalties(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 0.0
    }
}

fn validate_additive_penalty(
    field: &'static str,
    value: f32,
) -> std::result::Result<(), SamplingParamsValidationError> {
    if !value.is_finite() {
        return Err(SamplingParamsValidationError::new(
            field,
            value,
            "must be finite",
        ));
    }
    if !(-2.0..=2.0).contains(&value) {
        return Err(SamplingParamsValidationError::new(
            field,
            value,
            "must be in [-2, 2]",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SamplingParamsValidationError {
    field: &'static str,
    value: String,
    reason: &'static str,
}

impl SamplingParamsValidationError {
    fn new(field: &'static str, value: impl fmt::Debug, reason: &'static str) -> Self {
        Self {
            field,
            value: format!("{value:?}"),
            reason,
        }
    }
}

impl fmt::Display for SamplingParamsValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            ".{}={} is invalid: {}",
            self.field, self.value, self.reason
        )
    }
}

impl std::error::Error for SamplingParamsValidationError {}

/// Single-step categorical sampler. Stateless across steps; holds only an RNG
/// for the Gumbel-max noise source.
///
/// Construct with [`Sampler::new_with_seed`] for deterministic property tests
/// and reproducible runs. The composition root (`llm.rs`, T6) will pick the
/// seed; for `LLM::generate`'s public API a process-entropy seed is also
/// acceptable.
pub struct Sampler {
    rng: rand::rngs::StdRng,
    #[cfg(feature = "cuda")]
    cuda_workspace: Option<cuda::Workspace>,
}

impl Sampler {
    /// Deterministic sampler. Same seed → same token sequence for the same
    /// logits + params. Mandatory for property tests (T8 Q8.3) and for
    /// reproducible golden runs.
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            #[cfg(feature = "cuda")]
            cuda_workspace: None,
        }
    }

    /// Sample one token per row of `logits` (shape `[batch, vocab_size]`).
    ///
    /// `params` carries one [`SamplingParams`] per batch row — per-prompt
    /// overrides in a single batch is the nano-vllm list-of-params parity
    /// (issue #17: "Takes a per-sequence temperature tensor … so batch-mates
    /// with different `SamplingParams.temperature` sample correctly in one
    /// pass").
    ///
    /// `token_history` is the per-row list of token ids generated *so far*
    /// (prompt + completion); only consulted when `params[i].has_penalties()`.
    /// Pass empty vecs when no penalties are active.
    ///
    /// Returns a `[batch]` tensor of `u32` token ids on the same device as
    /// `logits`. The dtype is `U32` because candle's `Tensor::argmax`
    /// returns `U32`.
    pub fn forward(
        &mut self,
        logits: &Tensor,
        params: &[SamplingParams],
        token_history: &[Vec<u32>],
    ) -> Result<Tensor> {
        let rank = logits.rank();
        if rank != 2 {
            return Err(CandleError::msg(format!(
                "sampler.forward: logits must be rank-2 [batch, vocab], got rank {rank}"
            )));
        }
        let batch = logits.dim(0)?;
        let vocab = logits.dim(1)?;
        if params.len() != batch {
            return Err(CandleError::msg(format!(
                "sampler.forward: params.len() = {} but batch = {batch}",
                params.len()
            )));
        }
        if token_history.len() != batch {
            return Err(CandleError::msg(format!(
                "sampler.forward: token_history.len() = {} but batch = {batch}",
                token_history.len()
            )));
        }

        if sampling_execution(logits.device().is_cuda()) == SamplingExecution::Device {
            return self.forward_cuda(logits, params, token_history, batch);
        }

        self.forward_host(logits, params, token_history, batch, vocab)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &mut self,
        logits: &Tensor,
        params: &[SamplingParams],
        token_history: &[Vec<u32>],
        batch: usize,
    ) -> Result<Tensor> {
        // Consume exactly one scalar seed per row, including greedy rows. A
        // batch-mate changing sampling mode therefore cannot shift another
        // row's random stream within this call. Sampling is transactional with
        // respect to RNG state: a failed preparation, allocation, or kernel
        // leaves the next retry on the same seeds.
        let mut pending_rng = self.rng.clone();
        let row_seeds = (0..batch)
            .map(|_| pending_rng.next_u64())
            .collect::<Vec<_>>();
        let selected = cuda::sample(
            logits,
            params,
            token_history,
            &row_seeds,
            &mut self.cuda_workspace,
        )?;
        self.rng = pending_rng;
        Ok(selected)
    }

    #[cfg(not(feature = "cuda"))]
    fn forward_cuda(
        &mut self,
        _logits: &Tensor,
        _params: &[SamplingParams],
        _token_history: &[Vec<u32>],
        _batch: usize,
    ) -> Result<Tensor> {
        Err(CandleError::msg(
            "sampler.forward: CUDA tensor requires the crate `cuda` feature",
        ))
    }

    fn forward_host(
        &mut self,
        logits: &Tensor,
        params: &[SamplingParams],
        token_history: &[Vec<u32>],
        batch: usize,
        vocab: usize,
    ) -> Result<Tensor> {
        let device = logits.device();
        let mut out_ids = Vec::with_capacity(batch);

        // Slicing one row at a time keeps batch independence provable: each
        // row's params and token_history only touch that row's logits slice.
        // The full-batch form (broadcast-div a per-row temperature tensor,
        // then a single multinomial) is the v0.2 optimisation once property
        // tests pin the contract.
        for row in 0..batch {
            let row_tensor = logits.get(row)?;
            let sampled =
                self.sample_row_host(&row_tensor, &params[row], &token_history[row], vocab)?;
            // Vocabulary size is bounded by u32::MAX in every realistic model
            // (Qwen3 vocab is ~150k); the cast preserves all valid token ids.
            #[allow(clippy::cast_possible_truncation)]
            out_ids.push(sampled as u32);
        }

        Tensor::from_vec(out_ids, batch, device)
    }

    /// Per-row pipeline: 1-D logits `[vocab]` → sampled token id.
    ///
    /// Order matches the issue #17 spec exactly: upcast → penalties → greedy
    /// short-circuit → temperature → top-k → top-p → sample. Greedy
    /// short-circuits skip the temperature/top-k/top-p/sample path entirely
    /// (they would all reduce to argmax anyway).
    ///
    /// The CPU adapter pulls the row to host (`to_vec1::<f32>()`) and keeps
    /// the independently validated reference implementation in pure Rust.
    /// CUDA never calls this method; its private adapter implements the same
    /// ordered pipeline in project kernels without a full-logit D2H copy.
    fn sample_row_host(
        &mut self,
        logits_1d: &Tensor,
        params: &SamplingParams,
        history: &[u32],
        vocab: usize,
    ) -> Result<usize> {
        // Step 1: upcast to F32 then pull to host. nano-vllm parity — BF16
        // logits lose precision under low-temperature softmax scaling.
        let logits_f32 = logits_1d.to_dtype(DType::F32)?;
        let mut buf = full_logits_to_host(&logits_f32)?;

        // Step 2: apply penalties (additive presence/frequency, multiplicative
        // repetition) BEFORE temperature scaling — issue #17 acceptance
        // criterion: "Penalties applied before temperature scaling".
        if params.has_penalties() && !history.is_empty() {
            apply_penalties(&mut buf, params, history, vocab);
        }

        // Step 3: greedy short-circuit. temperature=0 OR top_k=1 → argmax.
        // Both paths converge to the same deterministic answer; the property
        // test suite verifies each independently (T8 Q8.3 invariants 1+2).
        if params.is_greedy() {
            return Ok(argmax_buf(&buf));
        }

        // Step 4: scale by temperature. nano-vllm parity: divide (not
        // multiply by 1/T) so temperature = +inf → uniform-sample corner
        // case is preserved.
        let inv_t = 1.0_f32 / params.temperature;
        for v in buf.iter_mut() {
            *v *= inv_t;
        }

        // Step 5: top-k mask. Some(1) is handled by the greedy branch above.
        if let Some(k) = params.top_k {
            if k >= 1 && k < vocab {
                mask_top_k(&mut buf, k);
            }
        }

        // Step 6: top-p nucleus. Sort descending, walk cumulative softmax,
        // mask everything past the cutoff to -inf.
        if let Some(p) = params.top_p {
            if p > 0.0 && p < 1.0 {
                mask_top_p(&mut buf, p);
            }
        }

        // Step 7: Gumbel-max sample. argmax(logit + Gumbel(0,1)) is a draw
        // from softmax(logits). Same distribution as nano-vllm's
        // `probs / exponential_(1)` form (exponential Gumbel trick); we use
        // additive Gumbel because it composes cleanly with the host buffer.
        Ok(self.gumbel_argmax_buf(&buf))
    }

    /// Gumbel-max draw over a host-side logit buffer. Same RNG call order
    /// as a multinomial — stable across runs given the same seed.
    fn gumbel_argmax_buf(&mut self, logits: &[f32]) -> usize {
        // Gumbel(0,1) location 0, scale 1. The argmax of (logit + g) over
        // g ~ iid Gumbel(0,1) is a categorical draw with P(i) = softmax(logit)_i.
        let Ok(gumbel) = Gumbel::new(0.0, 1.0) else {
            // Gumbel::new only errors on non-positive scale; scale=1 is
            // statically valid. Fallback to plain argmax if construction
            // somehow fails.
            return argmax_buf(logits);
        };
        let mut best = (f32::NEG_INFINITY, 0_usize);
        for (i, &logit) in logits.iter().enumerate() {
            // Gumbel distribution samples f64; downcast to f32 to match the
            // logit buffer's dtype. Precision loss here is acceptable: the
            // Gumbel noise is itself noise, not a calibrated value.
            #[allow(clippy::cast_possible_truncation)]
            let g = gumbel.sample(&mut self.rng) as f32;
            let score = logit + g;
            if score > best.0 {
                best = (score, i);
            }
        }
        best.1
    }
}

/// Argmax of a host-side logit buffer. Ties pick the lowest index — matches
/// numpy/torch/HF tie-breaking (the `Iterator::max_by` builtin returns the
/// last-equal element instead, so the explicit loop below is load-bearing).
fn argmax_buf(buf: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in buf.iter().enumerate() {
        // Strict `>` keeps the first occurrence on ties — the load-bearing
        // detail vs `>=` (which would pick the last).
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Apply presence / frequency / repetition penalties in-place. All three
/// skipped if `history` empty (no tokens to penalise).
///
/// Math matches the HF Transformers convention:
/// - presence:   `logit[t] -= presence_penalty`             (once per unique t ∈ history)
/// - frequency:  `logit[t] -= frequency_penalty * count(t)` (once per unique t ∈ history)
/// - repetition: `logit[t] /= repetition_penalty`           (once per unique t ∈ history,
///   skipped when repetition_penalty == 0.0)
///
/// **Per-unique-token application is load-bearing**: a naive `for tok in
/// history` loop would apply presence/repetition N times and frequency
/// N²·count times for a token appearing N times. The regression test
/// `penalties::penalty_application_is_unique_per_token` pins this.
fn apply_penalties(buf: &mut [f32], params: &SamplingParams, history: &[u32], vocab: usize) {
    // Deduplicate history → per-token count. HashMap alloc is O(history) and
    // dominated by the matmul that produced these logits at any realistic
    // batch size; a sort+dedupe would also work but is mutation-heavy.
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &tok in history {
        if (tok as usize) < vocab {
            *counts.entry(tok).or_insert(0) += 1;
        }
    }

    for (&tok, &count) in &counts {
        let idx = tok as usize;
        if params.repetition_penalty != 0.0 {
            // Spec-default 0.0 = no-op (issue #17). Non-zero values divide:
            // > 1.0 discourages repetition, < 1.0 encourages it.
            buf[idx] /= params.repetition_penalty;
        }
        buf[idx] -= params.presence_penalty;
        if params.frequency_penalty != 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let count_f = count as f32;
            buf[idx] -= params.frequency_penalty * count_f;
        }
    }
}

/// Top-k mask: set every logit strictly below the k-th largest to -inf.
/// Ties at the k-th-largest threshold are kept (HF / nano-vllm convention).
///
/// Uses `select_nth_unstable_by` (O(n) partial sort) instead of a full
/// `sort_by` (O(n log n)) — the only value we need is the threshold itself.
fn mask_top_k(buf: &mut [f32], k: usize) {
    let mut idx: Vec<usize> = (0..buf.len()).collect();
    // Partition so idx[..k] are the top-k indices (unordered); idx[k-1] is
    // the k-th-largest value's index. Descending comparator (b vs a).
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        buf[b]
            .partial_cmp(&buf[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = buf[idx[k.saturating_sub(1)]];
    for v in buf.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (nucleus) mask: keep the smallest token set whose cumulative
/// softmax probability ≥ p. Sort descending, accumulate, mask everything
/// past the cutoff to -inf. Operates on the (already top-k-masked, already
/// temperature-scaled) logits.
///
/// Softmax computed in f64 for numerical headroom even though the buffer
/// is already F32 — matches the F32-upcast invariant's spirit.
fn mask_top_p(buf: &mut [f32], p: f32) {
    let mut idx: Vec<usize> = (0..buf.len()).collect();
    idx.sort_by(|&a, &b| {
        buf[b]
            .partial_cmp(&buf[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max = buf[idx[0]];
    let exp_sum: f64 = buf.iter().map(|&v| ((v - max) as f64).exp()).sum();

    // rank_of_token[t] = position of token t in the descending-sorted order.
    let mut rank_of_token = vec![0_usize; buf.len()];
    for (rank, &tok) in idx.iter().enumerate() {
        rank_of_token[tok] = rank;
    }

    let mut cumsum: f64 = 0.0;
    let mut cutoff = buf.len(); // keep all if p is large
    for (rank, &tok) in idx.iter().enumerate() {
        let prob = ((buf[tok] - max) as f64).exp() / exp_sum;
        cumsum += prob;
        if cumsum >= p as f64 {
            cutoff = rank + 1;
            break;
        }
    }

    for (tok, v) in buf.iter_mut().enumerate() {
        if rank_of_token[tok] >= cutoff {
            *v = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use candle_core::Device;

    mod host_transfer_seam {
        use super::*;

        #[test]
        fn legacy_sampler_records_one_full_vocabulary_host_transfer() {
            HOST_TRANSFERS.with(|transfers| transfers.borrow_mut().clear());

            let logits =
                Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 0.0], (1, 4), &Device::Cpu).unwrap();
            let mut sampler = Sampler::new_with_seed(0);
            sampler
                .forward(
                    &logits,
                    std::slice::from_ref(&SamplingParams::default()),
                    &[Vec::new()],
                )
                .unwrap();

            let observed = HOST_TRANSFERS.with(|transfers| transfers.borrow().clone());
            assert_eq!(
                observed,
                vec![HostTransfer::FullLogits { elements: 4 }],
                "the base sampler must expose its full-vocabulary host copy at the seam"
            );
        }

        #[test]
        fn cuda_dispatch_is_device_resident() {
            assert_eq!(
                sampling_execution(true),
                SamplingExecution::Device,
                "CUDA sampling must not enter the full-logits host path"
            );
        }

        #[test]
        fn selected_token_transfer_is_distinct_from_full_logits() {
            HOST_TRANSFERS.with(|transfers| transfers.borrow_mut().clear());

            let token_ids = Tensor::from_vec(vec![7_u32, 11], 2, &Device::Cpu).unwrap();
            assert_eq!(selected_token_ids_to_host(&token_ids).unwrap(), vec![7, 11]);

            let observed = HOST_TRANSFERS.with(|transfers| transfers.borrow().clone());
            assert_eq!(observed, vec![HostTransfer::SelectedTokens { elements: 2 }]);
        }

        #[test]
        fn validation_error_records_no_host_transfer() {
            HOST_TRANSFERS.with(|transfers| transfers.borrow_mut().clear());

            let logits =
                Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 0.0], (1, 4), &Device::Cpu).unwrap();
            let mut sampler = Sampler::new_with_seed(0);
            let error = sampler.forward(&logits, &[], &[Vec::new()]).unwrap_err();

            assert!(error.to_string().contains("params.len()"));
            let observed = HOST_TRANSFERS.with(|transfers| transfers.borrow().clone());
            assert!(observed.is_empty(), "error path must not report a transfer");
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda_device {
        use super::*;

        fn cuda() -> Device {
            Device::new_cuda(0).unwrap()
        }

        fn repeat_row(row: &[f32], draws: usize, device: &Device) -> Tensor {
            let values = (0..draws)
                .flat_map(|_| row.iter().copied())
                .collect::<Vec<_>>();
            Tensor::from_vec(values, (draws, row.len()), device).unwrap()
        }

        fn sample_repeated(
            row: &[f32],
            draws: usize,
            params: &SamplingParams,
            seed: u64,
            device: &Device,
        ) -> Vec<u32> {
            let logits = repeat_row(row, draws, device);
            let params = vec![params.clone(); draws];
            let histories = vec![Vec::new(); draws];
            let mut sampler = Sampler::new_with_seed(seed);
            let selected = sampler.forward(&logits, &params, &histories).unwrap();
            selected_token_ids_to_host(&selected).unwrap()
        }

        #[test]
        fn mixed_device_paths_are_isolated_and_transfer_only_selected_tokens() {
            HOST_TRANSFERS.with(|transfers| transfers.borrow_mut().clear());
            let device = cuda();
            let rows: [[f32; 8]; 11] = [
                [1.0, 3.0, 2.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [1.0, 3.0, 2.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [2.0, 1.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [2.0, 1.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [2.0, 1.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [10.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [0.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
                [2.0, 1.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [0.0, 0.5, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [0.0, 0.5, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                [0.4, 0.6, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            ];
            let logits = Tensor::from_vec(
                rows.into_iter().flatten().collect::<Vec<_>>(),
                (rows.len(), rows[0].len()),
                &device,
            )
            .unwrap();
            let params = vec![
                SamplingParams::default(),
                SamplingParams {
                    temperature: 1.0,
                    top_k: Some(1),
                    ..SamplingParams::default()
                },
                SamplingParams {
                    presence_penalty: 2.0,
                    ..SamplingParams::default()
                },
                SamplingParams {
                    frequency_penalty: 1.0,
                    ..SamplingParams::default()
                },
                SamplingParams {
                    repetition_penalty: 4.0,
                    ..SamplingParams::default()
                },
                SamplingParams {
                    temperature: 1.0,
                    top_p: Some(0.5),
                    ..SamplingParams::default()
                },
                SamplingParams {
                    temperature: 1.0,
                    top_k: Some(2),
                    ..SamplingParams::default()
                },
                SamplingParams {
                    repetition_penalty: 0.0,
                    ..SamplingParams::default()
                },
                SamplingParams {
                    presence_penalty: -1.0,
                    ..SamplingParams::default()
                },
                SamplingParams {
                    frequency_penalty: -0.4,
                    ..SamplingParams::default()
                },
                SamplingParams {
                    repetition_penalty: 0.5,
                    ..SamplingParams::default()
                },
            ];
            let histories = vec![
                vec![],
                vec![],
                vec![0],
                vec![0, 0],
                vec![0],
                vec![],
                vec![],
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![0],
                vec![0, 0],
                vec![0],
            ];
            let mut sampler = Sampler::new_with_seed(7);

            let selected = sampler.forward(&logits, &params, &histories).unwrap();
            assert!(
                HOST_TRANSFERS.with(|transfers| transfers.borrow().is_empty()),
                "device sampling must not transfer logits"
            );
            let workspace_identity = sampler
                .cuda_workspace
                .as_ref()
                .unwrap()
                .allocation_identity();
            assert_eq!(
                sampler.cuda_workspace.as_ref().unwrap().history_capacity(),
                8,
                "no-op penalty rows must not inflate or upload history workspace"
            );
            let second = sampler.forward(&logits, &params, &histories).unwrap();
            assert_eq!(
                sampler
                    .cuda_workspace
                    .as_ref()
                    .unwrap()
                    .allocation_identity(),
                workspace_identity,
                "same device/vocabulary/history capacity must reuse scratch"
            );
            drop(second);

            let ids = selected_token_ids_to_host(&selected).unwrap();
            assert_eq!(&ids[..6], &[1, 1, 1, 1, 1, 0]);
            assert!(matches!(ids[6], 1 | 2));
            assert_eq!(ids[7], 0, "repetition_penalty=0 must remain a no-op");
            assert_eq!(
                &ids[8..],
                &[0, 0, 0],
                "negative additive and sub-one repetition penalties must encourage the seen token"
            );
            let observed = HOST_TRANSFERS.with(|transfers| transfers.borrow().clone());
            assert_eq!(
                observed,
                vec![HostTransfer::SelectedTokens {
                    elements: rows.len()
                }]
            );

            HOST_TRANSFERS.with(|transfers| transfers.borrow_mut().clear());
            let invalid = SamplingParams {
                temperature: 1.0,
                top_p: Some(0.0),
                ..SamplingParams::default()
            };
            let error = sampler
                .forward(
                    &logits.get(0).unwrap().unsqueeze(0).unwrap(),
                    &[invalid],
                    &[vec![]],
                )
                .unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("sampling params at row 0.top_p=0.0"),
                "missing synchronous row context: {message}"
            );
            assert!(
                HOST_TRANSFERS.with(|transfers| transfers.borrow().is_empty()),
                "failed CUDA sampling must not fall back or transfer data"
            );

            for stage in [3, 11] {
                let observed = crate::sampler::cuda::injected_error_observation(stage).unwrap();
                assert!(observed.contains("error observed during"));
                assert!(
                    !observed.contains("row"),
                    "runtime/CUB error observations must never claim an origin row: {observed}"
                );
            }

            let retry_logits = logits.get(0).unwrap().unsqueeze(0).unwrap();
            let valid = SamplingParams {
                temperature: 1.0,
                ..SamplingParams::default()
            };
            let mut retry_sampler = Sampler::new_with_seed(2_718);
            retry_sampler
                .forward(
                    &retry_logits,
                    &[SamplingParams {
                        temperature: 1.0,
                        top_p: Some(0.0),
                        ..SamplingParams::default()
                    }],
                    &[vec![]],
                )
                .unwrap_err();
            let retried = retry_sampler
                .forward(&retry_logits, std::slice::from_ref(&valid), &[vec![]])
                .unwrap();
            let retried = selected_token_ids_to_host(&retried).unwrap();
            let mut clean_sampler = Sampler::new_with_seed(2_718);
            let clean = clean_sampler
                .forward(&retry_logits, std::slice::from_ref(&valid), &[vec![]])
                .unwrap();
            let clean = selected_token_ids_to_host(&clean).unwrap();
            assert_eq!(
                retried, clean,
                "a failed CUDA attempt must not consume the retry's row seed"
            );
        }

        #[test]
        fn seeded_device_sampling_is_deterministic_and_statistically_sane() {
            let device = cuda();
            let probabilities = [0.7_f32, 0.2, 0.1];
            let logits = probabilities.map(f32::ln);
            let params = SamplingParams {
                temperature: 1.0,
                ..SamplingParams::default()
            };
            let draws = 4_096;
            let first = sample_repeated(&logits, draws, &params, 99, &device);
            let replay = sample_repeated(&logits, draws, &params, 99, &device);
            assert_eq!(first, replay, "same seed and row order must replay exactly");

            let two_rows = repeat_row(&logits, 2, &device);
            let stochastic = params.clone();
            let mut first_sampler = Sampler::new_with_seed(101);
            let first_batch = first_sampler
                .forward(
                    &two_rows,
                    &[SamplingParams::default(), stochastic.clone()],
                    &[vec![], vec![]],
                )
                .unwrap();
            let first_batch = selected_token_ids_to_host(&first_batch).unwrap();
            let mut second_sampler = Sampler::new_with_seed(101);
            let second_batch = second_sampler
                .forward(
                    &two_rows,
                    &[
                        SamplingParams {
                            temperature: 1.0,
                            top_k: Some(2),
                            ..SamplingParams::default()
                        },
                        stochastic,
                    ],
                    &[vec![], vec![]],
                )
                .unwrap();
            let second_batch = selected_token_ids_to_host(&second_batch).unwrap();
            assert_eq!(
                first_batch[1], second_batch[1],
                "one row changing sampling mode must not shift another row's scalar seed"
            );

            let mut counts = [0_usize; 3];
            for token in first {
                counts[token as usize] += 1;
            }
            for (token, expected) in probabilities.into_iter().enumerate() {
                let observed = counts[token] as f64 / draws as f64;
                assert!(
                    (observed - f64::from(expected)).abs() < 0.05,
                    "token {token}: expected {expected}, observed {observed}"
                );
            }

            let nucleus_probabilities = [0.6_f32, 0.25, 0.1, 0.05];
            let nucleus_logits = nucleus_probabilities.map(f32::ln);
            let nucleus_params = SamplingParams {
                temperature: 1.0,
                top_p: Some(0.8),
                ..SamplingParams::default()
            };
            let nucleus = sample_repeated(&nucleus_logits, 2_048, &nucleus_params, 17, &device);
            assert!(nucleus.iter().all(|&token| matches!(token, 0 | 1)));
            assert!(nucleus.contains(&0) && nucleus.contains(&1));
            let token_zero_fraction =
                nucleus.iter().filter(|&&token| token == 0).count() as f64 / nucleus.len() as f64;
            assert!((token_zero_fraction - (0.6 / 0.85)).abs() < 0.07);

            let minimal_params = SamplingParams {
                temperature: 1.0,
                top_p: Some(0.59),
                ..SamplingParams::default()
            };
            let minimal = sample_repeated(&nucleus_logits, 256, &minimal_params, 18, &device);
            assert!(
                minimal.iter().all(|&token| token == 0),
                "minimal nucleus must stop after its first probability crosses p"
            );

            let tied = sample_repeated(
                &[0.0, 0.0, 0.0, -20.0],
                1_024,
                &SamplingParams {
                    temperature: 1.0,
                    top_k: Some(2),
                    ..SamplingParams::default()
                },
                23,
                &device,
            );
            assert!(tied.iter().all(|&token| token < 3));
            assert!((0_u32..3).all(|token| tied.contains(&token)));

            let uniform = sample_repeated(
                &[20.0, 2.0, -4.0, -30.0],
                2_048,
                &SamplingParams {
                    temperature: f32::INFINITY,
                    top_k: Some(usize::MAX),
                    ..SamplingParams::default()
                },
                29,
                &device,
            );
            let mut uniform_counts = [0_usize; 4];
            for token in uniform {
                uniform_counts[token as usize] += 1;
            }
            for (token, count) in uniform_counts.into_iter().enumerate() {
                let fraction = count as f64 / 2_048.0;
                assert!(
                    (fraction - 0.25).abs() < 0.06,
                    "uniform token {token}: observed {fraction}"
                );
            }
        }

        #[test]
        fn qwen_vocabulary_filtering_reuses_bounded_workspace_without_full_d2h() {
            const QWEN3_VOCAB_SIZE: usize = 151_936;
            HOST_TRANSFERS.with(|transfers| transfers.borrow_mut().clear());
            let device = cuda();
            let values = (0..QWEN3_VOCAB_SIZE)
                .map(|token| -(token as f32) * 0.0001)
                .collect::<Vec<_>>();
            let logits = Tensor::from_vec(values, (1, QWEN3_VOCAB_SIZE), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let params = SamplingParams {
                temperature: 0.7,
                top_k: Some(50),
                top_p: Some(0.9),
                ..SamplingParams::default()
            };
            let mut sampler = Sampler::new_with_seed(31);

            let selected = sampler
                .forward(&logits, std::slice::from_ref(&params), &[vec![]])
                .unwrap();
            let workspace = sampler.cuda_workspace.as_ref().unwrap();
            let identity = workspace.allocation_identity();
            assert!(workspace.allocated_bytes() < 64 * 1024 * 1024);
            assert!(
                HOST_TRANSFERS.with(|transfers| transfers.borrow().is_empty()),
                "Qwen-sized filtering must remain entirely on device"
            );
            let token = selected_token_ids_to_host(&selected).unwrap()[0];
            assert!(token < 50, "top-k result {token} escaped the retained set");

            drop(
                sampler
                    .forward(&logits, std::slice::from_ref(&params), &[vec![]])
                    .unwrap(),
            );
            assert_eq!(
                sampler
                    .cuda_workspace
                    .as_ref()
                    .unwrap()
                    .allocation_identity(),
                identity
            );
            assert_eq!(
                HOST_TRANSFERS.with(|transfers| transfers.borrow().clone()),
                vec![HostTransfer::SelectedTokens { elements: 1 }]
            );
        }
    }

    mod sampling_params {
        use super::*;

        #[test]
        fn default_is_greedy() {
            let p = SamplingParams::default();
            assert!(p.is_greedy(), "default must short-circuit to argmax");
            assert_eq!(p.temperature, 0.0);
            assert_eq!(p.top_k, None);
            assert_eq!(p.max_tokens, 16);
            assert!(!p.ignore_eos);
            assert!(!p.has_penalties());
        }

        #[test]
        fn top_k_one_is_greedy_even_at_nonzero_temperature() {
            let p = SamplingParams {
                temperature: 1.0,
                top_k: Some(1),
                ..SamplingParams::default()
            };
            assert!(p.is_greedy());
        }

        #[test]
        fn temperature_zero_is_greedy_even_with_large_top_k() {
            let p = SamplingParams {
                temperature: 0.0,
                top_k: Some(100),
                ..SamplingParams::default()
            };
            assert!(p.is_greedy());
        }

        #[test]
        fn nonzero_temperature_with_top_k_gt_one_is_not_greedy() {
            let p = SamplingParams {
                temperature: 0.7,
                top_k: Some(40),
                ..SamplingParams::default()
            };
            assert!(!p.is_greedy());
        }
    }

    // T8 Q8.3 invariant 1+2: temp=0 ≡ greedy; top-k=1 ≡ greedy.
    mod greedy_equiv {
        use super::*;

        fn logits(batch: usize, vocab: usize, device: &Device) -> Tensor {
            let data: Vec<f32> = (0..batch * vocab).map(|i| (i as f32) * 0.1 - 5.0).collect();
            Tensor::from_vec(data, (batch, vocab), device).unwrap()
        }

        #[test]
        fn temp_zero_matches_argmax() {
            let device = Device::Cpu;
            let l = logits(3, 16, &device);
            let params = vec![SamplingParams::default(); 3];
            let history = vec![Vec::new(); 3];
            let mut sampler = Sampler::new_with_seed(0);

            let sampled = sampler.forward(&l, &params, &history).unwrap();
            let ids = sampled.to_vec1::<u32>().unwrap();

            for row in 0..3 {
                let row_l = l.get(row).unwrap();
                let row_f32 = row_l.to_dtype(DType::F32).unwrap();
                let buf = row_f32.to_vec1::<f32>().unwrap();
                let expected = argmax_buf(&buf) as u32;
                assert_eq!(ids[row], expected, "row {row}: temp=0 must equal argmax");
            }
        }

        #[test]
        fn top_k_one_matches_argmax() {
            let device = Device::Cpu;
            let l = logits(3, 16, &device);
            // top_k=1 with nonzero temperature must still hit the greedy
            // short-circuit per issue #17 acceptance criterion.
            let params = vec![
                SamplingParams {
                    temperature: 0.99,
                    top_k: Some(1),
                    ..SamplingParams::default()
                };
                3
            ];
            let history = vec![Vec::new(); 3];
            let mut sampler = Sampler::new_with_seed(0);

            let sampled = sampler.forward(&l, &params, &history).unwrap();
            let ids = sampled.to_vec1::<u32>().unwrap();

            for row in 0..3 {
                let row_l = l.get(row).unwrap();
                let row_f32 = row_l.to_dtype(DType::F32).unwrap();
                let buf = row_f32.to_vec1::<f32>().unwrap();
                let expected = argmax_buf(&buf) as u32;
                assert_eq!(
                    ids[row], expected,
                    "row {row}: top_k=1 must equal argmax even at nonzero temp"
                );
            }
        }

        #[test]
        fn temp_zero_and_top_k_one_agree() {
            let device = Device::Cpu;
            let l = logits(2, 8, &device);
            let history = vec![Vec::new(); 2];

            let mut s_a = Sampler::new_with_seed(1);
            let p_a = vec![SamplingParams::default(); 2];
            let a = s_a
                .forward(&l, &p_a, &history)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();

            let mut s_b = Sampler::new_with_seed(1);
            let p_b = vec![
                SamplingParams {
                    temperature: 1.0,
                    top_k: Some(1),
                    ..SamplingParams::default()
                };
                2
            ];
            let b = s_b
                .forward(&l, &p_b, &history)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();

            assert_eq!(a, b, "temp=0 and top_k=1 must produce identical tokens");
        }
    }

    // T8 Q8.3 invariant 3: sampled ∈ top-k when set.
    mod topk_membership {
        use super::*;
        use std::collections::HashSet;

        #[test]
        fn sampled_token_is_always_within_top_k() {
            let device = Device::Cpu;
            let vocab = 32;
            // Top-3 set is {4,3,2} (logits 14,13,12 — the three largest).
            let data: Vec<f32> = (0..vocab)
                .map(|i| {
                    if i < 5 {
                        (i as f32) + 10.0
                    } else {
                        -(i as f32)
                    }
                })
                .collect();
            let row_data: Vec<f32> = data.clone();
            let logits = Tensor::from_vec(data, (1, vocab), &device).unwrap();

            // Ground-truth top-3 set computed directly from raw logits.
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.select_nth_unstable_by(2, |&a, &b| {
                row_data[b]
                    .partial_cmp(&row_data[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let allowed: HashSet<u32> = idx[..3].iter().map(|&i| i as u32).collect();

            let params = SamplingParams {
                temperature: 1.0,
                top_k: Some(3),
                ..SamplingParams::default()
            };

            let mut sampler = Sampler::new_with_seed(42);
            for _ in 0..200 {
                let out = sampler
                    .forward(&logits, std::slice::from_ref(&params), &[Vec::new()])
                    .unwrap();
                let tok = out.to_vec1::<u32>().unwrap()[0];
                assert!(
                    allowed.contains(&tok),
                    "sampled token {tok} not in top-3 {allowed:?}"
                );
            }
        }
    }

    // T8 Q8.3 invariant 6: batch independence.
    mod batch_independence {
        use super::*;

        #[test]
        fn one_row_params_do_not_affect_another() {
            let device = Device::Cpu;
            let vocab = 16;
            let row: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.5).collect();
            let both: Vec<f32> = row.iter().copied().chain(row.iter().copied()).collect();
            let logits = Tensor::from_vec(both, (2, vocab), &device).unwrap();

            let history = vec![Vec::new(); 2];

            let mut s1 = Sampler::new_with_seed(7);
            let p1 = vec![
                SamplingParams::default(),
                SamplingParams {
                    temperature: 1.5,
                    top_k: Some(5),
                    ..SamplingParams::default()
                },
            ];
            let r1 = s1
                .forward(&logits, &p1, &history)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();

            let mut s2 = Sampler::new_with_seed(7);
            let p2 = vec![
                SamplingParams::default(),
                SamplingParams {
                    temperature: 0.5,
                    top_k: None,
                    top_p: Some(0.9),
                    ..SamplingParams::default()
                },
            ];
            let r2 = s2
                .forward(&logits, &p2, &history)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();

            assert_eq!(
                r1[0], r2[0],
                "row 0 (greedy) must not depend on row 1's params"
            );
        }

        #[test]
        fn row_result_matches_single_row_call() {
            // Stronger form: a row in a batch must equal that row sampled
            // alone, given the same RNG seed. Catches leaky sampler state.
            let device = Device::Cpu;
            let vocab = 24;
            let row_a: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.3).collect();
            let row_b: Vec<f32> = (0..vocab).map(|i| -(i as f32) * 0.2).collect();
            let batch_data: Vec<f32> = row_a.iter().chain(row_b.iter()).copied().collect();
            let batch = Tensor::from_vec(batch_data, (2, vocab), &device).unwrap();
            let params = vec![
                SamplingParams {
                    temperature: 1.0,
                    top_k: Some(8),
                    ..SamplingParams::default()
                },
                SamplingParams {
                    temperature: 0.8,
                    top_k: Some(4),
                    ..SamplingParams::default()
                },
            ];

            let mut sampler = Sampler::new_with_seed(99);
            let batched = sampler
                .forward(&batch, &params, &[Vec::new(), Vec::new()])
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();

            let single = Tensor::from_vec(row_a, (1, vocab), &device).unwrap();
            let mut sampler2 = Sampler::new_with_seed(99);
            let solo = sampler2
                .forward(&single, std::slice::from_ref(&params[0]), &[Vec::new()])
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();

            assert_eq!(batched[0], solo[0]);
        }
    }

    // T8 Q8.3 invariant 4: EOS argmax respected.
    mod eos_argmax {
        use super::*;

        #[test]
        fn sampler_returns_eos_when_eos_is_argmax() {
            // The engine loop (#21) is what stops generation on EOS. The
            // sampler-level contract is "if EOS is the argmax of the
            // logits, sampler returns EOS" — the loop above then interprets
            // it. A row whose argmax is at EOS-token-index must sample EOS.
            let device = Device::Cpu;
            let vocab = 8;
            let eos_id = 7_u32;
            let mut row = vec![-5.0_f32; vocab];
            row[eos_id as usize] = 10.0;

            let logits = Tensor::from_vec(row, (1, vocab), &device).unwrap();
            let mut sampler = Sampler::new_with_seed(0);
            let out = sampler
                .forward(
                    &logits,
                    std::slice::from_ref(&SamplingParams::default()),
                    &[Vec::new()],
                )
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(out[0], eos_id);
        }
    }

    // T8 Q8.3: FP32 upcast invariant.
    mod fp32_upcast {
        use super::*;

        #[test]
        fn bf16_input_uses_f32_argmax() {
            // Issue #17: "Upcasts logits to FP32 before scaling (nano-vllm
            // parity — low-temperature BF16 loses precision otherwise)".
            // Pick logits where BF16 rounding flips the argmax: adjacent
            // top values differing by less than BF16 epsilon (~0.008 at
            // this scale).
            let device = Device::Cpu;
            let vocab = 16;
            let mut data = vec![-1.0_f32; vocab];
            data[3] = 0.0_f32;
            data[4] = 0.001_f32;
            let f32_logits = Tensor::from_vec(data, (1, vocab), &device).unwrap();
            let bf16_logits = f32_logits.to_dtype(DType::BF16).unwrap();

            let mut sampler = Sampler::new_with_seed(0);
            let out = sampler
                .forward(
                    &bf16_logits,
                    std::slice::from_ref(&SamplingParams::default()),
                    &[Vec::new()],
                )
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(out[0], 4, "BF16 input must use F32-upcast argmax");
        }
    }

    // Penalties: presence, frequency, repetition correctness.
    mod penalties {
        use super::*;

        #[test]
        fn presence_penalty_suppresses_seen_token() {
            // argmax = 0; presence_penalty > 0 with token 0 in history must
            // suppress it enough to flip argmax to the runner-up (1).
            let device = Device::Cpu;
            let vocab = 4;
            let row = vec![2.0_f32, 1.0, 0.0, -1.0];
            let logits = Tensor::from_vec(row, (1, vocab), &device).unwrap();

            let params = SamplingParams {
                presence_penalty: 5.0,
                ..SamplingParams::default()
            };
            let mut sampler = Sampler::new_with_seed(0);
            let out = sampler
                .forward(&logits, std::slice::from_ref(&params), &[vec![0]])
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(out[0], 1, "presence penalty must flip argmax 0 → 1");
        }

        #[test]
        fn repetition_penalty_above_one_suppresses_seen() {
            let device = Device::Cpu;
            let vocab = 4;
            let row = vec![2.0_f32, 1.0, 0.0, -1.0];
            let logits = Tensor::from_vec(row, (1, vocab), &device).unwrap();

            // repetition_penalty > 1.0 divides the seen-token's logit,
            // pushing argmax off it.
            let params = SamplingParams {
                repetition_penalty: 4.0,
                ..SamplingParams::default()
            };
            let mut sampler = Sampler::new_with_seed(0);
            let out = sampler
                .forward(&logits, std::slice::from_ref(&params), &[vec![0]])
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(out[0], 1, "repetition_penalty > 1 must suppress seen token");
        }

        #[test]
        fn no_penalties_when_history_empty() {
            // Defensive: non-zero penalties with empty history is a no-op
            // (no tokens to penalise).
            let device = Device::Cpu;
            let vocab = 4;
            let row = vec![2.0_f32, 1.0, 0.0, -1.0];
            let logits = Tensor::from_vec(row, (1, vocab), &device).unwrap();

            let params = SamplingParams {
                presence_penalty: 100.0,
                frequency_penalty: 100.0,
                repetition_penalty: 100.0,
                ..SamplingParams::default()
            };
            let mut sampler = Sampler::new_with_seed(0);
            let out = sampler
                .forward(&logits, std::slice::from_ref(&params), &[Vec::new()])
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(out[0], 0, "empty history must short-circuit penalties");
        }

        #[test]
        fn penalty_application_is_unique_per_token() {
            // Regression: pre-fix bug applied presence/repetition N times and
            // frequency N²·count times for a token appearing N times in
            // history. After dedupe: presence/repetition once, freq·count once.
            // Direct test of `apply_penalties` since the bug only manifests
            // for repeated tokens (forward-level tests use count=1).
            let vocab = 4;
            let params = SamplingParams {
                presence_penalty: 1.0,
                frequency_penalty: 1.0,
                repetition_penalty: 2.0,
                ..SamplingParams::default()
            };
            let history = vec![0_u32, 0, 0]; // token 0 appears 3×
            let mut buf = vec![3.0_f32, 2.0, 1.0, 0.0];
            apply_penalties(&mut buf, &params, &history, vocab);

            // Correct math: (3.0 / 2) − 1.0 − (1.0 × 3) = 1.5 − 1 − 3 = −2.5
            // Pre-fix bug gave: (3.0 / 2³) − 3×1 − 9 = 0.375 − 12 = −11.625
            assert!(
                (buf[0] - (-2.5_f32)).abs() < 1e-5,
                "expected -2.5 (deduped), got {} (over-applied?)",
                buf[0]
            );
            // Untouched tokens stay at their original logits.
            assert_eq!(buf[1], 2.0);
            assert_eq!(buf[2], 1.0);
            assert_eq!(buf[3], 0.0);
        }

        #[test]
        fn presence_only_scales_with_unique_count_not_occurrences() {
            // Stronger regression: vary N occurrences of token 0; the penalty
            // delta vs no-penalty baseline must be exactly `presence` (1.0),
            // not N×presence.
            let vocab = 4;
            for n in [1_usize, 2, 3, 5, 10] {
                let params = SamplingParams {
                    presence_penalty: 1.0,
                    ..SamplingParams::default()
                };
                let history = vec![0_u32; n];
                let mut buf = vec![3.0_f32, 2.0, 1.0, 0.0];
                apply_penalties(&mut buf, &params, &history, vocab);
                assert!(
                    (buf[0] - 2.0_f32).abs() < 1e-5,
                    "n={n}: expected 3.0 − 1.0 = 2.0, got {} (over-applied?)",
                    buf[0]
                );
            }
        }
    }

    // Direct unit tests for the per-row mask/sort helpers. The forward-level
    // tests above exercise these transitively, but the helpers have sharp
    // edges (threshold ties, nucleus cutoff, descending comparator) where a
    // direct assertion is much easier to read than reconstructing it through
    // `forward`.
    mod mask_helpers {
        use super::*;

        #[test]
        fn argmax_buf_picks_lowest_index_on_ties() {
            assert_eq!(argmax_buf(&[1.0, 1.0, 1.0, 1.0]), 0);
            assert_eq!(argmax_buf(&[-1.0, 5.0, 5.0, -2.0]), 1);
            assert_eq!(argmax_buf(&[3.0]), 0);
        }

        #[test]
        fn mask_top_k_keeps_ties_at_threshold() {
            // Two tokens tied at the threshold value (1.0) both stay when k=3
            // — HF / nano-vllm convention: top-k is "≥ k-th largest", not
            // "exactly k tokens".
            let mut buf = vec![5.0_f32, 1.0, 1.0, -5.0, -10.0];
            mask_top_k(&mut buf, 3);
            // Threshold = 3rd-largest = 1.0. Indices 0, 1, 2 (≥ 1.0) stay;
            // 3, 4 (−5, −10) get masked.
            assert_eq!(buf[0], 5.0);
            assert_eq!(buf[1], 1.0);
            assert_eq!(buf[2], 1.0);
            assert!(buf[3].is_infinite() && buf[3].is_sign_negative());
            assert!(buf[4].is_infinite() && buf[4].is_sign_negative());
        }

        #[test]
        fn mask_top_k_smallest_k_is_identity() {
            let mut buf = vec![1.0_f32, 2.0, 3.0];
            mask_top_k(&mut buf, 1);
            // k=1 keeps only the maximum (3.0); everything else → -inf.
            assert!(buf[0].is_infinite() && buf[0].is_sign_negative());
            assert!(buf[1].is_infinite() && buf[1].is_sign_negative());
            assert_eq!(buf[2], 3.0);
        }

        #[test]
        fn mask_top_p_keeps_nucleus_only() {
            // Logits [10, 0, 0, 0] → softmax ≈ [0.9999, 1/3e-4, …].
            // p=0.9 keeps only index 0.
            let mut buf = vec![10.0_f32, 0.0, 0.0, 0.0];
            mask_top_p(&mut buf, 0.9);
            assert_eq!(buf[0], 10.0);
            for v in &buf[1..] {
                assert!(v.is_infinite() && v.is_sign_negative());
            }
        }

        #[test]
        fn mask_top_p_one_is_noop_branch() {
            // p=1.0 is filtered out at the call site (`p < 1.0` guard), but
            // verify the helper doesn't blow up if invoked directly: keep all.
            let mut buf = vec![1.0_f32, 2.0, 3.0, 4.0];
            mask_top_p(&mut buf, 1.0);
            // p=1.0 means cumsum reaches 1.0 at the last token, cutoff = all.
            assert_eq!(buf, vec![1.0_f32, 2.0, 3.0, 4.0]);
        }
    }

    // Cross-reference for the 6th T8 invariant (max_tokens boundary). The
    // sampler is single-step (one token per `forward` call) and stateless
    // across steps, so it cannot enforce a token-count boundary itself —
    // that contract lives in EngineCore.step() (#21 / `engine/mod.rs`),
    // which counts `step()` returns and stops calling `forward` once a
    // sequence's sampled-token count reaches `params.max_tokens`. The #21
    // engine test `engine::tests::max_tokens_boundary_stops_generation`
    // pins this; until #21 lands, the field is exercised here only
    // through `SamplingParams::default().max_tokens == 16`.
    mod max_tokens_boundary {
        use super::*;

        #[test]
        fn default_max_tokens_is_set() {
            // Smoke check: the field travels with SamplingParams and has a
            // sane default. Boundary enforcement is engine-level (#21) — see
            // module doc comment above.
            assert_eq!(SamplingParams::default().max_tokens, 16);
            assert_eq!(
                SamplingParams {
                    max_tokens: 4096,
                    ..SamplingParams::default()
                }
                .max_tokens,
                4096
            );
        }
    }
}
