//! Composition root for the contracted v0.2.0 generation interface
//! (ADR-0004, ADR-0011).
//!
//! The ONLY module that simultaneously imports `engine`, `models::registry`,
//! `loader`, `sampler`, and `attention`. Port of nano-vllm `llm.py` /
//! `llm_engine.py`.

mod initialization;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer as HFTokenizer;

use crate::attention::PagedKVCache;
use crate::config::Source;
use crate::engine::{
    scheduler::{
        DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS,
    },
    EngineCore, KvCacheManager, RequestOutput, Scheduler,
};
use crate::loader::model_identity::ResolvedModel;
use crate::models::registry::{resolved_factory, BuiltModel};
use crate::sampler::{Sampler, SamplingParams};

use initialization::initialize_model;

#[cfg(feature = "internal-golden")]
use crate::golden_capture::{BenchmarkSession, CaptureSession};

/// Construction-time configuration for `LLM::new`.
/// Mirrors the supported single-GPU subset of nano-vllm's `Config`.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// Maximum number of tokens processed in one prefill step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of concurrently-running sequences.
    pub max_num_seqs: usize,
    /// Maximum model context length (prompt + completion).
    pub max_model_len: usize,
    /// Fraction of free GPU memory to allocate to the KV cache pool (0.0–1.0).
    pub gpu_memory_utilization: f32,
    /// Must remain `true`; CUDA Graphs are outside the v0.2.0 boundary.
    pub enforce_eager: bool,
    /// Override the dtype read from `config.json`'s `torch_dtype`. `None`
    /// means "use the checkpoint's dtype" (BF16 for Qwen3).
    pub dtype: Option<DType>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_model_len: 4096,
            gpu_memory_utilization: DEFAULT_GPU_MEMORY_UTILIZATION,
            enforce_eager: true,
            dtype: None,
        }
    }
}

/// Input prompt for `LLM::generate`. Supports both natural-language text
/// and pre-tokenized fixtures (useful for golden comparison tests).
#[derive(Debug, Clone)]
pub enum Prompt {
    Text(String),
    TokenIds(Vec<u32>),
}

/// The composition root: owns the engine core, tokenizer, and shared
/// `PagedKVCache`. Constructed via `LLM::new`; generate via `LLM::generate`.
pub struct LLM {
    engine: EngineCore,
    tokenizer: HFTokenizer,
    _resolved_model: ResolvedModel,
    _paged_kv: Arc<Mutex<PagedKVCache>>,
    device: Device,
    max_model_len: usize,
}

fn build_resolved_model(
    resolved_model: &ResolvedModel,
    device: &Device,
    max_model_len: usize,
) -> Result<BuiltModel> {
    let model_factory = resolved_factory(resolved_model.config_json())?;
    model_factory(resolved_model, device, max_model_len)
}

impl LLM {
    /// Build the full inference stack — model, tokenizer, engine — from a
    /// model source (`Local` directory or `Hub` repo) and `EngineOptions`.
    ///
    /// Side effects: resolves architecture via registry, loads weights, sizes
    /// and allocates one KV cache from resolved geometry and remaining free GPU
    /// memory, runs representative prefill and paged-decode forwards with logits,
    /// and validates the CUDA device compute capability (≥ sm_89). No request is
    /// admitted until all initialization work succeeds.
    pub fn new(model: impl Into<Source>, options: EngineOptions) -> Result<Self> {
        let source: Source = model.into();

        let device = Device::cuda_if_available(0).map_err(|e| {
            anyhow!("CUDA device unavailable: {e}. vllm-oxide v0.2.0 requires one CUDA GPU.")
        })?;
        if !device.is_cuda() {
            bail!("vllm-oxide v0.2.0 requires one CUDA GPU. CPU-only inference is not supported.");
        }

        #[cfg(feature = "cuda")]
        validate_sm_version(&device)?;

        let resolved_model = ResolvedModel::resolve(source, options.dtype)?;
        let config_bytes = resolved_model.config_json();

        let max_pos = read_max_position_embeddings(config_bytes)?;
        if options.max_model_len > max_pos {
            bail!(
                "max_model_len ({}) exceeds the model's max_position_embeddings ({}) — \
                 RoPE positions would be out of range",
                options.max_model_len,
                max_pos,
            );
        }

        let dtype = resolved_model.dtype();

        tracing::info!(?dtype, identity = %resolved_model.identity(), "loading model");

        let tokenizer = resolved_model.load_tokenizer()?;
        let special_token_ids = resolved_model.special_token_ids(&tokenizer)?;
        tracing::debug!(
            config = %resolved_model.config_path().display(),
            tokenizer = %resolved_model.tokenizer_path().display(),
            bos_token_id = ?special_token_ids.bos(),
            pad_token_id = ?special_token_ids.pad(),
            "validated resolved model artifact contract"
        );

        let BuiltModel {
            mut model,
            attn_ctx,
        } = build_resolved_model(&resolved_model, &device, options.max_model_len)?;

        let cache_allocation = initialize_model(model.as_mut(), &attn_ctx, &device, &options)?;
        let num_gpu_blocks = cache_allocation.num_blocks;

        tracing::info!(
            free_mb = cache_allocation.free_bytes / (1024 * 1024),
            total_mb = cache_allocation.total_bytes / (1024 * 1024),
            kv_pool_mb = cache_allocation.pool_bytes / (1024 * 1024),
            bytes_per_block = cache_allocation.bytes_per_block,
            num_gpu_blocks,
            warmup_prefill_tokens = cache_allocation.warmup_prefill_tokens,
            "allocated and warmed KV cache pool"
        );

        let scheduler = Scheduler::new_with_eos_token_ids(
            options.max_num_batched_tokens,
            options.max_num_seqs,
            options.gpu_memory_utilization,
            special_token_ids.eos().to_vec(),
        );

        let kv_cache_manager = KvCacheManager::new(num_gpu_blocks, 256, attn_ctx.paged_kv.clone());

        let sampler = Sampler::new_with_seed(0);

        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            sampler,
            attn_ctx.clone(),
            device.clone(),
        );

        tracing::info!(
            vocab_size = tokenizer.get_vocab_size(true),
            "tokenizer loaded"
        );

        Ok(Self {
            engine,
            tokenizer,
            _resolved_model: resolved_model,
            _paged_kv: attn_ctx.paged_kv,
            device,
            max_model_len: options.max_model_len,
        })
    }

    /// Run inference on a batch of prompts with per-prompt sampling parameters.
    ///
    /// Text prompts (`Prompt::Text`) are tokenized using the loaded tokenizer;
    /// pre-tokenized prompts (`Prompt::TokenIds`) are used directly. All
    /// prompts in the batch are continuous-batched together.
    ///
    /// Returns one `RequestOutput` per prompt, in the same order, with both
    /// decoded text and raw token IDs.
    pub fn generate(
        &mut self,
        prompts: &[Prompt],
        sampling_params: &[SamplingParams],
    ) -> Result<Vec<RequestOutput>> {
        if sampling_params.len() != prompts.len() {
            bail!(
                "generate: expected {} sampling_params, got {}",
                prompts.len(),
                sampling_params.len(),
            );
        }
        for (batch_position, params) in sampling_params.iter().enumerate() {
            params
                .validate()
                .map_err(|error| anyhow!("generate: sampling_params[{batch_position}]{error}"))?;
        }
        #[cfg(feature = "internal-golden")]
        let mut capture = CaptureSession::from_env()
            .context("generate: invalid internal golden capture configuration")?;
        #[cfg(feature = "internal-golden")]
        let mut benchmark = BenchmarkSession::from_env()
            .context("generate: invalid internal benchmark telemetry configuration")?;
        #[cfg(feature = "internal-golden")]
        if capture.is_some() && benchmark.is_some() {
            bail!("generate: raw-logit capture and benchmark telemetry are mutually exclusive");
        }
        #[cfg(feature = "internal-golden")]
        if std::env::var_os(crate::golden_capture::behavior::PLAN_ENV).is_some() {
            if !prompts.is_empty()
                || capture.is_some()
                || benchmark.is_some()
                || crate::golden_capture::operators::requested()
                || std::env::var_os(crate::golden_capture::fixed_prefix::PLAN_ENV).is_some()
            {
                bail!("behavior verification requires an isolated unforced empty generate call");
            }
            crate::golden_capture::behavior::run_from_env(self)?;
            return Ok(Vec::new());
        }
        #[cfg(feature = "internal-golden")]
        if crate::golden_capture::operators::requested() {
            if !prompts.is_empty()
                || capture.is_some()
                || benchmark.is_some()
                || std::env::var_os(crate::golden_capture::fixed_prefix::PLAN_ENV).is_some()
            {
                bail!("operator verification requires an isolated empty generate call");
            }
            crate::golden_capture::operators::run_from_env(&self.device)?;
            return Ok(Vec::new());
        }
        if prompts.is_empty() {
            #[cfg(feature = "internal-golden")]
            if benchmark.is_some() {
                bail!("generate: benchmark telemetry requires at least one prompt");
            }
            #[cfg(feature = "internal-golden")]
            if let Some(capture) = capture.as_mut() {
                capture.bind_requests(&[])?;
            }
            #[cfg(feature = "internal-golden")]
            if let Some(capture) = capture.take() {
                capture.finish(&[])?;
            }
            return Ok(Vec::new());
        }

        let tokenized_prompts = prompts
            .iter()
            .map(|prompt| tokenize_prompt(prompt, &self.tokenizer))
            .collect::<Result<Vec<_>>>()?;
        for (position, (tokens, params)) in
            tokenized_prompts.iter().zip(sampling_params).enumerate()
        {
            if tokens.is_empty() {
                bail!("generate: prompt[{position}] must not be empty");
            }
            if tokens
                .len()
                .checked_add(params.max_tokens)
                .map_or(true, |length| length > self.max_model_len)
            {
                bail!(
                    "generate: prompt[{position}] context budget exceeds max_model_len {}",
                    self.max_model_len
                );
            }
        }
        #[cfg(feature = "internal-golden")]
        let mut replay = crate::golden_capture::fixed_prefix::ReplaySession::from_env(
            &tokenized_prompts,
            sampling_params,
        )?;
        #[cfg(feature = "internal-golden")]
        if replay.is_some() && (capture.is_some() || benchmark.is_some()) {
            bail!("fixed-prefix, legacy capture and benchmark modes are mutually exclusive");
        }
        #[cfg(feature = "internal-golden")]
        let replay_cache_blocks = if replay.is_some() {
            self._paged_kv
                .lock()
                .map_err(|error| anyhow!("cache identity: {error}"))?
                .num_blocks()
        } else {
            0
        };
        let prompt_lens = tokenized_prompts.iter().map(Vec::len).collect::<Vec<_>>();
        let mut request_ids = Vec::with_capacity(prompts.len());
        #[cfg(feature = "internal-golden")]
        let admission = Instant::now();
        for (token_ids, params) in tokenized_prompts.into_iter().zip(sampling_params.iter()) {
            let request_id = self.engine.add_request(token_ids, params.clone());
            request_ids.push(request_id);
        }
        #[cfg(feature = "internal-golden")]
        if let Err(error) = crate::golden_capture::behavior::record_binding(
            &request_ids,
            &prompt_lens,
            self.engine.scheduler.diagnostic_eos_token_ids(),
            self.max_model_len,
            &self.device,
        ) {
            return Err(self.abort_after_capture_error(error));
        }
        #[cfg(feature = "internal-golden")]
        if let Some(replay) = replay.as_mut() {
            if let Err(error) = replay.bind_requests(
                &request_ids,
                replay_cache_blocks,
                self.engine.scheduler.diagnostic_eos_token_ids(),
                self.max_model_len,
                &self.device,
            ) {
                return Err(self.abort_after_capture_error(error));
            }
        }
        #[cfg(feature = "internal-golden")]
        if let Some(capture) = capture.as_mut() {
            if let Err(error) = capture.bind_requests(&request_ids) {
                return Err(self.abort_after_capture_error(error));
            }
        }
        #[cfg(feature = "internal-golden")]
        if let Some(benchmark) = benchmark.as_mut() {
            if let Err(error) = benchmark.bind_requests(&request_ids) {
                return Err(self.abort_after_capture_error(error));
            }
        }

        let start = Instant::now();
        let mut step_count: usize = 0;

        let mut completed_outputs = Vec::with_capacity(prompts.len());

        while self.engine.is_running() {
            #[cfg(feature = "internal-golden")]
            if capture.is_some() || benchmark.is_some() || replay.is_some() {
                if let Err(error) = require_diagnostic_host_ram_floor() {
                    return Err(self.abort_after_capture_error(error));
                }
            }
            #[cfg(feature = "internal-golden")]
            let outputs = if let Some(replay) = replay.as_mut() {
                self.engine
                    .step_with_fixed_prefix(replay)
                    .context("generate: fixed-prefix execution failed")?
            } else if let Some(benchmark) = benchmark.as_mut() {
                if let Err(error) = self.device.synchronize() {
                    return Err(self.abort_after_capture_error(
                        anyhow!(error).context("generate: synchronizing benchmark step start"),
                    ));
                }
                let started_ns = duration_ns(admission.elapsed())?;
                let (outputs, step_telemetry) =
                    self.engine.step_with_telemetry().with_context(|| {
                        format!(
                            "generate: runtime failure for {}",
                            format_request_diagnostics(&request_ids, sampling_params)
                        )
                    })?;
                if let Err(error) = self.device.synchronize() {
                    return Err(self.abort_after_capture_error(
                        anyhow!(error).context("generate: synchronizing benchmark step end"),
                    ));
                }
                let ended_ns = duration_ns(admission.elapsed())?;
                if let Err(error) =
                    benchmark.record_engine_step(step_telemetry, started_ns, ended_ns)
                {
                    return Err(self.abort_after_capture_error(error));
                }
                outputs
            } else if let Some(capture) = capture.as_mut() {
                let (outputs, step_capture) =
                    self.engine.step_with_capture().with_context(|| {
                        format!(
                            "generate: runtime failure for {}",
                            format_request_diagnostics(&request_ids, sampling_params)
                        )
                    })?;
                if let Err(error) = capture.record_engine_step(step_capture) {
                    return Err(self.abort_after_capture_error(error));
                }
                outputs
            } else {
                self.engine.step().with_context(|| {
                    format!(
                        "generate: runtime failure for {}",
                        format_request_diagnostics(&request_ids, sampling_params)
                    )
                })?
            };
            #[cfg(not(feature = "internal-golden"))]
            let outputs = self.engine.step().with_context(|| {
                format!(
                    "generate: runtime failure for {}",
                    format_request_diagnostics(&request_ids, sampling_params)
                )
            })?;
            step_count += 1;

            completed_outputs.extend(outputs);
        }

        let elapsed = start.elapsed();
        let mut results =
            order_request_outputs(&request_ids, completed_outputs).with_context(|| {
                format!(
                    "generate: completion failure for {}",
                    format_request_diagnostics(&request_ids, sampling_params)
                )
            })?;
        for (output, params) in results.iter_mut().zip(sampling_params) {
            output.text = self
                .tokenizer
                .decode(&output.token_ids, true)
                .map_err(|error| {
                    anyhow!(
                        "generate: detokenization failed for request_id={}, sampling_params={params:?}: {error}",
                        output.request_id
                    )
                })?;
        }
        let total_tokens: usize = prompt_lens.iter().sum::<usize>()
            + results.iter().map(|o| o.token_ids.len()).sum::<usize>();

        if total_tokens > 0 {
            // total_tokens ≤ max_model_len (4096 default) ≪ 2^53 f64 mantissa
            #[allow(clippy::cast_precision_loss)]
            let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
            tracing::info!(
                total_tokens,
                tok_per_sec = format!("{tok_per_sec:.1}"),
                elapsed_ms = elapsed.as_millis(),
                steps = step_count,
                "generate complete"
            );
        }

        #[cfg(feature = "internal-golden")]
        if let Some(replay) = replay.take() {
            replay.finish(&results)?;
        }
        #[cfg(feature = "internal-golden")]
        if let Some(capture) = capture.take() {
            capture
                .finish(&results)
                .context("generate: publishing internal golden capture")?;
        }
        #[cfg(feature = "internal-golden")]
        if let Some(benchmark) = benchmark.take() {
            benchmark
                .finish(&results)
                .context("generate: publishing internal benchmark telemetry")?;
        }

        Ok(results)
    }

    #[cfg(feature = "internal-golden")]
    fn abort_after_capture_error(&mut self, error: anyhow::Error) -> anyhow::Error {
        match self.engine.abort_generation() {
            Ok(()) => error.context("generate: internal golden capture failed"),
            Err(cleanup_error) => error.context(format!(
                "generate: internal golden capture failed; request cleanup also failed: \
                 {cleanup_error}"
            )),
        }
    }
}

#[cfg(feature = "internal-golden")]
fn duration_ns(duration: std::time::Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).context("benchmark duration exceeds u64 nanoseconds")
}

#[cfg(feature = "internal-golden")]
fn require_diagnostic_host_ram_floor() -> Result<()> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .context("reading host memory guard from /proc/meminfo")?;
    let available_kib = meminfo
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
        })
        .ok_or_else(|| anyhow!("MemAvailable is unavailable; refusing diagnostic GPU work"))?;
    if available_kib < 16 * 1024 * 1024 {
        bail!("available host RAM fell below the 16 GiB diagnostic floor");
    }
    Ok(())
}

fn format_request_diagnostics(request_ids: &[usize], sampling_params: &[SamplingParams]) -> String {
    request_ids
        .iter()
        .zip(sampling_params)
        .map(|(request_id, params)| format!("request_id={request_id}, sampling_params={params:?}"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn order_request_outputs(
    request_ids: &[usize],
    completed_outputs: Vec<RequestOutput>,
) -> Result<Vec<RequestOutput>> {
    let mut input_positions = HashMap::with_capacity(request_ids.len());
    for (input_position, &request_id) in request_ids.iter().enumerate() {
        if input_positions.insert(request_id, input_position).is_some() {
            bail!("generate: request {request_id} was accepted more than once");
        }
    }

    let mut ordered_outputs = vec![None; request_ids.len()];
    for output in completed_outputs {
        let input_position = input_positions
            .get(&output.request_id)
            .copied()
            .ok_or_else(|| anyhow!("generate: completed unknown request {}", output.request_id))?;
        if ordered_outputs[input_position].replace(output).is_some() {
            bail!(
                "generate: request {} completed more than once",
                request_ids[input_position]
            );
        }
    }

    ordered_outputs
        .into_iter()
        .enumerate()
        .map(|(input_position, output)| {
            output.ok_or_else(|| {
                anyhow!(
                    "generate: request {} at input position {input_position} did not complete",
                    request_ids[input_position]
                )
            })
        })
        .collect()
}

impl Drop for LLM {
    fn drop(&mut self) {
        if self.device.is_cuda() {
            if let Err(e) = self.device.synchronize() {
                tracing::warn!("CUDA synchronize on drop failed: {e}");
            }
        }
    }
}

/// Tokenize a `Prompt` into `Vec<u32>` token IDs.
fn tokenize_prompt(prompt: &Prompt, tokenizer: &HFTokenizer) -> Result<Vec<u32>> {
    match prompt {
        Prompt::Text(text) => {
            let encoding = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
            Ok(encoding.get_ids().to_vec())
        }
        Prompt::TokenIds(ids) => Ok(ids.clone()),
    }
}

/// Extract `max_position_embeddings` from config.json bytes.
fn read_max_position_embeddings(config_json: &[u8]) -> Result<usize> {
    #[derive(serde::Deserialize)]
    struct PosCheck {
        max_position_embeddings: Option<usize>,
    }
    let parsed: PosCheck = serde_json::from_slice(config_json)
        .context("parsing config.json for max_position_embeddings")?;
    parsed
        .max_position_embeddings
        .ok_or_else(|| anyhow!("config.json has no `max_position_embeddings` field"))
}

/// Validate CUDA device compute capability ≥ sm_89.
///
/// Flash-attention paged kernels and Qwen3 BF16 matmuls require Hopper (sm_90)
/// or Ada Lovelace (sm_89) or newer. Older GPUs will hit opaque kernel launch
/// failures — this check gives a clear error before loading weights.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn validate_sm_version(_device: &Device) -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let mut cu_device: sys::CUdevice = 0;
    let result = unsafe { sys::cuDeviceGet(&mut cu_device, 0) };
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!(
            "cuDeviceGet(0) failed with error code {}. Is the CUDA driver installed?",
            result as i32
        );
    }

    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    let result = unsafe { sys::cuDeviceComputeCapability(&mut major, &mut minor, cu_device) };
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!(
            "cuDeviceComputeCapability failed with error code {}. \
             Cannot determine GPU compute capability.",
            result as i32
        );
    }

    let sm = major * 10 + minor;
    if sm < 89 {
        bail!(
            "GPU compute capability sm_{major}{minor} (sm_{sm}) is below the minimum \
             required sm_89. vllm-oxide v0.2.0 flash-attention kernels and Qwen3 BF16 \
             matmuls require Ada Lovelace (sm_89) or Hopper (sm_90) architecture. \
             Supported GPUs: RTX 40-series (Ada), H100/H200 (Hopper), and newer."
        );
    }

    tracing::info!(sm_version = sm, "CUDA device compute capability validated");
    Ok(())
}

/// Stub: SM validation is only meaningful with `--features cuda`. Without
/// CUDA features, the program would have already failed at
/// `Device::cuda_if_available(0)`.
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn validate_sm_version(_device: &Device) {}

/// Query CUDA free and total memory (in bytes) via the CUDA driver API.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn cuda_mem_info() -> Result<(usize, usize)> {
    use candle_core::cuda::cudarc::driver::sys;

    let mut free: usize = 0;
    let mut total: usize = 0;
    let result = unsafe { sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize) };
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("cuMemGetInfo_v2 failed with error code {}", result as i32);
    }
    Ok((free, total))
}

#[cfg(not(feature = "cuda"))]
fn cuda_mem_info() -> Result<(usize, usize)> {
    bail!("CUDA memory information requires the `cuda` feature")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::attention::{AttentionContext, AttnMetadata, PagedKVCacheGeometry};
    use crate::causal_lm::CausalLM;
    use crate::engine::sequence::BLOCK_SIZE;
    use candle_core::Tensor;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokenizers::models::bpe::{Vocab, BPE};

    struct MockModel {
        device: Device,
        fail_forward: bool,
    }

    struct ControlledLogitsModel {
        device: Device,
    }

    struct CacheAwareModel {
        device: Device,
        attn_ctx: AttentionContext,
        physical_tokens: Arc<Mutex<HashMap<usize, u32>>>,
    }

    #[derive(Clone)]
    struct CausalFingerprintControls {
        fail_next: Arc<AtomicBool>,
        seen_metadata: Arc<Mutex<Vec<AttnMetadata>>>,
    }

    impl CausalFingerprintControls {
        fn take_metadata(&self) -> Vec<AttnMetadata> {
            std::mem::take(&mut *self.seen_metadata.lock().unwrap())
        }
    }

    struct CausalFingerprintModel {
        device: Device,
        attn_ctx: AttentionContext,
        physical_tokens: HashMap<usize, u32>,
        controls: CausalFingerprintControls,
    }

    fn cached_token_at(
        physical_tokens: &HashMap<usize, u32>,
        block_table: &[i32],
        block_size: usize,
        logical_position: usize,
    ) -> candle_core::Result<u32> {
        let block_id = *block_table
            .get(logical_position / block_size)
            .ok_or_else(|| candle_core::Error::Msg("cache block table is too short".into()))?;
        let block_id = usize::try_from(block_id)
            .map_err(|_| candle_core::Error::Msg("cache block id is negative".into()))?;
        let physical_slot = block_id
            .checked_mul(block_size)
            .and_then(|base| base.checked_add(logical_position % block_size))
            .ok_or_else(|| candle_core::Error::Msg("cache physical slot overflow".into()))?;
        physical_tokens
            .get(&physical_slot)
            .copied()
            .ok_or_else(|| candle_core::Error::Msg("read an unwritten cache slot".into()))
    }

    impl CausalFingerprintModel {
        fn push_fingerprint(fingerprint: u8, token_id: u32) -> u8 {
            let fingerprint = u32::from(fingerprint)
                .wrapping_mul(31)
                .wrapping_add(token_id.wrapping_add(1))
                % 97;
            u8::try_from(fingerprint).unwrap()
        }
    }

    impl CausalLM for CausalFingerprintModel {
        fn forward(
            &mut self,
            input_ids: &Tensor,
            positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            let metadata = self
                .attn_ctx
                .prepared_for_bound_consumer()?
                .logical()
                .clone();
            self.controls
                .seen_metadata
                .lock()
                .unwrap()
                .push(metadata.clone());
            if self.controls.fail_next.swap(false, Ordering::SeqCst) {
                candle_core::bail!("injected prefix-cache execution failure");
            }
            let input_ids = input_ids.to_vec1::<u32>()?;
            let positions = positions.to_vec1::<u32>()?;
            if input_ids.len() != positions.len() || input_ids.len() != metadata.slot_mapping.len()
            {
                candle_core::bail!("causal model received inconsistent step metadata");
            }

            for (&token_id, &slot) in input_ids.iter().zip(&metadata.slot_mapping) {
                let slot = usize::try_from(slot).map_err(|_| {
                    candle_core::Error::Msg("causal model received a negative slot".into())
                })?;
                self.physical_tokens.insert(slot, token_id);
            }

            let batch_size = metadata.cu_seqlens_q.len().saturating_sub(1);
            if metadata.cu_seqlens_k.len() != batch_size + 1 {
                candle_core::bail!("causal model received inconsistent sequence lengths");
            }
            let mut hidden = Vec::with_capacity(input_ids.len());
            if !metadata.uses_paged_kv() {
                if metadata.cu_seqlens_q != metadata.cu_seqlens_k {
                    candle_core::bail!(
                        "initial prefill cannot consume an existing paged prefix in #37"
                    );
                }
                for sequence_index in 0..batch_size {
                    let query_start = metadata.cu_seqlens_q[sequence_index] as usize;
                    let query_end = metadata.cu_seqlens_q[sequence_index + 1] as usize;
                    let mut fingerprint = 0;
                    for &token_id in &input_ids[query_start..query_end] {
                        fingerprint = Self::push_fingerprint(fingerprint, token_id);
                        hidden.push(f32::from(fingerprint));
                    }
                }
            } else {
                if metadata.block_table.len() != batch_size {
                    candle_core::bail!("causal model received inconsistent block tables");
                }
                let block_size = self.attn_ctx.paged_kv.lock().unwrap().block_size();
                for sequence_index in 0..batch_size {
                    let query_start = metadata.cu_seqlens_q[sequence_index] as usize;
                    let query_end = metadata.cu_seqlens_q[sequence_index + 1] as usize;
                    let query_len = query_end - query_start;
                    let key_start = metadata.cu_seqlens_k[sequence_index] as usize;
                    let key_end = metadata.cu_seqlens_k[sequence_index + 1] as usize;
                    let key_len = key_end - key_start;
                    if query_len > key_len {
                        candle_core::bail!("causal model query is longer than its KV context");
                    }
                    for query_offset in 0..query_len {
                        let context_len = key_len - query_len + query_offset + 1;
                        let mut fingerprint = 0;
                        for logical_position in 0..context_len {
                            let token_id = cached_token_at(
                                &self.physical_tokens,
                                &metadata.block_table[sequence_index],
                                block_size,
                                logical_position,
                            )?;
                            fingerprint = Self::push_fingerprint(fingerprint, token_id);
                        }
                        hidden.push(f32::from(fingerprint));
                    }
                }
            }

            Tensor::from_vec(hidden, (input_ids.len(), 1), &self.device)
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        fn compute_logits(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            let rows = hidden_states.dim(0)?;
            let hidden = hidden_states.flatten_all()?.to_vec1::<f32>()?;
            let mut logits = vec![-100.0_f32; rows * 100];
            for (row, value) in hidden.into_iter().enumerate() {
                let target = value.round() as usize % 100;
                logits[row * 100 + target] = 100.0;
            }
            Tensor::from_vec(logits, (rows, 100), &self.device)
        }

        fn vocab_size(&self) -> usize {
            100
        }
    }

    impl CausalLM for CacheAwareModel {
        #[allow(clippy::cast_precision_loss)]
        fn forward(
            &mut self,
            input_ids: &Tensor,
            positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            let input_ids = input_ids.to_vec1::<u32>()?;
            let positions = positions.to_vec1::<u32>()?;
            let metadata = self
                .attn_ctx
                .prepared_for_bound_consumer()?
                .logical()
                .clone();
            if input_ids.len() != positions.len() || input_ids.len() != metadata.slot_mapping.len()
            {
                candle_core::bail!("cache-aware model received inconsistent step metadata");
            }
            let block_size = self.attn_ctx.paged_kv.lock().unwrap().block_size();
            let mut physical_tokens = self.physical_tokens.lock().unwrap();
            for ((&token_id, &position), &slot) in
                input_ids.iter().zip(&positions).zip(&metadata.slot_mapping)
            {
                let slot = usize::try_from(slot).map_err(|_| {
                    candle_core::Error::Msg("cache-aware model received a negative slot".into())
                })?;
                physical_tokens.insert(slot, token_id.wrapping_add(position));
            }

            let hidden =
                if metadata.is_prefill {
                    input_ids
                        .iter()
                        .zip(&positions)
                        .map(|(&token_id, &position)| token_id.wrapping_add(position) as f32)
                        .collect::<Vec<_>>()
                } else {
                    let context_len =
                        metadata.cu_seqlens_k.last().copied().ok_or_else(|| {
                            candle_core::Error::Msg("missing decode context".into())
                        })? as usize;
                    let block_table = metadata.block_table.first().ok_or_else(|| {
                        candle_core::Error::Msg("missing decode block table".into())
                    })?;
                    let mut fingerprint = 0u32;
                    for logical_position in 0..context_len {
                        fingerprint = fingerprint.wrapping_add(cached_token_at(
                            &physical_tokens,
                            block_table,
                            block_size,
                            logical_position,
                        )?);
                    }
                    vec![(fingerprint % 100) as f32]
                };
            Tensor::from_vec(hidden, (input_ids.len(), 1), &self.device)?.to_dtype(DType::F16)
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        fn compute_logits(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            let rows = hidden_states.dim(0)?;
            let hidden = hidden_states
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut logits = vec![-100.0_f32; rows * 100];
            for (row, value) in hidden.into_iter().enumerate() {
                let target = value.round() as usize % 100;
                logits[row * 100 + target] = 100.0;
            }
            Tensor::from_vec(logits, (rows, 100), &self.device)
        }

        fn vocab_size(&self) -> usize {
            100
        }
    }

    impl CausalLM for MockModel {
        fn forward(
            &mut self,
            input_ids: &Tensor,
            _positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            if self.fail_forward {
                return Err(candle_core::Error::Msg(
                    "injected model execution failure".to_string(),
                ));
            }
            Tensor::zeros((input_ids.dim(0)?, 1), DType::F32, &self.device)
        }

        fn compute_logits(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            let rows = hidden_states.dim(0)?;
            let mut logits = vec![-100.0_f32; rows * 100];
            for row in 0..rows {
                logits[row * 100 + 42] = 100.0;
            }
            Tensor::from_vec(logits, (rows, 100), &self.device)
        }

        fn vocab_size(&self) -> usize {
            100
        }
    }

    impl CausalLM for ControlledLogitsModel {
        fn forward(
            &mut self,
            input_ids: &Tensor,
            _positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            Tensor::zeros((input_ids.dim(0)?, 1), DType::F32, &self.device)
        }

        fn compute_logits(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            let rows = hidden_states.dim(0)?;
            let mut logits = vec![-100.0_f32; rows * 100];
            for row in 0..rows {
                logits[row * 100 + 2] = 2.0;
                logits[row * 100 + 3] = 3.0;
            }
            Tensor::from_vec(logits, (rows, 100), &self.device)
        }

        fn vocab_size(&self) -> usize {
            100
        }
    }

    fn finish_test_llm(
        engine: EngineCore,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        device: Device,
    ) -> LLM {
        let vocab = (0..100_u32)
            .map(|token_id| (format!("token-{token_id}"), token_id))
            .collect::<Vocab>();
        let tokenizer = HFTokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, Vec::new())
                .build()
                .unwrap(),
        );
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            br#"{"torch_dtype":"bfloat16"}"#,
        )
        .unwrap();
        tokenizer
            .save(model_dir.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(model_dir.path().join("model.safetensors"), b"").unwrap();
        let resolved_model =
            ResolvedModel::resolve(Source::Local(model_dir.path().to_path_buf()), None).unwrap();

        LLM {
            engine,
            tokenizer,
            _resolved_model: resolved_model,
            _paged_kv: paged_kv,
            device,
            max_model_len: 4096,
        }
    }

    fn test_llm_with_configuration(fail_forward: bool) -> LLM {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(
            PagedKVCache::new(1, 32, BLOCK_SIZE, 1, 1, DType::F32, &device).unwrap(),
        ));
        let attn_ctx = AttentionContext::new(paged_kv.clone());
        let scheduler = Scheduler::new(16, 16, 0.9);
        let kv_cache_manager = KvCacheManager::new(32, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        let model: Box<dyn CausalLM> = Box::new(MockModel {
            device: device.clone(),
            fail_forward,
        });
        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            Sampler::new_with_seed(0),
            attn_ctx,
            device.clone(),
        );
        finish_test_llm(engine, paged_kv, device)
    }

    fn test_llm_with_forward_failure(fail_forward: bool) -> LLM {
        test_llm_with_configuration(fail_forward)
    }

    fn test_llm() -> LLM {
        test_llm_with_forward_failure(false)
    }

    fn controlled_logits_test_llm(eos_token_ids: Vec<u32>) -> LLM {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(
            PagedKVCache::new(1, 32, BLOCK_SIZE, 1, 1, DType::F32, &device).unwrap(),
        ));
        let attn_ctx = AttentionContext::new(paged_kv.clone());
        let scheduler = Scheduler::new_with_eos_token_ids(16, 16, 0.9, eos_token_ids);
        let kv_cache_manager = KvCacheManager::new(32, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        let model: Box<dyn CausalLM> = Box::new(ControlledLogitsModel {
            device: device.clone(),
        });
        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            Sampler::new_with_seed(0),
            attn_ctx,
            device.clone(),
        );
        finish_test_llm(engine, paged_kv, device)
    }

    fn cache_aware_test_llm(run_warmup: bool) -> LLM {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(PagedKVCache::deferred(PagedKVCacheGeometry {
            num_layers: 1,
            block_size: BLOCK_SIZE,
            num_kv_heads: 1,
            head_dim: 1,
            dtype: DType::F16,
        })));
        let attn_ctx = AttentionContext::new(paged_kv.clone());
        let physical_tokens = Arc::new(Mutex::new(HashMap::new()));
        let mut model: Box<dyn CausalLM> = Box::new(CacheAwareModel {
            device: device.clone(),
            attn_ctx: attn_ctx.clone(),
            physical_tokens: physical_tokens.clone(),
        });
        let options = EngineOptions {
            max_num_batched_tokens: 2,
            max_num_seqs: 1,
            max_model_len: 8,
            gpu_memory_utilization: 0.5,
            enforce_eager: true,
            dtype: Some(DType::F16),
        };

        let num_blocks = if run_warmup {
            let allocation = initialization::initialize_model_with_available_memory(
                model.as_mut(),
                &attn_ctx,
                &device,
                initialization::DeviceMemory::new(4096, 8192),
                &options,
            )
            .unwrap();
            assert_eq!(physical_tokens.lock().unwrap().len(), 3);
            allocation.num_blocks
        } else {
            paged_kv.lock().unwrap().allocate(2, &device).unwrap();
            assert!(physical_tokens.lock().unwrap().is_empty());
            2
        };
        assert_eq!(paged_kv.lock().unwrap().allocation_count(), 1);

        let scheduler = Scheduler::new(
            options.max_num_batched_tokens,
            options.max_num_seqs,
            options.gpu_memory_utilization,
        );
        let kv_cache_manager =
            KvCacheManager::new(num_blocks, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            Sampler::new_with_seed(0),
            attn_ctx,
            device.clone(),
        );
        finish_test_llm(engine, paged_kv, device)
    }

    fn causal_fingerprint_test_harness(
        max_num_batched_tokens: usize,
        max_num_seqs: usize,
        prefix_cache_enabled: bool,
    ) -> (LLM, CausalFingerprintControls) {
        causal_fingerprint_test_harness_with_capacity(
            max_num_batched_tokens,
            max_num_seqs,
            32,
            prefix_cache_enabled,
        )
    }

    fn causal_fingerprint_test_harness_with_capacity(
        max_num_batched_tokens: usize,
        max_num_seqs: usize,
        num_blocks: usize,
        prefix_cache_enabled: bool,
    ) -> (LLM, CausalFingerprintControls) {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(
            PagedKVCache::new(1, 32, BLOCK_SIZE, 1, 1, DType::F32, &device).unwrap(),
        ));
        let attn_ctx = AttentionContext::new(paged_kv.clone());
        let controls = CausalFingerprintControls {
            fail_next: Arc::new(AtomicBool::new(false)),
            seen_metadata: Arc::new(Mutex::new(Vec::new())),
        };
        let scheduler = Scheduler::new(max_num_batched_tokens, max_num_seqs, 0.9);
        let kv_cache_manager = KvCacheManager::new_with_prefix_cache(
            num_blocks,
            BLOCK_SIZE,
            paged_kv.clone(),
            prefix_cache_enabled,
        );
        let model: Box<dyn CausalLM> = Box::new(CausalFingerprintModel {
            device: device.clone(),
            attn_ctx: attn_ctx.clone(),
            physical_tokens: HashMap::new(),
            controls: controls.clone(),
        });
        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            Sampler::new_with_seed(0),
            attn_ctx,
            device.clone(),
        );
        (finish_test_llm(engine, paged_kv, device), controls)
    }

    fn causal_fingerprint_test_llm_with_limits(
        max_num_batched_tokens: usize,
        max_num_seqs: usize,
    ) -> LLM {
        causal_fingerprint_test_harness(max_num_batched_tokens, max_num_seqs, true).0
    }

    fn causal_fingerprint_test_llm(max_num_batched_tokens: usize) -> LLM {
        causal_fingerprint_test_llm_with_limits(max_num_batched_tokens, 16)
    }

    fn deterministic_causal_params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            max_tokens,
            ignore_eos: true,
            ..SamplingParams::default()
        }
    }

    mod engine_options {
        use super::*;

        #[test]
        fn default_has_sane_values() {
            let opts = EngineOptions::default();
            assert!(opts.max_num_batched_tokens > 0);
            assert!(opts.max_num_seqs > 0);
            assert!(opts.max_model_len > 0);
            assert!(opts.gpu_memory_utilization > 0.0 && opts.gpu_memory_utilization <= 1.0);
            assert!(opts.enforce_eager);
            assert!(opts.dtype.is_none());
        }

        #[test]
        fn custom_override() {
            let opts = EngineOptions {
                max_num_batched_tokens: 100,
                max_num_seqs: 10,
                max_model_len: 2048,
                gpu_memory_utilization: 0.5,
                enforce_eager: true,
                dtype: Some(DType::F16),
            };
            assert_eq!(opts.max_num_batched_tokens, 100);
            assert_eq!(opts.max_num_seqs, 10);
            assert_eq!(opts.max_model_len, 2048);
            assert_eq!(opts.gpu_memory_utilization, 0.5);
            assert_eq!(opts.dtype, Some(DType::F16));
        }
    }

    mod model_warmup {
        use super::*;

        #[test]
        fn first_generation_matches_the_non_mutating_reference_scenario() {
            let mut warmed = cache_aware_test_llm(true);
            let mut reference = cache_aware_test_llm(false);
            let prompts = [Prompt::TokenIds(vec![1, 2, 3])];
            let params = [SamplingParams {
                temperature: 0.0,
                max_tokens: 3,
                ignore_eos: true,
                ..SamplingParams::default()
            }];

            let warmed_output = warmed.generate(&prompts, &params).unwrap();
            let reference_output = reference.generate(&prompts, &params).unwrap();

            assert_eq!(warmed_output.len(), 1);
            assert_eq!(warmed_output[0].request_id, 0);
            assert_eq!(warmed_output[0].token_ids, vec![5, 17, 38]);
            assert_eq!(warmed_output[0].request_id, reference_output[0].request_id);
            assert_eq!(warmed_output[0].token_ids, reference_output[0].token_ids);
            assert_eq!(warmed_output[0].text, reference_output[0].text);
            assert_eq!(warmed_output[0].finished, reference_output[0].finished);
        }
    }

    mod chunked_prefill {
        use super::*;

        fn deterministic_params(max_tokens: usize) -> SamplingParams {
            SamplingParams {
                temperature: 0.0,
                top_k: Some(5),
                top_p: Some(0.9),
                max_tokens,
                ignore_eos: true,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                repetition_penalty: 1.0,
            }
        }

        #[test]
        fn chunked_generate_matches_equivalent_unchunked_generation() {
            let prompt = [Prompt::TokenIds(vec![7, 11, 13, 17, 19])];
            let params = [deterministic_params(3)];
            let mut chunked = causal_fingerprint_test_llm(2);
            let mut unchunked = causal_fingerprint_test_llm(32);

            let chunked_output = chunked.generate(&prompt, &params).unwrap();
            let unchunked_output = unchunked.generate(&prompt, &params).unwrap();

            assert_eq!(chunked_output.len(), 1);
            assert_eq!(chunked_output[0].request_id, 0);
            assert!(chunked_output[0].finished);
            assert_eq!(chunked_output[0].token_ids.len(), 3);
            assert_eq!(chunked_output[0].token_ids, unchunked_output[0].token_ids);
            assert_eq!(chunked_output[0].text, unchunked_output[0].text);
        }

        #[test]
        fn mixed_chunked_generate_preserves_identity_order_and_per_request_output() {
            let prompts = [
                Prompt::TokenIds(vec![2]),
                Prompt::TokenIds(vec![7, 11, 13, 17, 19]),
                Prompt::TokenIds(vec![23, 29, 31]),
            ];
            let params = [
                deterministic_params(1),
                deterministic_params(2),
                deterministic_params(3),
            ];
            let mut mixed = causal_fingerprint_test_llm(3);

            let mixed_outputs = mixed.generate(&prompts, &params).unwrap();

            assert_eq!(
                mixed_outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![0, 1, 2]
            );
            for (input_position, (prompt, params)) in prompts.iter().zip(params.iter()).enumerate()
            {
                let mut isolated = causal_fingerprint_test_llm(32);
                let isolated_output = isolated
                    .generate(std::slice::from_ref(prompt), std::slice::from_ref(params))
                    .unwrap();
                assert!(mixed_outputs[input_position].finished);
                assert_eq!(
                    mixed_outputs[input_position].token_ids,
                    isolated_output[0].token_ids
                );
                assert_eq!(mixed_outputs[input_position].text, isolated_output[0].text);
            }
        }
    }

    mod recompute_preemption {
        use super::*;

        #[test]
        fn low_capacity_generation_matches_unpreempted_output_and_completes_waiters() {
            let block_size = u32::try_from(BLOCK_SIZE).unwrap();
            let recovery_len = u32::try_from(BLOCK_SIZE + 1).unwrap();
            let prompts = [
                Prompt::TokenIds((0..block_size).collect()),
                Prompt::TokenIds((1_000..1_000 + block_size).collect()),
                Prompt::TokenIds(vec![7, 11, 13]),
            ];
            let params = [
                deterministic_causal_params(3),
                deterministic_causal_params(3),
                deterministic_causal_params(2),
            ];
            let (mut preempted, controls) =
                causal_fingerprint_test_harness_with_capacity(BLOCK_SIZE + 1, 2, 3, false);
            let (mut unpreempted, _) =
                causal_fingerprint_test_harness_with_capacity(BLOCK_SIZE + 1, 2, 4, false);

            let preempted_outputs = preempted.generate(&prompts, &params).unwrap();
            let unpreempted_outputs = unpreempted.generate(&prompts, &params).unwrap();

            assert_eq!(preempted_outputs.len(), prompts.len());
            for (actual, expected) in preempted_outputs.iter().zip(&unpreempted_outputs) {
                assert_eq!(actual.request_id, expected.request_id);
                assert_eq!(actual.token_ids, expected.token_ids);
                assert_eq!(actual.text, expected.text);
                assert_eq!(actual.finished, expected.finished);
            }
            assert!(controls
                .take_metadata()
                .iter()
                .any(|metadata| metadata.cu_seqlens_q == vec![0, recovery_len]));
        }
    }

    mod prefix_cached_prefill {
        use super::*;

        fn assert_same_output(actual: &RequestOutput, expected: &RequestOutput) {
            assert_eq!(actual.request_id, expected.request_id);
            assert_eq!(actual.token_ids, expected.token_ids);
            assert_eq!(actual.text, expected.text);
            assert_eq!(actual.finished, expected.finished);
        }

        #[test]
        fn cache_enabled_and_disabled_match_for_the_same_request_output() {
            let target = Prompt::TokenIds((0..513).collect());
            let params = deterministic_causal_params(3);
            let (mut cache_enabled, enabled_controls) =
                causal_fingerprint_test_harness(1_024, 16, true);
            let (mut cache_disabled, disabled_controls) =
                causal_fingerprint_test_harness(1_024, 16, false);

            cache_enabled
                .generate(std::slice::from_ref(&target), std::slice::from_ref(&params))
                .unwrap();
            cache_disabled
                .generate(std::slice::from_ref(&target), std::slice::from_ref(&params))
                .unwrap();
            enabled_controls.take_metadata();
            disabled_controls.take_metadata();

            let enabled = cache_enabled
                .generate(std::slice::from_ref(&target), std::slice::from_ref(&params))
                .unwrap();
            let disabled = cache_disabled
                .generate(std::slice::from_ref(&target), std::slice::from_ref(&params))
                .unwrap();

            assert_eq!(enabled.len(), 1);
            assert_eq!(enabled[0].request_id, 1);
            assert_same_output(&enabled[0], &disabled[0]);
            let enabled_metadata = enabled_controls.take_metadata();
            let disabled_metadata = disabled_controls.take_metadata();
            assert_eq!(enabled_metadata[0].cu_seqlens_q, vec![0, 1]);
            assert_eq!(enabled_metadata[0].cu_seqlens_k, vec![0, 513]);
            assert_eq!(enabled_metadata[0].block_table.len(), 1);
            assert_eq!(disabled_metadata[0].cu_seqlens_q, vec![0, 513]);
            assert_eq!(disabled_metadata[0].cu_seqlens_k, vec![0, 513]);
            assert!(disabled_metadata[0].block_table.is_empty());
            assert_eq!(cache_enabled.engine.kv_cache_manager.num_free_blocks(), 32);
            assert_eq!(cache_disabled.engine.kv_cache_manager.num_free_blocks(), 32);
        }

        #[test]
        fn mixed_full_partial_and_miss_match_isolated_cache_misses() {
            let block_size = u32::try_from(BLOCK_SIZE).unwrap();
            let first_block = (0..block_size).collect::<Vec<_>>();
            let second_block = (1_000..1_000 + block_size).collect::<Vec<_>>();
            let alternate_second = (2_000..2_000 + block_size).collect::<Vec<_>>();
            let mut full_prompt = first_block.clone();
            full_prompt.extend_from_slice(&second_block);
            full_prompt.push(9);
            let mut partial_prompt = first_block;
            partial_prompt.extend_from_slice(&alternate_second);
            partial_prompt.push(10);
            let mut miss_prompt = (3_000..3_000 + block_size).collect::<Vec<_>>();
            miss_prompt.push(11);
            let prompts = [
                Prompt::TokenIds(full_prompt.clone()),
                Prompt::TokenIds(partial_prompt),
                Prompt::TokenIds(miss_prompt),
            ];
            let params = [
                deterministic_causal_params(1),
                deterministic_causal_params(2),
                deterministic_causal_params(3),
            ];
            let mut mixed = causal_fingerprint_test_llm(1_024);
            mixed
                .generate(
                    &[Prompt::TokenIds(full_prompt)],
                    &[deterministic_causal_params(1)],
                )
                .unwrap();

            let outputs = mixed.generate(&prompts, &params).unwrap();

            assert_eq!(outputs.len(), 3);
            for (index, (prompt, params)) in prompts.iter().zip(&params).enumerate() {
                let mut isolated = causal_fingerprint_test_llm(1_024);
                let expected = isolated
                    .generate(std::slice::from_ref(prompt), std::slice::from_ref(params))
                    .unwrap();
                assert_eq!(outputs[index].request_id, index + 1);
                assert_eq!(outputs[index].token_ids, expected[0].token_ids);
                assert_eq!(outputs[index].text, expected[0].text);
                assert_eq!(outputs[index].finished, expected[0].finished);
            }
            assert_eq!(mixed.engine.kv_cache_manager.num_free_blocks(), 32);
        }

        #[test]
        fn rejected_request_does_not_borrow_or_discard_cached_prefix_ownership() {
            let (mut llm, controls) = causal_fingerprint_test_harness(1_024, 16, true);
            let prompt = [Prompt::TokenIds((0..513).collect())];
            let valid = [deterministic_causal_params(1)];
            llm.generate(&prompt, &valid).unwrap();
            let ownership_before = llm.engine.kv_cache_manager.ownership_snapshot();
            assert_eq!(ownership_before.free_block_ids.len(), 32);
            assert!(ownership_before.used_block_ids.is_empty());
            assert!(ownership_before.ref_counts.iter().all(|&count| count == 0));
            assert_eq!(ownership_before.cache_entries.len(), 2);

            let error = llm
                .generate(
                    &prompt,
                    &[SamplingParams {
                        max_tokens: 0,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap_err();

            assert!(error.to_string().contains("max_tokens=0"));
            assert!(!llm.engine.is_running());
            assert_eq!(
                llm.engine.kv_cache_manager.ownership_snapshot(),
                ownership_before
            );

            controls.take_metadata();
            let accepted = llm.generate(&prompt, &valid).unwrap();
            assert_eq!(accepted[0].request_id, 1);
            assert!(accepted[0].finished);
            let metadata = controls.take_metadata();
            assert_eq!(metadata[0].cu_seqlens_q, vec![0, 1]);
            assert_eq!(metadata[0].cu_seqlens_k, vec![0, 513]);
            assert_eq!(metadata[0].block_table.len(), 1);
            assert_eq!(llm.engine.kv_cache_manager.num_free_blocks(), 32);
        }
    }

    mod continuous_batching {
        use super::*;

        #[test]
        fn generate_preserves_order_and_outputs_when_admission_interleaves_with_decode() {
            let prompts = [
                Prompt::TokenIds(vec![2]),
                Prompt::TokenIds(vec![7]),
                Prompt::TokenIds(vec![11, 13, 17, 19, 23]),
            ];
            let params = [
                deterministic_causal_params(1),
                deterministic_causal_params(3),
                deterministic_causal_params(2),
            ];
            let mut mixed = causal_fingerprint_test_llm_with_limits(4, 2);

            let outputs = mixed.generate(&prompts, &params).unwrap();

            assert_eq!(outputs.len(), prompts.len());
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![0, 1, 2]
            );
            for (input_position, (prompt, params)) in prompts.iter().zip(&params).enumerate() {
                let mut isolated = causal_fingerprint_test_llm(32);
                let isolated_output = isolated
                    .generate(std::slice::from_ref(prompt), std::slice::from_ref(params))
                    .unwrap();
                assert!(outputs[input_position].finished);
                assert_eq!(
                    outputs[input_position].token_ids,
                    isolated_output[0].token_ids
                );
                assert_eq!(outputs[input_position].text, isolated_output[0].text);
            }
        }
    }

    mod prompt {
        use super::*;

        #[test]
        fn text_variant_carries_string() {
            match Prompt::Text("hello".into()) {
                Prompt::Text(s) => assert_eq!(s, "hello"),
                _ => panic!("expected Text variant"),
            }
        }

        #[test]
        fn token_ids_variant_carries_vec() {
            match Prompt::TokenIds(vec![1, 2, 3]) {
                Prompt::TokenIds(ids) => assert_eq!(ids, vec![1, 2, 3]),
                _ => panic!("expected TokenIds variant"),
            }
        }

        #[test]
        fn token_ids_path_bypasses_tokenization_entirely() {
            let ids = vec![42, 99, 151_645];
            let prompt = Prompt::TokenIds(ids.clone());
            match &prompt {
                Prompt::TokenIds(existing) => assert_eq!(existing, &ids),
                Prompt::Text(_) => panic!("expected TokenIds"),
            }
        }
    }

    mod read_max_position_embeddings {
        use super::*;

        #[test]
        fn parses_qwen3_style_config() {
            let json = br#"{"max_position_embeddings": 40960, "hidden_size": 1024}"#;
            assert_eq!(read_max_position_embeddings(json).unwrap(), 40960);
        }

        #[test]
        fn parses_small_config() {
            let json = br#"{"max_position_embeddings": 128}"#;
            assert_eq!(read_max_position_embeddings(json).unwrap(), 128);
        }

        #[test]
        fn missing_field_errors() {
            let json = br#"{"hidden_size": 1024}"#;
            assert!(read_max_position_embeddings(json).is_err());
        }

        #[test]
        fn null_field_errors() {
            let json = br#"{"max_position_embeddings": null}"#;
            assert!(read_max_position_embeddings(json).is_err());
        }

        #[test]
        fn empty_json_errors() {
            let json = br#"{}"#;
            assert!(read_max_position_embeddings(json).is_err());
        }
    }

    mod request_output {
        use crate::engine::RequestOutput;

        #[test]
        fn text_field_defaults_empty() {
            let output = RequestOutput {
                request_id: 0,
                token_ids: vec![1, 2, 3],
                text: String::new(),
                finished: true,
            };
            assert!(output.text.is_empty());
            assert_eq!(output.token_ids.len(), 3);
            assert!(output.finished);
        }

        #[test]
        fn text_field_can_be_populated() {
            let output = RequestOutput {
                request_id: 1,
                token_ids: vec![42],
                text: "hello".into(),
                finished: true,
            };
            assert_eq!(output.text, "hello");
            assert_eq!(output.request_id, 1);
        }
    }

    mod order_request_outputs {
        use super::*;

        fn output(request_id: usize, token_id: u32) -> RequestOutput {
            RequestOutput {
                request_id,
                token_ids: vec![token_id],
                text: String::new(),
                finished: true,
            }
        }

        #[test]
        fn restores_input_order_from_reversed_completion_order() {
            let outputs = order_request_outputs(
                &[10, 20, 30],
                vec![output(30, 3), output(20, 2), output(10, 1)],
            )
            .unwrap();

            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![10, 20, 30]
            );
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.token_ids[0])
                    .collect::<Vec<_>>(),
                vec![1, 2, 3]
            );
        }

        #[test]
        fn rejects_an_unknown_completion() {
            let error = order_request_outputs(&[10], vec![output(99, 1)]).unwrap_err();

            assert_eq!(error.to_string(), "generate: completed unknown request 99");
        }

        #[test]
        fn rejects_a_duplicate_completion() {
            let error =
                order_request_outputs(&[10], vec![output(10, 1), output(10, 2)]).unwrap_err();

            assert_eq!(
                error.to_string(),
                "generate: request 10 completed more than once"
            );
        }

        #[test]
        fn rejects_a_missing_completion() {
            let error = order_request_outputs(&[10, 20], vec![output(10, 1)]).unwrap_err();

            assert_eq!(
                error.to_string(),
                "generate: request 20 at input position 1 did not complete"
            );
        }
    }

    mod generate_empty_batch {
        use super::*;

        #[test]
        fn engine_options_defaults_match_scheduler_constants() {
            let opts = EngineOptions::default();
            assert_eq!(
                opts.max_num_batched_tokens,
                crate::engine::scheduler::DEFAULT_MAX_NUM_BATCHED_TOKENS
            );
            assert_eq!(
                opts.max_num_seqs,
                crate::engine::scheduler::DEFAULT_MAX_NUM_SEQS
            );
            assert_eq!(
                opts.gpu_memory_utilization,
                crate::engine::scheduler::DEFAULT_GPU_MEMORY_UTILIZATION
            );
        }
    }

    mod generate_repeated_calls {
        use super::*;

        #[test]
        fn second_call_returns_a_real_output() {
            let mut llm = test_llm();
            let prompts = [Prompt::TokenIds(vec![1])];
            let params = [SamplingParams {
                max_tokens: 1,
                ..SamplingParams::default()
            }];

            let first = llm.generate(&prompts, &params).unwrap();
            assert_eq!(first[0].token_ids, vec![42]);

            let second = llm.generate(&prompts, &params).unwrap();
            assert_eq!(second[0].token_ids, vec![42]);
            assert!(second[0].finished);
        }

        #[test]
        fn one_instance_returns_complete_outputs_for_one_hundred_calls() {
            let mut llm = test_llm();
            let prompts = [Prompt::TokenIds(vec![1])];
            let params = [SamplingParams {
                max_tokens: 1,
                ..SamplingParams::default()
            }];

            for expected_request_id in 0..100 {
                let outputs = llm.generate(&prompts, &params).unwrap();
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].request_id, expected_request_id);
                assert_eq!(outputs[0].token_ids, vec![42]);
                assert!(outputs[0].finished);
            }
        }
    }

    mod generate_batch_order {
        use super::*;

        #[test]
        fn mixed_batch_stays_in_input_order_when_requests_finish_out_of_order() {
            let mut llm = test_llm();
            let warmup_prompts = [Prompt::TokenIds(vec![99])];
            let warmup_params = [SamplingParams {
                max_tokens: 1,
                ..SamplingParams::default()
            }];
            llm.generate(&warmup_prompts, &warmup_params).unwrap();

            let prompts = [
                Prompt::TokenIds(vec![1]),
                Prompt::TokenIds(vec![2, 3, 4]),
                Prompt::TokenIds(vec![5, 6]),
            ];
            let params = [3, 1, 2].map(|max_tokens| SamplingParams {
                max_tokens,
                ..SamplingParams::default()
            });

            let outputs = llm.generate(&prompts, &params).unwrap();

            assert_eq!(outputs.len(), prompts.len());
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![1, 2, 3]
            );
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.token_ids.len())
                    .collect::<Vec<_>>(),
                vec![3, 1, 2]
            );
            assert!(outputs.iter().all(|output| output.finished));
        }
    }

    mod generate_sampling_params {
        use super::*;

        #[test]
        fn presence_penalty_reaches_sampling_through_generate() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![3])],
                    &[SamplingParams {
                        presence_penalty: 2.0,
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();

            assert_eq!(output[0].token_ids, vec![2]);
        }

        #[test]
        fn invalid_batch_is_rejected_before_any_request_is_admitted() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let error = llm
                .generate(
                    &[Prompt::TokenIds(vec![1]), Prompt::TokenIds(vec![2])],
                    &[
                        SamplingParams {
                            max_tokens: 1,
                            ..SamplingParams::default()
                        },
                        SamplingParams {
                            temperature: -0.5,
                            max_tokens: 1,
                            ..SamplingParams::default()
                        },
                    ],
                )
                .unwrap_err();

            assert_eq!(
                error.to_string(),
                "generate: sampling_params[1].temperature=-0.5 is invalid: must be greater than or equal to 0"
            );

            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();
            assert_eq!(output[0].request_id, 0);
        }

        #[test]
        fn temperature_reaches_sampling_through_generate() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let prompts = vec![Prompt::TokenIds(vec![1]); 16];
            let params = vec![
                SamplingParams {
                    temperature: f32::INFINITY,
                    max_tokens: 1,
                    ..SamplingParams::default()
                };
                prompts.len()
            ];
            let output = llm.generate(&prompts, &params).unwrap();

            assert!(
                output.iter().any(|request| request.token_ids != vec![3]),
                "positive infinity must preserve the uniform pre-filter path"
            );
        }

        #[test]
        fn top_k_reaches_sampling_through_generate() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        temperature: f32::INFINITY,
                        top_k: Some(1),
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();

            assert_eq!(output[0].token_ids, vec![3]);
        }

        #[test]
        fn top_p_reaches_sampling_through_generate() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        temperature: 1.0,
                        top_p: Some(0.5),
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();

            assert_eq!(output[0].token_ids, vec![3]);
        }

        #[test]
        fn frequency_penalty_reaches_sampling_through_generate() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![3, 3])],
                    &[SamplingParams {
                        frequency_penalty: 0.75,
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();

            assert_eq!(output[0].token_ids, vec![2]);
        }

        #[test]
        fn repetition_penalty_reaches_sampling_through_generate() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![3])],
                    &[SamplingParams {
                        repetition_penalty: 2.0,
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();

            assert_eq!(output[0].token_ids, vec![2]);
        }

        #[test]
        fn empty_prompt_is_rejected_without_consuming_request_identity() {
            let mut llm = test_llm();
            assert!(llm
                .generate(
                    &[Prompt::TokenIds(vec![])],
                    &[deterministic_causal_params(1)]
                )
                .is_err());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[deterministic_causal_params(1)],
                )
                .unwrap();
            assert_eq!(output[0].request_id, 0);
        }

        #[test]
        fn context_budget_is_rejected_before_any_request_is_admitted() {
            let mut llm = test_llm();
            let error = llm
                .generate(
                    &[Prompt::TokenIds(vec![1; 4096])],
                    &[deterministic_causal_params(1)],
                )
                .unwrap_err();
            assert!(error.to_string().contains("context budget"));
            assert_eq!(llm.engine.scheduler.num_waiting(), 0);
            assert_eq!(llm.engine.scheduler.num_running(), 0);
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![1; 4095])],
                    &[deterministic_causal_params(1)],
                )
                .unwrap();
            assert_eq!(output[0].request_id, 0);
            assert_eq!(output[0].token_ids.len(), 1);
        }

        #[test]
        fn max_tokens_counts_only_completion_tokens() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![1, 1, 1, 1])],
                    &[SamplingParams {
                        max_tokens: 3,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();

            assert_eq!(output[0].token_ids, vec![3, 3, 3]);
        }

        #[test]
        fn ignore_eos_only_bypasses_resolved_eos() {
            let mut stops_on_eos = controlled_logits_test_llm(vec![3]);
            let stopped = stops_on_eos
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        max_tokens: 2,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();
            assert_eq!(stopped[0].token_ids, vec![3]);

            let mut ignores_eos = controlled_logits_test_llm(vec![3]);
            let ignored = ignores_eos
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        max_tokens: 2,
                        ignore_eos: true,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();
            assert_eq!(ignored[0].token_ids, vec![3, 3]);
        }

        #[test]
        fn valid_fields_compose_without_cross_request_contamination() {
            let mut llm = controlled_logits_test_llm(vec![3]);
            let output = llm
                .generate(
                    &[
                        Prompt::TokenIds(vec![1]),
                        Prompt::TokenIds(vec![3]),
                        Prompt::TokenIds(vec![1]),
                    ],
                    &[
                        SamplingParams {
                            max_tokens: 1,
                            ..SamplingParams::default()
                        },
                        SamplingParams {
                            presence_penalty: 2.0,
                            max_tokens: 2,
                            ignore_eos: true,
                            ..SamplingParams::default()
                        },
                        SamplingParams {
                            temperature: 0.0,
                            top_k: Some(1),
                            top_p: Some(0.01),
                            max_tokens: 3,
                            ignore_eos: true,
                            ..SamplingParams::default()
                        },
                    ],
                )
                .unwrap();

            assert_eq!(output[0].token_ids, vec![3]);
            assert_eq!(output[1].token_ids, vec![2, 3]);
            assert_eq!(output[2].token_ids, vec![3, 3, 3]);
        }

        #[test]
        fn greedy_is_deterministic_for_repeated_calls_on_one_resolved_model() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let prompt = [Prompt::TokenIds(vec![3])];
            let params = [SamplingParams {
                temperature: 0.0,
                top_k: Some(100),
                top_p: Some(0.01),
                max_tokens: 3,
                ignore_eos: true,
                presence_penalty: 0.25,
                frequency_penalty: 0.5,
                repetition_penalty: 1.25,
            }];

            let first = llm.generate(&prompt, &params).unwrap();
            let second = llm.generate(&prompt, &params).unwrap();

            assert_eq!(first[0].token_ids, second[0].token_ids);
            assert_eq!(first[0].text, second[0].text);
        }

        #[test]
        fn every_invalid_field_reports_position_value_and_reason() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let invalid_cases = [
                (
                    SamplingParams {
                        temperature: f32::NAN,
                        ..SamplingParams::default()
                    },
                    "temperature=NaN",
                    "must not be NaN",
                ),
                (
                    SamplingParams {
                        top_k: Some(0),
                        ..SamplingParams::default()
                    },
                    "top_k=0",
                    "must be at least 1 when set",
                ),
                (
                    SamplingParams {
                        top_p: Some(f32::INFINITY),
                        ..SamplingParams::default()
                    },
                    "top_p=inf",
                    "must be finite",
                ),
                (
                    SamplingParams {
                        top_p: Some(0.0),
                        ..SamplingParams::default()
                    },
                    "top_p=0.0",
                    "must be in (0, 1]",
                ),
                (
                    SamplingParams {
                        top_p: Some(1.1),
                        ..SamplingParams::default()
                    },
                    "top_p=1.1",
                    "must be in (0, 1]",
                ),
                (
                    SamplingParams {
                        presence_penalty: 2.1,
                        ..SamplingParams::default()
                    },
                    "presence_penalty=2.1",
                    "must be in [-2, 2]",
                ),
                (
                    SamplingParams {
                        presence_penalty: f32::NAN,
                        ..SamplingParams::default()
                    },
                    "presence_penalty=NaN",
                    "must be finite",
                ),
                (
                    SamplingParams {
                        frequency_penalty: f32::NEG_INFINITY,
                        ..SamplingParams::default()
                    },
                    "frequency_penalty=-inf",
                    "must be finite",
                ),
                (
                    SamplingParams {
                        frequency_penalty: -2.1,
                        ..SamplingParams::default()
                    },
                    "frequency_penalty=-2.1",
                    "must be in [-2, 2]",
                ),
                (
                    SamplingParams {
                        repetition_penalty: -0.1,
                        ..SamplingParams::default()
                    },
                    "repetition_penalty=-0.1",
                    "must be greater than or equal to 0",
                ),
                (
                    SamplingParams {
                        repetition_penalty: f32::INFINITY,
                        ..SamplingParams::default()
                    },
                    "repetition_penalty=inf",
                    "must be finite",
                ),
                (
                    SamplingParams {
                        max_tokens: 0,
                        ..SamplingParams::default()
                    },
                    "max_tokens=0",
                    "must be at least 1",
                ),
            ];

            for (params, field_and_value, reason) in invalid_cases {
                let error = llm
                    .generate(&[Prompt::TokenIds(vec![1])], &[params])
                    .unwrap_err();
                let message = error.to_string();
                assert!(message.contains("sampling_params[0]"), "{message}");
                assert!(message.contains(field_and_value), "{message}");
                assert!(message.contains(reason), "{message}");
            }

            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();
            assert_eq!(output[0].request_id, 0);
        }

        #[test]
        fn accepted_boundary_values_are_not_rejected() {
            let mut llm = controlled_logits_test_llm(Vec::new());
            let output = llm
                .generate(
                    &[Prompt::TokenIds(vec![2])],
                    &[SamplingParams {
                        temperature: f32::INFINITY,
                        top_k: Some(1_000),
                        top_p: Some(1.0),
                        max_tokens: 1,
                        ignore_eos: true,
                        presence_penalty: -2.0,
                        frequency_penalty: 2.0,
                        repetition_penalty: 0.0,
                    }],
                )
                .unwrap();

            assert_eq!(output.len(), 1);
            assert_eq!(output[0].token_ids.len(), 1);
        }
    }

    mod generate_errors {
        use super::*;

        #[test]
        fn prefix_hit_execution_failure_releases_all_ownership_and_allows_retry() {
            let (mut llm, controls) = causal_fingerprint_test_harness(1_024, 16, true);
            let prompt = Prompt::TokenIds((0..513).collect());
            let params = deterministic_causal_params(1);
            let warmup = llm
                .generate(std::slice::from_ref(&prompt), std::slice::from_ref(&params))
                .unwrap();
            let ownership_before = llm.engine.kv_cache_manager.ownership_snapshot();
            assert_eq!(ownership_before.free_block_ids.len(), 32);
            assert!(ownership_before.used_block_ids.is_empty());
            assert!(ownership_before.ref_counts.iter().all(|&count| count == 0));
            assert_eq!(ownership_before.cache_entries.len(), 2);
            controls.take_metadata();

            controls.fail_next.store(true, Ordering::SeqCst);
            let error = llm
                .generate(
                    &[prompt.clone(), prompt.clone()],
                    &[params.clone(), params.clone()],
                )
                .unwrap_err();

            assert!(format!("{error:#}").contains("injected prefix-cache execution failure"));
            let failed_metadata = controls.take_metadata();
            assert_eq!(failed_metadata.len(), 1);
            assert_eq!(failed_metadata[0].cu_seqlens_q, vec![0, 1, 2]);
            assert_eq!(failed_metadata[0].cu_seqlens_k, vec![0, 513, 1_026]);
            assert_eq!(failed_metadata[0].block_table.len(), 2);
            assert_eq!(
                &failed_metadata[0].block_table[0][..2],
                &failed_metadata[0].block_table[1][..2],
                "both active requests must borrow the same two cached prefix blocks"
            );
            assert!(!llm.engine.is_running());
            assert_eq!(llm.engine.scheduler.num_waiting(), 0);
            assert_eq!(llm.engine.scheduler.num_running(), 0);
            assert_eq!(
                llm.engine.kv_cache_manager.ownership_snapshot(),
                ownership_before,
                "failure cleanup must restore refcounts, used/free ownership, and cache identities"
            );

            let retry = llm
                .generate(std::slice::from_ref(&prompt), std::slice::from_ref(&params))
                .unwrap();
            assert_eq!(retry[0].request_id, 3);
            assert_eq!(retry[0].token_ids, warmup[0].token_ids);
            assert_eq!(retry[0].text, warmup[0].text);
            assert!(retry[0].finished);
            let retry_metadata = controls.take_metadata();
            assert_eq!(retry_metadata[0].cu_seqlens_q, vec![0, 1]);
            assert_eq!(retry_metadata[0].cu_seqlens_k, vec![0, 513]);
            assert_eq!(retry_metadata[0].block_table.len(), 1);
            assert_eq!(llm.engine.kv_cache_manager.num_free_blocks(), 32);
        }

        #[test]
        fn model_execution_failure_is_returned_as_an_error() {
            let mut llm = test_llm_with_forward_failure(true);
            let params = SamplingParams {
                temperature: 0.75,
                top_k: Some(8),
                top_p: Some(0.9),
                max_tokens: 3,
                ignore_eos: true,
                presence_penalty: 0.25,
                frequency_penalty: -0.5,
                repetition_penalty: 1.25,
            };
            let error = llm
                .generate(&[Prompt::TokenIds(vec![1])], &[params])
                .unwrap_err();

            let message = format!("{error:#}");
            assert!(message.contains("injected model execution failure"));
            assert!(message.contains("request_id=0"));
            assert!(message.contains("temperature: 0.75"));
            assert!(message.contains("top_k: Some(8)"));
            assert!(message.contains("top_p: Some(0.9)"));
            assert!(message.contains("max_tokens: 3"));
            assert!(message.contains("ignore_eos: true"));
            assert!(message.contains("presence_penalty: 0.25"));
            assert!(message.contains("frequency_penalty: -0.5"));
            assert!(message.contains("repetition_penalty: 1.25"));
            assert!(!message.contains("sequence_id"));
        }
    }

    #[cfg(feature = "internal-golden")]
    mod internal_golden_capture {
        use super::*;
        use std::ffi::OsString;
        use std::os::unix::fs::PermissionsExt;

        const CHILD_ENV: &str = "VLLM_OXIDE_INTERNAL_GOLDEN_TEST_CHILD";

        struct EnvironmentRestore {
            previous: Vec<(&'static str, Option<OsString>)>,
        }

        impl EnvironmentRestore {
            fn install(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
                let mut previous = Vec::with_capacity(values.len());
                for &(name, value) in values {
                    previous.push((name, std::env::var_os(name)));
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
                Self { previous }
            }
        }

        impl Drop for EnvironmentRestore {
            fn drop(&mut self) {
                for (name, value) in self.previous.drain(..) {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }

        fn enter_isolated_test(test_name: &str) -> bool {
            if std::env::var_os(CHILD_ENV).is_some() {
                return true;
            }
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", test_name, "--nocapture"])
                .env(CHILD_ENV, "1")
                .status()
                .unwrap();
            assert!(status.success(), "isolated capture test process failed");
            false
        }

        #[test]
        fn behavior_runner_preserves_real_errors_and_repeated_public_outputs() {
            if !enter_isolated_test("llm::tests::internal_golden_capture::behavior_runner_preserves_real_errors_and_repeated_public_outputs") { return; }
            let root = tempfile::tempdir().unwrap();
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let plan = root.path().join("behavior.json");
            let output = root.path().join("behavior.capture.json");
            std::fs::write(&plan,serde_json::to_vec(&serde_json::json!({"calls":[
                {"call_id":"invalid","prompts":[[1]],"params":[{"max_tokens":1,"temperature":-1.0}],"expected":"error","error_contains":"temperature"},
                {"call_id":"first","prompts":[[1]],"params":[{"max_tokens":4}],"expected":"success"},
                {"call_id":"second","prompts":[[1]],"params":[{"max_tokens":2,"ignore_eos":true}],"expected":"success"}
            ]})).unwrap()).unwrap();
            std::env::set_var("VLLM_OXIDE_INTERNAL_BEHAVIOR_PLAN", &plan);
            std::env::set_var("VLLM_OXIDE_INTERNAL_BEHAVIOR_OUTPUT", &output);
            let mut llm = controlled_logits_test_llm(vec![3]);
            assert!(llm.generate(&[], &[]).unwrap().is_empty());
            let capture: serde_json::Value =
                serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
            assert!(capture["calls"][0]["error"]
                .as_str()
                .unwrap()
                .contains("temperature"));
            assert!(capture["calls"][0]["binding"].is_null());
            assert_eq!(
                capture["calls"][1]["outputs"][0]["token_ids"],
                serde_json::json!([3])
            );
            assert_eq!(
                capture["calls"][2]["outputs"][0]["token_ids"],
                serde_json::json!([3, 3])
            );
            assert_eq!(
                capture["calls"][2]["binding"]["request_ids"],
                serde_json::json!([1])
            );
            assert_eq!(capture["calls"][2]["binding"]["forcing_enabled"], false);
        }

        #[test]
        fn behavior_binding_observes_unforced_public_generation_and_resolved_eos() {
            if !enter_isolated_test("llm::tests::internal_golden_capture::behavior_binding_observes_unforced_public_generation_and_resolved_eos") { return; }
            let root = tempfile::tempdir().unwrap();
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let path = root.path().join("binding.json");
            let _environment = EnvironmentRestore::install(&[(
                "VLLM_OXIDE_INTERNAL_BEHAVIOR_BINDING",
                Some(path.as_os_str()),
            )]);
            let mut llm = controlled_logits_test_llm(vec![3]);
            let outputs = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        max_tokens: 4,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap();
            assert_eq!(outputs[0].token_ids, vec![3]);
            let binding: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            assert_eq!(
                binding["request_ids"],
                serde_json::json!([outputs[0].request_id])
            );
            assert_eq!(binding["eos_token_ids"], serde_json::json!([3]));
            assert_eq!(binding["forcing_enabled"], false);
        }

        #[test]
        fn fixed_prefix_mixed_batch_admits_waiter_and_uses_advanced_kv_history() {
            if !enter_isolated_test("llm::tests::internal_golden_capture::fixed_prefix_mixed_batch_admits_waiter_and_uses_advanced_kv_history") { return; }
            let root = tempfile::tempdir().unwrap();
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let plan = root.path().join("plan.json");
            let destination = root.path().join("capture.json");
            std::fs::write(&plan,serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,"execution_group_id":"waiting","call_id":"call","vocab_size":100,
                "members":[{"case_id":"short","member_id":"a","prompt":[1],"continuation":[7,8]},
                    {"case_id":"long","member_id":"b","prompt":[2],"continuation":[9,10,11,12]},
                    {"case_id":"waiting","member_id":"c","prompt":[3,4],"continuation":[13,14]}]}).to_string()).unwrap();
            let _environment = EnvironmentRestore::install(&[
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN",
                    Some(plan.as_os_str()),
                ),
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_OUTPUT",
                    Some(destination.as_os_str()),
                ),
            ]);
            let (mut llm, _) = causal_fingerprint_test_harness(128, 2, true);
            let outputs = llm
                .generate(
                    &[
                        Prompt::TokenIds(vec![1]),
                        Prompt::TokenIds(vec![2]),
                        Prompt::TokenIds(vec![3, 4]),
                    ],
                    &[
                        deterministic_causal_params(2),
                        deterministic_causal_params(4),
                        deterministic_causal_params(2),
                    ],
                )
                .unwrap();
            assert_eq!(
                outputs
                    .iter()
                    .map(|o| o.token_ids.clone())
                    .collect::<Vec<_>>(),
                vec![vec![7, 8], vec![9, 10, 11, 12], vec![13, 14]]
            );
            let capture: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(destination).unwrap()).unwrap();
            let rows = capture["rows"].as_array().unwrap();
            assert_eq!(rows.len(), 8);
            assert_eq!(
                rows.iter()
                    .filter(|r| r["member_id"] == "a")
                    .map(|r| r["predicted_token_id"].as_u64().unwrap())
                    .collect::<Vec<_>>(),
                vec![2, 70]
            );
            assert!(capture["execution_events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|event| {
                    let members = event["members"].as_array().unwrap();
                    members
                        .iter()
                        .any(|m| m["request_id"] == 2 && m["phase"] == "prefill")
                        && members
                            .iter()
                            .any(|m| m["request_id"] == 1 && m["phase"] == "decode")
                }));
        }

        #[test]
        fn private_operator_verification_runs_real_cpu_operators_without_generation() {
            if !enter_isolated_test("llm::tests::internal_golden_capture::private_operator_verification_runs_real_cpu_operators_without_generation") { return; }
            let root = tempfile::tempdir().unwrap();
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let plan = root.path().join("operators.json");
            let output = root.path().join("result.json");
            std::fs::write(&plan,serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
                "profiles":[{"profile_id":"rms","rule_id":"materialized_halfway_sum_v1"},
                {"profile_id":"silu","rule_id":"gate_2_up_0.515625_v1"},
                {"profile_id":"sampler","rule_id":"unique_and_multiway_max_with_filter_penalties_v1"}]}).to_string()).unwrap();
            let _environment = EnvironmentRestore::install(&[
                ("VLLM_OXIDE_INTERNAL_OPERATOR_PLAN", Some(plan.as_os_str())),
                (
                    "VLLM_OXIDE_INTERNAL_OPERATOR_OUTPUT",
                    Some(output.as_os_str()),
                ),
            ]);
            let mut llm = test_llm();
            assert!(llm.generate(&[], &[]).unwrap().is_empty());
            let value: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();
            assert_eq!(value["device"], "cpu");
            assert_eq!(
                value["operator_checks"][0]["values"],
                serde_json::json!([1.0, 1.0, 1.0, 1.0])
            );
            assert_eq!(
                value["operator_checks"][1]["values"],
                serde_json::json!([0.90625])
            );
            assert_eq!(
                value["operator_checks"][2]["values"],
                serde_json::json!([7.0, 12.0, 29.0])
            );
            assert!(!llm.engine.is_running());
        }

        #[test]
        fn fixed_prefix_capacity_cap_keeps_real_warmup_and_one_cache_allocation() {
            if !enter_isolated_test("llm::tests::internal_golden_capture::fixed_prefix_capacity_cap_keeps_real_warmup_and_one_cache_allocation") { return; }
            let flag = OsString::from("1");
            let _environment = EnvironmentRestore::install(&[
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CACHE_BLOCKS",
                    Some(flag.as_os_str()),
                ),
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN",
                    Some(flag.as_os_str()),
                ),
            ]);
            let llm = cache_aware_test_llm(true);
            let cache = llm._paged_kv.lock().unwrap();
            assert_eq!(cache.num_blocks(), 1);
            assert_eq!(cache.allocation_count(), 1);
        }

        #[test]
        fn fixed_prefix_advances_frozen_tokens_without_changing_raw_prediction() {
            if !enter_isolated_test("llm::tests::internal_golden_capture::fixed_prefix_advances_frozen_tokens_without_changing_raw_prediction") { return; }
            let temp = tempfile::tempdir().unwrap();
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let plan = temp.path().join("plan.json");
            let output = temp.path().join("fixed.json");
            std::fs::write(
                &plan,
                serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
                "execution_group_id":"toy-group","call_id":"fixed-call","vocab_size":100,
                "members":[{"case_id":"toy","member_id":"a","prompt":[1],"continuation":[3,4]}]})
                .to_string(),
            )
            .unwrap();
            let _environment = EnvironmentRestore::install(&[
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN",
                    Some(plan.as_os_str()),
                ),
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_OUTPUT",
                    Some(output.as_os_str()),
                ),
            ]);
            let mut llm = test_llm();
            let result = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[deterministic_causal_params(2)],
                )
                .unwrap();
            assert_eq!(result[0].token_ids, vec![3, 4]);
            let capture: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();
            assert_eq!(capture["rows"].as_array().unwrap().len(), 2);
            assert_eq!(capture["rows"][0]["predicted_token_id"], 42);
            assert_eq!(capture["rows"][1]["advance_token_id"], 4);
            assert_eq!(capture["rows"][1]["effective_length"], 2);
            assert_eq!(capture["rows"][1]["position"], 1);
            assert_eq!(capture["execution_events"].as_array().unwrap().len(), 2);
            assert_eq!(
                capture["execution_events"][1]["members"][0]["input_token_ids"],
                serde_json::json!([3])
            );
            assert_eq!(
                capture["execution_events"][1]["members"][0]["phase"],
                "decode"
            );
            let control_path = temp.path().join("control.json");
            let control_flag = OsString::from("1");
            let _control_env = EnvironmentRestore::install(&[
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_OUTPUT",
                    Some(control_path.as_os_str()),
                ),
                (
                    "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CONTROL",
                    Some(control_flag.as_os_str()),
                ),
            ]);
            let control = test_llm()
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[deterministic_causal_params(2)],
                )
                .unwrap();
            assert_eq!(control[0].token_ids, vec![42, 42]);
            let returned_text = control[0].text.clone();
            let control: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(control_path).unwrap()).unwrap();
            assert_eq!(control["mode"], "collection_control");
            assert_eq!(
                control["public_call"]["outputs"][0]["token_ids"],
                serde_json::json!([42, 42])
            );
            assert_eq!(control["public_call"]["outputs"][0]["text"], returned_text);
            assert_eq!(control["public_call"]["binding"]["forcing_enabled"], false);
            assert_eq!(
                control["public_call"]["binding"]["request_ids"],
                serde_json::json!([0])
            );
            assert_eq!(control["rows"][0]["logits"], capture["rows"][0]["logits"]);
        }

        #[test]
        fn configured_generate_publishes_complete_capture_through_the_normal_method() {
            if !enter_isolated_test(
                "llm::tests::internal_golden_capture::configured_generate_publishes_complete_capture_through_the_normal_method",
            ) {
                return;
            }
            let temp = tempfile::tempdir().unwrap();
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let destination = OsString::from("capture.jsonl");
            let call_id = OsString::from("llm-generate-call-1");
            let _environment = EnvironmentRestore::install(&[
                (
                    crate::golden_capture::TEMP_DIR_ENV,
                    Some(temp.path().as_os_str()),
                ),
                (
                    crate::golden_capture::DESTINATION_ENV,
                    Some(destination.as_os_str()),
                ),
                (
                    crate::golden_capture::CALL_ID_ENV,
                    Some(call_id.as_os_str()),
                ),
            ]);
            let mut llm = test_llm();

            let outputs = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[deterministic_causal_params(2)],
                )
                .unwrap();

            assert_eq!(outputs[0].token_ids, vec![42, 42]);
            let artifact = std::fs::read_to_string(temp.path().join(destination)).unwrap();
            let lines = artifact.lines().collect::<Vec<_>>();
            assert_eq!(lines.len(), 4);
            assert!(lines[0].contains("llm-generate-call-1"));
            assert!(lines[3].contains("\"complete\":true"));
            assert!(lines[3].contains("\"tensor_shape\":[2,100]"));
        }

        #[test]
        fn partial_capture_configuration_fails_before_request_admission() {
            if !enter_isolated_test(
                "llm::tests::internal_golden_capture::partial_capture_configuration_fails_before_request_admission",
            ) {
                return;
            }
            let temp = tempfile::tempdir().unwrap();
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let _environment = EnvironmentRestore::install(&[
                (
                    crate::golden_capture::TEMP_DIR_ENV,
                    Some(temp.path().as_os_str()),
                ),
                (crate::golden_capture::DESTINATION_ENV, None),
                (crate::golden_capture::CALL_ID_ENV, None),
            ]);
            let mut llm = test_llm();

            let error = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[deterministic_causal_params(1)],
                )
                .unwrap_err();

            assert!(format!("{error:#}").contains("configuration is incomplete"));
            assert!(!llm.engine.is_running());
            assert_eq!(llm.engine.scheduler.num_waiting(), 0);
            assert_eq!(llm.engine.scheduler.num_running(), 0);
        }
    }
}
