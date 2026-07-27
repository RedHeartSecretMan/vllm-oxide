//! `engine/` — V1 three-layer data model (ADR-0004).
//!
//! `Scheduler` / `BlockPool` / `KVCacheManager` / `EngineCore` / `Sequence`.
//! The potential `engine ↔ attention` cycle is broken by the rule: engine
//! holds `Arc<Mutex<PagedKVCache>>` and builds `AttnMetadata` from its own
//! scheduler state; `attention/` never imports `engine/`.
//!
//! # Current status
//!
//! - **T2/#20 (landed)**: `Sequence`, `SequenceGroup`, `SequenceStatus`,
//!   `Block`, `BlockPool`, `KVCacheManager` — the full data-model leaf below
//!   the scheduler.
//! - **T2/#21 (this)**: `EngineCore`, `Scheduler`, `RequestOutput` — the
//!   control loop that wires data model → attention → model → sampler.
//!
//! `EngineCore` itself collapses V1/nano-vllm `ModelRunner` — `step()`
//! performs scheduler → tensor prep → `model.forward()` → sampler → KV
//! update in one method. No `model_runner` sub-module at v0.1. Split trigger
//! (ADR-0004 R5): `step()` exceeds ~300 LOC or CUDA graph capture lands.

#![allow(dead_code)]

pub mod block_pool;
pub mod kv_cache_manager;
pub mod scheduler;
pub mod sequence;

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor};

use crate::Sampler;
use crate::SamplingParams;
use crate::attention::{PagedKVCache, AttnMetadata, build_prefill_metadata, build_decode_metadata};
use crate::models::CausalLM;

pub use block_pool::BlockPoolError;
pub use kv_cache_manager::KvCacheManager;
pub use scheduler::{RequestOutput, ScheduleMode, ScheduleOutput, Scheduler};
pub use sequence::{Sequence, SequenceGroup, SequenceStatus};

/// In-process engine core — collapses V1/nano-vllm's `ModelRunner` (ADR-0004).
///
/// `step()` performs scheduler → tensor prep → `model.forward()` →
/// sampler → KV update in one method. Holds the shared `PagedKVCache`
/// and `AttnMetadata` via `Arc<Mutex<>>` so attention layers can read
/// them during the forward pass.
pub struct EngineCore {
    pub scheduler: Scheduler,
    pub kv_cache_manager: KvCacheManager,
    model: Box<dyn CausalLM>,
    sampler: Sampler,
    paged_kv: Arc<Mutex<PagedKVCache>>,
    attn_meta: Arc<Mutex<AttnMetadata>>,
    device: Device,
}

impl EngineCore {
    pub fn new(
        scheduler: Scheduler,
        kv_cache_manager: KvCacheManager,
        model: Box<dyn CausalLM>,
        sampler: Sampler,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        attn_meta: Arc<Mutex<AttnMetadata>>,
        device: Device,
    ) -> Self {
        Self {
            scheduler,
            kv_cache_manager,
            model,
            sampler,
            paged_kv,
            attn_meta,
            device,
        }
    }

    /// Add a new inference request and schedule it.
    pub fn add_request(&mut self, prompt: Vec<u32>, params: SamplingParams) {
        self.scheduler.add_request(prompt, params);
    }

    /// One step of the engine loop: schedule → forward → sample → KV update.
    ///
    /// Returns `RequestOutput`s for any sequences that finished this step.
    /// When no work remains, returns `Ok(Vec::new())`.
    pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
        let (outputs, _logits) = self.step_with_logits()?;
        Ok(outputs)
    }

    /// Like [`step`], but also returns the pre-sampling logits tensor.
    ///
    /// The returned tensor has shape `[batch, vocab_size]` and dtype FP32.
    /// Used by `LLM::generate_logits` (T12/#23) for L2 golden comparison.
    ///
    /// When no work remains, returns `Ok((Vec::new(), Tensor::zeros(...)))`.
    pub fn step_with_logits(&mut self) -> Result<(Vec<RequestOutput>, Tensor)> {
        let output = self.scheduler.schedule(&mut self.kv_cache_manager);

        if self.scheduler.num_running() == 0 {
            let empty = Tensor::zeros((0, 0), DType::F32, &self.device)?;
            return Ok((Vec::new(), empty));
        }

        let logits = match output.mode {
            ScheduleMode::Prefill => self.forward_prefill()?,
            ScheduleMode::Decode => self.forward_decode()?,
        };

        let (outputs, logits_clone) = self.sample_and_postprocess_with_logits(&logits)?;
        Ok((outputs, logits_clone))
    }

    /// Whether there are any pending or running sequences.
    pub fn is_running(&self) -> bool {
        self.scheduler.is_running()
    }

    /// Sample from logits, postprocess sequences, and return both the
    /// finished outputs and a clone of the pre-sampling logits (FP32,
    /// shape `[batch, vocab_size]`).
    fn sample_and_postprocess_with_logits(
        &mut self,
        logits: &Tensor,
    ) -> Result<(Vec<RequestOutput>, Tensor)> {
        let params: Vec<SamplingParams> = self
            .scheduler
            .running_seqs()
            .map(|g| {
                let s = g.seq();
                SamplingParams {
                    temperature: s.temperature,
                    max_tokens: s.max_tokens,
                    ignore_eos: s.ignore_eos,
                    ..SamplingParams::default()
                }
            })
            .collect();

        let token_histories: Vec<Vec<u32>> = self
            .scheduler
            .running_seqs()
            .map(|g| g.seq().token_ids.clone())
            .collect();

        let sampled = self.sampler.forward(logits, &params, &token_histories)?;
        let sampled_ids = sampled.to_vec1::<u32>()?;

        let outputs = self
            .scheduler
            .postprocess(&sampled_ids, &mut self.kv_cache_manager);

        // Convert to FP32 for numerical comparison (matches oracle format).
        let logits_fp32 = logits.to_dtype(DType::F32)?;

        Ok((outputs, logits_fp32))
    }

    /// Run the prefill forward pass and return pre-sampling logits
    /// (shape `[batch, vocab_size]`). Does NOT sample — the caller
    /// is responsible for calling `sample_and_postprocess_with_logits`.
    fn forward_prefill(&mut self) -> Result<Tensor> {
        let batch = self.scheduler.num_running();
        if batch == 0 {
            return Tensor::zeros((0, 0), DType::F32, &self.device);
        }

        let mut all_input_ids: Vec<u32> = Vec::new();
        let mut all_positions: Vec<u32> = Vec::new();
        let mut seq_starts: Vec<usize> = Vec::new();
        let mut seq_lens: Vec<usize> = Vec::new();
        let mut slot_mapping: Vec<i64> = Vec::new();
        let mut kv_lengths: Vec<u32> = Vec::new();

        let mut cumsum = 0usize;
        for group in self.scheduler.running_seqs() {
            let seq = group.seq();
            let n = seq.num_scheduled_tokens;
            let start = seq.num_cached_tokens.saturating_sub(n);

            let tokens = &seq.token_ids[start..start + n];
            all_input_ids.extend_from_slice(tokens);

            let pos_base = seq.num_cached_tokens.saturating_sub(n) as u32;
            for k in 0..n {
                all_positions.push(pos_base + k as u32);
            }

            seq_starts.push(cumsum);
            cumsum += n;
            seq_lens.push(n);

            let sm = self
                .kv_cache_manager
                .compute_slot_mapping(seq, start, n);
            slot_mapping.extend(sm);

            kv_lengths.push((seq.num_cached_tokens.saturating_sub(n) + n) as u32);
        }

        let total_tokens = all_input_ids.len();

        let input_ids = Tensor::from_vec(all_input_ids, total_tokens, &self.device)?;
        let positions = Tensor::from_vec(all_positions, total_tokens, &self.device)?;

        let scheduled_tokens: Vec<u32> = seq_lens.iter().map(|&l| l as u32).collect();

        let meta = build_prefill_metadata(&scheduled_tokens, &kv_lengths, &slot_mapping);
        {
            let mut lock = self.attn_meta.lock().unwrap();
            *lock = meta;
        }

        let hidden = self.model.forward(&input_ids, &positions)?;

        let mut last_hiddens = Vec::new();
        for (i, &start) in seq_starts.iter().enumerate() {
            let len = seq_lens[i];
            let last_idx = start + len - 1;
            let h = hidden.get(last_idx)?;
            last_hiddens.push(h);
        }

        let logits_hidden = Tensor::stack(&last_hiddens.iter().collect::<Vec<&Tensor>>(), 0)?;
        self.model.compute_logits(&logits_hidden)
    }

    /// Run the decode forward pass and return pre-sampling logits
    /// (shape `[batch, vocab_size]`). Does NOT sample — the caller
    /// is responsible for calling `sample_and_postprocess_with_logits`.
    fn forward_decode(&mut self) -> Result<Tensor> {
        let batch = self.scheduler.num_running();
        if batch == 0 {
            return Tensor::zeros((0, 0), DType::F32, &self.device);
        }

        let mut last_tokens: Vec<u32> = Vec::new();
        let mut positions: Vec<u32> = Vec::new();
        let mut context_lens: Vec<u32> = Vec::new();
        let mut slot_mapping: Vec<i64> = Vec::new();
        let mut block_tables: Vec<Vec<i32>> = Vec::new();

        for group in self.scheduler.running_seqs() {
            let seq = group.seq();

            last_tokens.push(seq.last_token);
            let pos = seq.num_tokens.saturating_sub(1) as u32;
            positions.push(pos);
            context_lens.push(seq.num_tokens as u32);

            let sm = self.kv_cache_manager.compute_slot_mapping(seq, pos as usize, 1);
            slot_mapping.extend(sm);

            let bt: Vec<i32> = seq.block_table.iter().map(|&id| id as i32).collect();
            block_tables.push(bt);
        }

        let input_ids = Tensor::from_vec(last_tokens, batch, &self.device)?;
        let pos_tensor = Tensor::from_vec(positions, batch, &self.device)?;

        let meta = build_decode_metadata(&context_lens, &block_tables, &slot_mapping);
        {
            let mut lock = self.attn_meta.lock().unwrap();
            *lock = meta;
        }

        let hidden = self.model.forward(&input_ids, &pos_tensor)?;
        self.model.compute_logits(&hidden)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use candle_core::DType;
    use crate::engine::sequence::BLOCK_SIZE;
    use crate::Sampler;

    /// A mock CausalLM that always returns hidden states where token 7
    /// (not the Qwen3 EOS token 151645) has the highest logit. This lets
    /// the greedy sampler produce a predictable non-EOS token sequence.
    struct MockModel {
        hidden_size: usize,
        vocab_size: usize,
        target_token: u32,
        device: Device,
    }

    impl MockModel {
        fn new(target_token: u32, device: Device) -> Self {
            Self {
                hidden_size: 64,
                vocab_size: 100,
                target_token,
                device: device.clone(),
            }
        }
    }

    impl CausalLM for MockModel {
        fn forward(&mut self, input_ids: &Tensor, _positions: &Tensor) -> Result<Tensor> {
            let n_tokens = input_ids.dim(0)?;
            Tensor::zeros((n_tokens, self.hidden_size), DType::F32, &self.device)
        }

        fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
            let n = hidden_states.dim(0)?;
            let vocab = self.vocab_size;
            let mut data = vec![-100.0_f32; n * vocab];

            for i in 0..n {
                data[i * vocab + self.target_token as usize] = 100.0;
            }

            Tensor::from_vec(data, (n, vocab), &self.device)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn make_engine(
        scheduler: Scheduler,
        kv_mgr: KvCacheManager,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        attn_meta: Arc<Mutex<AttnMetadata>>,
        device: &Device,
    ) -> EngineCore {
        let model = Box::new(MockModel::new(42, device.clone()));
        let sampler = Sampler::new_with_seed(0);
        EngineCore::new(scheduler, kv_mgr, model, sampler, paged_kv, attn_meta, device.clone())
    }

    fn make_fake_cache() -> Arc<Mutex<PagedKVCache>> {
        Arc::new(Mutex::new(
            PagedKVCache::new(1, 32, 256, 1, 64, DType::F32, &Device::Cpu).unwrap(),
        ))
    }

    fn make_fake_meta() -> Arc<Mutex<AttnMetadata>> {
        Arc::new(Mutex::new(build_prefill_metadata(&[], &[], &[])))
    }

    /// T8 Q8.3 invariant #6: max_tokens boundary respected.
    ///
    /// Cross-reference with #17: sampler is single-step and cannot enforce
    /// the boundary itself, so EngineCore.step() must stop calling
    /// `Sampler::forward` once a sequence's sampled-token count reaches
    /// `params.max_tokens`.
    #[test]
    fn max_tokens_boundary_stops_generation() {
        let device = Device::Cpu;

        let mut scheduler = Scheduler::with_defaults();
        let paged_kv = make_fake_cache();
        let attn_meta = make_fake_meta();
        let kv_mgr = KvCacheManager::new(100, BLOCK_SIZE, paged_kv.clone());

        scheduler.add_request(
            vec![1, 2, 3],
            SamplingParams {
                max_tokens: 5,
                ..SamplingParams::default()
            },
        );

        let mut engine = make_engine(scheduler, kv_mgr, paged_kv, attn_meta, &device);

        let outputs = engine.step().unwrap();
        assert!(outputs.is_empty(), "prefill should not finish with max_tokens=5");

        let mut total_outputs = 0;
        for step_num in 1..=10 {
            if !engine.is_running() {
                break;
            }
            let outputs = engine.step().unwrap();
            total_outputs += outputs.len();

            if step_num == 5 {
                assert!(total_outputs > 0, "should finish by step 5");
                break;
            }
        }

        assert!(!engine.is_running(), "engine should stop after max_tokens reached");
    }

    #[test]
    fn empty_engine_returns_empty() {
        let device = Device::Cpu;
        let scheduler = Scheduler::with_defaults();
        let paged_kv = make_fake_cache();
        let attn_meta = make_fake_meta();
        let kv_mgr = KvCacheManager::new(10, BLOCK_SIZE, paged_kv.clone());

        let mut engine = make_engine(scheduler, kv_mgr, paged_kv, attn_meta, &device);
        let outputs = engine.step().unwrap();
        assert!(outputs.is_empty());
        assert!(!engine.is_running());
    }

    #[test]
    fn add_request_makes_engine_running() {
        let device = Device::Cpu;
        let mut scheduler = Scheduler::with_defaults();
        let paged_kv = make_fake_cache();
        let attn_meta = make_fake_meta();
        let kv_mgr = KvCacheManager::new(10, BLOCK_SIZE, paged_kv.clone());

        scheduler.add_request(vec![1, 2, 3], SamplingParams::default());
        let engine = make_engine(scheduler, kv_mgr, paged_kv, attn_meta, &device);
        assert!(engine.is_running());
    }
}
