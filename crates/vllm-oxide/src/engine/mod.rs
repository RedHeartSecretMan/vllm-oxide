//! `engine/` — V1 three-layer data model (ADR-0004).
//!
//! `Scheduler` / `BlockPool` / `KVCacheManager` / `EngineCore` / `Sequence`.
//! The potential `engine ↔ attention` cycle is broken by the rule: engine
//! holds `Arc<Mutex<PagedKVCache>>` and builds `AttnMetadata` from its own
//! scheduler state; `attention/` never imports `engine/`.
//!
//! Scheduler-owned state is captured in one immutable `StepPlan`.
//! `EngineCore` executes only that plan and returns a corresponding
//! `StepResult`; the Scheduler applies the result exactly once. This keeps
//! scheduling and lifecycle decisions out of model execution (ADR-0004).

#![allow(dead_code)]

pub mod block_pool;
pub mod kv_cache_manager;
pub mod scheduler;
pub mod sequence;
mod step;

use candle_core::{DType, Device, Result, Tensor};

use crate::attention::AttentionContext;
use crate::causal_lm::CausalLM;
use crate::sampler::selected_token_ids_to_host;
use crate::Sampler;
use crate::SamplingParams;

pub use block_pool::BlockPoolError;
pub use kv_cache_manager::KvCacheManager;
pub use scheduler::{RequestOutput, Scheduler};
pub use sequence::{Sequence, SequenceStatus};

pub(crate) use step::{
    AdmissionBlockedReason, BlockedAdmission, CacheOperation, SequenceCachePlan, SequencePhase,
    SequenceStepPlan, SequenceStepResult, StepPhase, StepPlan, StepPlanError, StepResult,
};

/// In-process engine core — collapses V1/nano-vllm's `ModelRunner` (ADR-0004).
///
/// Holds the model-execution side of the internal `StepPlan` / `StepResult`
/// seam and the shared attention state (`AttentionContext`). Scheduling,
/// lifecycle, and result application remain owned by [`Scheduler`].
pub struct EngineCore {
    pub scheduler: Scheduler,
    pub kv_cache_manager: KvCacheManager,
    model: Box<dyn CausalLM>,
    sampler: Sampler,
    attn_ctx: AttentionContext,
    device: Device,
}

impl EngineCore {
    pub fn new(
        scheduler: Scheduler,
        kv_cache_manager: KvCacheManager,
        model: Box<dyn CausalLM>,
        sampler: Sampler,
        attn_ctx: AttentionContext,
        device: Device,
    ) -> Self {
        Self {
            scheduler,
            kv_cache_manager,
            model,
            sampler,
            attn_ctx,
            device,
        }
    }

    /// Add a new inference request and return its stable public request identity.
    pub fn add_request(&mut self, prompt: Vec<u32>, params: SamplingParams) -> usize {
        self.scheduler.add_request(prompt, params)
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
        let plan = match self.scheduler.plan_step(&mut self.kv_cache_manager) {
            Ok(plan) => plan,
            Err(error) => {
                return Err(self.cleanup_failed_step(candle_core::Error::msg(error)));
            }
        };
        let Some(plan) = plan else {
            let empty = Tensor::zeros((0, 0), DType::F32, &self.device)?;
            return Ok((Vec::new(), empty));
        };

        let (result, logits) = match self.execute_plan(&plan) {
            Ok(executed) => executed,
            Err(error) => return Err(self.cleanup_failed_step(error)),
        };
        let outputs = match self
            .scheduler
            .apply_step_result(&result, &mut self.kv_cache_manager)
        {
            Ok(outputs) => outputs,
            Err(error) => {
                return Err(self.cleanup_failed_step(candle_core::Error::msg(error)));
            }
        };
        Ok((outputs, logits))
    }

    fn cleanup_failed_step(&mut self, error: candle_core::Error) -> candle_core::Error {
        match self
            .scheduler
            .abort_all_requests(&mut self.kv_cache_manager)
        {
            Ok(()) => error,
            Err(cleanup_error) => candle_core::Error::Msg(format!(
                "{error}; cache ownership cleanup after engine failure also failed: {cleanup_error}"
            )),
        }
    }

    /// Execute exactly the immutable work captured in `plan`.
    fn execute_plan(&mut self, plan: &StepPlan) -> Result<(StepResult, Tensor)> {
        if plan.sequences.is_empty() || plan.token_budget == 0 {
            candle_core::bail!("cannot execute an empty step plan")
        }

        let input_token_ids = plan
            .sequences
            .iter()
            .flat_map(|sequence| sequence.input_token_ids.iter().copied())
            .collect::<Vec<_>>();
        let logical_positions = plan
            .sequences
            .iter()
            .flat_map(|sequence| sequence.logical_positions.clone())
            .map(|position| {
                u32::try_from(position).map_err(|_| {
                    candle_core::Error::Msg("logical position does not fit u32".to_string())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if input_token_ids.len() != plan.token_budget
            || logical_positions.len() != plan.token_budget
        {
            candle_core::bail!("step plan token budget does not match its inputs")
        }

        let input_ids = Tensor::from_vec(input_token_ids, plan.token_budget, &self.device)?;
        let positions = Tensor::from_vec(logical_positions, plan.token_budget, &self.device)?;
        {
            // Engine execution is single-threaded; this mutex is never poisoned.
            #[allow(clippy::unwrap_used)]
            let mut attention = self.attn_ctx.attn_meta.lock().unwrap();
            *attention = plan.attention.clone();
        }

        let hidden = self.model.forward(&input_ids, &positions)?;
        let mut offset = 0usize;
        let mut sample_hiddens = Vec::new();
        let mut sampling_params = Vec::new();
        let mut token_histories = Vec::new();
        for sequence in &plan.sequences {
            if sequence.sampling_allowed {
                sample_hiddens.push(hidden.get(offset + sequence.token_budget - 1)?);
                sampling_params.push(sequence.sampling_params.clone());
                token_histories.push(sequence.token_history.clone());
            }
            offset += sequence.token_budget;
        }

        let (sampled_tokens, logits) = if sample_hiddens.is_empty() {
            (Vec::new(), Tensor::zeros((0, 0), DType::F32, &self.device)?)
        } else {
            let refs = sample_hiddens.iter().collect::<Vec<_>>();
            let logits = self
                .model
                .compute_logits(&Tensor::stack(&refs, 0)?)?
                .to_dtype(DType::F32)
                .map_err(|error| {
                    candle_core::Error::msg(format!(
                        "sampling FP32 upcast failed for {}: {error}",
                        sampling_diagnostics(plan)
                    ))
                })?;
            let sampled_device = self
                .sampler
                .forward(&logits, &sampling_params, &token_histories)
                .map_err(|error| {
                    candle_core::Error::msg(format!(
                        "sampling failed for {}: {error}",
                        sampling_diagnostics(plan)
                    ))
                })?;
            let sampled = selected_token_ids_to_host(&sampled_device).map_err(|error| {
                candle_core::Error::msg(format!(
                    "selected-token transfer failed for {}: {error}",
                    sampling_diagnostics(plan)
                ))
            })?;
            (sampled, logits)
        };

        let mut sampled_tokens = sampled_tokens.into_iter();
        let sequences = plan
            .sequences
            .iter()
            .map(|sequence| SequenceStepResult {
                request_id: sequence.request_id,
                sequence_id: sequence.sequence_id,
                sampled_token: sequence
                    .sampling_allowed
                    .then(|| sampled_tokens.next())
                    .flatten(),
            })
            .collect();

        Ok((
            StepResult {
                plan_id: plan.id,
                sequences,
            },
            logits,
        ))
    }

    /// Whether there are any pending or running sequences.
    pub fn is_running(&self) -> bool {
        self.scheduler.is_running()
    }
}

fn sampling_diagnostics(plan: &StepPlan) -> String {
    plan.sequences
        .iter()
        .filter(|sequence| sequence.sampling_allowed)
        .enumerate()
        .map(|(sampling_row, sequence)| {
            format!(
                "sampling_row={sampling_row}, request_id={}, sampling_params={:?}",
                sequence.request_id, sequence.sampling_params,
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::attention::{build_prefill_metadata, AttnMetadata, PagedKVCache};
    use crate::engine::sequence::BLOCK_SIZE;
    use crate::Sampler;
    use candle_core::DType;

    /// A mock CausalLM that always returns hidden states where token 7
    /// (not the Qwen3 EOS token 151645) has the highest logit. This lets
    /// the greedy sampler produce a predictable non-EOS token sequence.
    struct MockModel {
        hidden_size: usize,
        vocab_size: usize,
        target_token: u32,
        device: Device,
    }

    struct RecordingModel {
        seen_input_ids: Arc<Mutex<Vec<Vec<u32>>>>,
        device: Device,
    }

    struct InvalidSamplingLogitsModel {
        device: Device,
    }

    impl CausalLM for RecordingModel {
        fn forward(&mut self, input_ids: &Tensor, _positions: &Tensor) -> Result<Tensor> {
            self.seen_input_ids
                .lock()
                .unwrap()
                .push(input_ids.to_vec1::<u32>()?);
            Tensor::zeros((input_ids.dim(0)?, 64), DType::F32, &self.device)
        }

        fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
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

        fn device(&self) -> &Device {
            &self.device
        }
    }

    impl CausalLM for InvalidSamplingLogitsModel {
        fn forward(&mut self, input_ids: &Tensor, _positions: &Tensor) -> Result<Tensor> {
            Tensor::zeros((input_ids.dim(0)?, 64), DType::F32, &self.device)
        }

        fn compute_logits(&self, _hidden_states: &Tensor) -> Result<Tensor> {
            Tensor::zeros(100, DType::F32, &self.device)
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn device(&self) -> &Device {
            &self.device
        }
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
        attn_ctx: AttentionContext,
        device: &Device,
    ) -> EngineCore {
        let model = Box::new(MockModel::new(42, device.clone()));
        let sampler = Sampler::new_with_seed(0);
        EngineCore::new(scheduler, kv_mgr, model, sampler, attn_ctx, device.clone())
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
        let attn_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let kv_mgr = KvCacheManager::new(100, BLOCK_SIZE, attn_ctx.paged_kv.clone());

        scheduler.add_request(
            vec![1, 2, 3],
            SamplingParams {
                max_tokens: 5,
                ..SamplingParams::default()
            },
        );

        let mut engine = make_engine(scheduler, kv_mgr, attn_ctx, &device);

        let outputs = engine.step().unwrap();
        assert!(
            outputs.is_empty(),
            "prefill should not finish with max_tokens=5"
        );

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

        assert!(
            !engine.is_running(),
            "engine should stop after max_tokens reached"
        );
    }

    #[test]
    fn empty_engine_returns_empty() {
        let device = Device::Cpu;
        let scheduler = Scheduler::with_defaults();
        let attn_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let kv_mgr = KvCacheManager::new(10, BLOCK_SIZE, attn_ctx.paged_kv.clone());

        let mut engine = make_engine(scheduler, kv_mgr, attn_ctx, &device);
        let outputs = engine.step().unwrap();
        assert!(outputs.is_empty());
        assert!(!engine.is_running());
    }

    #[test]
    fn add_request_makes_engine_running() {
        let device = Device::Cpu;
        let mut scheduler = Scheduler::with_defaults();
        let attn_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let kv_mgr = KvCacheManager::new(10, BLOCK_SIZE, attn_ctx.paged_kv.clone());

        scheduler.add_request(vec![1, 2, 3], SamplingParams::default());
        let engine = make_engine(scheduler, kv_mgr, attn_ctx, &device);
        assert!(engine.is_running());
    }

    #[test]
    fn execute_plan_uses_only_plan_membership() {
        let device = Device::Cpu;
        let source_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let mut source_scheduler = Scheduler::with_defaults();
        let mut source_kv = KvCacheManager::new(100, BLOCK_SIZE, source_ctx.paged_kv.clone());
        source_scheduler.add_request(vec![11, 12, 13], SamplingParams::default());
        let plan = source_scheduler.plan_step(&mut source_kv).unwrap().unwrap();

        let engine_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let mut decoy_scheduler = Scheduler::with_defaults();
        decoy_scheduler.add_request(vec![99], SamplingParams::default());
        let engine_kv = KvCacheManager::new(100, BLOCK_SIZE, engine_ctx.paged_kv.clone());
        let seen_input_ids = Arc::new(Mutex::new(Vec::new()));
        let model = Box::new(RecordingModel {
            seen_input_ids: seen_input_ids.clone(),
            device: device.clone(),
        });
        let mut engine = EngineCore::new(
            decoy_scheduler,
            engine_kv,
            model,
            Sampler::new_with_seed(0),
            engine_ctx,
            device,
        );

        let (result, _logits) = engine.execute_plan(&plan).unwrap();

        assert_eq!(*seen_input_ids.lock().unwrap(), vec![vec![11, 12, 13]]);
        assert_eq!(result.plan_id, plan.id);
        assert_eq!(result.sequences.len(), 1);
        assert_eq!(result.sequences[0].request_id, 0);
        assert_eq!(result.sequences[0].sequence_id, 0);
        assert_eq!(result.sequences[0].sampled_token, Some(42));
    }

    #[test]
    fn sampling_tensor_error_retains_request_context() {
        let device = Device::Cpu;
        let mut scheduler = Scheduler::with_defaults();
        let attn_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let kv_mgr = KvCacheManager::new(100, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        scheduler.add_request(
            vec![11],
            SamplingParams {
                temperature: 0.75,
                top_k: Some(8),
                top_p: Some(0.9),
                presence_penalty: 0.25,
                frequency_penalty: -0.5,
                repetition_penalty: 1.25,
                ..SamplingParams::default()
            },
        );
        scheduler.add_request(
            vec![22],
            SamplingParams {
                temperature: 1.0,
                top_k: Some(2),
                ..SamplingParams::default()
            },
        );
        let mut engine = EngineCore::new(
            scheduler,
            kv_mgr,
            Box::new(InvalidSamplingLogitsModel {
                device: device.clone(),
            }),
            Sampler::new_with_seed(0),
            attn_ctx,
            device,
        );

        let error = engine.step().unwrap_err().to_string();

        assert!(error.contains("sampling failed"));
        assert!(error.contains("sampling_row=0, request_id=0"));
        assert!(error.contains("sampling_row=1, request_id=1"));
        assert!(error.contains("temperature: 0.75"));
        assert!(error.contains("top_k: Some(8)"));
        assert!(error.contains("top_p: Some(0.9)"));
        assert!(error.contains("presence_penalty: 0.25"));
        assert!(error.contains("frequency_penalty: -0.5"));
        assert!(error.contains("repetition_penalty: 1.25"));
        assert!(!error.contains("sequence_id"));
        assert!(!engine.is_running());
    }

    #[test]
    fn engine_step_obeys_plan_sampling_permission() {
        let device = Device::Cpu;
        let mut scheduler = Scheduler::new(2, 512, 0.9);
        let attn_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let kv_mgr = KvCacheManager::new(100, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        scheduler.add_request(vec![11, 12, 13], SamplingParams::default());
        let mut engine = make_engine(scheduler, kv_mgr, attn_ctx, &device);

        let outputs = engine.step().unwrap();

        assert!(outputs.is_empty());
        let sequence = engine.scheduler.running_seqs().next().unwrap();
        assert_eq!(sequence.num_cached_tokens, 2);
        assert_eq!(sequence.num_completion_tokens(), 0);
    }

    #[test]
    fn batched_greedy_generation_remains_deterministic_through_step_plans() {
        let device = Device::Cpu;
        let mut scheduler = Scheduler::with_defaults();
        let attn_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let kv_mgr = KvCacheManager::new(100, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        let params = SamplingParams {
            max_tokens: 2,
            ..SamplingParams::default()
        };
        scheduler.add_request(vec![11, 12, 13], params.clone());
        scheduler.add_request(vec![21], params);
        let mut engine = make_engine(scheduler, kv_mgr, attn_ctx, &device);
        let mut outputs = Vec::new();

        while engine.is_running() {
            outputs.extend(engine.step().unwrap());
        }
        outputs.sort_by_key(|output| output.request_id);

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].request_id, 0);
        assert_eq!(outputs[0].token_ids, vec![42, 42]);
        assert_eq!(outputs[1].request_id, 1);
        assert_eq!(outputs[1].token_ids, vec![42, 42]);
    }

    #[test]
    fn staggered_arrivals_interleave_and_complete_once_with_stable_identity() {
        let device = Device::Cpu;
        let scheduler = Scheduler::new(3, 3, 0.9);
        let attn_ctx = AttentionContext {
            paged_kv: make_fake_cache(),
            attn_meta: make_fake_meta(),
        };
        let kv_mgr = KvCacheManager::new(100, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        let seen_input_ids = Arc::new(Mutex::new(Vec::new()));
        let model = Box::new(RecordingModel {
            seen_input_ids: seen_input_ids.clone(),
            device: device.clone(),
        });
        let mut engine = EngineCore::new(
            scheduler,
            kv_mgr,
            model,
            Sampler::new_with_seed(0),
            attn_ctx,
            device,
        );
        engine.add_request(
            vec![1],
            SamplingParams {
                max_tokens: 3,
                ..SamplingParams::default()
            },
        );
        assert!(engine.step().unwrap().is_empty());
        engine.add_request(
            vec![4, 5],
            SamplingParams {
                max_tokens: 1,
                ..SamplingParams::default()
            },
        );
        engine.add_request(
            vec![6, 7, 8, 9],
            SamplingParams {
                max_tokens: 2,
                ..SamplingParams::default()
            },
        );

        let mut outputs = Vec::new();
        for _ in 0..10 {
            if !engine.is_running() {
                break;
            }
            outputs.extend(engine.step().unwrap());
        }

        assert!(!engine.is_running());
        assert_eq!(
            &seen_input_ids.lock().unwrap()[..3],
            &[vec![1], vec![42, 4, 5], vec![42, 6, 7]]
        );
        outputs.sort_by_key(|output| output.request_id);
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs
                .iter()
                .map(|output| (output.request_id, output.token_ids.clone(), output.finished))
                .collect::<Vec<_>>(),
            vec![
                (0, vec![42, 42, 42], true),
                (1, vec![42], true),
                (2, vec![42, 42], true),
            ]
        );
    }
}
