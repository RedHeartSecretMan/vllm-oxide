//! Token-level scheduler — `Scheduler` (V1 parity, nano-vllm algorithm).
//!
//! One `plan_step()` is either **pure-prefill** or **pure-decode**
//! (nano-vllm parity — simplifies attention metadata). Every participating
//! sequence spends from one global token budget, and sequences with no
//! positive work remain outside the plan. Preemption is **recompute-only**:
//! deallocate the sequence's blocks and requeue at the front of `waiting`
//! (V1 parity; v0.1 never swaps KV to host — V0 `swap_space` dropped).
//!
//! Each decision is captured in an immutable `StepPlan`; applying its
//! corresponding `StepResult` updates block hashes, advances cached tokens,
//! appends sampled tokens, and finalises sequences on EOS or `max_tokens`
//! exactly once. Scheduler talks to `KvCacheManager`, never to `BlockPool`
//! or `PagedKVCache` directly (ADR-0004 seam).

use std::collections::VecDeque;

use crate::attention::{build_decode_metadata, build_prefill_metadata};
use crate::engine::kv_cache_manager::KvCacheManager;
use crate::engine::sequence::{Sequence, SequenceStatus};
use crate::engine::{
    CacheOperation, SequenceCachePlan, SequenceStepPlan, StepPhase, StepPlan, StepPlanError,
    StepResult,
};
use crate::SamplingParams;

/// Default `max_num_batched_tokens` — the maximum number of tokens the engine
/// can process in one step. User story #21 (v0.1-spec).
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 16384;

/// Default `max_num_seqs` — the maximum number of concurrently-running
/// sequences. User story #22 (v0.1-spec).
pub const DEFAULT_MAX_NUM_SEQS: usize = 512;

/// Default `gpu_memory_utilization` — fraction of GPU memory to allocate
/// to the KV cache pool. User story #23 (v0.1-spec).
pub const DEFAULT_GPU_MEMORY_UTILIZATION: f32 = 0.9;

fn usize_to_u32(value: usize, name: &str) -> Result<u32, StepPlanError> {
    u32::try_from(value).map_err(|_| StepPlanError::invalid(format!("{name} does not fit u32")))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkSelection {
    phase: StepPhase,
}

/// Token-level scheduler — the algorithmic heart of the engine.
///
/// Maintains `waiting` / `running` deques of `Sequence`s. One `plan_step()`
/// call produces either a pure-prefill or pure-decode [`StepPlan`].
/// `apply_step_result()` advances state and produces `RequestOutput`s for
/// finished sequences.
///
/// # V1 three-layer split
///
/// Scheduler talks to `KvCacheManager`, never to `BlockPool` or
/// `PagedKVCache` directly (ADR-0004 seam).
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
    max_num_batched_tokens: usize,
    max_num_seqs: usize,
    eos_token_ids: Vec<u32>,
    /// Fraction of GPU memory reserved for the KV cache pool.
    /// Used by EngineCore to compute pool size; the scheduler itself
    /// does not allocate the pool.
    pub gpu_memory_utilization: f32,

    // Monotonic counters
    next_seq_id: usize,
    next_request_id: usize,
    next_plan_id: u64,
    in_flight: Option<StepPlan>,
}

impl Scheduler {
    /// Create a new scheduler with configurable budgets.
    pub fn new(
        max_num_batched_tokens: usize,
        max_num_seqs: usize,
        gpu_memory_utilization: f32,
    ) -> Self {
        Self::new_with_eos_token_ids(
            max_num_batched_tokens,
            max_num_seqs,
            gpu_memory_utilization,
            Vec::new(),
        )
    }

    /// Create a scheduler whose stop condition uses model-resolved EOS ids.
    pub(crate) fn new_with_eos_token_ids(
        max_num_batched_tokens: usize,
        max_num_seqs: usize,
        gpu_memory_utilization: f32,
        eos_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            max_num_batched_tokens,
            max_num_seqs,
            eos_token_ids,
            gpu_memory_utilization,
            next_seq_id: 0,
            next_request_id: 0,
            next_plan_id: 0,
            in_flight: None,
        }
    }

    /// Create a scheduler with default budgets (user stories #21–#23).
    pub fn with_defaults() -> Self {
        Self::new(
            DEFAULT_MAX_NUM_BATCHED_TOKENS,
            DEFAULT_MAX_NUM_SEQS,
            DEFAULT_GPU_MEMORY_UTILIZATION,
        )
    }

    /// Add a new inference request. Creates a `Sequence` directly
    /// (the former 1:1 `SequenceGroup` wrapper has been absorbed).
    ///
    /// The new sequence starts in `Waiting` status. Returns the stable public
    /// request identity assigned to the accepted prompt.
    pub fn add_request(&mut self, prompt_token_ids: Vec<u32>, params: SamplingParams) -> usize {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let seq = Sequence::new(request_id, seq_id, prompt_token_ids, &params);
        self.waiting.push_back(seq);
        request_id
    }

    /// Select runnable work and capture its phase and cache-hit count before
    /// `plan_step()` snapshots the immutable execution inputs.
    ///
    /// # Algorithm (nano-vllm parity)
    ///
    /// 1. **Preempt** running sequences if there aren't enough free blocks
    ///    (recompute-only: deallocate + requeue at front of waiting).
    /// 2. If any running seq still needs prefill (chunked prefill
    ///    continuation): **prefill** step — schedule more prompt tokens.
    /// 3. If `running` is non-empty (all prefilling done): **decode**
    ///    step — schedule at most 1 token per sequence within the global budget.
    /// 4. If `running` is empty and `waiting` is non-empty: **prefill**
    ///    step — schedule waiting sequences within the global budget.
    fn select_work(&mut self, kv_mgr: &mut KvCacheManager) -> Result<WorkSelection, StepPlanError> {
        self.preempt_if_needed(kv_mgr)?;

        // Chunked prefill continuation: running sequences that still need
        // more prompt tokens before they can decode.
        if self.running.iter().any(|s| s.is_prefill) {
            return Ok(self.schedule_prefill_continue());
        }

        if !self.running.is_empty() {
            return self.schedule_decode(kv_mgr);
        }

        self.schedule_prefill(kv_mgr)
    }

    /// Select work and capture everything EngineCore needs in one immutable plan.
    pub(crate) fn plan_step(
        &mut self,
        kv_mgr: &mut KvCacheManager,
    ) -> Result<Option<StepPlan>, StepPlanError> {
        if let Some(plan) = &self.in_flight {
            return Err(StepPlanError::PlanAlreadyInFlight { plan_id: plan.id });
        }

        let output = self.select_work(kv_mgr)?;
        let mut sequences = Vec::new();

        for sequence in &mut self.running {
            if sequence.num_scheduled_tokens == 0 {
                continue;
            }
            let (token_range, sampling_allowed) = match output.phase {
                StepPhase::Prefill => {
                    let start = sequence.num_cached_tokens;
                    let end = start
                        .checked_add(sequence.num_scheduled_tokens)
                        .ok_or_else(|| StepPlanError::invalid("scheduled token range overflow"))?
                        .min(sequence.num_prompt_tokens);
                    (start..end, end == sequence.num_prompt_tokens)
                }
                StepPhase::Decode => {
                    let start = sequence
                        .num_tokens
                        .checked_sub(1)
                        .ok_or_else(|| StepPlanError::invalid("decode sequence has no tokens"))?;
                    (start..sequence.num_tokens, true)
                }
            };

            let token_budget = token_range.len();
            if token_budget == 0 {
                return Err(StepPlanError::invalid(format!(
                    "sequence {} has zero-token scheduled work",
                    sequence.seq_id
                )));
            }
            sequence.num_scheduled_tokens = token_budget;

            let slot_mapping =
                kv_mgr.compute_slot_mapping(sequence, token_range.start, token_budget);
            let cache = SequenceCachePlan {
                num_cached_tokens: sequence.num_cached_tokens,
                kv_length: token_range.end,
                block_table: sequence.block_table.clone(),
                slot_mapping,
            };
            sequences.push(SequenceStepPlan {
                request_id: sequence.request_id,
                sequence_id: sequence.seq_id,
                logical_positions: token_range.clone(),
                input_token_ids: sequence.token_ids[token_range.clone()].to_vec(),
                token_range,
                token_budget,
                cache,
                sampling_allowed,
                sampling_params: SamplingParams {
                    temperature: sequence.temperature,
                    max_tokens: sequence.max_tokens,
                    ignore_eos: sequence.ignore_eos,
                    ..SamplingParams::default()
                },
                token_history: sequence.token_ids.clone(),
            });
        }

        if sequences.is_empty() {
            if self.is_running() {
                return Err(StepPlanError::NoProgress {
                    waiting_sequences: self.waiting.len(),
                    running_sequences: self.running.len(),
                    token_budget: self.max_num_batched_tokens,
                    free_blocks: kv_mgr.num_free_blocks(),
                });
            }
            return Ok(None);
        }

        let token_budget = sequences.iter().map(|sequence| sequence.token_budget).sum();
        let slot_mapping = sequences
            .iter()
            .flat_map(|sequence| sequence.cache.slot_mapping.iter().copied())
            .collect::<Vec<_>>();
        let attention = match output.phase {
            StepPhase::Prefill => {
                let scheduled_tokens = sequences
                    .iter()
                    .map(|sequence| usize_to_u32(sequence.token_budget, "scheduled token count"))
                    .collect::<Result<Vec<_>, _>>()?;
                let kv_lengths = sequences
                    .iter()
                    .map(|sequence| usize_to_u32(sequence.cache.kv_length, "KV length"))
                    .collect::<Result<Vec<_>, _>>()?;
                build_prefill_metadata(&scheduled_tokens, &kv_lengths, &slot_mapping)
            }
            StepPhase::Decode => {
                let context_lengths = sequences
                    .iter()
                    .map(|sequence| usize_to_u32(sequence.cache.kv_length, "context length"))
                    .collect::<Result<Vec<_>, _>>()?;
                let block_tables = sequences
                    .iter()
                    .map(|sequence| {
                        sequence
                            .cache
                            .block_table
                            .iter()
                            .map(|&block| {
                                i32::try_from(block).map_err(|_| {
                                    StepPlanError::invalid("cache block id does not fit i32")
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                build_decode_metadata(&context_lengths, &block_tables, &slot_mapping)
            }
        };

        let plan = StepPlan {
            id: self.next_plan_id,
            phase: output.phase,
            sequences,
            token_budget,
            attention,
        };
        self.next_plan_id += 1;
        self.in_flight = Some(plan.clone());
        Ok(Some(plan))
    }

    /// Apply the result for the currently in-flight plan exactly once.
    pub(crate) fn apply_step_result(
        &mut self,
        result: &StepResult,
        kv_mgr: &mut KvCacheManager,
    ) -> Result<Vec<RequestOutput>, StepPlanError> {
        let plan = self
            .in_flight
            .clone()
            .ok_or(StepPlanError::NoPlanInFlight {
                result_plan_id: result.plan_id,
            })?;
        if result.plan_id != plan.id {
            return Err(StepPlanError::StaleResult {
                expected_plan_id: plan.id,
                result_plan_id: result.plan_id,
            });
        }
        if result.sequences.len() != plan.sequences.len() {
            return Err(StepPlanError::ResultMismatch {
                plan_id: plan.id,
                reason: "sequence count differs".to_string(),
            });
        }

        for (planned, executed) in plan.sequences.iter().zip(&result.sequences) {
            if planned.request_id != executed.request_id
                || planned.sequence_id != executed.sequence_id
            {
                return Err(StepPlanError::ResultMismatch {
                    plan_id: plan.id,
                    reason: format!(
                        "expected request {}/sequence {}, got request {}/sequence {}",
                        planned.request_id,
                        planned.sequence_id,
                        executed.request_id,
                        executed.sequence_id
                    ),
                });
            }
            if planned.sampling_allowed != executed.sampled_token.is_some() {
                return Err(StepPlanError::ResultMismatch {
                    plan_id: plan.id,
                    reason: format!(
                        "sequence {} sampling result does not match permission",
                        planned.sequence_id
                    ),
                });
            }
            let Some(sequence) = self.running.iter().find(|sequence| {
                sequence.request_id == planned.request_id && sequence.seq_id == planned.sequence_id
            }) else {
                return Err(StepPlanError::ResultMismatch {
                    plan_id: plan.id,
                    reason: format!("sequence {} is no longer running", planned.sequence_id),
                });
            };
            if sequence.status != SequenceStatus::Running {
                return Err(StepPlanError::ResultMismatch {
                    plan_id: plan.id,
                    reason: format!("sequence {} is not running", planned.sequence_id),
                });
            }
        }

        let mut outputs = Vec::new();
        let mut finished_sequence_ids = Vec::new();
        for (planned, executed) in plan.sequences.iter().zip(&result.sequences) {
            // Membership was validated above, so the planned sequence must exist.
            #[allow(clippy::unwrap_used)]
            let sequence = self
                .running
                .iter_mut()
                .find(|sequence| {
                    sequence.request_id == planned.request_id
                        && sequence.seq_id == planned.sequence_id
                })
                .unwrap();

            kv_mgr.hash_blocks(sequence);
            sequence.num_cached_tokens =
                (sequence.num_cached_tokens + planned.token_budget).min(sequence.num_tokens);
            sequence.num_scheduled_tokens = 0;

            if let Some(token_id) = executed.sampled_token {
                sequence.append_token(token_id);
                let hit_max_tokens = sequence.num_completion_tokens() >= sequence.max_tokens;
                let hit_eos = self.eos_token_ids.contains(&token_id) && !sequence.ignore_eos;
                if hit_max_tokens || hit_eos {
                    sequence.status = SequenceStatus::Finished;
                    finished_sequence_ids.push(sequence.seq_id);
                    outputs.push(RequestOutput {
                        request_id: sequence.request_id,
                        token_ids: sequence.completion_token_ids().to_vec(),
                        text: String::new(),
                        finished: true,
                    });
                }
            }
        }

        for sequence_id in finished_sequence_ids {
            // The id was collected from a running sequence above.
            #[allow(clippy::unwrap_used)]
            let index = self
                .running
                .iter()
                .position(|sequence| sequence.seq_id == sequence_id)
                .unwrap();
            kv_mgr
                .deallocate(&mut self.running[index])
                .map_err(|source| {
                    StepPlanError::cache(CacheOperation::Deallocation, sequence_id, source)
                })?;
            self.running.remove(index);
        }

        self.in_flight = None;
        Ok(outputs)
    }

    /// Number of sequences currently waiting to be processed.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Number of sequences currently being processed (running).
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Whether there are any waiting or running sequences.
    pub fn is_running(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Iterator over running sequences (read-only) for diagnostics.
    pub fn running_seqs(&self) -> impl Iterator<Item = &Sequence> {
        self.running.iter()
    }

    /// Iterator over waiting sequences (read-only).
    pub fn waiting_seqs(&self) -> impl Iterator<Item = &Sequence> {
        self.waiting.iter()
    }

    // ── Private helpers ─────────────────────────────────────────────

    /// Preempt running sequences when there aren't enough free blocks.
    ///
    /// If the next budget-bounded decode batch cannot reserve all required
    /// append blocks, preempt the last running sequence (lowest priority).
    /// Recompute-only: deallocate blocks, requeue at the front of waiting.
    fn preempt_if_needed(&mut self, kv_mgr: &mut KvCacheManager) -> Result<(), StepPlanError> {
        if self.running.iter().any(|sequence| sequence.is_prefill) {
            return Ok(());
        }

        loop {
            let decode_batch = self
                .running
                .iter()
                .take(self.max_num_batched_tokens)
                .collect::<Vec<_>>();
            if kv_mgr.can_append(&decode_batch) {
                return Ok(());
            }
            let Some(mut victim) = self.running.pop_back() else {
                return Ok(());
            };
            let sequence_id = victim.seq_id;
            if let Err(source) = kv_mgr.deallocate(&mut victim) {
                self.running.push_back(victim);
                return Err(StepPlanError::cache(
                    CacheOperation::Deallocation,
                    sequence_id,
                    source,
                ));
            }
            victim.status = SequenceStatus::Waiting;
            victim.num_scheduled_tokens = 0;
            self.waiting.push_front(victim);
        }
    }

    /// Schedule a decode step: at most one token per running sequence and no
    /// more than the global token budget across the plan.
    fn schedule_decode(
        &mut self,
        kv_mgr: &mut KvCacheManager,
    ) -> Result<WorkSelection, StepPlanError> {
        for sequence in &mut self.running {
            sequence.num_scheduled_tokens = 0;
        }

        let scheduled_count = self.running.len().min(self.max_num_batched_tokens);
        let mut selected = Vec::with_capacity(scheduled_count);
        for _ in 0..scheduled_count {
            // scheduled_count is bounded by running.len().
            #[allow(clippy::unwrap_used)]
            selected.push(self.running.pop_front().unwrap());
        }
        let selected_sequence_id = selected.first().map_or(0, |sequence| sequence.seq_id);
        if let Err(source) = kv_mgr.may_append(&mut selected) {
            for sequence in selected.into_iter().rev() {
                self.running.push_front(sequence);
            }
            return Err(StepPlanError::cache(
                CacheOperation::AppendAllocation,
                selected_sequence_id,
                source,
            ));
        }
        for sequence in &mut selected {
            sequence.num_scheduled_tokens = 1;
            sequence.is_prefill = false;
        }
        for sequence in selected.into_iter().rev() {
            self.running.push_front(sequence);
        }

        Ok(WorkSelection {
            phase: StepPhase::Decode,
        })
    }

    /// Continue a chunked prefill: schedule more prompt tokens for
    /// running sequences that still have `is_prefill` set.
    fn schedule_prefill_continue(&mut self) -> WorkSelection {
        let mut total_tokens = 0;
        for seq in self.running.iter_mut() {
            if !seq.is_prefill {
                seq.num_scheduled_tokens = 0;
                continue;
            }
            let remaining = seq.num_prompt_tokens.saturating_sub(seq.num_cached_tokens);
            if remaining == 0 {
                seq.is_prefill = false;
                seq.num_scheduled_tokens = 0;
                continue;
            }
            let budget = self.max_num_batched_tokens.saturating_sub(total_tokens);
            let n_tokens = remaining.min(budget);
            if n_tokens == 0 {
                seq.num_scheduled_tokens = 0;
                continue;
            }
            seq.num_scheduled_tokens = n_tokens;
            total_tokens += n_tokens;
            if seq.num_cached_tokens + n_tokens >= seq.num_prompt_tokens {
                seq.is_prefill = false;
            }
        }
        WorkSelection {
            phase: StepPhase::Prefill,
        }
    }

    /// Schedule a prefill step: pick sequences from `waiting`, spend one global
    /// token budget in queue order, allocate blocks, and move them to `running`.
    fn schedule_prefill(
        &mut self,
        kv_mgr: &mut KvCacheManager,
    ) -> Result<WorkSelection, StepPlanError> {
        if self.waiting.is_empty() {
            return Ok(WorkSelection {
                phase: StepPhase::Prefill,
            });
        }

        let max_running = self.max_num_seqs.saturating_sub(self.running.len());
        let mut total_tokens: usize = 0;
        let mut to_schedule: Vec<(usize, usize)> = Vec::new(); // (waiting_index, n_tokens)

        for i in 0..self.waiting.len().min(max_running) {
            let remaining_before_allocation = self.waiting[i]
                .num_prompt_tokens
                .saturating_sub(self.waiting[i].num_cached_tokens);
            if remaining_before_allocation == 0 {
                continue;
            }
            let budget = self.max_num_batched_tokens.saturating_sub(total_tokens);
            if budget == 0 {
                break;
            }

            match kv_mgr.can_allocate(&self.waiting[i]) {
                None => break,
                Some(num_cached) => {
                    if let Err(source) = kv_mgr.allocate(&mut self.waiting[i], num_cached) {
                        for &(allocated_index, _) in to_schedule.iter().rev() {
                            kv_mgr
                                .deallocate(&mut self.waiting[allocated_index])
                                .map_err(|rollback| {
                                    StepPlanError::cache(
                                        CacheOperation::AllocationRollback,
                                        self.waiting[allocated_index].seq_id,
                                        rollback,
                                    )
                                })?;
                        }
                        return Err(StepPlanError::cache(
                            CacheOperation::Allocation,
                            self.waiting[i].seq_id,
                            source,
                        ));
                    }

                    let remaining = self.waiting[i]
                        .num_prompt_tokens
                        .saturating_sub(self.waiting[i].num_cached_tokens);
                    let n_tokens = remaining.min(budget);
                    if n_tokens == 0 {
                        kv_mgr.deallocate(&mut self.waiting[i]).map_err(|source| {
                            StepPlanError::cache(
                                CacheOperation::EmptyAllocationRollback,
                                self.waiting[i].seq_id,
                                source,
                            )
                        })?;
                        continue;
                    }
                    total_tokens += n_tokens;
                    to_schedule.push((i, n_tokens));
                }
            }
        }

        let mut selected = Vec::with_capacity(to_schedule.len());
        for &(idx, n_tokens) in to_schedule.iter().rev() {
            // idx comes from valid indices into waiting (verified earlier in this fn)
            #[allow(clippy::unwrap_used)]
            let seq = self.waiting.remove(idx).unwrap();
            selected.push((seq, n_tokens));
        }

        for (mut seq, n_tokens) in selected.into_iter().rev() {
            seq.num_scheduled_tokens = n_tokens;
            let fully_prefilled = seq.num_cached_tokens + n_tokens >= seq.num_prompt_tokens;
            seq.is_prefill = !fully_prefilled;
            seq.status = SequenceStatus::Running;
            self.running.push_back(seq);
        }

        Ok(WorkSelection {
            phase: StepPhase::Prefill,
        })
    }
}

/// Per-step return value containing the completion tokens for a finished
/// request. Accumulated by `LLM::generate` until `is_finished()`.
///
/// The `text` field is populated by the composition root (`llm.rs`) during
/// detokenization — the scheduler does not have access to a tokenizer.
#[derive(Debug, Clone)]
pub struct RequestOutput {
    /// Stable public identity assigned when the prompt is accepted.
    pub request_id: usize,
    pub token_ids: Vec<u32>,
    pub text: String,
    pub finished: bool,
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use crate::attention::PagedKVCache;
    use crate::engine::sequence::BLOCK_SIZE;
    use std::sync::{Arc, Mutex};

    const TEST_EOS_TOKEN_ID: u32 = 777;

    fn fake_cache() -> Arc<Mutex<PagedKVCache>> {
        Arc::new(Mutex::new(
            PagedKVCache::new(
                1,
                32,
                256,
                1,
                64,
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            )
            .unwrap(),
        ))
    }

    fn make_params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            max_tokens,
            ..SamplingParams::default()
        }
    }

    fn make_scheduler() -> Scheduler {
        Scheduler::new_with_eos_token_ids(
            DEFAULT_MAX_NUM_BATCHED_TOKENS,
            DEFAULT_MAX_NUM_SEQS,
            DEFAULT_GPU_MEMORY_UTILIZATION,
            vec![TEST_EOS_TOKEN_ID],
        )
    }

    fn make_kv_mgr(num_blocks: usize) -> KvCacheManager {
        KvCacheManager::new(num_blocks, BLOCK_SIZE, fake_cache())
    }

    fn result_for_plan(plan: &StepPlan, sampled_token: u32) -> StepResult {
        StepResult {
            plan_id: plan.id,
            sequences: plan
                .sequences
                .iter()
                .map(|sequence| crate::engine::SequenceStepResult {
                    request_id: sequence.request_id,
                    sequence_id: sequence.sequence_id,
                    sampled_token: sequence.sampling_allowed.then_some(sampled_token),
                })
                .collect(),
        }
    }

    mod add_request {
        use super::*;

        #[test]
        fn adds_to_waiting() {
            let mut s = make_scheduler();
            assert_eq!(s.num_waiting(), 0);
            s.add_request(vec![1, 2, 3], make_params(16));
            assert_eq!(s.num_waiting(), 1);
        }

        #[test]
        fn assigns_monotonic_ids() {
            let mut s = make_scheduler();
            s.add_request(vec![1], make_params(16));
            s.add_request(vec![2], make_params(16));
            assert_eq!(s.waiting[0].seq_id, 0);
            assert_eq!(s.waiting[1].seq_id, 1);
            assert_eq!(s.waiting[0].request_id(), 0);
            assert_eq!(s.waiting[1].request_id(), 1);
        }

        #[test]
        fn sequence_starts_waiting() {
            let mut s = make_scheduler();
            s.add_request(vec![1, 2, 3], make_params(16));
            assert_eq!(s.waiting[0].status, SequenceStatus::Waiting);
        }
    }

    mod schedule_prefill {
        use super::*;

        #[test]
        fn prefills_single_waiting_sequence() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request((0..BLOCK_SIZE as u32 + 10).collect(), make_params(16));
            let output = s.select_work(&mut kv).unwrap();
            assert_eq!(output.phase, StepPhase::Prefill);
            assert_eq!(s.num_running(), 1);
            assert_eq!(s.num_waiting(), 0);

            let seq = &s.running[0];
            assert!(
                !seq.is_prefill,
                "fully prefilled seq should have is_prefill=false"
            );
            assert_eq!(seq.status, SequenceStatus::Running);
            assert_eq!(seq.num_scheduled_tokens, BLOCK_SIZE + 10);
        }

        #[test]
        fn chunked_prefill_when_prompt_exceeds_budget() {
            let mut s = Scheduler::new(100, 512, 0.9); // small budget
            let mut kv = make_kv_mgr(100);
            s.add_request((0..500u32).collect(), make_params(16));
            let output = s.select_work(&mut kv).unwrap();
            assert_eq!(output.phase, StepPhase::Prefill);
            let seq = &s.running[0];
            assert!(seq.is_prefill);
            // Should be chunked: budget is 100, so only 100 scheduled.
            assert_eq!(seq.num_scheduled_tokens, 100);
            assert!(seq.num_scheduled_tokens < seq.num_prompt_tokens);
        }

        #[test]
        fn empty_waiting_selects_prefill_without_running_work() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(10);
            let output = s.select_work(&mut kv).unwrap();
            assert_eq!(output.phase, StepPhase::Prefill);
            assert_eq!(s.num_running(), 0);
        }

        #[test]
        fn pure_prefill_no_running_allowed() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            // Add one request, prefill it.
            s.add_request(vec![1, 2, 3], make_params(16));
            s.select_work(&mut kv).unwrap();
            assert_eq!(s.num_running(), 1);
            assert_eq!(s.num_waiting(), 0);

            // Add another request. Since running is non-empty, next schedule
            // should be decode (not prefill).
            s.add_request(vec![4, 5, 6], make_params(16));
            let output = s.select_work(&mut kv).unwrap();
            assert_eq!(output.phase, StepPhase::Decode);
            // waiting should still have the new request.
            assert_eq!(s.num_waiting(), 1);
        }

        #[test]
        fn allocation_failure_is_returned_without_admitting_the_sequence() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(4);
            scheduler.add_request(vec![1, 2, 3], make_params(16));
            scheduler.waiting[0].block_table.push(0);
            let free_before = kv.num_free_blocks();

            let error = scheduler.plan_step(&mut kv).unwrap_err();

            assert!(error
                .to_string()
                .contains("cache allocation failed for sequence 0"));
            assert_eq!(kv.num_free_blocks(), free_before);
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(scheduler.num_running(), 0);
            assert_eq!(scheduler.waiting[0].status, SequenceStatus::Waiting);
            assert_eq!(scheduler.waiting[0].num_scheduled_tokens, 0);
            assert_eq!(scheduler.waiting[0].block_table, vec![0]);
            assert!(scheduler.in_flight.is_none());
        }
    }

    mod step_plan_contract {
        use super::*;

        #[test]
        fn prefill_plan_names_exact_scheduled_work() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(16));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Prefill);
            assert_eq!(plan.token_budget, 3);
            assert_eq!(plan.sequences.len(), 1);

            let sequence = &plan.sequences[0];
            assert_eq!(sequence.request_id, 0);
            assert_eq!(sequence.sequence_id, 0);
            assert_eq!(sequence.token_range, 0..3);
            assert_eq!(sequence.logical_positions, 0..3);
            assert_eq!(sequence.token_budget, 3);
            assert_eq!(sequence.cache.num_cached_tokens, 0);
            assert_eq!(sequence.cache.kv_length, 3);
            assert_eq!(sequence.cache.block_table.len(), 1);
            assert_eq!(sequence.cache.slot_mapping.len(), 3);
            assert!(sequence.sampling_allowed);

            assert!(plan.attention.is_prefill);
            assert_eq!(plan.attention.cu_seqlens_q, vec![0, 3]);
            assert_eq!(plan.attention.cu_seqlens_k, vec![0, 3]);
            assert_eq!(plan.attention.slot_mapping, sequence.cache.slot_mapping);
        }

        #[test]
        fn step_result_advances_state_once_and_rejects_duplicate_application() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(2));
            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let result = crate::engine::StepResult {
                plan_id: plan.id,
                sequences: vec![crate::engine::SequenceStepResult {
                    request_id: 0,
                    sequence_id: 0,
                    sampled_token: Some(42),
                }],
            };

            let outputs = scheduler.apply_step_result(&result, &mut kv).unwrap();
            assert!(outputs.is_empty());
            assert_eq!(scheduler.running[0].num_cached_tokens, 3);
            assert_eq!(scheduler.running[0].completion_token_ids(), &[42]);

            let duplicate = scheduler.apply_step_result(&result, &mut kv).unwrap_err();
            assert_eq!(
                duplicate,
                StepPlanError::NoPlanInFlight {
                    result_plan_id: plan.id
                }
            );
            assert_eq!(scheduler.running[0].num_cached_tokens, 3);
            assert_eq!(scheduler.running[0].completion_token_ids(), &[42]);
        }

        #[test]
        fn stale_result_is_rejected_without_consuming_the_current_plan() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(2));
            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let mut stale = result_for_plan(&plan, 42);
            stale.plan_id += 1;

            let error = scheduler.apply_step_result(&stale, &mut kv).unwrap_err();
            assert_eq!(
                error,
                StepPlanError::StaleResult {
                    expected_plan_id: plan.id,
                    result_plan_id: stale.plan_id,
                }
            );
            assert_eq!(scheduler.running[0].num_cached_tokens, 0);
            assert!(scheduler.running[0].completion_token_ids().is_empty());

            scheduler
                .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                .unwrap();
            assert_eq!(scheduler.running[0].num_cached_tokens, 3);
            assert_eq!(scheduler.running[0].completion_token_ids(), &[42]);
        }

        #[test]
        fn scheduler_rejects_a_new_plan_while_one_is_in_flight() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(2));
            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            let error = scheduler.plan_step(&mut kv).unwrap_err();

            assert_eq!(
                error,
                StepPlanError::PlanAlreadyInFlight { plan_id: plan.id }
            );
            assert_eq!(scheduler.running[0].num_cached_tokens, 0);
            assert!(scheduler.running[0].completion_token_ids().is_empty());
        }

        #[test]
        fn mismatched_result_membership_is_rejected_before_state_changes() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(2));
            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let mut mismatch = result_for_plan(&plan, 42);
            mismatch.sequences[0].sequence_id = 99;

            let error = scheduler.apply_step_result(&mismatch, &mut kv).unwrap_err();

            assert!(matches!(error, StepPlanError::ResultMismatch { .. }));
            assert_eq!(scheduler.running[0].num_cached_tokens, 0);
            assert!(scheduler.running[0].completion_token_ids().is_empty());
        }

        #[test]
        fn decode_plan_names_the_last_token_and_full_context() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(3));
            let prefill = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&prefill, 42), &mut kv)
                .unwrap();

            let decode = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(decode.phase, StepPhase::Decode);
            assert_eq!(decode.token_budget, 1);
            assert_eq!(decode.sequences.len(), 1);
            let sequence = &decode.sequences[0];
            assert_eq!(sequence.token_range, 3..4);
            assert_eq!(sequence.logical_positions, 3..4);
            assert_eq!(sequence.input_token_ids, vec![42]);
            assert_eq!(sequence.cache.num_cached_tokens, 3);
            assert_eq!(sequence.cache.kv_length, 4);
            assert_eq!(sequence.cache.slot_mapping.len(), 1);
            assert!(sequence.sampling_allowed);
            assert!(!decode.attention.is_prefill);
            assert_eq!(decode.attention.cu_seqlens_q, vec![0, 1]);
            assert_eq!(decode.attention.cu_seqlens_k, vec![0, 4]);
        }

        #[test]
        fn next_plan_excludes_running_sequences_with_no_scheduled_tokens() {
            let mut scheduler = Scheduler::new(5, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(2));
            scheduler.add_request(vec![21, 22, 23, 24], make_params(2));
            let first = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&first, 42), &mut kv)
                .unwrap();

            let next = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(scheduler.num_running(), 2);
            assert_eq!(next.sequences.len(), 1);
            assert_eq!(next.sequences[0].sequence_id, 1);
            assert_eq!(next.sequences[0].token_range, 2..4);
            assert_eq!(next.sequences[0].token_budget, 2);
            assert!(next
                .sequences
                .iter()
                .all(|sequence| sequence.token_budget > 0));
        }

        #[test]
        fn pending_work_with_zero_budget_returns_a_contextual_stall() {
            let mut scheduler = Scheduler::new(0, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11], make_params(2));

            let error = scheduler.plan_step(&mut kv).unwrap_err();

            assert_eq!(
                error.to_string(),
                "scheduler made no progress: waiting=1, running=0, token_budget=0, free_blocks=100"
            );
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(scheduler.num_running(), 0);
            assert!(scheduler.in_flight.is_none());
        }
    }

    mod schedule_decode {
        use super::*;

        #[test]
        fn decode_schedules_one_token_per_running_seq() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(16));
            // Prefill first.
            s.select_work(&mut kv).unwrap();
            assert_eq!(s.num_running(), 1);

            // Next step should be decode.
            let output = s.select_work(&mut kv).unwrap();
            assert_eq!(output.phase, StepPhase::Decode);
            let seq = &s.running[0];
            assert!(!seq.is_prefill, "decode must set is_prefill=false");
            assert_eq!(seq.num_scheduled_tokens, 1);
        }

        #[test]
        fn decode_does_not_touch_waiting() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(16));
            s.select_work(&mut kv).unwrap(); // prefill → running
            s.add_request(vec![4, 5, 6], make_params(16)); // stays waiting

            let output = s.select_work(&mut kv).unwrap(); // decode
            assert_eq!(output.phase, StepPhase::Decode);
            assert_eq!(s.num_running(), 1);
            assert_eq!(s.num_waiting(), 1);
        }
    }

    mod apply_step_result {
        use super::*;

        #[test]
        fn advances_num_cached_tokens() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(16));
            let plan = s.plan_step(&mut kv).unwrap().unwrap();

            let outputs = s
                .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                .unwrap();
            let seq = &s.running[0];
            assert_eq!(seq.num_cached_tokens, 3); // 3 prompt + 1 generated
            assert!(outputs.is_empty(), "not yet finished");
        }

        #[test]
        fn finishes_on_eos() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(64));
            let plan = s.plan_step(&mut kv).unwrap().unwrap();

            let outputs = s
                .apply_step_result(&result_for_plan(&plan, TEST_EOS_TOKEN_ID), &mut kv)
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert!(outputs[0].finished);
            assert_eq!(s.num_running(), 0, "finished seq should be removed");
        }

        #[test]
        fn finishes_on_max_tokens() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(1)); // max_tokens=1
            let plan = s.plan_step(&mut kv).unwrap().unwrap();

            let outputs = s
                .apply_step_result(&result_for_plan(&plan, 99), &mut kv)
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert!(outputs[0].finished);
            assert_eq!(s.num_running(), 0);
        }

        #[test]
        fn ignore_eos_does_not_finish() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(
                vec![1, 2, 3],
                SamplingParams {
                    max_tokens: 64,
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
            );
            let plan = s.plan_step(&mut kv).unwrap().unwrap();

            let outputs = s
                .apply_step_result(&result_for_plan(&plan, TEST_EOS_TOKEN_ID), &mut kv)
                .unwrap();
            assert!(outputs.is_empty(), "ignore_eos should not finish on EOS");
            assert_eq!(s.num_running(), 1);
        }

        #[test]
        fn configured_model_eos_ids_control_completion() {
            let mut scheduler = Scheduler::new_with_eos_token_ids(16_384, 512, 0.9, vec![7, 8]);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![1, 2, 3], make_params(64));
            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            let outputs = scheduler
                .apply_step_result(&result_for_plan(&plan, 8), &mut kv)
                .unwrap();

            assert_eq!(outputs.len(), 1);
            assert!(outputs[0].finished);
        }

        #[test]
        fn unrelated_former_qwen_eos_value_does_not_finish() {
            let mut scheduler = Scheduler::new_with_eos_token_ids(16_384, 512, 0.9, vec![7]);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![1, 2, 3], make_params(64));
            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            let outputs = scheduler
                .apply_step_result(&result_for_plan(&plan, 151_645), &mut kv)
                .unwrap();

            assert!(outputs.is_empty());
            assert_eq!(scheduler.num_running(), 1);
        }
    }

    mod preemption {
        use super::*;

        #[test]
        fn preempts_when_no_free_blocks_for_decode() {
            let mut s = Scheduler::with_defaults();
            let mut kv = make_kv_mgr(1);

            let tokens: Vec<u32> = (0..BLOCK_SIZE as u32).collect();
            s.add_request(tokens.clone(), make_params(64));
            s.select_work(&mut kv).unwrap();
            assert_eq!(s.num_running(), 1);
            assert_eq!(kv.num_free_blocks(), 0);

            {
                let seq = &mut s.running[0];
                while seq.num_tokens < BLOCK_SIZE + 1 {
                    seq.append_token(99);
                }
            }

            s.select_work(&mut kv).unwrap();
            assert!(s.num_waiting() >= 1, "preempted seq should be in waiting");
        }

        #[test]
        fn exact_capacity_chunked_prefill_does_not_restart_forever() {
            let mut scheduler = Scheduler::new(1, 512, 0.9);
            let mut kv = make_kv_mgr(2);
            scheduler.add_request((0..=BLOCK_SIZE as u32).collect(), make_params(1));
            let mut completed = Vec::new();

            for _ in 0..=BLOCK_SIZE {
                let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
                assert_eq!(plan.token_budget, 1);
                completed.extend(
                    scheduler
                        .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                        .unwrap(),
                );
                if !scheduler.is_running() {
                    break;
                }
            }

            assert!(!scheduler.is_running(), "bounded prefill must complete");
            assert_eq!(completed.len(), 1);
            assert_eq!(completed[0].request_id, 0);
            assert_eq!(completed[0].token_ids, vec![42]);
            assert_eq!(kv.num_free_blocks(), 2);
        }

        #[test]
        fn aggregate_decode_capacity_is_reserved_before_any_append() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = make_kv_mgr(3);
            for sequence_id in 0..2 {
                let mut sequence = Sequence::new(
                    sequence_id,
                    sequence_id,
                    (0..BLOCK_SIZE as u32).collect(),
                    &make_params(16),
                );
                kv.allocate(&mut sequence, 0).unwrap();
                sequence.append_token(42);
                sequence.status = SequenceStatus::Running;
                sequence.is_prefill = false;
                scheduler.running.push_back(sequence);
            }

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Decode);
            assert_eq!(plan.token_budget, 1);
            assert_eq!(plan.sequences[0].sequence_id, 0);
            assert_eq!(scheduler.num_running(), 1);
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(kv.num_free_blocks(), 1);
        }
    }

    mod is_running {
        use super::*;

        #[test]
        fn false_when_empty() {
            let s = make_scheduler();
            assert!(!s.is_running());
        }

        #[test]
        fn true_when_waiting_not_empty() {
            let mut s = make_scheduler();
            s.add_request(vec![1], make_params(16));
            assert!(s.is_running());
        }

        #[test]
        fn true_when_running_not_empty() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(16));
            s.select_work(&mut kv).unwrap();
            assert!(s.is_running());
        }
    }

    mod budgets {
        use super::*;

        #[test]
        fn mixed_prefill_plan_spends_one_global_budget_in_queue_order() {
            let mut scheduler = Scheduler::new(5, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![10, 11, 12], make_params(16));
            scheduler.add_request(vec![20, 21, 22, 23], make_params(16));
            scheduler.add_request(vec![30], make_params(16));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.token_budget, 5);
            assert_eq!(
                plan.sequences
                    .iter()
                    .map(|sequence| (sequence.sequence_id, sequence.token_budget))
                    .collect::<Vec<_>>(),
                vec![(0, 3), (1, 2)]
            );
            assert!(plan
                .sequences
                .iter()
                .all(|sequence| sequence.token_budget > 0));
            assert_eq!(scheduler.num_waiting(), 1);
        }

        #[test]
        fn cache_hit_spends_budget_on_each_sequences_actual_uncached_work() {
            let mut scheduler = Scheduler::new(100, 512, 0.9);
            let mut kv = make_kv_mgr(10);
            let shared_prompt = (0..300).collect::<Vec<u32>>();
            scheduler.add_request(shared_prompt.clone(), make_params(1));
            while scheduler.is_running() {
                let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
                scheduler
                    .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                    .unwrap();
            }

            scheduler.add_request(shared_prompt, make_params(1));
            scheduler.add_request((1_000..1_056).collect(), make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.token_budget, 100);
            assert_eq!(
                plan.sequences
                    .iter()
                    .map(|sequence| (sequence.sequence_id, sequence.token_budget))
                    .collect::<Vec<_>>(),
                vec![(1, 44), (2, 56)]
            );
            assert_eq!(plan.sequences[0].cache.num_cached_tokens, 256);
        }

        #[test]
        fn decode_plan_caps_membership_at_the_global_budget() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = make_kv_mgr(100);

            for sequence_id in 0..3 {
                let mut sequence =
                    Sequence::new(sequence_id, sequence_id, vec![10, 11], &make_params(16));
                kv.allocate(&mut sequence, 0).unwrap();
                sequence.status = SequenceStatus::Running;
                sequence.is_prefill = false;
                scheduler.running.push_back(sequence);
            }

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Decode);
            assert_eq!(plan.token_budget, 2);
            assert_eq!(
                plan.sequences
                    .iter()
                    .map(|sequence| sequence.sequence_id)
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            assert_eq!(scheduler.running[2].num_scheduled_tokens, 0);
        }

        #[test]
        fn tight_budget_mixed_lengths_complete_without_empty_or_oversized_plans() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = make_kv_mgr(8);
            scheduler.add_request(vec![10], make_params(1));
            scheduler.add_request(vec![20, 21], make_params(1));
            scheduler.add_request(vec![30, 31, 32, 33, 34], make_params(1));
            let mut completed_request_ids = Vec::new();

            for _ in 0..8 {
                if !scheduler.is_running() {
                    break;
                }
                let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
                assert!((1..=2).contains(&plan.token_budget));
                assert_eq!(
                    plan.token_budget,
                    plan.sequences
                        .iter()
                        .map(|sequence| sequence.token_budget)
                        .sum::<usize>()
                );
                assert!(plan
                    .sequences
                    .iter()
                    .all(|sequence| sequence.token_budget > 0));
                completed_request_ids.extend(
                    scheduler
                        .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                        .unwrap()
                        .into_iter()
                        .map(|output| output.request_id),
                );
            }

            assert!(!scheduler.is_running(), "mixed requests must not loop");
            assert_eq!(completed_request_ids, vec![0, 1, 2]);
            assert_eq!(kv.num_free_blocks(), 8);
        }

        #[test]
        fn respects_max_num_batched_tokens() {
            let mut s = Scheduler::new(50, 512, 0.9); // tight budget
            let mut kv = make_kv_mgr(100);
            s.add_request((0..200u32).collect(), make_params(16));
            s.select_work(&mut kv).unwrap();
            let tokens = s.running[0].num_scheduled_tokens;
            assert!(tokens <= 50, "must respect max_num_batched_tokens (50)");
        }

        #[test]
        fn respects_max_num_seqs() {
            let mut s = Scheduler::new(16384, 2, 0.9);
            let mut kv = make_kv_mgr(100);
            for _ in 0..5 {
                s.add_request(vec![1, 2, 3], make_params(16));
            }
            s.select_work(&mut kv).unwrap();
            // Only 2 should have been scheduled (max_num_seqs).
            assert!(s.num_running() <= 2);
            assert!(s.num_waiting() > 0, "remaining should stay waiting");
        }
    }
}
