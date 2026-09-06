//! Token-level scheduler — `Scheduler` (V1 parity, nano-vllm algorithm).
//!
//! One `plan_step()` may mix prefill and decode work while every participating
//! sequence spends from one global token budget and sequences with no positive
//! work remain outside the plan. Preemption is **recompute-only**:
//! deallocate the sequence's blocks and requeue at the front of `waiting`
//! (V1 parity; v0.1 never swaps KV to host — V0 `swap_space` dropped).
//!
//! Each decision is captured in an immutable `StepPlan`; applying its
//! corresponding `StepResult` updates block hashes, advances cached tokens,
//! appends sampled tokens, and finalises sequences on EOS or `max_tokens`
//! exactly once. Scheduler talks to `KvCacheManager`, never to `BlockPool`
//! or `PagedKVCache` directly (ADR-0004 seam).

use std::collections::VecDeque;

use crate::attention::{
    build_continued_prefill_metadata, build_decode_metadata, build_prefill_metadata, AttnMetadata,
};
use crate::engine::kv_cache_manager::{KvCacheError, KvCacheManager};
use crate::engine::sequence::{Sequence, SequenceStatus};
use crate::engine::{
    AdmissionBlockedReason, BlockedAdmission, CacheOperation, SequenceCachePlan, SequencePhase,
    SequenceStepPlan, StepPhase, StepPlan, StepPlanError, StepResult,
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

fn block_tables_for_plan(sequences: &[SequenceStepPlan]) -> Result<Vec<Vec<i32>>, StepPlanError> {
    sequences
        .iter()
        .map(|sequence| {
            sequence
                .cache
                .block_table
                .iter()
                .map(|&block| {
                    i32::try_from(block)
                        .map_err(|_| StepPlanError::invalid("cache block id does not fit i32"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn build_paged_prefill_plan_metadata(
    sequences: &[SequenceStepPlan],
    slot_mapping: &[i64],
) -> Result<AttnMetadata, StepPlanError> {
    let scheduled_tokens = sequences
        .iter()
        .map(|sequence| usize_to_u32(sequence.token_budget, "scheduled token count"))
        .collect::<Result<Vec<_>, _>>()?;
    let kv_lengths = sequences
        .iter()
        .map(|sequence| usize_to_u32(sequence.cache.kv_length, "KV length"))
        .collect::<Result<Vec<_>, _>>()?;
    let block_tables = block_tables_for_plan(sequences)?;
    Ok(build_continued_prefill_metadata(
        &scheduled_tokens,
        &kv_lengths,
        &block_tables,
        slot_mapping,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkSelection {
    phase: StepPhase,
    is_prefill_continuation: bool,
    blocked_admission: Option<BlockedAdmission>,
}

/// Token-level scheduler — the algorithmic heart of the engine.
///
/// Maintains `waiting` / `running` deques of `Sequence`s. One `plan_step()`
/// call produces an immutable [`StepPlan`] that may mix prefill and decode.
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
    #[cfg(feature = "internal-golden")]
    pub(crate) fn diagnostic_eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

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
    /// # Algorithm
    ///
    /// 1. **Preempt** running sequences if there aren't enough free blocks
    ///    (recompute-only: deallocate + requeue at front of waiting).
    /// 2. Reserve one token for the FIFO waiting head whenever token, sequence,
    ///    and cache capacity all permit admission.
    /// 3. Schedule bounded running continuation or decode work from the same
    ///    global token budget, then spend the reservation on waiting prefill.
    /// 4. Reserve decode append blocks transactionally. Under aggregate cache
    ///    pressure, roll admissions back from the FIFO tail until the maximal
    ///    feasible prefix can coexist with decode progress.
    ///
    /// FIFO arrivals never overtake one another. Therefore, while every
    /// decision retains enough token, sequence, and cache capacity for at least
    /// one admission, a waiter initially at zero-based position `p` is selected
    /// within at most `p + 1` decisions.
    fn select_work(&mut self, kv_mgr: &mut KvCacheManager) -> Result<WorkSelection, StepPlanError> {
        self.preempt_if_needed(kv_mgr)?;

        let blocked_before_selection = self.admission_block(kv_mgr, 0);
        let reserve_admission_token =
            !self.waiting.is_empty() && blocked_before_selection.is_none();
        let running_budget = self
            .max_num_batched_tokens
            .saturating_sub(usize::from(reserve_admission_token));

        if !self.running.is_empty() {
            let running_before_admission = self.running.len();
            self.schedule_running(running_budget);
            let running_tokens = self.scheduled_tokens();
            if let Err(error) = self.schedule_prefill_with_budget(
                kv_mgr,
                self.max_num_batched_tokens.saturating_sub(running_tokens),
            ) {
                self.reset_scheduled_work();
                return Err(error);
            }
            loop {
                match self.reserve_decode_cache(kv_mgr) {
                    Ok(()) => break,
                    Err(error) if self.running.len() == running_before_admission => {
                        self.reset_scheduled_work();
                        return Err(error);
                    }
                    Err(_) => {
                        if let Err(error) = self.rollback_last_admission(kv_mgr) {
                            self.reset_scheduled_work();
                            return Err(error);
                        }
                    }
                }
            }
            let blocked_admission = blocked_before_selection
                .or_else(|| self.admission_block(kv_mgr, self.scheduled_tokens()));
            return Ok(self.selected_work(running_before_admission, blocked_admission));
        }

        let mut prefill = self.schedule_prefill(kv_mgr)?;
        prefill.blocked_admission = blocked_before_selection
            .or_else(|| self.admission_block(kv_mgr, self.scheduled_tokens()));
        Ok(prefill)
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
        if let Some(blocked) = output.blocked_admission {
            tracing::debug!(
                request_id = blocked.request_id,
                reason = %blocked.reason,
                "request remains waiting because admission capacity is unavailable"
            );
        }
        let mut sequences = Vec::new();

        for sequence in &mut self.running {
            if sequence.num_scheduled_tokens == 0 {
                continue;
            }
            let sequence_phase = Self::sequence_phase(sequence);
            let (token_range, sampling_allowed) = match sequence_phase {
                SequencePhase::Prefill => {
                    let start = sequence.num_cached_tokens;
                    let prefill_target = sequence.prefill_target_tokens();
                    let end = start
                        .checked_add(sequence.num_scheduled_tokens)
                        .ok_or_else(|| StepPlanError::invalid("scheduled token range overflow"))?
                        .min(prefill_target);
                    (start..end, end == prefill_target)
                }
                SequencePhase::Decode => {
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
                cached_token_range: 0..sequence.num_cached_tokens,
                kv_length: token_range.end,
                block_table: sequence.block_table.clone(),
                slot_mapping,
            };
            sequences.push(SequenceStepPlan {
                request_id: sequence.request_id,
                sequence_id: sequence.seq_id,
                phase: sequence_phase,
                logical_positions: token_range.clone(),
                input_token_ids: sequence.token_ids[token_range.clone()].to_vec(),
                token_range,
                token_budget,
                cache,
                sampling_allowed,
                sampling_params: sequence.sampling_params().clone(),
                token_history: sequence.token_ids.clone(),
                completion_step: sequence.num_completion_tokens(),
            });
        }

        if sequences.is_empty() {
            if self.is_running() {
                return Err(StepPlanError::NoProgress {
                    waiting_sequences: self.waiting.len(),
                    running_sequences: self.running.len(),
                    token_budget: self.max_num_batched_tokens,
                    free_blocks: kv_mgr.num_free_blocks(),
                    blocked_admission: output.blocked_admission,
                });
            }
            return Ok(None);
        }

        let token_budget = sequences.iter().map(|sequence| sequence.token_budget).sum();
        let slot_mapping = sequences
            .iter()
            .flat_map(|sequence| sequence.cache.slot_mapping.iter().copied())
            .collect::<Vec<_>>();
        let has_cached_prefill = sequences.iter().any(|sequence| {
            sequence.phase == SequencePhase::Prefill
                && !sequence.cache.cached_token_range.is_empty()
        });
        // A same-request continuation is identified during work selection.
        // A newly admitted prefix hit is not a continuation, but its captured
        // cached range independently requires the same paged causal path.
        let attention = match output.phase {
            StepPhase::Prefill if output.is_prefill_continuation || has_cached_prefill => {
                build_paged_prefill_plan_metadata(&sequences, &slot_mapping)?
            }
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
                let block_tables = block_tables_for_plan(&sequences)?;
                build_decode_metadata(&context_lengths, &block_tables, &slot_mapping)
            }
            StepPhase::Mixed => build_paged_prefill_plan_metadata(&sequences, &slot_mapping)?,
        };

        let plan = StepPlan {
            id: self.next_plan_id,
            phase: output.phase,
            sequences,
            token_budget,
            attention,
            blocked_admission: output.blocked_admission,
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
        let mut recovered_sequence_ids = Vec::new();
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
            if planned.phase == SequencePhase::Prefill
                && planned.sampling_allowed
                && sequence.recompute_target_tokens == Some(sequence.num_cached_tokens)
            {
                recovered_sequence_ids.push(sequence.seq_id);
            }

            if let Some(token_id) = executed.sampled_token {
                sequence.append_token(token_id);
                let hit_max_tokens =
                    sequence.num_completion_tokens() >= sequence.sampling_params().max_tokens;
                let hit_eos = self.eos_token_ids.contains(&token_id)
                    && !sequence.sampling_params().ignore_eos;
                if hit_max_tokens || hit_eos {
                    sequence.status = SequenceStatus::Finished;
                    tracing::debug!(
                        request_id = sequence.request_id,
                        sampling_params = ?sequence.sampling_params(),
                        hit_eos,
                        hit_max_tokens,
                        "request completed"
                    );
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

        if let Some(&first_finished_id) = finished_sequence_ids.first() {
            let finished_ids = finished_sequence_ids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>();
            let mut finished = self
                .running
                .iter()
                .filter(|sequence| finished_ids.contains(&sequence.seq_id))
                .cloned()
                .collect::<Vec<_>>();
            kv_mgr.deallocate_batch(&mut finished).map_err(|source| {
                StepPlanError::cache(CacheOperation::Deallocation, first_finished_id, source)
            })?;
            self.running
                .retain(|sequence| !finished_ids.contains(&sequence.seq_id));
        }

        for sequence in &mut self.running {
            if recovered_sequence_ids.contains(&sequence.seq_id) {
                sequence.recompute_target_tokens = None;
            }
        }

        self.in_flight = None;
        Ok(outputs)
    }

    /// Abandon every request after an engine-step failure.
    ///
    /// `LLM::generate` is synchronous, so a failed step invalidates the whole
    /// active call. All block tables are released in one transaction before
    /// queues or the in-flight plan are cleared.
    pub(crate) fn abort_all_requests(
        &mut self,
        kv_mgr: &mut KvCacheManager,
    ) -> Result<(), StepPlanError> {
        let mut allocated = self
            .running
            .iter()
            .chain(&self.waiting)
            .filter(|sequence| !sequence.block_table.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        if !allocated.is_empty() {
            let sequence_id = allocated[0].seq_id;
            kv_mgr.deallocate_batch(&mut allocated).map_err(|source| {
                StepPlanError::cache(CacheOperation::FailureCleanup, sequence_id, source)
            })?;
        }
        self.running.clear();
        self.waiting.clear();
        self.in_flight = None;
        Ok(())
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

    fn scheduled_tokens(&self) -> usize {
        self.running
            .iter()
            .map(|sequence| sequence.num_scheduled_tokens)
            .sum()
    }

    fn sequence_phase(sequence: &Sequence) -> SequencePhase {
        if sequence.num_cached_tokens < sequence.prefill_target_tokens() {
            SequencePhase::Prefill
        } else {
            SequencePhase::Decode
        }
    }

    fn selected_work(
        &self,
        running_before_admission: usize,
        blocked_admission: Option<BlockedAdmission>,
    ) -> WorkSelection {
        let mut has_prefill = false;
        let mut has_decode = false;
        let mut has_continued_prefill = false;
        for (index, sequence) in self.running.iter().enumerate() {
            if sequence.num_scheduled_tokens == 0 {
                continue;
            }
            match Self::sequence_phase(sequence) {
                SequencePhase::Prefill => {
                    has_prefill = true;
                    has_continued_prefill |= index < running_before_admission;
                }
                SequencePhase::Decode => has_decode = true,
            }
        }
        let phase = match (has_prefill, has_decode) {
            (true, true) => StepPhase::Mixed,
            (true, false) => StepPhase::Prefill,
            (false, true) => StepPhase::Decode,
            (false, false) if self.running.iter().any(|sequence| sequence.is_prefill) => {
                StepPhase::Prefill
            }
            (false, false) => StepPhase::Decode,
        };
        WorkSelection {
            phase,
            is_prefill_continuation: has_prefill && (has_decode || has_continued_prefill),
            blocked_admission,
        }
    }

    fn schedule_running(&mut self, token_budget: usize) {
        let mut scheduled_tokens = 0;
        for sequence in &mut self.running {
            sequence.num_scheduled_tokens = 0;
            let budget = token_budget.saturating_sub(scheduled_tokens);
            if budget == 0 {
                continue;
            }
            let num_tokens = match Self::sequence_phase(sequence) {
                SequencePhase::Prefill => sequence
                    .prefill_target_tokens()
                    .saturating_sub(sequence.num_cached_tokens)
                    .min(budget),
                SequencePhase::Decode => 1,
            };
            sequence.num_scheduled_tokens = num_tokens;
            sequence.is_prefill = sequence.num_cached_tokens.saturating_add(num_tokens)
                < sequence.prefill_target_tokens();
            scheduled_tokens += num_tokens;
        }
    }

    fn reset_scheduled_work(&mut self) {
        for sequence in &mut self.running {
            sequence.num_scheduled_tokens = 0;
            sequence.is_prefill = sequence.num_cached_tokens < sequence.prefill_target_tokens();
        }
    }

    fn reserve_decode_cache(&mut self, kv_mgr: &mut KvCacheManager) -> Result<(), StepPlanError> {
        let decode_indices = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, sequence)| {
                sequence.num_scheduled_tokens > 0
                    && Self::sequence_phase(sequence) == SequencePhase::Decode
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if decode_indices.is_empty() {
            return Ok(());
        }
        let mut selected = decode_indices
            .iter()
            .map(|&index| self.running[index].clone())
            .collect::<Vec<_>>();
        let selected_sequence_id = selected[0].seq_id;
        kv_mgr.may_append_batch(&mut selected).map_err(|source| {
            StepPlanError::cache(
                CacheOperation::AppendAllocation,
                selected_sequence_id,
                source,
            )
        })?;
        for (&index, reserved) in decode_indices.iter().zip(selected) {
            self.running[index].block_table = reserved.block_table;
        }
        Ok(())
    }

    fn rollback_last_admission(
        &mut self,
        kv_mgr: &mut KvCacheManager,
    ) -> Result<(), StepPlanError> {
        let Some(mut sequence) = self.running.pop_back() else {
            return Err(StepPlanError::invalid(
                "cannot roll back admission from an empty running queue",
            ));
        };
        let sequence_id = sequence.seq_id;
        if let Err(source) = kv_mgr.deallocate(&mut sequence).map_err(KvCacheError::from) {
            self.running.push_back(sequence);
            return Err(StepPlanError::cache(
                CacheOperation::AllocationRollback,
                sequence_id,
                source,
            ));
        }
        sequence.status = SequenceStatus::Waiting;
        sequence.num_scheduled_tokens = 0;
        sequence.is_prefill = true;
        self.waiting.push_front(sequence);
        Ok(())
    }

    fn admission_block(
        &self,
        kv_mgr: &KvCacheManager,
        scheduled_tokens: usize,
    ) -> Option<BlockedAdmission> {
        let sequence = self.waiting.front()?;
        let reason = if scheduled_tokens >= self.max_num_batched_tokens {
            AdmissionBlockedReason::TokenBudget
        } else if self.running.len() >= self.max_num_seqs {
            AdmissionBlockedReason::SequenceLimit
        } else if kv_mgr.can_allocate(sequence).is_none() {
            AdmissionBlockedReason::KvCache
        } else {
            return None;
        };
        Some(BlockedAdmission {
            request_id: sequence.request_id,
            reason,
        })
    }

    /// Preempt running sequences when there aren't enough free blocks.
    ///
    /// If the next budget-bounded decode batch cannot reserve all required
    /// append blocks, preempt the last running sequence (lowest priority).
    /// Recompute-only: transactionally deallocate blocks, freeze the complete
    /// logical history length as the recovery target, and requeue at the front
    /// of `waiting` without changing request identity, tokens, or parameters.
    fn preempt_if_needed(&mut self, kv_mgr: &mut KvCacheManager) -> Result<(), StepPlanError> {
        if self
            .running
            .iter()
            .any(|sequence| sequence.is_prefill && sequence.recompute_target_tokens.is_none())
        {
            return Ok(());
        }

        loop {
            let decode_batch = self
                .running
                .iter()
                .take(self.max_num_batched_tokens)
                .collect::<Vec<_>>();
            if kv_mgr.can_append_batch(&decode_batch) {
                return Ok(());
            }
            let Some(mut victim) = self.running.pop_back() else {
                return Ok(());
            };
            let sequence_id = victim.seq_id;
            if let Err(source) = kv_mgr.deallocate(&mut victim).map_err(KvCacheError::from) {
                self.running.push_back(victim);
                return Err(StepPlanError::cache(
                    CacheOperation::Deallocation,
                    sequence_id,
                    source,
                ));
            }
            victim.status = SequenceStatus::Waiting;
            victim.num_scheduled_tokens = 0;
            victim.recompute_target_tokens = Some(
                victim
                    .recompute_target_tokens
                    .unwrap_or_default()
                    .max(victim.num_tokens),
            );
            victim.is_prefill = true;
            self.waiting.push_front(victim);
        }
    }

    /// Schedule a prefill step: pick sequences from `waiting`, spend one global
    /// token budget in queue order, allocate blocks, and move them to `running`.
    fn schedule_prefill(
        &mut self,
        kv_mgr: &mut KvCacheManager,
    ) -> Result<WorkSelection, StepPlanError> {
        self.schedule_prefill_with_budget(kv_mgr, self.max_num_batched_tokens)
    }

    fn schedule_prefill_with_budget(
        &mut self,
        kv_mgr: &mut KvCacheManager,
        token_budget: usize,
    ) -> Result<WorkSelection, StepPlanError> {
        if self.waiting.is_empty() {
            return Ok(WorkSelection {
                phase: StepPhase::Prefill,
                is_prefill_continuation: false,
                blocked_admission: None,
            });
        }

        let max_running = self.max_num_seqs.saturating_sub(self.running.len());
        let mut total_tokens: usize = 0;
        let mut to_schedule: Vec<(usize, usize)> = Vec::new(); // (waiting_index, n_tokens)

        for i in 0..self.waiting.len().min(max_running) {
            let remaining_before_allocation = self.waiting[i]
                .prefill_target_tokens()
                .saturating_sub(self.waiting[i].num_cached_tokens);
            if remaining_before_allocation == 0 {
                continue;
            }
            let budget = token_budget.saturating_sub(total_tokens);
            if budget == 0 {
                break;
            }

            match kv_mgr.can_allocate(&self.waiting[i]) {
                None => break,
                Some(num_cached) => {
                    if let Err(source) = kv_mgr
                        .allocate(&mut self.waiting[i], num_cached)
                        .map_err(KvCacheError::from)
                    {
                        for &(allocated_index, _) in to_schedule.iter().rev() {
                            kv_mgr
                                .deallocate(&mut self.waiting[allocated_index])
                                .map_err(KvCacheError::from)
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
                        .prefill_target_tokens()
                        .saturating_sub(self.waiting[i].num_cached_tokens);
                    let n_tokens = remaining.min(budget);
                    if n_tokens == 0 {
                        kv_mgr
                            .deallocate(&mut self.waiting[i])
                            .map_err(KvCacheError::from)
                            .map_err(|source| {
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
            let fully_prefilled = seq.num_cached_tokens + n_tokens >= seq.prefill_target_tokens();
            seq.is_prefill = !fully_prefilled;
            seq.status = SequenceStatus::Running;
            self.running.push_back(seq);
        }

        Ok(WorkSelection {
            phase: StepPhase::Prefill,
            is_prefill_continuation: false,
            blocked_admission: None,
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

    fn assert_sampling_params(actual: &SamplingParams, expected: &SamplingParams) {
        assert_eq!(actual.temperature, expected.temperature);
        assert_eq!(actual.top_k, expected.top_k);
        assert_eq!(actual.top_p, expected.top_p);
        assert_eq!(actual.max_tokens, expected.max_tokens);
        assert_eq!(actual.ignore_eos, expected.ignore_eos);
        assert_eq!(actual.presence_penalty, expected.presence_penalty);
        assert_eq!(actual.frequency_penalty, expected.frequency_penalty);
        assert_eq!(actual.repetition_penalty, expected.repetition_penalty);
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

    fn begin_decoding(
        scheduler: &mut Scheduler,
        kv: &mut KvCacheManager,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampled_token: u32,
    ) {
        scheduler.add_request(prompt, make_params(max_tokens));
        let prefill = scheduler.plan_step(kv).unwrap().unwrap();
        scheduler
            .apply_step_result(&result_for_plan(&prefill, sampled_token), kv)
            .unwrap();
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
            assert_eq!(sequence.cache.cached_token_range, 0..0);
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
        fn continued_prefill_plan_carries_paged_causal_context() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13, 14, 15], make_params(2));

            let first = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&first, 42), &mut kv)
                .unwrap();
            let continuation = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(continuation.phase, StepPhase::Prefill);
            assert_eq!(continuation.token_budget, 2);
            assert_eq!(continuation.sequences.len(), 1);
            let sequence = &continuation.sequences[0];
            assert_eq!(sequence.token_range, 2..4);
            assert_eq!(sequence.logical_positions, 2..4);
            assert_eq!(sequence.input_token_ids, vec![13, 14]);
            assert_eq!(sequence.cache.cached_token_range, 0..2);
            assert_eq!(sequence.cache.kv_length, 4);
            assert!(!sequence.sampling_allowed);
            assert_eq!(continuation.attention.cu_seqlens_q, vec![0, 2]);
            assert_eq!(continuation.attention.cu_seqlens_k, vec![0, 4]);
            assert_eq!(continuation.attention.max_seqlen_q, 2);
            assert_eq!(continuation.attention.max_seqlen_k, 4);
            assert_eq!(
                continuation.attention.block_table,
                vec![sequence
                    .cache
                    .block_table
                    .iter()
                    .map(|&block| i32::try_from(block).unwrap())
                    .collect::<Vec<_>>()]
            );
        }

        #[test]
        fn initial_maximal_prefix_hit_carries_paged_causal_context() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            let prompt = (0..513).collect::<Vec<u32>>();
            scheduler.add_request(prompt.clone(), make_params(1));
            let warmup = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&warmup, 42), &mut kv)
                .unwrap();

            scheduler.add_request(prompt, make_params(1));
            let hit = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(hit.phase, StepPhase::Prefill);
            assert_eq!(hit.token_budget, 1);
            let sequence = &hit.sequences[0];
            // "Full hit" means the maximal reusable prefix. The final logical
            // block is deliberately recomputed so sampling has a hidden state.
            assert_eq!(sequence.cache.cached_token_range, 0..2 * BLOCK_SIZE);
            assert_eq!(sequence.token_range, 2 * BLOCK_SIZE..513);
            assert_eq!(sequence.logical_positions, 2 * BLOCK_SIZE..513);
            assert_eq!(sequence.input_token_ids, vec![512]);
            assert_eq!(sequence.cache.kv_length, 513);
            assert!(sequence.sampling_allowed);
            assert_eq!(hit.attention.cu_seqlens_q, vec![0, 1]);
            assert_eq!(hit.attention.cu_seqlens_k, vec![0, 513]);
            assert_eq!(
                hit.attention.block_table,
                vec![sequence
                    .cache
                    .block_table
                    .iter()
                    .map(|&block| i32::try_from(block).unwrap())
                    .collect::<Vec<_>>()]
            );
        }

        #[test]
        fn initial_full_partial_and_miss_keep_per_request_causal_mappings() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            let first_block = (0..BLOCK_SIZE as u32).collect::<Vec<_>>();
            let second_block = (1_000..1_000 + BLOCK_SIZE as u32).collect::<Vec<_>>();
            let alternate_second = (2_000..2_000 + BLOCK_SIZE as u32).collect::<Vec<_>>();
            let miss_prefix = (3_000..3_000 + BLOCK_SIZE as u32).collect::<Vec<_>>();

            let mut full_prompt = first_block.clone();
            full_prompt.extend_from_slice(&second_block);
            full_prompt.push(9);
            scheduler.add_request(full_prompt.clone(), make_params(1));
            let warmup = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&warmup, 42), &mut kv)
                .unwrap();

            let mut partial_prompt = first_block;
            partial_prompt.extend_from_slice(&alternate_second);
            partial_prompt.push(10);
            let mut miss_prompt = miss_prefix;
            miss_prompt.push(11);
            scheduler.add_request(full_prompt, make_params(1));
            scheduler.add_request(partial_prompt, make_params(1));
            scheduler.add_request(miss_prompt, make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Prefill);
            assert_eq!(plan.token_budget, 1 + 257 + 257);
            assert_eq!(plan.sequences.len(), 3);
            let expected = [
                (2 * BLOCK_SIZE, 513, vec![9]),
                (BLOCK_SIZE, 513, {
                    let mut tokens = alternate_second;
                    tokens.push(10);
                    tokens
                }),
                (0, 257, {
                    let mut tokens = (3_000..3_000 + BLOCK_SIZE as u32).collect::<Vec<_>>();
                    tokens.push(11);
                    tokens
                }),
            ];
            for (sequence, (cached, end, input)) in plan.sequences.iter().zip(expected) {
                assert_eq!(sequence.cache.cached_token_range, 0..cached);
                assert_eq!(sequence.token_range, cached..end);
                assert_eq!(sequence.logical_positions, cached..end);
                assert_eq!(sequence.input_token_ids, input);
                assert_eq!(sequence.cache.kv_length, end);
                assert_eq!(
                    sequence.cache.slot_mapping.len(),
                    sequence.token_range.len()
                );
                assert!(sequence.sampling_allowed);
            }
            assert_eq!(plan.attention.cu_seqlens_q, vec![0, 1, 258, 515]);
            assert_eq!(plan.attention.cu_seqlens_k, vec![0, 513, 1_026, 1_283]);
            assert_eq!(plan.attention.block_table.len(), 3);
            assert_eq!(
                plan.sequences[0].cache.block_table[0], plan.sequences[1].cache.block_table[0],
                "requests with the same first logical block must share its physical prefix block"
            );
            assert_eq!(
                plan.attention.slot_mapping,
                plan.sequences
                    .iter()
                    .flat_map(|sequence| sequence.cache.slot_mapping.iter().copied())
                    .collect::<Vec<_>>()
            );

            let outputs = scheduler
                .apply_step_result(&result_for_plan(&plan, 43), &mut kv)
                .unwrap();
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![1, 2, 3]
            );
            assert_eq!(kv.num_free_blocks(), 100);
        }

        #[test]
        fn block_boundary_prompt_recomputes_its_final_logical_block_for_sampling() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            let prompt = (0..(2 * BLOCK_SIZE) as u32).collect::<Vec<_>>();
            scheduler.add_request(prompt.clone(), make_params(1));
            let warmup = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&warmup, 42), &mut kv)
                .unwrap();

            scheduler.add_request(prompt.clone(), make_params(1));
            let hit = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let sequence = &hit.sequences[0];

            assert_eq!(sequence.cache.cached_token_range, 0..BLOCK_SIZE);
            assert_eq!(sequence.token_range, BLOCK_SIZE..2 * BLOCK_SIZE);
            assert_eq!(sequence.logical_positions, BLOCK_SIZE..2 * BLOCK_SIZE);
            assert_eq!(sequence.input_token_ids, prompt[BLOCK_SIZE..]);
            assert_eq!(sequence.cache.kv_length, 2 * BLOCK_SIZE);
            assert_eq!(hit.token_budget, BLOCK_SIZE);
            assert!(sequence.sampling_allowed);
            assert_eq!(hit.attention.cu_seqlens_q, vec![0, BLOCK_SIZE as u32]);
            assert_eq!(hit.attention.cu_seqlens_k, vec![0, (2 * BLOCK_SIZE) as u32]);
            assert_eq!(hit.attention.block_table.len(), 1);
        }

        #[test]
        fn initial_partial_hit_becomes_continuation_without_sampling_early() {
            let mut scheduler = make_scheduler();
            let mut kv = make_kv_mgr(100);
            let warm_prompt = (0..513).collect::<Vec<u32>>();
            scheduler.add_request(warm_prompt, make_params(1));
            let warmup = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&warmup, 42), &mut kv)
                .unwrap();

            scheduler.max_num_batched_tokens = 100;
            let mut prompt = (0..BLOCK_SIZE as u32).collect::<Vec<_>>();
            prompt.extend(2_000..2_344);
            scheduler.add_request(prompt, make_params(1));

            let initial_hit = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let initial = &initial_hit.sequences[0];
            assert_eq!(initial.cache.cached_token_range, 0..BLOCK_SIZE);
            assert_eq!(initial.token_range, BLOCK_SIZE..BLOCK_SIZE + 100);
            assert_eq!(initial.logical_positions, initial.token_range);
            assert_eq!(initial.cache.kv_length, BLOCK_SIZE + 100);
            assert!(!initial.sampling_allowed);
            assert_eq!(initial_hit.attention.block_table.len(), 1);
            scheduler
                .apply_step_result(&result_for_plan(&initial_hit, 43), &mut kv)
                .unwrap();

            let continuation = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let continued = &continuation.sequences[0];
            assert_eq!(continued.cache.cached_token_range, 0..BLOCK_SIZE + 100);
            assert_eq!(continued.token_range, BLOCK_SIZE + 100..BLOCK_SIZE + 200);
            assert_eq!(continued.logical_positions, continued.token_range);
            assert_eq!(continued.cache.kv_length, BLOCK_SIZE + 200);
            assert!(!continued.sampling_allowed);
            assert_eq!(continuation.attention.block_table.len(), 1);
            scheduler.abort_all_requests(&mut kv).unwrap();
            assert_eq!(kv.num_free_blocks(), 100);
        }

        #[test]
        fn one_token_prompt_is_one_final_prefill_chunk() {
            let mut scheduler = Scheduler::new(1, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Prefill);
            assert_eq!(plan.token_budget, 1);
            assert_eq!(plan.sequences[0].token_range, 0..1);
            assert_eq!(plan.sequences[0].logical_positions, 0..1);
            assert!(plan.sequences[0].sampling_allowed);
        }

        #[test]
        fn exact_budget_prompt_is_one_final_prefill_chunk() {
            let mut scheduler = Scheduler::new(3, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Prefill);
            assert_eq!(plan.token_budget, 3);
            assert_eq!(plan.sequences[0].token_range, 0..3);
            assert_eq!(plan.sequences[0].logical_positions, 0..3);
            assert!(plan.sequences[0].sampling_allowed);
        }

        #[test]
        fn multi_chunk_prompt_advances_exact_ranges_and_samples_only_on_the_final_chunk() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            let prompt = vec![11, 12, 13, 14, 15];
            scheduler.add_request(prompt.clone(), make_params(1));
            let mut ranges = Vec::new();
            let mut positions = Vec::new();
            let mut inputs = Vec::new();
            let mut sampling_permissions = Vec::new();
            let mut completed = Vec::new();

            while scheduler.is_running() {
                let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
                assert_eq!(plan.phase, StepPhase::Prefill);
                assert!(plan.token_budget <= 2);
                let sequence = &plan.sequences[0];
                ranges.push(sequence.token_range.clone());
                positions.push(sequence.logical_positions.clone());
                inputs.extend_from_slice(&sequence.input_token_ids);
                sampling_permissions.push(sequence.sampling_allowed);
                completed.extend(
                    scheduler
                        .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                        .unwrap(),
                );
            }

            assert_eq!(ranges, vec![0..2, 2..4, 4..5]);
            assert_eq!(positions, ranges);
            assert_eq!(inputs, prompt);
            assert_eq!(sampling_permissions, vec![false, false, true]);
            assert_eq!(completed.len(), 1);
            assert_eq!(completed[0].request_id, 0);
            assert_eq!(completed[0].token_ids, vec![42]);
        }

        #[test]
        fn sampled_token_for_incomplete_prefill_is_rejected_without_advancing_state() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![11, 12, 13], make_params(1));
            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert!(!plan.sequences[0].sampling_allowed);
            let invalid = StepResult {
                plan_id: plan.id,
                sequences: vec![crate::engine::SequenceStepResult {
                    request_id: 0,
                    sequence_id: 0,
                    sampled_token: Some(42),
                }],
            };

            let error = scheduler.apply_step_result(&invalid, &mut kv).unwrap_err();

            assert_eq!(
                error,
                StepPlanError::ResultMismatch {
                    plan_id: plan.id,
                    reason: "sequence 0 sampling result does not match permission".to_string(),
                }
            );
            assert_eq!(scheduler.running[0].num_cached_tokens, 0);
            assert!(scheduler.running[0].completion_token_ids().is_empty());
            assert_eq!(scheduler.in_flight.as_ref().unwrap().id, plan.id);
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
            assert_eq!(sequence.cache.cached_token_range, 0..3);
            assert_eq!(sequence.cache.kv_length, 4);
            assert_eq!(sequence.cache.slot_mapping.len(), 1);
            assert!(sequence.sampling_allowed);
            assert!(!decode.attention.is_prefill);
            assert_eq!(decode.attention.cu_seqlens_q, vec![0, 1]);
            assert_eq!(decode.attention.cu_seqlens_k, vec![0, 4]);
        }

        #[test]
        fn next_plan_mixes_only_positive_running_work() {
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
            assert_eq!(next.phase, StepPhase::Mixed);
            assert_eq!(next.sequences.len(), 2);
            assert_eq!(next.sequences[0].sequence_id, 0);
            assert_eq!(next.sequences[0].phase, SequencePhase::Decode);
            assert_eq!(next.sequences[0].token_budget, 1);
            assert_eq!(next.sequences[1].sequence_id, 1);
            assert_eq!(next.sequences[1].phase, SequencePhase::Prefill);
            assert_eq!(next.sequences[1].token_range, 2..4);
            assert_eq!(next.sequences[1].token_budget, 2);
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
                "scheduler made no progress: waiting=1, running=0, token_budget=0, free_blocks=100, blocked_admission=request 0 (token budget)"
            );
            assert!(matches!(
                error,
                StepPlanError::NoProgress {
                    blocked_admission: Some(BlockedAdmission {
                        request_id: 0,
                        reason: AdmissionBlockedReason::TokenBudget,
                    }),
                    ..
                }
            ));
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
            let prefill = s.plan_step(&mut kv).unwrap().unwrap();
            s.apply_step_result(&result_for_plan(&prefill, 42), &mut kv)
                .unwrap();
            assert_eq!(s.num_running(), 1);

            // Next step should be decode.
            let plan = s.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(plan.phase, StepPhase::Decode);
            let seq = &s.running[0];
            assert!(!seq.is_prefill, "decode must set is_prefill=false");
            assert_eq!(seq.num_scheduled_tokens, 1);
        }

        #[test]
        fn waiting_prefill_joins_decode_plan_when_budgets_allow() {
            let mut scheduler = Scheduler::new(4, 4, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![1], make_params(2));
            let prefill = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&prefill, 42), &mut kv)
                .unwrap();
            scheduler.add_request(vec![4, 5], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.token_budget, 3);
            assert_eq!(
                plan.sequences
                    .iter()
                    .map(|sequence| sequence.sequence_id)
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            assert_eq!(scheduler.num_waiting(), 0);
            assert_eq!(scheduler.num_running(), 2);
        }
    }

    mod continuous_admission {
        use super::*;

        #[test]
        fn fifo_waiters_are_admitted_within_their_position_bound() {
            let mut scheduler = Scheduler::new(2, 8, 0.9);
            let mut kv = make_kv_mgr(100);
            begin_decoding(&mut scheduler, &mut kv, vec![1], 10, 42);
            for prompt_start in [10, 20, 30] {
                scheduler.add_request(
                    vec![prompt_start, prompt_start + 1, prompt_start + 2],
                    make_params(1),
                );
            }

            let mut admitted_at = [None; 3];
            let mut admission_order = Vec::new();
            for decision in 0..3 {
                let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
                for sequence in &plan.sequences {
                    if let Some(waiter_index) = sequence.request_id.checked_sub(1) {
                        if waiter_index < admitted_at.len() && admitted_at[waiter_index].is_none() {
                            admitted_at[waiter_index] = Some(decision);
                            admission_order.push(sequence.request_id);
                        }
                    }
                }
                scheduler
                    .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                    .unwrap();
                if decision == 0 {
                    scheduler.add_request(vec![40, 41, 42], make_params(1));
                }
            }

            assert_eq!(admitted_at, [Some(0), Some(1), Some(2)]);
            assert_eq!(admission_order, vec![1, 2, 3]);
        }

        #[test]
        fn cache_blocked_waiter_stays_waiting_with_a_plan_reason() {
            let mut scheduler = Scheduler::new(4, 4, 0.9);
            let mut kv = make_kv_mgr(1);
            begin_decoding(&mut scheduler, &mut kv, vec![1], 2, 42);
            scheduler.add_request(vec![4, 5], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(
                plan.blocked_admission,
                Some(BlockedAdmission {
                    request_id: 1,
                    reason: AdmissionBlockedReason::KvCache,
                })
            );
            assert_eq!(
                plan.sequences
                    .iter()
                    .map(|sequence| sequence.request_id)
                    .collect::<Vec<_>>(),
                vec![0]
            );
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(scheduler.waiting[0].status, SequenceStatus::Waiting);
            assert_eq!(scheduler.waiting[0].num_scheduled_tokens, 0);
            assert!(scheduler.waiting[0].block_table.is_empty());
        }

        #[test]
        fn mixed_admission_failure_leaves_decode_cache_and_selection_unchanged() {
            let mut scheduler = Scheduler::new(300, 4, 0.9);
            let mut kv = make_kv_mgr(3);
            begin_decoding(
                &mut scheduler,
                &mut kv,
                (0..BLOCK_SIZE as u32).collect(),
                2,
                42,
            );
            scheduler.add_request(vec![4], make_params(1));
            scheduler.waiting[0].block_table.push(0);
            let free_before = kv.num_free_blocks();
            let decode_blocks_before = scheduler.running[0].block_table.clone();

            let error = scheduler.plan_step(&mut kv).unwrap_err();

            assert!(error
                .to_string()
                .contains("cache allocation failed for sequence 1"));
            assert_eq!(kv.num_free_blocks(), free_before);
            assert_eq!(scheduler.running[0].block_table, decode_blocks_before);
            assert_eq!(scheduler.running[0].num_scheduled_tokens, 0);
            assert_eq!(scheduler.num_running(), 1);
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(scheduler.waiting[0].status, SequenceStatus::Waiting);
            assert_eq!(scheduler.waiting[0].num_scheduled_tokens, 0);
            assert!(scheduler.in_flight.is_none());
        }

        #[test]
        fn aggregate_cache_pressure_defers_admission_but_keeps_decode_progress() {
            let mut scheduler = Scheduler::new(300, 4, 0.9);
            let mut kv = make_kv_mgr(2);
            begin_decoding(
                &mut scheduler,
                &mut kv,
                (0..BLOCK_SIZE as u32).collect(),
                2,
                42,
            );
            scheduler.add_request(vec![4], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Decode);
            assert_eq!(plan.token_budget, 1);
            assert_eq!(plan.sequences[0].request_id, 0);
            assert_eq!(
                plan.blocked_admission,
                Some(BlockedAdmission {
                    request_id: 1,
                    reason: AdmissionBlockedReason::KvCache,
                })
            );
            assert_eq!(scheduler.num_running(), 1);
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(scheduler.waiting[0].status, SequenceStatus::Waiting);
            assert_eq!(scheduler.waiting[0].num_scheduled_tokens, 0);
            assert!(scheduler.waiting[0].block_table.is_empty());
            assert_eq!(kv.num_free_blocks(), 0);
        }

        #[test]
        fn aggregate_cache_backoff_keeps_the_maximal_fifo_admission_prefix() {
            let mut scheduler = Scheduler::new(300, 4, 0.9);
            let mut kv = make_kv_mgr(3);
            begin_decoding(
                &mut scheduler,
                &mut kv,
                (0..BLOCK_SIZE as u32).collect(),
                2,
                42,
            );
            scheduler.max_num_batched_tokens = 3;
            scheduler.add_request(vec![4], make_params(1));
            scheduler.add_request(vec![5], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Mixed);
            assert_eq!(plan.token_budget, 2);
            assert_eq!(
                plan.sequences
                    .iter()
                    .map(|sequence| sequence.request_id)
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            assert_eq!(
                plan.blocked_admission,
                Some(BlockedAdmission {
                    request_id: 2,
                    reason: AdmissionBlockedReason::KvCache,
                })
            );
            assert_eq!(scheduler.num_running(), 2);
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(scheduler.waiting[0].request_id, 2);
            assert_eq!(scheduler.waiting[0].status, SequenceStatus::Waiting);
            assert_eq!(scheduler.waiting[0].num_scheduled_tokens, 0);
            assert!(scheduler.waiting[0].block_table.is_empty());
            assert_eq!(kv.num_free_blocks(), 0);
        }

        #[test]
        fn mixed_plan_preserves_per_sequence_phase_and_causal_state() {
            let mut scheduler = Scheduler::new(3, 4, 0.9);
            let mut kv = make_kv_mgr(100);
            begin_decoding(&mut scheduler, &mut kv, vec![1], 2, 42);
            scheduler.add_request(vec![4, 5], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Mixed);
            assert_eq!(plan.token_budget, 3);
            assert_eq!(
                plan.sequences
                    .iter()
                    .map(|sequence| (
                        sequence.request_id,
                        sequence.phase,
                        sequence.token_range.clone(),
                        sequence.sampling_allowed,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, SequencePhase::Decode, 1..2, true),
                    (1, SequencePhase::Prefill, 0..2, true),
                ]
            );
            assert_eq!(plan.attention.cu_seqlens_q, vec![0, 1, 3]);
            assert_eq!(plan.attention.cu_seqlens_k, vec![0, 2, 4]);
            assert_eq!(plan.attention.block_table.len(), 2);
            let outputs = scheduler
                .apply_step_result(&result_for_plan(&plan, 77), &mut kv)
                .unwrap();
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| (output.request_id, output.token_ids.clone()))
                    .collect::<Vec<_>>(),
                vec![(0, vec![42, 77]), (1, vec![77])]
            );
            assert!(!scheduler.is_running());
        }

        #[test]
        fn sequence_limit_defers_waiter_with_a_plan_reason() {
            let mut scheduler = Scheduler::new(4, 1, 0.9);
            let mut kv = make_kv_mgr(100);
            begin_decoding(&mut scheduler, &mut kv, vec![1], 2, 42);
            scheduler.add_request(vec![4], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(
                plan.blocked_admission,
                Some(BlockedAdmission {
                    request_id: 1,
                    reason: AdmissionBlockedReason::SequenceLimit,
                })
            );
            assert_eq!(plan.sequences.len(), 1);
            assert_eq!(plan.sequences[0].request_id, 0);
            assert_eq!(scheduler.num_waiting(), 1);
        }

        #[test]
        fn cache_only_stall_is_not_reported_as_successful_progress() {
            let mut scheduler = Scheduler::new(4, 4, 0.9);
            let mut kv = make_kv_mgr(0);
            scheduler.add_request(vec![1], make_params(1));

            let error = scheduler.plan_step(&mut kv).unwrap_err();

            assert!(matches!(
                error,
                StepPlanError::NoProgress {
                    blocked_admission: Some(BlockedAdmission {
                        request_id: 0,
                        reason: AdmissionBlockedReason::KvCache,
                    }),
                    ..
                }
            ));
            assert_eq!(scheduler.num_waiting(), 1);
            assert_eq!(scheduler.num_running(), 0);
            assert_eq!(scheduler.waiting[0].num_scheduled_tokens, 0);
            assert!(scheduler.in_flight.is_none());
        }

        #[test]
        fn mixed_incomplete_prefill_cannot_sample_or_change_decode_history() {
            let mut scheduler = Scheduler::new(3, 4, 0.9);
            let mut kv = make_kv_mgr(100);
            begin_decoding(&mut scheduler, &mut kv, vec![1], 3, 42);
            scheduler.add_request(vec![4, 5, 6], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Mixed);
            assert_eq!(plan.token_budget, 3);
            assert!(plan.sequences[0].sampling_allowed);
            assert!(!plan.sequences[1].sampling_allowed);
            let outputs = scheduler
                .apply_step_result(&result_for_plan(&plan, 77), &mut kv)
                .unwrap();
            assert!(outputs.is_empty());
            let decoder = scheduler
                .running
                .iter()
                .find(|sequence| sequence.request_id == 0)
                .unwrap();
            assert_eq!(decoder.completion_token_ids(), &[42, 77]);
            let prefilling = scheduler
                .running
                .iter()
                .find(|sequence| sequence.request_id == 1)
                .unwrap();
            assert_eq!(prefilling.num_cached_tokens, 2);
            assert!(prefilling.completion_token_ids().is_empty());
            assert!(prefilling.is_prefill);
        }

        #[test]
        fn decode_continues_while_an_admitted_prefill_finishes_its_next_chunk() {
            let mut scheduler = Scheduler::new(3, 4, 0.9);
            let mut kv = make_kv_mgr(100);
            begin_decoding(&mut scheduler, &mut kv, vec![1], 3, 42);
            scheduler.add_request(vec![4, 5, 6, 7], make_params(1));
            let admission = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(admission.phase, StepPhase::Mixed);
            scheduler
                .apply_step_result(&result_for_plan(&admission, 77), &mut kv)
                .unwrap();

            let continuation = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(continuation.phase, StepPhase::Mixed);
            assert_eq!(continuation.token_budget, 3);
            assert_eq!(
                continuation
                    .sequences
                    .iter()
                    .map(|sequence| (
                        sequence.request_id,
                        sequence.phase,
                        sequence.token_range.clone(),
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, SequencePhase::Decode, 2..3),
                    (1, SequencePhase::Prefill, 2..4),
                ]
            );
            assert_eq!(continuation.attention.cu_seqlens_q, vec![0, 1, 3]);
            assert_eq!(continuation.attention.cu_seqlens_k, vec![0, 3, 7]);
            assert_eq!(continuation.attention.block_table.len(), 2);
        }

        #[test]
        fn one_token_horizon_admits_prefill_without_mislabeling_unscheduled_decode() {
            let mut scheduler = Scheduler::new(1, 4, 0.9);
            let mut kv = make_kv_mgr(100);
            begin_decoding(&mut scheduler, &mut kv, vec![1], 2, 42);
            scheduler.add_request(vec![4], make_params(1));

            let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(plan.phase, StepPhase::Prefill);
            assert_eq!(plan.token_budget, 1);
            assert_eq!(plan.sequences.len(), 1);
            assert_eq!(plan.sequences[0].request_id, 1);
            assert_eq!(plan.sequences[0].phase, SequencePhase::Prefill);
            assert_eq!(scheduler.running[0].num_scheduled_tokens, 0);
            assert_eq!(scheduler.running[0].completion_token_ids(), &[42]);
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

        fn prepare_decoder_before_first_decode(
            kv: &mut KvCacheManager,
            mut sequence: Sequence,
            completion_token: u32,
        ) -> Sequence {
            kv.allocate(&mut sequence, 0).unwrap();
            sequence.num_scheduled_tokens = sequence.num_prompt_tokens;
            kv.hash_blocks(&mut sequence);
            sequence.num_cached_tokens = sequence.num_prompt_tokens;
            sequence.num_scheduled_tokens = 0;
            sequence.append_token(completion_token);
            sequence.status = SequenceStatus::Running;
            sequence.is_prefill = false;
            sequence
        }

        #[test]
        fn cache_pressure_during_chunked_recovery_preempts_the_recovery_victim() {
            let mut scheduler = Scheduler::new(BLOCK_SIZE / 2 + 1, 512, 0.9);
            let mut kv = KvCacheManager::new_with_prefix_cache(3, BLOCK_SIZE, fake_cache(), false);

            let decoder = prepare_decoder_before_first_decode(
                &mut kv,
                Sequence::new(10, 10, (0..BLOCK_SIZE as u32).collect(), &make_params(2)),
                90,
            );
            scheduler.running.push_back(decoder);

            let mut recovery = prepare_decoder_before_first_decode(
                &mut kv,
                Sequence::new(
                    11,
                    11,
                    (1_000..1_000 + BLOCK_SIZE as u32).collect(),
                    &make_params(3),
                ),
                91,
            );
            recovery.num_cached_tokens = BLOCK_SIZE / 2;
            recovery.recompute_target_tokens = Some(BLOCK_SIZE + 1);
            recovery.is_prefill = true;
            scheduler.running.push_back(recovery);

            let decoder_plan = scheduler.plan_step(&mut kv).unwrap().unwrap();

            assert_eq!(decoder_plan.sequences.len(), 1);
            assert_eq!(decoder_plan.sequences[0].request_id, 10);
            assert_eq!(decoder_plan.token_budget, 1);
            assert_eq!(
                decoder_plan.blocked_admission,
                Some(BlockedAdmission {
                    request_id: 11,
                    reason: AdmissionBlockedReason::KvCache,
                })
            );
            let waiting = &scheduler.waiting[0];
            assert_eq!(waiting.request_id, 11);
            assert_eq!(waiting.status, SequenceStatus::Waiting);
            assert_eq!(waiting.num_cached_tokens, 0);
            assert_eq!(waiting.num_scheduled_tokens, 0);
            assert_eq!(waiting.recompute_target_tokens, Some(BLOCK_SIZE + 1));
            assert_eq!(waiting.completion_token_ids(), &[91]);
            assert!(waiting.block_table.is_empty());

            scheduler
                .apply_step_result(&result_for_plan(&decoder_plan, 42), &mut kv)
                .unwrap();
            let resumed = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(resumed.sequences[0].request_id, 11);
            assert_eq!(resumed.sequences[0].token_range, 0..BLOCK_SIZE / 2 + 1);
            assert!(!resumed.sequences[0].sampling_allowed);
        }

        #[test]
        fn recovery_plan_replays_prompt_and_completion_before_sampling() {
            let mut scheduler = Scheduler::new(BLOCK_SIZE + 1, 512, 0.9);
            let mut kv = KvCacheManager::new_with_prefix_cache(3, BLOCK_SIZE, fake_cache(), false);
            let survivor_params = make_params(2);
            let victim_params = SamplingParams {
                temperature: 0.25,
                top_k: Some(7),
                top_p: Some(0.85),
                max_tokens: 3,
                ignore_eos: true,
                presence_penalty: 0.5,
                frequency_penalty: -0.25,
                repetition_penalty: 1.1,
            };
            for sequence_id in 0..2 {
                let params = if sequence_id == 0 {
                    &survivor_params
                } else {
                    &victim_params
                };
                let sequence = prepare_decoder_before_first_decode(
                    &mut kv,
                    Sequence::new(
                        sequence_id,
                        sequence_id,
                        (0..BLOCK_SIZE as u32).collect(),
                        params,
                    ),
                    90 + sequence_id as u32,
                );
                scheduler.running.push_back(sequence);
            }

            let survivor = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(survivor.sequences.len(), 1);
            assert_eq!(survivor.sequences[0].request_id, 0);
            scheduler
                .apply_step_result(&result_for_plan(&survivor, 42), &mut kv)
                .unwrap();

            let recovery = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let sequence = &recovery.sequences[0];
            let expected_history = (0..BLOCK_SIZE as u32)
                .chain(std::iter::once(91))
                .collect::<Vec<_>>();
            assert_eq!(sequence.request_id, 1);
            assert_eq!(sequence.phase, SequencePhase::Prefill);
            assert_eq!(sequence.token_range, 0..BLOCK_SIZE + 1);
            assert_eq!(sequence.logical_positions, 0..BLOCK_SIZE + 1);
            assert_eq!(sequence.input_token_ids, expected_history);
            assert!(sequence.sampling_allowed);
            assert_sampling_params(&sequence.sampling_params, &victim_params);
        }

        #[test]
        fn recovery_reuses_an_exact_cached_prefix_but_replays_its_completion() {
            let mut scheduler = Scheduler::new(BLOCK_SIZE + 1, 512, 0.9);
            let mut kv = make_kv_mgr(3);
            let mut victim_block = None;
            for sequence_id in 0..2 {
                let sequence = prepare_decoder_before_first_decode(
                    &mut kv,
                    Sequence::new(
                        sequence_id,
                        sequence_id,
                        (sequence_id as u32 * 1_000
                            ..sequence_id as u32 * 1_000 + BLOCK_SIZE as u32)
                            .collect(),
                        &make_params(if sequence_id == 0 { 2 } else { 3 }),
                    ),
                    90 + sequence_id as u32,
                );
                if sequence_id == 1 {
                    victim_block = sequence.block_table.first().copied();
                }
                scheduler.running.push_back(sequence);
            }
            let victim_block = victim_block.unwrap();

            let survivor = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let released = kv.ownership_snapshot();
            assert!(released.free_block_ids.contains(&victim_block));
            assert!(!released.used_block_ids.contains(&victim_block));
            assert_eq!(released.ref_counts[victim_block], 0);
            assert!(scheduler.waiting[0].block_table.is_empty());
            scheduler
                .apply_step_result(&result_for_plan(&survivor, 42), &mut kv)
                .unwrap();

            let recovery = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let sequence = &recovery.sequences[0];
            assert_eq!(sequence.cache.cached_token_range, 0..BLOCK_SIZE);
            assert_eq!(sequence.cache.block_table[0], victim_block);
            assert_eq!(sequence.token_range, BLOCK_SIZE..BLOCK_SIZE + 1);
            assert_eq!(sequence.logical_positions, BLOCK_SIZE..BLOCK_SIZE + 1);
            assert_eq!(sequence.input_token_ids, vec![91]);
            assert_eq!(sequence.cache.kv_length, BLOCK_SIZE + 1);
            assert!(sequence.sampling_allowed);
            assert_eq!(recovery.attention.cu_seqlens_q, vec![0, 1]);
            assert_eq!(
                recovery.attention.cu_seqlens_k,
                vec![0, (BLOCK_SIZE + 1) as u32]
            );

            scheduler
                .apply_step_result(&result_for_plan(&recovery, 42), &mut kv)
                .unwrap();
            assert_eq!(scheduler.running[0].completion_token_ids(), &[91, 42]);
            assert_eq!(scheduler.running[0].recompute_target_tokens, None);
        }

        #[test]
        fn chunked_recovery_samples_once_only_after_the_last_history_token() {
            let mut scheduler = Scheduler::new(BLOCK_SIZE / 2, 512, 0.9);
            let mut kv = KvCacheManager::new_with_prefix_cache(3, BLOCK_SIZE, fake_cache(), false);
            for sequence_id in 0..2 {
                let sequence = prepare_decoder_before_first_decode(
                    &mut kv,
                    Sequence::new(
                        sequence_id,
                        sequence_id,
                        (0..BLOCK_SIZE as u32).collect(),
                        &make_params(if sequence_id == 0 { 2 } else { 3 }),
                    ),
                    90 + sequence_id as u32,
                );
                scheduler.running.push_back(sequence);
            }

            let survivor = scheduler.plan_step(&mut kv).unwrap().unwrap();
            scheduler
                .apply_step_result(&result_for_plan(&survivor, 42), &mut kv)
                .unwrap();

            for expected_range in [0..BLOCK_SIZE / 2, BLOCK_SIZE / 2..BLOCK_SIZE] {
                let chunk = scheduler.plan_step(&mut kv).unwrap().unwrap();
                let sequence = &chunk.sequences[0];
                assert_eq!(sequence.phase, SequencePhase::Prefill);
                assert_eq!(sequence.token_range, expected_range);
                assert!(!sequence.sampling_allowed);
                scheduler
                    .apply_step_result(&result_for_plan(&chunk, 99), &mut kv)
                    .unwrap();
                assert_eq!(scheduler.running[0].completion_token_ids(), &[91]);
                assert_eq!(
                    scheduler.running[0].recompute_target_tokens,
                    Some(BLOCK_SIZE + 1)
                );
            }

            let final_chunk = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let planned = &final_chunk.sequences[0];
            assert_eq!(planned.phase, SequencePhase::Prefill);
            assert_eq!(planned.token_range, BLOCK_SIZE..BLOCK_SIZE + 1);
            assert_eq!(planned.input_token_ids, vec![91]);
            assert_eq!(planned.token_history.len(), BLOCK_SIZE + 1);
            assert!(planned.sampling_allowed);

            let rejected = StepResult {
                plan_id: final_chunk.id,
                sequences: vec![crate::engine::SequenceStepResult {
                    request_id: planned.request_id,
                    sequence_id: planned.sequence_id,
                    sampled_token: None,
                }],
            };
            assert!(matches!(
                scheduler.apply_step_result(&rejected, &mut kv),
                Err(StepPlanError::ResultMismatch { .. })
            ));
            assert_eq!(scheduler.running[0].num_cached_tokens, BLOCK_SIZE);
            assert_eq!(scheduler.running[0].completion_token_ids(), &[91]);
            assert_eq!(
                scheduler.running[0].recompute_target_tokens,
                Some(BLOCK_SIZE + 1)
            );

            scheduler
                .apply_step_result(&result_for_plan(&final_chunk, 42), &mut kv)
                .unwrap();
            assert_eq!(scheduler.running[0].num_cached_tokens, BLOCK_SIZE + 1);
            assert_eq!(scheduler.running[0].completion_token_ids(), &[91, 42]);
            assert_eq!(scheduler.running[0].recompute_target_tokens, None);

            let decode = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(decode.sequences[0].phase, SequencePhase::Decode);
            assert_eq!(decode.sequences[0].input_token_ids, vec![42]);
            let completed = scheduler
                .apply_step_result(&result_for_plan(&decode, 43), &mut kv)
                .unwrap();
            assert_eq!(completed.len(), 1);
            assert_eq!(completed[0].request_id, 1);
            assert_eq!(completed[0].token_ids, vec![91, 42, 43]);
            assert_eq!(kv.num_free_blocks(), 3);
        }

        #[test]
        fn blocked_recovery_stays_waiting_without_claiming_progress() {
            let mut scheduler = Scheduler::new(64, 512, 0.9);
            let mut kv = KvCacheManager::new_with_prefix_cache(1, BLOCK_SIZE, fake_cache(), false);
            let params = make_params(3);
            let mut recovery = Sequence::new(44, 7, (0..BLOCK_SIZE as u32).collect(), &params);
            recovery.append_token(91);
            recovery.recompute_target_tokens = Some(BLOCK_SIZE + 1);
            scheduler.waiting.push_back(recovery);
            let ownership_before = kv.ownership_snapshot();

            let error = scheduler.plan_step(&mut kv).unwrap_err();

            assert!(matches!(
                error,
                StepPlanError::NoProgress {
                    waiting_sequences: 1,
                    running_sequences: 0,
                    token_budget: 64,
                    free_blocks: 1,
                    blocked_admission: Some(BlockedAdmission {
                        request_id: 44,
                        reason: AdmissionBlockedReason::KvCache,
                    }),
                }
            ));
            assert_eq!(kv.ownership_snapshot(), ownership_before);
            assert!(scheduler.in_flight.is_none());
            assert_eq!(scheduler.num_waiting(), 1);
            let waiting = &scheduler.waiting[0];
            assert_eq!(waiting.status, SequenceStatus::Waiting);
            assert_eq!(waiting.request_id, 44);
            assert_eq!(waiting.seq_id, 7);
            assert_eq!(waiting.num_cached_tokens, 0);
            assert_eq!(waiting.num_scheduled_tokens, 0);
            assert!(waiting.block_table.is_empty());
            assert_eq!(waiting.completion_token_ids(), &[91]);
            assert_eq!(waiting.recompute_target_tokens, Some(BLOCK_SIZE + 1));
            assert_sampling_params(waiting.sampling_params(), &params);
        }

        #[test]
        fn failed_victim_release_is_transactional_and_keeps_running_order() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = KvCacheManager::new_with_prefix_cache(2, BLOCK_SIZE, fake_cache(), false);
            for sequence_id in 0..2 {
                let sequence = prepare_decoder_before_first_decode(
                    &mut kv,
                    Sequence::new(
                        sequence_id,
                        sequence_id,
                        (0..BLOCK_SIZE as u32).collect(),
                        &make_params(3),
                    ),
                    90 + sequence_id as u32,
                );
                scheduler.running.push_back(sequence);
            }
            let victim_block = scheduler.running[1].block_table[0];
            scheduler.running[1].block_table.push(victim_block);
            let ownership_before = kv.ownership_snapshot();

            let error = scheduler.plan_step(&mut kv).unwrap_err();

            assert!(matches!(
                error,
                StepPlanError::Cache {
                    operation: CacheOperation::Deallocation,
                    sequence_id: 1,
                    ..
                }
            ));
            assert_eq!(kv.ownership_snapshot(), ownership_before);
            assert_eq!(
                scheduler
                    .running
                    .iter()
                    .map(|sequence| sequence.request_id)
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            let victim = &scheduler.running[1];
            assert_eq!(victim.status, SequenceStatus::Running);
            assert_eq!(victim.block_table, vec![victim_block, victim_block]);
            assert_eq!(victim.num_cached_tokens, BLOCK_SIZE);
            assert_eq!(victim.num_scheduled_tokens, 0);
            assert_eq!(victim.completion_token_ids(), &[91]);
            assert_eq!(victim.recompute_target_tokens, None);
            assert!(scheduler.waiting.is_empty());
            assert!(scheduler.in_flight.is_none());
        }

        #[test]
        fn repeated_preemption_advances_the_same_history_without_leaks_or_duplicates() {
            let mut scheduler = Scheduler::new(2 * BLOCK_SIZE + 2, 512, 0.9);
            let mut kv = KvCacheManager::new_with_prefix_cache(5, BLOCK_SIZE, fake_cache(), false);
            for sequence_id in 0..2 {
                let prompt_start = sequence_id as u32 * 1_000;
                let sequence = prepare_decoder_before_first_decode(
                    &mut kv,
                    Sequence::new(
                        sequence_id,
                        sequence_id,
                        (prompt_start..prompt_start + (2 * BLOCK_SIZE - 1) as u32).collect(),
                        &make_params(if sequence_id == 0 { 3 } else { 5 }),
                    ),
                    700 + sequence_id as u32,
                );
                scheduler.running.push_back(sequence);
            }

            let shared_decode = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(shared_decode.token_budget, 2);
            scheduler
                .apply_step_result(&result_for_plan(&shared_decode, 88), &mut kv)
                .unwrap();

            let first_preemption = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(first_preemption.sequences[0].request_id, 0);
            let first_target = scheduler.waiting[0].recompute_target_tokens.unwrap();
            assert_eq!(first_target, 2 * BLOCK_SIZE + 1);
            assert_eq!(scheduler.waiting[0].completion_token_ids(), &[701, 88]);
            assert!(scheduler.waiting[0].block_table.is_empty());
            scheduler
                .apply_step_result(&result_for_plan(&first_preemption, 88), &mut kv)
                .unwrap();

            let first_recovery = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(first_recovery.token_budget, first_target);
            assert_eq!(first_recovery.sequences[0].token_range, 0..first_target);
            assert!(first_recovery.sequences[0].sampling_allowed);
            scheduler
                .apply_step_result(&result_for_plan(&first_recovery, 88), &mut kv)
                .unwrap();
            assert_eq!(scheduler.running[0].request_id, 1);
            assert_eq!(scheduler.running[0].completion_token_ids(), &[701, 88, 88]);
            assert_eq!(scheduler.running[0].recompute_target_tokens, None);

            let competitor = prepare_decoder_before_first_decode(
                &mut kv,
                Sequence::new(
                    2,
                    2,
                    (2_000..2_000 + (2 * BLOCK_SIZE) as u32).collect(),
                    &make_params(2),
                ),
                702,
            );
            scheduler.running.push_front(competitor);

            let second_preemption = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(second_preemption.sequences[0].request_id, 2);
            let second_target = scheduler.waiting[0].recompute_target_tokens.unwrap();
            assert_eq!(second_target, 2 * BLOCK_SIZE + 2);
            assert!(second_target > first_target);
            assert_eq!(scheduler.waiting[0].request_id, 1);
            assert_eq!(scheduler.waiting[0].completion_token_ids(), &[701, 88, 88]);
            assert!(scheduler.waiting[0].block_table.is_empty());
            scheduler
                .apply_step_result(&result_for_plan(&second_preemption, 88), &mut kv)
                .unwrap();

            let second_recovery = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(second_recovery.token_budget, second_target);
            assert_eq!(second_recovery.sequences[0].token_range, 0..second_target);
            scheduler
                .apply_step_result(&result_for_plan(&second_recovery, 88), &mut kv)
                .unwrap();
            assert_eq!(
                scheduler.running[0].completion_token_ids(),
                &[701, 88, 88, 88]
            );
            assert_eq!(scheduler.running[0].recompute_target_tokens, None);

            let final_decode = scheduler.plan_step(&mut kv).unwrap().unwrap();
            let completed = scheduler
                .apply_step_result(&result_for_plan(&final_decode, 88), &mut kv)
                .unwrap();
            assert_eq!(completed.len(), 1);
            assert_eq!(completed[0].request_id, 1);
            assert_eq!(completed[0].token_ids, vec![701, 88, 88, 88, 88]);
            assert!(!scheduler.is_running());
            assert_eq!(kv.num_free_blocks(), 5);
            assert_eq!(
                kv.ownership_snapshot().ref_counts,
                vec![0; 5],
                "every physical block must be released after repeated recovery"
            );
        }

        #[test]
        fn recovery_shares_the_global_budget_without_starving_fifo_waiters() {
            let mut scheduler = Scheduler::new(3, 3, 0.9);
            let mut kv = KvCacheManager::new_with_prefix_cache(3, BLOCK_SIZE, fake_cache(), false);

            let decoder = prepare_decoder_before_first_decode(
                &mut kv,
                Sequence::new(10, 10, vec![1], &make_params(4)),
                10,
            );
            scheduler.running.push_back(decoder);

            let mut recovery = Sequence::new(11, 11, vec![2, 3, 4, 5], &make_params(2));
            recovery.append_token(9);
            recovery.recompute_target_tokens = Some(5);
            scheduler.waiting.push_back(recovery);
            scheduler
                .waiting
                .push_back(Sequence::new(12, 12, vec![6], &make_params(1)));

            let first = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(first.token_budget, 3);
            assert_eq!(
                first
                    .sequences
                    .iter()
                    .map(|sequence| (sequence.request_id, sequence.token_budget))
                    .collect::<Vec<_>>(),
                vec![(10, 1), (11, 2)]
            );
            assert_eq!(
                first.blocked_admission,
                Some(BlockedAdmission {
                    request_id: 12,
                    reason: AdmissionBlockedReason::TokenBudget,
                })
            );
            scheduler
                .apply_step_result(&result_for_plan(&first, 42), &mut kv)
                .unwrap();
            assert_eq!(scheduler.running[1].completion_token_ids(), &[9]);
            assert_eq!(scheduler.running[1].recompute_target_tokens, Some(5));

            let second = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(second.token_budget, 3);
            assert_eq!(
                second
                    .sequences
                    .iter()
                    .map(|sequence| (sequence.request_id, sequence.token_budget))
                    .collect::<Vec<_>>(),
                vec![(10, 1), (11, 1), (12, 1)]
            );
            assert!(!second.sequences[1].sampling_allowed);
            let second_outputs = scheduler
                .apply_step_result(&result_for_plan(&second, 42), &mut kv)
                .unwrap();
            assert_eq!(second_outputs.len(), 1);
            assert_eq!(second_outputs[0].request_id, 12);

            let final_step = scheduler.plan_step(&mut kv).unwrap().unwrap();
            assert_eq!(final_step.token_budget, 3);
            assert_eq!(
                final_step
                    .sequences
                    .iter()
                    .map(|sequence| (sequence.request_id, sequence.token_budget))
                    .collect::<Vec<_>>(),
                vec![(10, 1), (11, 2)]
            );
            assert_eq!(final_step.sequences[1].token_range, 3..5);
            assert_eq!(final_step.sequences[1].input_token_ids, vec![5, 9]);
            assert!(final_step.sequences[1].sampling_allowed);
            let final_outputs = scheduler
                .apply_step_result(&result_for_plan(&final_step, 42), &mut kv)
                .unwrap();
            assert_eq!(
                final_outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![10, 11]
            );
            assert!(!scheduler.is_running());
            assert_eq!(kv.num_free_blocks(), 3);
        }

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
                sequence.num_cached_tokens = BLOCK_SIZE;
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
            assert_eq!(plan.sequences[0].cache.cached_token_range, 0..256);
            assert_eq!(plan.attention.cu_seqlens_q, vec![0, 44, 100]);
            assert_eq!(plan.attention.cu_seqlens_k, vec![0, 300, 356]);
            assert_eq!(plan.attention.block_table.len(), 2);
        }

        #[test]
        fn mixed_prefill_steps_preserve_ids_and_one_global_budget() {
            let mut scheduler = Scheduler::new(3, 512, 0.9);
            let mut kv = make_kv_mgr(100);
            scheduler.add_request(vec![10, 11, 12, 13, 14], make_params(1));
            scheduler.add_request(vec![20], make_params(1));
            scheduler.add_request(vec![30, 31, 32, 33], make_params(1));
            let mut completed_request_ids = Vec::new();
            let prompt_lengths = [5, 1, 4];
            let mut next_prompt_offsets = [0, 0, 0];

            while scheduler.is_running() {
                let plan = scheduler.plan_step(&mut kv).unwrap().unwrap();
                assert!((1..=3).contains(&plan.token_budget));
                assert_eq!(
                    plan.token_budget,
                    plan.sequences
                        .iter()
                        .map(|sequence| sequence.token_budget)
                        .sum::<usize>()
                );
                for sequence in &plan.sequences {
                    assert!(sequence.request_id < prompt_lengths.len());
                    assert_eq!(
                        sequence.token_range.start,
                        next_prompt_offsets[sequence.request_id]
                    );
                    assert!(sequence.token_range.end <= prompt_lengths[sequence.request_id]);
                    assert_eq!(sequence.logical_positions, sequence.token_range);
                    next_prompt_offsets[sequence.request_id] = sequence.token_range.end;
                }
                completed_request_ids.extend(
                    scheduler
                        .apply_step_result(&result_for_plan(&plan, 42), &mut kv)
                        .unwrap()
                        .into_iter()
                        .map(|output| output.request_id),
                );
            }

            assert_eq!(completed_request_ids, vec![0, 1, 2]);
            assert_eq!(next_prompt_offsets, prompt_lengths);
            assert_eq!(kv.num_free_blocks(), 100);
        }

        #[test]
        fn decode_plan_caps_membership_at_the_global_budget() {
            let mut scheduler = Scheduler::new(2, 512, 0.9);
            let mut kv = make_kv_mgr(100);

            for sequence_id in 0..3 {
                let mut sequence =
                    Sequence::new(sequence_id, sequence_id, vec![10, 11], &make_params(16));
                kv.allocate(&mut sequence, 0).unwrap();
                sequence.num_cached_tokens = sequence.num_prompt_tokens;
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
