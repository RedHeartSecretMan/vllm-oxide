//! Token-level scheduler — `Scheduler` (V1 parity, nano-vllm algorithm).
//!
//! One `schedule()` step is either **pure-prefill** or **pure-decode**
//! (nano-vllm parity — simplifies attention metadata). Chunked prefill
//! applies **only to the first sequence** in a step when the token budget
//! is tight. Preemption is **recompute-only**: deallocate the sequence's
//! blocks and requeue at the front of `waiting` (V1 parity; v0.1 never
//! swaps KV to host — V0 `swap_space` dropped).
//!
//! `postprocess()` updates block hashes, advances `num_cached_tokens`,
//! appends sampled tokens, and finalises sequences on EOS or `max_tokens`.
//! Talks to `KvCacheManager`, never to `BlockPool` or `PagedKVCache`
//! directly (ADR-0004 seam). Reports `num_cached_blocks` for prefix-cache
//! hits counted before allocating.

use std::collections::VecDeque;

use crate::engine::kv_cache_manager::KvCacheManager;
use crate::engine::sequence::{Sequence, SequenceGroup, SequenceStatus};
use crate::SamplingParams;

/// Default `max_num_batched_tokens` — the maximum number of tokens the engine
/// can process in one prefill step. User story #21 (v0.1-spec).
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 16384;

/// Default `max_num_seqs` — the maximum number of concurrently-running
/// sequences. User story #22 (v0.1-spec).
pub const DEFAULT_MAX_NUM_SEQS: usize = 512;

/// Default `gpu_memory_utilization` — fraction of GPU memory to allocate
/// to the KV cache pool. User story #23 (v0.1-spec).
pub const DEFAULT_GPU_MEMORY_UTILIZATION: f32 = 0.9;

/// Output of a scheduling step — the scheduler tells the engine core
/// whether this step is prefill or decode. The actual scheduling state
/// is stored in the sequences themselves (`num_scheduled_tokens`,
/// `is_prefill`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleMode {
    /// All scheduled sequences are being prefilled (chunked or full).
    Prefill,
    /// All scheduled sequences are decoding (one token each).
    Decode,
}

#[derive(Debug, Clone)]
pub struct ScheduleOutput {
    pub mode: ScheduleMode,
    /// Number of prefix-cache blocks hit during allocation (prefill only).
    /// Set to 0 for decode steps.
    pub num_cached_blocks: usize,
}

/// Token-level scheduler — the algorithmic heart of the engine.
///
/// Maintains `waiting` / `running` deques of `SequenceGroup`s. One
/// `schedule()` call produces either a pure-prefill or pure-decode step.
/// `postprocess()` updates sequence state after the model forward pass
/// and sampling, producing `RequestOutput`s for finished sequences.
///
/// # V1 three-layer split
///
/// Scheduler talks to `KvCacheManager`, never to `BlockPool` or
/// `PagedKVCache` directly (ADR-0004 seam).
pub struct Scheduler {
    waiting: VecDeque<SequenceGroup>,
    running: VecDeque<SequenceGroup>,
    max_num_batched_tokens: usize,
    max_num_seqs: usize,
    /// Fraction of GPU memory reserved for the KV cache pool.
    /// Used by EngineCore to compute pool size; the scheduler itself
    /// does not allocate the pool.
    pub gpu_memory_utilization: f32,

    // Monotonic counters
    next_seq_id: usize,
    next_request_id: usize,
}

impl Scheduler {
    /// Create a new scheduler with configurable budgets.
    pub fn new(
        max_num_batched_tokens: usize,
        max_num_seqs: usize,
        gpu_memory_utilization: f32,
    ) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            max_num_batched_tokens,
            max_num_seqs,
            gpu_memory_utilization,
            next_seq_id: 0,
            next_request_id: 0,
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

    /// Add a new inference request. Creates a `Sequence` wrapped in a 1:1
    /// `SequenceGroup` (n>1 sampling deferred to v0.2).
    ///
    /// The new sequence starts in `Waiting` status.
    pub fn add_request(&mut self, prompt_token_ids: Vec<u32>, params: SamplingParams) {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let seq = Sequence::new(seq_id, prompt_token_ids, &params);
        let group = SequenceGroup::new(request_id, seq);
        self.waiting.push_back(group);
    }

    /// The primary scheduling step. Returns a `ScheduleOutput` indicating
    /// the step mode and cache-hit count. Mutates the internal sequence
    /// state in-place.
    ///
    /// # Algorithm (nano-vllm parity)
    ///
    /// 1. **Preempt** running sequences if there aren't enough free blocks
    ///    (recompute-only: deallocate + requeue at front of waiting).
    /// 2. If any running seq still needs prefill (chunked prefill
    ///    continuation): **prefill** step — schedule more prompt tokens.
    /// 3. If `running` is non-empty (all prefilling done): **decode**
    ///    step — schedule 1 token per running sequence.
    /// 4. If `running` is empty and `waiting` is non-empty: **prefill**
    ///    step — schedule waiting sequences with chunking for the first.
    pub fn schedule(&mut self, kv_mgr: &mut KvCacheManager) -> ScheduleOutput {
        self.preempt_if_needed(kv_mgr);

        // Chunked prefill continuation: running sequences that still need
        // more prompt tokens before they can decode.
        if self.running.iter().any(|g| g.seq().is_prefill) {
            return self.schedule_prefill_continue();
        }

        if !self.running.is_empty() {
            return self.schedule_decode(kv_mgr);
        }

        self.schedule_prefill(kv_mgr)
    }

    /// Post-process sequences after model forward + sampling: update
    /// `num_cached_tokens`, append sampled tokens, hash blocks, finalise
    /// on EOS or `max_tokens`. Returns `RequestOutput`s for sequences
    /// that finished this step.
    ///
    /// `sampled_tokens` must be in the same order as the running sequences
    /// at the time `schedule()` was called.
    pub fn postprocess(
        &mut self,
        sampled_tokens: &[u32],
        kv_mgr: &mut KvCacheManager,
    ) -> Vec<RequestOutput> {
        let batch = sampled_tokens.len();
        let mut outputs = Vec::new();

        let mut finished_indices = Vec::new();

        for (i, group) in self.running.iter_mut().enumerate() {
            if i >= batch {
                break;
            }
            let token_id = sampled_tokens[i];
            let seq = group.seq_mut();
            let scheduled = seq.num_scheduled_tokens;

            seq.append_token(token_id);

            let token_count = seq.num_completion_tokens();
            let hit_max_tokens = token_count >= seq.max_tokens;
            let hit_eos = token_id == EOS_TOKEN_ID && !seq.ignore_eos;

            if hit_max_tokens || hit_eos {
                seq.num_cached_tokens = (seq.num_cached_tokens + scheduled)
                    .min(seq.num_tokens);
                seq.num_scheduled_tokens = 0;
                seq.status = SequenceStatus::Finished;
                finished_indices.push(i);
                outputs.push(RequestOutput {
                    seq_id: seq.seq_id,
                    token_ids: seq.completion_token_ids().to_vec(),
                    text: String::new(),
                    finished: true,
                });
            } else {
                kv_mgr.hash_blocks(seq);
                seq.num_cached_tokens = (seq.num_cached_tokens + scheduled)
                    .min(seq.num_tokens);
                seq.num_scheduled_tokens = 0;
            }
        }

        for &idx in finished_indices.iter().rev() {
            let _ = kv_mgr.deallocate(self.running[idx].seq_mut());
            self.running.remove(idx);
        }

        outputs
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

    /// Iterator over running sequences (read-only). Used by EngineCore
    /// to build tensors for the forward pass.
    pub fn running_seqs(&self) -> impl Iterator<Item = &SequenceGroup> {
        self.running.iter()
    }

    /// Iterator over waiting sequences (read-only).
    pub fn waiting_seqs(&self) -> impl Iterator<Item = &SequenceGroup> {
        self.waiting.iter()
    }

    // ── Private helpers ─────────────────────────────────────────────

    /// Preempt running sequences when there aren't enough free blocks.
    ///
    /// The condition: if any running sequence can't `can_append` (would
    /// need a new block but none free), preempt the last running sequence
    /// (lowest priority). Recompute-only: deallocate blocks, requeue
    /// at the front of waiting.
    fn preempt_if_needed(&mut self, kv_mgr: &mut KvCacheManager) {
        let needs_preemption = self.running.iter().any(|g| !kv_mgr.can_append(g.seq()));

        if !needs_preemption {
            return;
        }

        while needs_preemption && !self.running.is_empty() {
            let mut victim = self.running.pop_back().unwrap();
            let _ = kv_mgr.deallocate(victim.seq_mut());
            let vseq = victim.seq_mut();
            vseq.status = SequenceStatus::Waiting;
            vseq.num_scheduled_tokens = 0;
            self.waiting.push_front(victim);

            // Re-check.
            let still_needed = self.running.iter().any(|g| !kv_mgr.can_append(g.seq()));
            if !still_needed {
                break;
            }
        }
    }

    /// Schedule a decode step: one token per running sequence.
    fn schedule_decode(&mut self, kv_mgr: &mut KvCacheManager) -> ScheduleOutput {
        for group in self.running.iter_mut() {
            let seq = group.seq_mut();

            if !kv_mgr.can_append(seq) {
                seq.num_scheduled_tokens = 0;
                continue;
            }

            let _ = kv_mgr.may_append(seq);
            seq.num_scheduled_tokens = 1;
            seq.is_prefill = false;
        }

        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].seq().num_scheduled_tokens == 0 {
                let mut victim = self.running.remove(i).unwrap();
                let _ = kv_mgr.deallocate(victim.seq_mut());
                let vseq = victim.seq_mut();
                vseq.status = SequenceStatus::Waiting;
                self.waiting.push_front(victim);
            } else {
                i += 1;
            }
        }

        ScheduleOutput {
            mode: ScheduleMode::Decode,
            num_cached_blocks: 0,
        }
    }

    /// Continue a chunked prefill: schedule more prompt tokens for
    /// running sequences that still have `is_prefill` set.
    fn schedule_prefill_continue(&mut self) -> ScheduleOutput {
        let mut total_tokens = 0;
        for group in self.running.iter_mut() {
            let seq = group.seq_mut();
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
        ScheduleOutput {
            mode: ScheduleMode::Prefill,
            num_cached_blocks: 0,
        }
    }

    /// Schedule a prefill step: pick sequences from `waiting`, chunk
    /// the first if the token budget is tight, allocate blocks, and
    /// move to `running`.
    fn schedule_prefill(&mut self, kv_mgr: &mut KvCacheManager) -> ScheduleOutput {
        if self.waiting.is_empty() {
            return ScheduleOutput {
                mode: ScheduleMode::Prefill,
                num_cached_blocks: 0,
            };
        }

        let max_running = self.max_num_seqs.saturating_sub(self.running.len());
        let mut total_tokens: usize = 0;
        let mut scheduled_count: usize = 0;
        let mut total_cached_blocks: usize = 0;

        let mut to_schedule: Vec<(usize, usize)> = Vec::new(); // (waiting_index, n_tokens)

        for i in 0..self.waiting.len().min(max_running) {
            let group = &self.waiting[i];
            let seq = group.seq();

            let remaining = seq.num_prompt_tokens.saturating_sub(seq.num_cached_tokens);
            if remaining == 0 {
                continue;
            }

            match kv_mgr.can_allocate(seq) {
                None => break,
                Some(num_cached_blocks) => {
                    let n_tokens = if scheduled_count == 0 {
                        let budget = self.max_num_batched_tokens.saturating_sub(total_tokens);
                        if remaining > budget && total_tokens == 0 {
                            budget
                        } else {
                            remaining.min(budget)
                        }
                    } else {
                        remaining
                    };

                    if n_tokens == 0 {
                        break;
                    }

                    total_tokens += n_tokens;
                    total_cached_blocks += num_cached_blocks;
                    scheduled_count += 1;
                    to_schedule.push((i, n_tokens));
                }
            }
        }

        for &(idx, n_tokens) in to_schedule.iter().rev() {
            let mut group = self.waiting.remove(idx).unwrap();
            let seq = group.seq_mut();

            match kv_mgr.can_allocate(seq) {
                Some(num_cached) => {
                    let _ = kv_mgr.allocate(seq, num_cached);
                    seq.num_scheduled_tokens = n_tokens;
                    let fully_prefilled =
                        seq.num_cached_tokens + n_tokens >= seq.num_prompt_tokens;
                    seq.is_prefill = !fully_prefilled;
                    seq.status = SequenceStatus::Running;
                    self.running.push_back(group);
                }
                None => {
                    seq.status = SequenceStatus::Waiting;
                    self.waiting.push_front(group);
                }
            }
        }

        ScheduleOutput {
            mode: ScheduleMode::Prefill,
            num_cached_blocks: total_cached_blocks,
        }
    }
}

/// Per-step return value containing the completion tokens for a finished
/// sequence. Accumulated by `LLM::generate` until `is_finished()`.
///
/// The `text` field is populated by the composition root (`llm.rs`) during
/// detokenization — the scheduler does not have access to a tokenizer.
#[derive(Debug, Clone)]
pub struct RequestOutput {
    pub seq_id: usize,
    pub token_ids: Vec<u32>,
    pub text: String,
    pub finished: bool,
}

/// End-of-sequence token id (used as a sentinel for the standard EOS token).
/// The actual EOS id depends on the model's tokenizer, but the scheduler
/// uses this constant for the stop-condition check. The engine loop
/// verifies against the model-specific value — see `EngineCore::step()`.
pub const EOS_TOKEN_ID: u32 = 151645; // Qwen3 default EOS = <|im_end|>

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
        Scheduler::with_defaults()
    }

    fn make_kv_mgr(num_blocks: usize) -> KvCacheManager {
        KvCacheManager::new(num_blocks, BLOCK_SIZE, fake_cache())
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
            assert_eq!(s.waiting[0].seq().seq_id, 0);
            assert_eq!(s.waiting[1].seq().seq_id, 1);
            assert_eq!(s.waiting[0].request_id(), 0);
            assert_eq!(s.waiting[1].request_id(), 1);
        }

        #[test]
        fn sequence_starts_waiting() {
            let mut s = make_scheduler();
            s.add_request(vec![1, 2, 3], make_params(16));
            assert_eq!(s.waiting[0].seq().status, SequenceStatus::Waiting);
        }
    }

    mod schedule_prefill {
        use super::*;

        #[test]
        fn prefills_single_waiting_sequence() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(
                (0..BLOCK_SIZE as u32 + 10).collect(),
                make_params(16),
            );
            let output = s.schedule(&mut kv);
            assert_eq!(output.mode, ScheduleMode::Prefill);
            assert_eq!(s.num_running(), 1);
            assert_eq!(s.num_waiting(), 0);

            let seq = s.running[0].seq();
            assert!(!seq.is_prefill, "fully prefilled seq should have is_prefill=false");
            assert_eq!(seq.status, SequenceStatus::Running);
            assert_eq!(seq.num_scheduled_tokens, BLOCK_SIZE + 10);
        }

        #[test]
        fn chunked_prefill_when_prompt_exceeds_budget() {
            let mut s = Scheduler::new(100, 512, 0.9); // small budget
            let mut kv = make_kv_mgr(100);
            s.add_request((0..500u32).collect(), make_params(16));
            let output = s.schedule(&mut kv);
            assert_eq!(output.mode, ScheduleMode::Prefill);
            let seq = s.running[0].seq();
            assert!(seq.is_prefill);
            // Should be chunked: budget is 100, so only 100 scheduled.
            assert_eq!(seq.num_scheduled_tokens, 100);
            assert!(seq.num_scheduled_tokens < seq.num_prompt_tokens);
        }

        #[test]
        fn empty_waiting_returns_prefill_with_zero_cached() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(10);
            let output = s.schedule(&mut kv);
            assert_eq!(output.mode, ScheduleMode::Prefill);
            assert_eq!(output.num_cached_blocks, 0);
            assert_eq!(s.num_running(), 0);
        }

        #[test]
        fn pure_prefill_no_running_allowed() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            // Add one request, prefill it.
            s.add_request(vec![1, 2, 3], make_params(16));
            s.schedule(&mut kv);
            assert_eq!(s.num_running(), 1);
            assert_eq!(s.num_waiting(), 0);

            // Add another request. Since running is non-empty, next schedule
            // should be decode (not prefill).
            s.add_request(vec![4, 5, 6], make_params(16));
            let output = s.schedule(&mut kv);
            assert_eq!(output.mode, ScheduleMode::Decode);
            // waiting should still have the new request.
            assert_eq!(s.num_waiting(), 1);
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
            s.schedule(&mut kv);
            assert_eq!(s.num_running(), 1);

            // Next step should be decode.
            let output = s.schedule(&mut kv);
            assert_eq!(output.mode, ScheduleMode::Decode);
            let seq = s.running[0].seq();
            assert!(!seq.is_prefill, "decode must set is_prefill=false");
            assert_eq!(seq.num_scheduled_tokens, 1);
        }

        #[test]
        fn decode_does_not_touch_waiting() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(16));
            s.schedule(&mut kv); // prefill → running
            s.add_request(vec![4, 5, 6], make_params(16)); // stays waiting

            let output = s.schedule(&mut kv); // decode
            assert_eq!(output.mode, ScheduleMode::Decode);
            assert_eq!(s.num_running(), 1);
            assert_eq!(s.num_waiting(), 1);
        }
    }

    mod postprocess {
        use super::*;

        #[test]
        fn advances_num_cached_tokens() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(16));
            s.schedule(&mut kv); // prefill

            let outputs = s.postprocess(&[42], &mut kv);
            let seq = s.running[0].seq();
            assert_eq!(seq.num_cached_tokens, 3); // 3 prompt + 1 generated
            assert!(outputs.is_empty(), "not yet finished");
        }

        #[test]
        fn finishes_on_eos() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(64));
            s.schedule(&mut kv); // prefill

            let outputs = s.postprocess(&[EOS_TOKEN_ID], &mut kv);
            assert_eq!(outputs.len(), 1);
            assert!(outputs[0].finished);
            assert_eq!(s.num_running(), 0, "finished seq should be removed");
        }

        #[test]
        fn finishes_on_max_tokens() {
            let mut s = make_scheduler();
            let mut kv = make_kv_mgr(100);
            s.add_request(vec![1, 2, 3], make_params(1)); // max_tokens=1
            s.schedule(&mut kv); // prefill

            let outputs = s.postprocess(&[99], &mut kv);
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
            s.schedule(&mut kv);

            let outputs = s.postprocess(&[EOS_TOKEN_ID], &mut kv);
            assert!(outputs.is_empty(), "ignore_eos should not finish on EOS");
            assert_eq!(s.num_running(), 1);
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
            s.schedule(&mut kv);
            assert_eq!(s.num_running(), 1);
            assert_eq!(kv.num_free_blocks(), 0);

            {
                let seq = s.running[0].seq_mut();
                while seq.num_tokens < BLOCK_SIZE + 1 {
                    seq.append_token(99);
                }
            }

            s.schedule(&mut kv);
            assert!(s.num_waiting() >= 1, "preempted seq should be in waiting");
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
            s.schedule(&mut kv);
            assert!(s.is_running());
        }
    }

    mod budgets {
        use super::*;

        #[test]
        fn respects_max_num_batched_tokens() {
            let mut s = Scheduler::new(50, 512, 0.9); // tight budget
            let mut kv = make_kv_mgr(100);
            s.add_request((0..200u32).collect(), make_params(16));
            s.schedule(&mut kv);
            let tokens = s.running[0].seq().num_scheduled_tokens;
            assert!(tokens <= 50, "must respect max_num_batched_tokens (50)");
        }

        #[test]
        fn respects_max_num_seqs() {
            let mut s = Scheduler::new(16384, 2, 0.9);
            let mut kv = make_kv_mgr(100);
            for _ in 0..5 {
                s.add_request(vec![1, 2, 3], make_params(16));
            }
            s.schedule(&mut kv);
            // Only 2 should have been scheduled (max_num_seqs).
            assert!(s.num_running() <= 2);
            assert!(s.num_waiting() > 0, "remaining should stay waiting");
        }
    }
}
