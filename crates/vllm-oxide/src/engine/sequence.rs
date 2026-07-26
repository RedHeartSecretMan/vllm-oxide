//! V1 sequence data model — `Sequence`, `SequenceGroup`, `SequenceStatus` (ADR-0004 M2).
//!
//! Mirrors `nano-vllm/nanovllm/engine/sequence.py` field-for-field, but with
//! `seq_id` as a constructor parameter (no global `itertools.count()`) and
//! the V1 three-layer split where `SequenceGroup` is a thin 1:1 wrapper
//! (n>1 sampling deferred to v0.2).
//!
//! # V1 three-layer split (ADR-0004)
//!
//! In V1 the data model lives below the scheduler: `Sequence` owns its
//! `block_table` and sampling scalars; `BlockPool` owns the physical block
//! lifetime; `KVCacheManager` is the only scheduler-facing seam. This file
//! contains the leaf types that all three layers reference.

use crate::SamplingParams;

/// Block size for KV cache blocks. Hardcoded per v0.1 spec
/// (`kvcache_block_size = 256`, constant, not per-arch).
pub const BLOCK_SIZE: usize = 256;

/// Sequence lifecycle status, mirroring `nanovllm.engine.sequence.SequenceStatus`.
///
/// Transitions: `Waiting → Running → Finished`. The engine loop (T2) drives
/// these transitions via `Scheduler::schedule()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

/// Per-sequence state — the V1 data-model leaf (ADR-0004 M2).
///
/// Mirrors `nanovllm.engine.sequence.Sequence` field-for-field, diverging only
/// in taking `seq_id` as a constructor param (no global counter). Owns its
/// `block_table` and copies of sampling scalars from `SamplingParams`.
///
/// # Block slicing
///
/// `block(i)` returns a slice of `token_ids` spanning `i * BLOCK_SIZE ..
/// (i + 1) * BLOCK_SIZE`. The precondition is `i < num_blocks()`. The final
/// block may be partial (fewer than `BLOCK_SIZE` tokens).
#[derive(Debug, Clone)]
pub struct Sequence {
    pub(crate) seq_id: usize,
    pub(crate) status: SequenceStatus,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) last_token: u32,
    pub(crate) num_tokens: usize,
    pub(crate) num_prompt_tokens: usize,
    pub(crate) num_cached_tokens: usize,
    pub(crate) num_scheduled_tokens: usize,
    pub(crate) is_prefill: bool,
    pub(crate) block_table: Vec<usize>,
    // Attached sampling scalars (V1 — Sequence owns copies, not a SamplingParams ref).
    pub(crate) temperature: f32,
    pub(crate) max_tokens: usize,
    pub(crate) ignore_eos: bool,
}

impl Sequence {
    /// Construct a new sequence in `Waiting` status.
    ///
    /// `seq_id` is caller-provided (EngineCore/Scheduler owns the counter in
    /// #21). Sampling scalars are read from `params` so the engine loop
    /// can access them without a `SamplingParams` ref.
    pub fn new(seq_id: usize, token_ids: Vec<u32>, params: &SamplingParams) -> Self {
        let num_tokens = token_ids.len();
        let last_token = *token_ids.last().unwrap_or(&0);
        Self {
            seq_id,
            status: SequenceStatus::Waiting,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_cached_tokens: 0,
            num_scheduled_tokens: 0,
            is_prefill: true,
            block_table: Vec::new(),
            temperature: params.temperature,
            max_tokens: params.max_tokens,
            ignore_eos: params.ignore_eos,
        }
    }

    /// Number of blocks needed to hold `num_tokens` tokens (rounding up).
    ///
    /// Mirrors `nanovllm.engine.sequence.Sequence.num_blocks`:
    /// `(num_tokens + block_size - 1) // block_size`. 0 tokens → 0 blocks.
    pub fn num_blocks(&self) -> usize {
        self.num_tokens.div_ceil(BLOCK_SIZE)
    }

    /// Number of tokens in the last (potentially partial) block.
    ///
    /// Mirrors `nanovllm.engine.sequence.Sequence.last_block_num_tokens`.
    /// Guaranteed to be in `[1, BLOCK_SIZE]` when `num_tokens > 0`.
    pub fn last_block_num_tokens(&self) -> usize {
        let nb = self.num_blocks();
        if nb == 0 {
            return 0;
        }
        self.num_tokens - (nb - 1) * BLOCK_SIZE
    }

    /// Return a slice of `token_ids` for block `i`.
    ///
    /// # Preconditions
    ///
    /// `i < self.num_blocks()`. Panics (debug assertion) otherwise.
    ///
    /// Full blocks span `BLOCK_SIZE` tokens; the final block returns whatever
    /// remains (1..=BLOCK_SIZE tokens).
    pub fn block(&self, i: usize) -> &[u32] {
        let start = i * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(self.num_tokens);
        &self.token_ids[start..end]
    }

    /// Append one token at the end of the sequence (decode step).
    ///
    /// Mirrors `nanovllm.engine.sequence.Sequence.append_token`: pushes to
    /// `token_ids`, updates `last_token`, increments `num_tokens`.
    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
    }

    /// Whether the sequence is in `Finished` status.
    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    /// Number of completion tokens generated so far (total − prompt).
    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    /// Prompt (input) portion of `token_ids`.
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.token_ids[..self.num_prompt_tokens]
    }

    /// Completion (generated) portion of `token_ids`.
    pub fn completion_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_prompt_tokens..]
    }
}

/// Thin 1:1 wrapper over a single `Sequence` sharing a `request_id`.
///
/// V1 parity: `SequenceGroup` groups one or more `Sequence`s that share a
/// logical request. v0.1 uses 1:1 (one group, one sequence) — n>1 sampling
/// (beam search / best-of / parallel sampling) is deferred to v0.2.
///
/// # v0.2 migration
///
/// When n>1 lands, this struct will hold `Vec<Sequence>` (or similar). The
/// accessor methods [`seq`], [`seq_mut`], and [`is_finished`] will operate on
/// the primary sequence (index 0). Callers should not rely on the 1:1 shape.
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    request_id: usize,
    seq: Sequence,
}

impl SequenceGroup {
    /// Create a 1:1 group wrapping one sequence.
    pub fn new(request_id: usize, seq: Sequence) -> Self {
        Self { request_id, seq }
    }

    /// Immutable access to the inner sequence.
    pub fn seq(&self) -> &Sequence {
        &self.seq
    }

    /// Mutable access to the inner sequence.
    pub fn seq_mut(&mut self) -> &mut Sequence {
        &mut self.seq
    }

    /// The logical request id shared across all sequences in this group.
    pub fn request_id(&self) -> usize {
        self.request_id
    }

    /// Delegates to the inner sequence's `is_finished()`.
    pub fn is_finished(&self) -> bool {
        self.seq.is_finished()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::SamplingParams;

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            max_tokens: 64,
            ..SamplingParams::default()
        }
    }

    fn make_seq(token_ids: Vec<u32>) -> Sequence {
        Sequence::new(0, token_ids, &greedy_params())
    }

    mod status_transitions {
        use super::*;

        #[test]
        fn new_sequence_is_waiting() {
            let s = make_seq(vec![1, 2, 3]);
            assert_eq!(s.status, SequenceStatus::Waiting);
        }

        #[test]
        fn can_transition_waiting_to_running() {
            let mut s = make_seq(vec![1, 2, 3]);
            s.status = SequenceStatus::Running;
            assert_eq!(s.status, SequenceStatus::Running);
        }

        #[test]
        fn can_transition_running_to_finished() {
            let mut s = make_seq(vec![1, 2, 3]);
            s.status = SequenceStatus::Running;
            s.status = SequenceStatus::Finished;
            assert_eq!(s.status, SequenceStatus::Finished);
        }

        #[test]
        fn is_finished_true_only_when_finished() {
            let mut s = make_seq(vec![1, 2, 3]);
            assert!(!s.is_finished());
            s.status = SequenceStatus::Running;
            assert!(!s.is_finished());
            s.status = SequenceStatus::Finished;
            assert!(s.is_finished());
        }
    }

    mod num_blocks {
        use super::*;

        #[test]
        fn zero_tokens_zero_blocks() {
            let s = make_seq(vec![]);
            assert_eq!(s.num_blocks(), 0);
        }

        #[test]
        fn one_to_block_size_is_one_block() {
            for n in 1..=BLOCK_SIZE {
                let tokens: Vec<u32> = (0..n as u32).collect();
                let s = make_seq(tokens);
                assert_eq!(s.num_blocks(), 1, "{n} tokens should give 1 block");
            }
        }

        #[test]
        fn block_size_plus_one_is_two_blocks() {
            let tokens: Vec<u32> = (0..=BLOCK_SIZE as u32).collect();
            let s = make_seq(tokens);
            assert_eq!(s.num_blocks(), 2);
        }

        #[test]
        fn exact_multiple_gives_exact_count() {
            for n in [BLOCK_SIZE, 2 * BLOCK_SIZE, 3 * BLOCK_SIZE] {
                let tokens: Vec<u32> = (0..n as u32).collect();
                let s = make_seq(tokens);
                assert_eq!(
                    s.num_blocks(),
                    n / BLOCK_SIZE,
                    "{n} tokens -> {n}/{BLOCK_SIZE} blocks"
                );
            }
        }
    }

    mod last_block_num_tokens {
        use super::*;

        #[test]
        fn zero_tokens_zero() {
            let s = make_seq(vec![]);
            assert_eq!(s.last_block_num_tokens(), 0);
        }

        #[test]
        fn full_block_returns_block_size() {
            for n in [BLOCK_SIZE, 2 * BLOCK_SIZE, 3 * BLOCK_SIZE] {
                let tokens: Vec<u32> = (0..n as u32).collect();
                let s = make_seq(tokens);
                assert_eq!(
                    s.last_block_num_tokens(),
                    BLOCK_SIZE,
                    "{n} tokens -> last full block"
                );
            }
        }

        #[test]
        fn partial_block_returns_remainder() {
            for k in 1..BLOCK_SIZE {
                let n = 2 * BLOCK_SIZE + k;
                let tokens: Vec<u32> = (0..n as u32).collect();
                let s = make_seq(tokens);
                assert_eq!(s.last_block_num_tokens(), k, "{n} tokens -> remainder {k}");
            }
        }
    }

    mod block_slice {
        use super::*;

        #[test]
        fn single_full_block() {
            let tokens: Vec<u32> = (0..BLOCK_SIZE as u32).collect();
            let s = make_seq(tokens.clone());
            assert_eq!(s.block(0), &tokens[..]);
        }

        #[test]
        fn two_full_blocks() {
            let tokens: Vec<u32> = (0..(2 * BLOCK_SIZE) as u32).collect();
            let s = make_seq(tokens.clone());
            let block0: Vec<u32> = tokens[..BLOCK_SIZE].to_vec();
            let block1: Vec<u32> = tokens[BLOCK_SIZE..2 * BLOCK_SIZE].to_vec();
            assert_eq!(s.block(0), &block0[..]);
            assert_eq!(s.block(1), &block1[..]);
        }

        #[test]
        fn final_partial_block() {
            let n = 2 * BLOCK_SIZE + 42;
            let tokens: Vec<u32> = (0..n as u32).collect();
            let s = make_seq(tokens.clone());
            assert_eq!(s.block(0), &tokens[..BLOCK_SIZE]);
            assert_eq!(s.block(1), &tokens[BLOCK_SIZE..2 * BLOCK_SIZE]);
            assert_eq!(s.block(2), &tokens[2 * BLOCK_SIZE..]);
        }
    }

    mod append_token {
        use super::*;

        #[test]
        fn append_bumps_num_tokens_and_sets_last_token() {
            let mut s = make_seq(vec![1, 2, 3]);
            let prev = s.num_tokens;
            s.append_token(42);
            assert_eq!(s.num_tokens, prev + 1);
            assert_eq!(s.last_token, 42);
        }

        #[test]
        fn append_leaves_num_prompt_tokens_unchanged() {
            let mut s = make_seq(vec![1, 2, 3]);
            s.append_token(42);
            s.append_token(99);
            assert_eq!(s.num_prompt_tokens, 3);
            assert_eq!(s.num_tokens, 5);
        }

        #[test]
        fn append_adds_to_token_ids() {
            let mut s = make_seq(vec![10, 20]);
            s.append_token(30);
            assert_eq!(s.token_ids, vec![10, 20, 30]);
        }
    }

    mod completion_tokens {
        use super::*;

        #[test]
        fn zero_completion_initially() {
            let s = make_seq(vec![1, 2, 3, 4, 5]);
            assert_eq!(s.num_completion_tokens(), 0);
            assert_eq!(s.prompt_token_ids(), &[1, 2, 3, 4, 5]);
            assert!(s.completion_token_ids().is_empty());
        }

        #[test]
        fn after_appends_partition_correctly() {
            let mut s = make_seq(vec![100, 200, 300]);
            s.append_token(400);
            s.append_token(500);
            assert_eq!(s.num_completion_tokens(), 2);
            assert_eq!(s.prompt_token_ids(), &[100, 200, 300]);
            assert_eq!(s.completion_token_ids(), &[400, 500]);
        }
    }

    mod group {
        use super::*;

        #[test]
        fn group_new_and_accessors() {
            let seq = make_seq(vec![1, 2, 3]);
            let group = SequenceGroup::new(42, seq);
            assert_eq!(group.request_id(), 42);
            assert_eq!(group.seq().seq_id, 0);
        }

        #[test]
        fn group_is_finished_delegates_to_seq() {
            let seq = make_seq(vec![1]);
            let mut group = SequenceGroup::new(0, seq);
            assert!(!group.is_finished());
            group.seq_mut().status = SequenceStatus::Finished;
            assert!(group.is_finished());
        }

        #[test]
        fn group_seq_mut_allows_mutation() {
            let seq = make_seq(vec![1, 2]);
            let mut group = SequenceGroup::new(0, seq);
            group.seq_mut().append_token(42);
            assert_eq!(group.seq().num_tokens, 3);
        }
    }
}
