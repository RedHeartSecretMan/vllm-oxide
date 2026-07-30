//! Physical block pool — `Block`, `BlockPool` (V1 prefix-cache leaf).
//!
//! Mirrors `nanovllm.engine.block_manager.BlockManager` algorithmically
//! (the class name in Rust is `BlockPool` for V1 parity). Owns the
//! free-list deque, the used-set, and the xxhash-chained prefix-cache
//! hashtable. CoW semantics for shared prefix blocks.
//!
//! # V1 three-layer split (ADR-0004)
//!
//! `BlockPool` is the middle leaf — below `KVCacheManager` (the only
//! scheduler-facing seam) and above the physical `PagedKVCache`. The
//! scheduler never imports `BlockPool` directly; it goes through
//! `KVCacheManager`. This file contains `BlockPool` and `Block`, which
//! are `pub(crate)` — the full `pub` lift happens in `lib.rs`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hasher;

use twox_hash::XxHash64;

use super::sequence::Sequence;

/// Error type for fallible `BlockPool` operations.
///
/// Mirrors nano-vllm's asserts for precondition violations (out of memory,
/// double-free, etc.) but returns `Result` instead of panicking, keeping
/// the workspace `panic = "warn"` lint clean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPoolError(pub String);

impl std::fmt::Display for BlockPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockPool: {}", self.0)
    }
}

impl std::error::Error for BlockPoolError {}

/// A physical KV-cache block.
///
/// Mirrors `nanovllm.engine.block_manager.Block`. `hash = -1` means
/// unhashed (nano-vllm sentinel convention). `ref_count` tracks the number
/// of sequences sharing this block (CoW prefix-cache semantics).
#[derive(Debug, Clone)]
pub struct Block {
    pub(crate) block_id: usize,
    pub(crate) ref_count: usize,
    pub(crate) hash: i64,
    pub(crate) token_ids: Vec<u32>,
}

impl Block {
    pub(crate) fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: -1,
            token_ids: Vec::new(),
        }
    }

    /// Set the hash and token_ids (called when a block is filled and hashed).
    pub(crate) fn update(&mut self, hash: i64, token_ids: Vec<u32>) {
        self.hash = hash;
        self.token_ids = token_ids;
    }

    /// Reset to allocated-but-empty state (ref_count = 1, hash = -1).
    ///
    /// Called by `_allocate_block` after popping from the free list.
    /// nano-vllm asserts `ref_count == 0` before this call.
    pub(crate) fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = -1;
        self.token_ids = Vec::new();
    }
}

/// Block pool with prefix-cache hashtable and free-list management.
///
/// Mirrors `nanovllm.engine.block_manager.BlockManager` algorithmically.
/// Key differences from nano-vllm:
///
/// - Class name is `BlockPool` (V1 parity).
/// - Fallible operations return `Result` instead of asserting.
/// - Token type is `u32` end-to-end (nano-vllm's numpy defaults to int64
///   for token arrays; we use u32 because our token type is u32 end-to-end).
///
/// # Chained prefix-cache hash
///
/// Each block's hash incorporates the previous block's hash so that
/// prefix-cache entries appear incrementally: `hash[i] = XXH64(prev_hash
/// || token_ids[i])`. This means a shared prefix of length N produces the
/// same chained hash sequence regardless of what follows.
#[derive(Debug, Clone)]
pub struct BlockPool {
    pub(crate) block_size: usize,
    pub(crate) blocks: Vec<Block>,
    pub(crate) hash_to_block_id: HashMap<i64, usize>,
    pub(crate) free_block_ids: VecDeque<usize>,
    pub(crate) used_block_ids: HashSet<usize>,
}

impl BlockPool {
    /// Create a pool with `num_blocks` pre-allocated (all free).
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let blocks: Vec<Block> = (0..num_blocks).map(Block::new).collect();
        let free_block_ids: VecDeque<usize> = (0..num_blocks).collect();
        Self {
            block_size,
            blocks,
            hash_to_block_id: HashMap::new(),
            free_block_ids,
            used_block_ids: HashSet::new(),
        }
    }

    /// Compute the chained xxhash for a set of token ids with the given
    /// prefix hash.
    ///
    /// Mirrors nano-vllm's `BlockManager.compute_hash`:
    /// `xxhash.xxh64().update(prefix.to_bytes(8, 'little'))` then
    /// `.update(np.array(token_ids).tobytes())`.
    ///
    /// nano-vllm's numpy defaults to int64 for token arrays; we use u32
    /// because our token type is u32 end-to-end — the hash is internally
    /// consistent (not cross-imp comparable, which is not a v0.1 release
    /// criterion).
    ///
    /// When `prefix == -1` (the nano-vllm sentinel for "no previous block"),
    /// the prefix bytes are omitted from the hash input. This matches
    /// nano-vllm's guard: `if prefix != -1: h.update(prefix.to_bytes(8, "little"))`.
    pub fn compute_hash(token_ids: &[u32], prefix: i64) -> i64 {
        let mut hasher = XxHash64::default();
        if prefix != -1 {
            hasher.write(&prefix.to_le_bytes());
        }
        for &t in token_ids {
            hasher.write(&t.to_le_bytes());
        }
        hasher.finish() as i64
    }

    /// Allocate a free block and return its id.
    ///
    /// Equivalent to nano-vllm's `_allocate_block`: pops from the front of
    /// the free deque, cleans the old hash entry if present, resets the
    /// block, and adds it to the used set.
    ///
    /// # Errors
    ///
    /// Returns `BlockPoolError` if the free list is empty (out of memory).
    fn allocate_block_private(&mut self) -> Result<usize, BlockPoolError> {
        let block_id = self
            .free_block_ids
            .pop_front()
            .ok_or_else(|| BlockPoolError("no free blocks available".to_string()))?;
        let block = &self.blocks[block_id];
        // nano-vllm asserts ref_count == 0 here.
        if block.hash != -1 {
            if let Some(&existing) = self.hash_to_block_id.get(&block.hash) {
                if existing == block_id {
                    self.hash_to_block_id.remove(&block.hash);
                }
            }
        }
        self.blocks[block_id].reset();
        self.used_block_ids.insert(block_id);
        Ok(block_id)
    }

    /// Return a used block to the free list (ref_count must be 0).
    ///
    /// Equivalent to nano-vllm's `_deallocate_block`. Panic-free: returns
    /// `Err` if the block is not in the used set.
    fn deallocate_block_private(&mut self, block_id: usize) -> Result<(), BlockPoolError> {
        if !self.used_block_ids.remove(&block_id) {
            return Err(BlockPoolError(format!(
                "block {block_id} is not in the used set (double-free?)"
            )));
        }
        self.free_block_ids.push_back(block_id);
        Ok(())
    }

    /// Check whether a sequence can be allocated and, if so, how many
    /// blocks are cache-hits.
    ///
    /// Mirrors nano-vllm's `BlockManager.can_allocate`. Returns
    /// `Some(num_cached_blocks)` when there is room, `None` when
    /// insufficient free blocks.
    ///
    /// Walks `seq.num_blocks() - 1` blocks (the last block is partial and
    /// never cached). For each block, computes the chained hash, checks
    /// the hashtable + token_ids match. Counts how many new blocks would
    /// be needed (fewer when a cached block is already in `used_block_ids`
    /// — shared, counts against free only if not currently used).
    pub fn can_allocate(&self, seq: &Sequence) -> Option<usize> {
        if seq.num_blocks() == 0 {
            return Some(0);
        }
        let mut h: i64 = -1;
        let mut num_cached_blocks: usize = 0;
        let num_blocks = seq.num_blocks();
        // The last block is partial — never cached.
        let check_until = if num_blocks > 0 { num_blocks - 1 } else { 0 };

        let mut num_new_blocks = num_blocks;
        for i in 0..check_until {
            let token_ids = seq.block(i);
            h = Self::compute_hash(token_ids, h);
            match self.hash_to_block_id.get(&h) {
                Some(&block_id) if self.blocks[block_id].token_ids == token_ids => {
                    num_cached_blocks += 1;
                    if self.used_block_ids.contains(&block_id) {
                        num_new_blocks -= 1;
                    }
                }
                _ => break,
            }
        }

        if self.free_block_ids.len() < num_new_blocks {
            return None;
        }
        Some(num_cached_blocks)
    }

    /// Allocate blocks for a sequence, using cached blocks where possible.
    ///
    /// Mirrors nano-vllm's `BlockManager.allocate`. For `num_cached_blocks`
    /// blocks that were found in `can_allocate`: bumps ref_count if already
    /// used, or moves from free to used with ref_count=1. For the remaining
    /// uncached blocks: calls `allocate_block_private`. Sets
    /// `seq.num_cached_tokens`.
    ///
    /// # Errors
    ///
    /// Returns `BlockPoolError` if a new block cannot be allocated (should
    /// not happen if `can_allocate` returned `Some`).
    pub fn allocate(
        &mut self,
        seq: &mut Sequence,
        num_cached_blocks: usize,
    ) -> Result<(), BlockPoolError> {
        if !seq.block_table.is_empty() {
            return Err(BlockPoolError(
                "sequence already has a block table".to_string(),
            ));
        }

        let mut h: i64 = -1;
        for i in 0..num_cached_blocks {
            let token_ids = seq.block(i).to_vec();
            h = Self::compute_hash(&token_ids, h);
            let block_id = *self
                .hash_to_block_id
                .get(&h)
                .ok_or_else(|| BlockPoolError("cached block hash not found".to_string()))?;
            let block = &mut self.blocks[block_id];
            if self.used_block_ids.contains(&block_id) {
                block.ref_count += 1;
            } else {
                // Block is in hash_to_block_id but not used — a previously
                // hashed-then-deallocated block; move from free to used.
                let idx = self
                    .free_block_ids
                    .iter()
                    .position(|&fid| fid == block_id)
                    .ok_or_else(|| BlockPoolError("cached block not in free list".to_string()))?;
                self.free_block_ids.remove(idx);
                block.ref_count = 1;
                self.used_block_ids.insert(block_id);
            }
            seq.block_table.push(block_id);
        }

        for _ in num_cached_blocks..seq.num_blocks() {
            let block_id = self.allocate_block_private()?;
            seq.block_table.push(block_id);
        }

        seq.num_cached_tokens = num_cached_blocks * self.block_size;
        Ok(())
    }

    /// Deallocate all blocks owned by a sequence.
    ///
    /// Mirrors nano-vllm's `BlockManager.deallocate`. Walks
    /// `seq.block_table` in reverse, decrementing ref_count and freeing
    /// blocks that reach 0. Clears `seq.block_table` and resets
    /// `num_cached_tokens` to 0.
    pub fn deallocate(&mut self, seq: &mut Sequence) -> Result<(), BlockPoolError> {
        for &block_id in seq.block_table.iter().rev() {
            let block = &mut self.blocks[block_id];
            if block.ref_count == 0 {
                return Err(BlockPoolError(format!(
                    "block {block_id} already has ref_count 0 (double-free?)"
                )));
            }
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.deallocate_block_private(block_id)?;
            }
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
        Ok(())
    }

    /// Check whether the pool has enough free blocks for the next
    /// append: if the next token starts a new block, we need 1 free block.
    pub fn can_append(&self, seq: &Sequence) -> bool {
        let need_new_block = usize::from(seq.num_tokens % self.block_size == 1);
        self.free_block_ids.len() >= need_new_block
    }

    /// Allocate a new block if the next append starts a new block
    /// (`num_tokens % block_size == 1`).
    ///
    /// Mirrors nano-vllm's `BlockManager.may_append`.
    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<(), BlockPoolError> {
        if seq.num_tokens % self.block_size == 1 {
            let block_id = self.allocate_block_private()?;
            seq.block_table.push(block_id);
        }
        Ok(())
    }

    /// Hash blocks that have been filled since the last hash_blocks call.
    ///
    /// Mirrors nano-vllm's `BlockManager.hash_blocks`. Operates on the
    /// range `[num_cached_tokens / block_size, (num_cached_tokens +
    /// num_scheduled_tokens) / block_size)`. Computes the chained hash
    /// from the previous block's hash (or -1 if starting from block 0),
    /// then updates each block and the hashtable.
    pub fn hash_blocks(&mut self, seq: &mut Sequence) {
        let start = seq.num_cached_tokens / self.block_size;
        let end = (seq.num_cached_tokens + seq.num_scheduled_tokens) / self.block_size;
        if start >= end {
            return;
        }
        // Retrieve the prefix hash from the block before `start`.
        let mut h: i64 = if start > 0 {
            self.blocks[seq.block_table[start - 1]].hash
        } else {
            -1
        };

        for i in start..end {
            let block_id = seq.block_table[i];
            let token_ids = seq.block(i).to_vec();
            h = Self::compute_hash(&token_ids, h);
            let block = &mut self.blocks[block_id];
            block.update(h, token_ids);
            self.hash_to_block_id.insert(h, block_id);
        }
    }

    /// Number of free (unused) blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_block_ids.len()
    }

    /// Total number of blocks in the pool.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]
mod tests {
    use super::*;

    fn make_seq(token_ids: Vec<u32>) -> Sequence {
        Sequence::new(
            0,
            0,
            token_ids,
            &crate::SamplingParams {
                max_tokens: 64,
                ..crate::SamplingParams::default()
            },
        )
    }

    mod compute_hash {
        use super::*;

        #[test]
        fn prefix_sensitive() {
            let tokens = vec![1u32, 2, 3];
            // Same tokens with different prefixes → different hashes.
            let h_a = BlockPool::compute_hash(&tokens, -1);
            let h_b = BlockPool::compute_hash(&tokens, 0);
            let h_c = BlockPool::compute_hash(&tokens, 42);
            assert_ne!(h_a, h_b, "prefix -1 vs 0 must differ");
            assert_ne!(h_a, h_c, "prefix -1 vs 42 must differ");
            assert_ne!(h_b, h_c, "prefix 0 vs 42 must differ");
        }

        #[test]
        fn same_tokens_and_prefix_produce_same_hash() {
            let tokens = vec![10u32, 20, 30, 40];
            let h1 = BlockPool::compute_hash(&tokens, 100);
            let h2 = BlockPool::compute_hash(&tokens, 100);
            assert_eq!(h1, h2);
        }

        #[test]
        fn token_order_sensitive() {
            let a = BlockPool::compute_hash(&[1u32, 2, 3], -1);
            let b = BlockPool::compute_hash(&[3u32, 2, 1], -1);
            assert_ne!(a, b, "[1,2,3] and [3,2,1] must hash differently");
        }

        #[test]
        fn empty_tokens_produce_deterministic_hash() {
            let h = BlockPool::compute_hash(&[], -1);
            // Just check it doesn't crash and is consistent.
            assert_eq!(BlockPool::compute_hash(&[], -1), h);
        }

        #[test]
        fn prefix_minus_one_omits_prefix_bytes() {
            // -1 prefix should produce the same hash as no-prefix (empty prefix),
            // because the guard skips the update entirely.
            let with_minus_one = BlockPool::compute_hash(&[7u32, 8], -1);
            let without_prefix = {
                let mut hasher = XxHash64::default();
                for &t in &[7u32, 8] {
                    hasher.write(&t.to_le_bytes());
                }
                hasher.finish() as i64
            };
            assert_eq!(with_minus_one, without_prefix);
        }

        #[test]
        fn non_minus_one_prefix_alters_hash() {
            // With prefix=0, the hash includes the prefix bytes.
            // This should differ from the prefix=-1 case.
            let with_prefix = BlockPool::compute_hash(&[7u32, 8], 0);
            let without_prefix = {
                let mut hasher = XxHash64::default();
                hasher.write(&0i64.to_le_bytes());
                for &t in &[7u32, 8] {
                    hasher.write(&t.to_le_bytes());
                }
                hasher.finish() as i64
            };
            assert_eq!(with_prefix, without_prefix);
        }
    }

    mod can_allocate {
        use super::*;

        #[test]
        fn returns_some_zero_for_empty_seq() {
            let pool = BlockPool::new(10, 256);
            let seq = make_seq(vec![]);
            assert_eq!(pool.can_allocate(&seq), Some(0));
        }

        #[test]
        fn returns_none_when_pool_exhausted() {
            let pool = BlockPool::new(2, 256);
            // A sequence needing 4 blocks should fail.
            let tokens: Vec<u32> = (0..(4 * 256 + 1) as u32).collect();
            let seq = make_seq(tokens);
            assert_eq!(pool.can_allocate(&seq), None);

            // But a 2-block sequence should succeed (needs 2 blocks, 2 free).
            let tokens2: Vec<u32> = (0..(2 * 256) as u32).collect();
            let seq2 = make_seq(tokens2);
            assert_eq!(pool.can_allocate(&seq2), Some(0));
        }

        #[test]
        fn counts_some_cached_blocks() {
            let mut pool = BlockPool::new(10, 256);
            // Populate the hashtable with 2 cached blocks.
            let tokens_b1: Vec<u32> = (0..256).collect();
            let tokens_b2: Vec<u32> = (256..512).collect();
            let h1 = BlockPool::compute_hash(&tokens_b1, -1);
            let h2 = BlockPool::compute_hash(&tokens_b2, h1);
            // Simulate that these blocks were hashed and are used.
            let b1 = pool.allocate_block_private().unwrap();
            let b2 = pool.allocate_block_private().unwrap();
            pool.blocks[b1].update(h1, tokens_b1);
            pool.blocks[b2].update(h2, tokens_b2);
            pool.hash_to_block_id.insert(h1, b1);
            pool.hash_to_block_id.insert(h2, b2);

            // Sequence with 3 full blocks. Should find 2 cached, need 1 new.
            let tokens_seq: Vec<u32> = (0..(3 * 256) as u32).collect();
            let seq = make_seq(tokens_seq);
            let result = pool.can_allocate(&seq);
            assert_eq!(result, Some(2));
        }
    }

    mod allocate {
        use super::*;

        #[test]
        fn sets_block_table_and_cached_tokens() {
            let mut pool = BlockPool::new(10, 256);
            let tokens: Vec<u32> = (0..(3 * 256 + 50) as u32).collect();
            let mut seq = make_seq(tokens);
            let num_cached = pool.can_allocate(&seq).unwrap();
            // No cached blocks since hashtable is empty.
            assert_eq!(num_cached, 0);
            pool.allocate(&mut seq, 0).unwrap();

            assert_eq!(seq.block_table.len(), 4); // 3 full + 1 partial
            assert_eq!(seq.num_cached_tokens, 0);
        }

        #[test]
        fn shared_block_bumps_ref_count() {
            let mut pool = BlockPool::new(10, 256);
            let prefix_tokens: Vec<u32> = (0..256).collect();
            let h = BlockPool::compute_hash(&prefix_tokens, -1);

            // Allocate and hash the prefix block.
            let mut seq_a = make_seq(prefix_tokens.clone());
            pool.allocate(&mut seq_a, 0).unwrap();
            // Manually hash the block.
            let block_id = seq_a.block_table[0];
            pool.blocks[block_id].update(h, prefix_tokens.clone());
            pool.hash_to_block_id.insert(h, block_id);
            // seq_a has ref_count=1 from allocate_block_private.

            // Allocate seq_b with same prefix. can_allocate should find it cached.
            let mut tokens_b: Vec<u32> = prefix_tokens.clone();
            tokens_b.push(999); // one extra token
            let mut seq_b = make_seq(tokens_b);
            let num_cached = pool.can_allocate(&seq_b).unwrap();
            assert_eq!(num_cached, 1);

            pool.allocate(&mut seq_b, 1).unwrap();
            // The shared prefix block should have ref_count=2.
            assert_eq!(pool.blocks[block_id].ref_count, 2);
            assert_eq!(seq_b.block_table[0], block_id);
        }

        #[test]
        fn allocate_twice_without_dealloc_errors() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq(vec![1u32, 2, 3]);
            pool.allocate(&mut seq, 0).unwrap();
            let result = pool.allocate(&mut seq, 0);
            assert!(result.is_err());
        }
    }

    mod deallocate {
        use super::*;

        #[test]
        fn releases_blocks_back_to_free() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq((0..(2 * 256 + 1) as u32).collect());
            let initial_free = pool.num_free_blocks();
            pool.allocate(&mut seq, 0).unwrap();
            assert_eq!(pool.num_free_blocks(), initial_free - 3);
            assert!(!seq.block_table.is_empty());

            pool.deallocate(&mut seq).unwrap();
            assert_eq!(pool.num_free_blocks(), initial_free);
            assert!(seq.block_table.is_empty());
            assert_eq!(seq.num_cached_tokens, 0);
        }

        #[test]
        fn at_ref_count_zero_only() {
            let mut pool = BlockPool::new(10, 256);
            let prefix_tokens: Vec<u32> = (0..256).collect();
            let h = BlockPool::compute_hash(&prefix_tokens, -1);

            // Allocate and hash the prefix.
            let mut seq_a = make_seq(prefix_tokens.clone());
            pool.allocate(&mut seq_a, 0).unwrap();
            let block_id = seq_a.block_table[0];
            pool.blocks[block_id].update(h, prefix_tokens.clone());
            pool.hash_to_block_id.insert(h, block_id);

            // Allocate seq_b with shared prefix.
            let mut tokens_b: Vec<u32> = prefix_tokens.clone();
            tokens_b.push(42);
            let mut seq_b = make_seq(tokens_b);
            let nc = pool.can_allocate(&seq_b).unwrap();
            pool.allocate(&mut seq_b, nc).unwrap();
            assert_eq!(pool.blocks[block_id].ref_count, 2);

            // Deallocate seq_a. ref_count should go 2→1, block still used.
            pool.deallocate(&mut seq_a).unwrap();
            assert_eq!(pool.blocks[block_id].ref_count, 1);
            assert!(pool.used_block_ids.contains(&block_id));

            // Deallocate seq_b. ref_count goes 1→0, block freed.
            pool.deallocate(&mut seq_b).unwrap();
            assert_eq!(pool.blocks[block_id].ref_count, 0);
            assert!(!pool.used_block_ids.contains(&block_id));
        }

        #[test]
        fn deallocating_twice_is_noop() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq(vec![1u32, 2, 3]);
            pool.allocate(&mut seq, 0).unwrap();
            pool.deallocate(&mut seq).unwrap();
            // nano-vllm clears block_table on deallocate; second call is a no-op.
            pool.deallocate(&mut seq).unwrap();
            assert!(seq.block_table.is_empty());
            assert_eq!(seq.num_cached_tokens, 0);
        }
    }

    mod cow {
        use super::*;

        #[test]
        fn shared_prefix_does_not_mutate_original() {
            let mut pool = BlockPool::new(10, 256);
            let prefix_tokens: Vec<u32> = (0..256).collect();
            let h = BlockPool::compute_hash(&prefix_tokens, -1);

            // Seq A: allocate and hash the prefix block.
            let mut seq_a = make_seq(prefix_tokens.clone());
            pool.allocate(&mut seq_a, 0).unwrap();
            let block_id = seq_a.block_table[0];
            pool.blocks[block_id].update(h, prefix_tokens.clone());
            pool.hash_to_block_id.insert(h, block_id);

            // Seq B: allocate with shared prefix + one more token.
            let mut tokens_b: Vec<u32> = prefix_tokens.clone();
            tokens_b.push(999);
            let mut seq_b = make_seq(tokens_b);
            let nc = pool.can_allocate(&seq_b).unwrap();
            assert_eq!(nc, 1);
            pool.allocate(&mut seq_b, nc).unwrap();

            // Seq B's exclusive block (index 1) has different content from
            // what any seq A block could reference.
            let b_block1_id = seq_b.block_table[1];
            assert_ne!(b_block1_id, block_id);
            // Seq A's block 0 should still have the original token_ids.
            assert_eq!(
                pool.blocks[block_id].token_ids, prefix_tokens,
                "shared prefix block must not be mutated by seq B's exclusive block"
            );
        }
    }

    mod can_append {
        use super::*;

        #[test]
        fn requires_free_only_at_block_boundary() {
            let pool = BlockPool::new(1, 256);
            let mut seq = make_seq((0..256).collect());
            // 256 tokens → 1 block. num_tokens % 256 == 0 → need_new_block=0 → can_append.
            assert!(pool.can_append(&seq));
            // 257 tokens: num_tokens % 256 == 1 → need_new_block=1 → needs 1 free.
            seq.append_token(42);
            assert_eq!(seq.num_tokens, 257);
            assert!(pool.can_append(&seq));
        }

        #[test]
        fn returns_false_when_new_block_needed_but_no_free() {
            // Pool with 1 block, fully used.
            let mut pool = BlockPool::new(1, 256);
            // Allocate the only block to a seq.
            let mut seq = make_seq(vec![1u32]);
            pool.allocate(&mut seq, 0).unwrap();
            // seq has 1 token, 1 block allocated. free=0.
            assert_eq!(pool.num_free_blocks(), 0);
            // num_tokens=1, 1%256=1 → needs new block, but free=0.
            assert!(!pool.can_append(&seq));
        }
    }

    mod may_append {
        use super::*;

        #[test]
        fn appends_only_at_boundary() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq((0..257).collect()); // 257 tokens → 2 blocks
                                                        // Allocate the 2 blocks.
            pool.allocate(&mut seq, 0).unwrap();
            assert_eq!(seq.block_table.len(), 2);
            let initial_free = pool.num_free_blocks();

            // num_tokens=257, 257%256=1 → may_append should add a block.
            pool.may_append(&mut seq).unwrap();
            assert_eq!(seq.block_table.len(), 3);
            assert_eq!(pool.num_free_blocks(), initial_free - 1);

            // num_tokens is still 257 (no append_token call).
            // After may_append added a block, calling again should be idempotent
            // since num_tokens hasn't changed.
            // 257 % 256 == 1 still → another block would be added.
            // So we need to actually append to change num_tokens.

            // Reset by deallocating and checking.
            pool.deallocate(&mut seq).unwrap();
        }

        #[test]
        fn does_not_append_away_from_boundary() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq((0..256).collect()); // 256 tokens → 1 block
            pool.allocate(&mut seq, 0).unwrap();
            assert_eq!(seq.block_table.len(), 1);
            // 256 % 256 == 0 → no append.
            let free_before = pool.num_free_blocks();
            pool.may_append(&mut seq).unwrap();
            assert_eq!(seq.block_table.len(), 1);
            assert_eq!(pool.num_free_blocks(), free_before);
        }

        #[test]
        fn may_append_idempotent_after_append_token() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq((0..256).collect());
            pool.allocate(&mut seq, 0).unwrap();
            assert_eq!(seq.block_table.len(), 1);

            // Append token → 257. 257%256=1 → should append.
            seq.append_token(42);
            assert_eq!(seq.num_tokens, 257);
            let free_before = pool.num_free_blocks();
            pool.may_append(&mut seq).unwrap();
            assert_eq!(seq.block_table.len(), 2);
            assert_eq!(pool.num_free_blocks(), free_before - 1);

            // Now num_tokens=258. 258%256=2 → no append.
            seq.append_token(43);
            let free_before2 = pool.num_free_blocks();
            pool.may_append(&mut seq).unwrap();
            assert_eq!(seq.block_table.len(), 2);
            assert_eq!(pool.num_free_blocks(), free_before2);
        }
    }

    mod hash_blocks {
        use super::*;

        fn setup_pool_and_seq() -> (BlockPool, Sequence) {
            let mut pool = BlockPool::new(20, 256);
            let tokens: Vec<u32> = (0..(4 * 256 + 50) as u32).collect();
            let mut seq = make_seq(tokens);
            pool.allocate(&mut seq, 0).unwrap();
            seq.num_cached_tokens = 0;
            seq.num_scheduled_tokens = 4 * 256; // 4 full blocks scheduled
            (pool, seq)
        }

        #[test]
        fn hashes_all_scheduled_blocks() {
            let (mut pool, mut seq) = setup_pool_and_seq();
            assert!(pool.hash_to_block_id.is_empty());

            pool.hash_blocks(&mut seq);

            // 4 blocks should have been hashed.
            for i in 0..4 {
                let block_id = seq.block_table[i];
                assert_ne!(pool.blocks[block_id].hash, -1, "block {i} should be hashed");
            }
            assert_eq!(pool.hash_to_block_id.len(), 4);
        }

        #[test]
        fn incremental_hashing_consistent() {
            let tokens: Vec<u32> = (0..(4 * 256 + 50) as u32).collect();

            // Pass 1: hash blocks 0..2
            let mut pool1 = BlockPool::new(20, 256);
            let mut seq1 = make_seq(tokens.clone());
            pool1.allocate(&mut seq1, 0).unwrap();
            seq1.num_cached_tokens = 0;
            seq1.num_scheduled_tokens = 2 * 256;
            pool1.hash_blocks(&mut seq1);

            // Pass 2: hash blocks 2..4
            seq1.num_cached_tokens = 2 * 256;
            seq1.num_scheduled_tokens = 2 * 256;
            pool1.hash_blocks(&mut seq1);

            // Pass 3: hash all 4 at once
            let mut pool2 = BlockPool::new(20, 256);
            let mut seq2 = make_seq(tokens);
            pool2.allocate(&mut seq2, 0).unwrap();
            seq2.num_cached_tokens = 0;
            seq2.num_scheduled_tokens = 4 * 256;
            pool2.hash_blocks(&mut seq2);

            // The hashes should be identical.
            for i in 0..4 {
                let h1 = pool1.blocks[seq1.block_table[i]].hash;
                let h2 = pool2.blocks[seq2.block_table[i]].hash;
                assert_eq!(
                    h1, h2,
                    "block {i} hash mismatch between incremental and one-pass"
                );
            }
        }

        #[test]
        fn idempotent() {
            let tokens: Vec<u32> = (0..(3 * 256) as u32).collect();
            let mut pool = BlockPool::new(20, 256);
            let mut seq = make_seq(tokens);
            pool.allocate(&mut seq, 0).unwrap();
            seq.num_cached_tokens = 0;
            seq.num_scheduled_tokens = 3 * 256;

            pool.hash_blocks(&mut seq);
            let state_before = pool.hash_to_block_id.clone();
            let hashes_before: Vec<i64> = (0..3)
                .map(|i| pool.blocks[seq.block_table[i]].hash)
                .collect();

            // Hash again with same state.
            pool.hash_blocks(&mut seq);

            assert_eq!(pool.hash_to_block_id, state_before);
            for i in 0..3 {
                assert_eq!(
                    pool.blocks[seq.block_table[i]].hash, hashes_before[i],
                    "block {i} hash changed on re-hash"
                );
            }
        }

        #[test]
        fn noop_when_no_scheduled_tokens() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq(vec![1u32, 2, 3]);
            pool.allocate(&mut seq, 0).unwrap();
            seq.num_cached_tokens = 0;
            seq.num_scheduled_tokens = 0;

            let state_before = pool.hash_to_block_id.clone();
            pool.hash_blocks(&mut seq);
            assert_eq!(
                pool.hash_to_block_id, state_before,
                "hash_blocks with 0 scheduled tokens must be a no-op"
            );
        }
    }

    mod num_free_blocks {
        use super::*;

        #[test]
        fn returns_total_at_construction() {
            let pool = BlockPool::new(42, 256);
            assert_eq!(pool.num_free_blocks(), 42);
        }

        #[test]
        fn decreases_on_allocate_increases_on_deallocate() {
            let mut pool = BlockPool::new(10, 256);
            let mut seq = make_seq((0..257).collect());
            assert_eq!(pool.num_free_blocks(), 10);
            pool.allocate(&mut seq, 0).unwrap();
            // 257 tokens → 2 blocks.
            assert_eq!(pool.num_free_blocks(), 8);
            pool.deallocate(&mut seq).unwrap();
            assert_eq!(pool.num_free_blocks(), 10);
        }
    }
}
