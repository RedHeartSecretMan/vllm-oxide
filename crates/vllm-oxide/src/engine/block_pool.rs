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
use std::sync::Arc;

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
/// of sequences sharing this block (CoW prefix-cache semantics). The private
/// identity disambiguates collisions across the complete prefix chain.
#[derive(Debug, Clone)]
pub struct Block {
    pub(crate) block_id: usize,
    pub(crate) ref_count: usize,
    pub(crate) hash: i64,
    pub(crate) token_ids: Vec<u32>,
    prefix_identity: Option<Arc<PrefixCacheIdentity>>,
}

/// Exact collision-safe identity for one cached logical prefix block.
///
/// Nodes share their parent through `Arc`, so retaining the complete token
/// chain costs one block of token ids per unique node rather than copying the
/// full prefix into every hashtable entry.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PrefixCacheIdentity {
    parent: Option<Arc<Self>>,
    token_ids: Vec<u32>,
}

impl PrefixCacheIdentity {
    fn extend(parent: Option<Arc<Self>>, token_ids: Vec<u32>) -> Arc<Self> {
        Arc::new(Self { parent, token_ids })
    }
}

impl Block {
    pub(crate) fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: -1,
            token_ids: Vec::new(),
            prefix_identity: None,
        }
    }

    /// Set the hash, current-block tokens, and complete chained identity.
    fn update(
        &mut self,
        hash: i64,
        token_ids: Vec<u32>,
        prefix_identity: Arc<PrefixCacheIdentity>,
    ) {
        self.hash = hash;
        self.token_ids = token_ids;
        self.prefix_identity = Some(prefix_identity);
    }

    /// Reset to allocated-but-empty state (ref_count = 1, hash = -1).
    ///
    /// Called by `_allocate_block` after popping from the free list.
    /// nano-vllm asserts `ref_count == 0` before this call.
    pub(crate) fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = -1;
        self.token_ids = Vec::new();
        self.prefix_identity = None;
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
    pub(crate) hash_to_block_id: HashMap<i64, Vec<usize>>,
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

    fn validate_free_block(&self, block_id: usize) -> Result<(), BlockPoolError> {
        let block = self
            .blocks
            .get(block_id)
            .ok_or_else(|| BlockPoolError(format!("free block id {block_id} is out of range")))?;
        if block.ref_count != 0 || self.used_block_ids.contains(&block_id) {
            return Err(BlockPoolError(format!(
                "free block {block_id} has inconsistent ownership"
            )));
        }
        Ok(())
    }

    fn cached_block_id(&self, hash: i64, identity: &PrefixCacheIdentity) -> Option<usize> {
        self.hash_to_block_id
            .get(&hash)?
            .iter()
            .copied()
            .find(|&block_id| {
                self.blocks.get(block_id).is_some_and(|block| {
                    block.hash == hash
                        && block.prefix_identity.as_deref() == Some(identity)
                        && block.token_ids == identity.token_ids
                })
            })
    }

    fn remove_cache_entry(&mut self, hash: i64, block_id: usize) {
        let mut remove_bucket = false;
        if let Some(block_ids) = self.hash_to_block_id.get_mut(&hash) {
            block_ids.retain(|&cached_id| cached_id != block_id);
            remove_bucket = block_ids.is_empty();
        }
        if remove_bucket {
            self.hash_to_block_id.remove(&hash);
        }
    }

    fn index_cache_entry(&mut self, hash: i64, block_id: usize) {
        let block_ids = self.hash_to_block_id.entry(hash).or_default();
        block_ids.retain(|&cached_id| cached_id != block_id);
        block_ids.push(block_id);
    }

    /// Allocate a free block and return its id.
    ///
    /// Equivalent to nano-vllm's `_allocate_block`: pops from the front of
    /// the free deque, cleans the old hash entry if present, resets the
    /// block, and adds it to the used set.
    ///
    /// # Errors
    ///
    /// Returns `BlockPoolError` without mutating the pool if the free-list head
    /// is missing or violates the pool's ownership invariants.
    fn allocate_block_private(&mut self) -> Result<usize, BlockPoolError> {
        let block_id = *self
            .free_block_ids
            .front()
            .ok_or_else(|| BlockPoolError("no free blocks available".to_string()))?;
        self.validate_free_block(block_id)?;
        let old_hash = self.blocks[block_id].hash;
        self.free_block_ids.pop_front();
        // nano-vllm asserts ref_count == 0 here.
        if old_hash != -1 {
            self.remove_cache_entry(old_hash, block_id);
        }
        self.blocks[block_id].reset();
        self.used_block_ids.insert(block_id);
        Ok(block_id)
    }

    /// Check whether a sequence can be allocated and, if so, how many
    /// blocks are cache-hits.
    ///
    /// Mirrors nano-vllm's `BlockManager.can_allocate`. Returns
    /// `Some(num_cached_blocks)` when there is room, `None` when
    /// insufficient free blocks.
    ///
    /// Walks `seq.num_blocks() - 1` blocks. The final logical block is never
    /// reused, even at an exact block boundary, so non-empty requests retain
    /// new model input from which sampling can obtain a hidden state. For each
    /// reusable block, computes the chained hash and requires
    /// a complete chained token-identity match within its hash bucket. Counts
    /// how many new blocks would be needed (fewer when a cached block is
    /// already in `used_block_ids` — shared, counts against free only if not
    /// currently used).
    pub fn can_allocate(&self, seq: &Sequence) -> Option<usize> {
        if seq.num_blocks() == 0 {
            return Some(0);
        }
        let mut h: i64 = -1;
        let mut identity = None;
        let mut num_cached_blocks: usize = 0;
        let num_blocks = seq.num_blocks();
        // Recompute the final logical block even when it is exactly full so a
        // non-empty request never becomes a zero-token sampling plan.
        let check_until = if num_blocks > 0 { num_blocks - 1 } else { 0 };

        let mut num_new_blocks = num_blocks;
        for i in 0..check_until {
            let token_ids = seq.block(i);
            h = Self::compute_hash(token_ids, h);
            let next_identity = PrefixCacheIdentity::extend(identity, token_ids.to_vec());
            match self.cached_block_id(h, &next_identity) {
                Some(block_id) => {
                    num_cached_blocks += 1;
                    if self.used_block_ids.contains(&block_id) {
                        num_new_blocks -= 1;
                    }
                    identity = Some(next_identity);
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
    /// Validates the complete allocation before changing ownership. Any
    /// `BlockPoolError` therefore leaves both the pool and `Sequence` unchanged.
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
        if num_cached_blocks > seq.num_blocks() {
            return Err(BlockPoolError(format!(
                "cached block count {num_cached_blocks} exceeds sequence block count {}",
                seq.num_blocks()
            )));
        }
        let mut h: i64 = -1;
        let mut identity = None;
        let mut cached_block_ids = Vec::with_capacity(num_cached_blocks);
        let mut distinct_cached_block_ids = HashSet::with_capacity(num_cached_blocks);
        for i in 0..num_cached_blocks {
            let token_ids = seq.block(i);
            h = Self::compute_hash(token_ids, h);
            let next_identity = PrefixCacheIdentity::extend(identity, token_ids.to_vec());
            let block_id = self
                .cached_block_id(h, &next_identity)
                .ok_or_else(|| BlockPoolError("cached block hash not found".to_string()))?;
            let block = self.blocks.get(block_id).ok_or_else(|| {
                BlockPoolError(format!("cached block {block_id} is out of range"))
            })?;
            if block.token_ids != token_ids {
                return Err(BlockPoolError(format!(
                    "cached block {block_id} token ids do not match"
                )));
            }
            if !distinct_cached_block_ids.insert(block_id) {
                return Err(BlockPoolError(format!(
                    "cached block {block_id} appears more than once"
                )));
            }
            if self.used_block_ids.contains(&block_id) {
                if block.ref_count == 0 {
                    return Err(BlockPoolError(format!(
                        "used cached block {block_id} has ref_count 0"
                    )));
                }
                block.ref_count.checked_add(1).ok_or_else(|| {
                    BlockPoolError(format!("cached block {block_id} ref_count overflow"))
                })?;
            } else if block.ref_count != 0 || !self.free_block_ids.iter().any(|&id| id == block_id)
            {
                return Err(BlockPoolError(format!(
                    "cached block {block_id} has inconsistent free-list ownership"
                )));
            }
            cached_block_ids.push(block_id);
            identity = Some(next_identity);
        }

        let required_uncached_blocks = seq.num_blocks() - num_cached_blocks;
        let cached_free_blocks = cached_block_ids
            .iter()
            .filter(|block_id| !self.used_block_ids.contains(block_id))
            .count();
        let required_free_blocks = required_uncached_blocks
            .checked_add(cached_free_blocks)
            .ok_or_else(|| BlockPoolError("required free block count overflow".to_string()))?;
        if self.free_block_ids.len() < required_free_blocks {
            return Err(BlockPoolError("no free blocks available".to_string()));
        }
        let new_block_ids = self
            .free_block_ids
            .iter()
            .filter(|block_id| !distinct_cached_block_ids.contains(block_id))
            .take(required_uncached_blocks)
            .copied()
            .collect::<Vec<_>>();
        if new_block_ids.len() != required_uncached_blocks {
            return Err(BlockPoolError("no free blocks available".to_string()));
        }
        let mut distinct_new_block_ids = HashSet::with_capacity(new_block_ids.len());
        for &block_id in &new_block_ids {
            if !distinct_new_block_ids.insert(block_id) {
                return Err(BlockPoolError(format!(
                    "free block {block_id} has inconsistent ownership"
                )));
            }
            self.validate_free_block(block_id)?;
        }

        for block_id in cached_block_ids {
            let block = &mut self.blocks[block_id];
            if self.used_block_ids.contains(&block_id) {
                block.ref_count += 1;
            } else {
                // Block is in hash_to_block_id but not used — a previously
                // hashed-then-deallocated block; move from free to used.
                self.free_block_ids.retain(|&free_id| free_id != block_id);
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
        self.deallocate_batch(std::slice::from_mut(seq))?;
        Ok(())
    }

    /// Deallocate a set of sequences as one ownership transaction.
    ///
    /// Every block id, per-sequence table, aggregate refcount decrement, and
    /// free/used ownership relation is validated before the first mutation.
    pub(crate) fn deallocate_batch(
        &mut self,
        sequences: &mut [Sequence],
    ) -> Result<(), BlockPoolError> {
        let mut release_counts = HashMap::<usize, usize>::new();
        let mut release_order = Vec::new();
        for sequence in sequences.iter() {
            let mut sequence_blocks = HashSet::with_capacity(sequence.block_table.len());
            for &block_id in sequence.block_table.iter().rev() {
                if !sequence_blocks.insert(block_id) {
                    return Err(BlockPoolError(format!(
                        "block {block_id} appears more than once in sequence {}",
                        sequence.seq_id
                    )));
                }
                if !release_counts.contains_key(&block_id) {
                    release_order.push(block_id);
                }
                let count = release_counts.entry(block_id).or_default();
                *count = count.checked_add(1).ok_or_else(|| {
                    BlockPoolError(format!("block {block_id} release count overflow"))
                })?;
            }
        }

        for (&block_id, &release_count) in &release_counts {
            let block = self.blocks.get(block_id).ok_or_else(|| {
                BlockPoolError(format!(
                    "block {block_id} is out of range during deallocation"
                ))
            })?;
            if block.ref_count == 0 {
                return Err(BlockPoolError(format!(
                    "block {block_id} already has ref_count 0 (double-free?)"
                )));
            }
            if block.ref_count < release_count {
                return Err(BlockPoolError(format!(
                    "block {block_id} ref_count {} is smaller than release count {release_count}",
                    block.ref_count
                )));
            }
            if !self.used_block_ids.contains(&block_id)
                || self
                    .free_block_ids
                    .iter()
                    .any(|&free_id| free_id == block_id)
            {
                return Err(BlockPoolError(format!(
                    "block {block_id} has inconsistent deallocation ownership"
                )));
            }
        }

        for block_id in release_order {
            let release_count = release_counts[&block_id];
            let block = &mut self.blocks[block_id];
            block.ref_count -= release_count;
            if block.ref_count == 0 {
                self.used_block_ids.remove(&block_id);
                self.free_block_ids.push_back(block_id);
            }
        }
        for sequence in sequences {
            sequence.num_cached_tokens = 0;
            sequence.num_scheduled_tokens = 0;
            sequence.block_table.clear();
        }
        Ok(())
    }

    /// Check whether the pool has enough room for one sequence's next append.
    pub fn can_append(&self, sequence: &Sequence) -> bool {
        self.can_append_batch(&[sequence])
    }

    /// Check whether the pool can reserve every missing block for a decode batch.
    pub(crate) fn can_append_batch(&self, sequences: &[&Sequence]) -> bool {
        let mut required_blocks = 0usize;
        for sequence in sequences {
            let Some(missing_blocks) = sequence
                .num_blocks()
                .checked_sub(sequence.block_table.len())
            else {
                return false;
            };
            if missing_blocks > 1 {
                return false;
            }
            let Some(total) = required_blocks.checked_add(missing_blocks) else {
                return false;
            };
            required_blocks = total;
        }
        self.free_block_ids.len() >= required_blocks
    }

    /// Allocate a block for one sequence's next append when needed.
    pub fn may_append(&mut self, sequence: &mut Sequence) -> Result<(), BlockPoolError> {
        self.may_append_batch(std::slice::from_mut(sequence))
    }

    /// Allocate any blocks needed by a decode batch as one transaction.
    ///
    /// Every required free block is validated before ownership changes. An
    /// error therefore leaves both the pool and every sequence unchanged.
    pub(crate) fn may_append_batch(
        &mut self,
        sequences: &mut [Sequence],
    ) -> Result<(), BlockPoolError> {
        let mut append_indices = Vec::new();
        for (index, sequence) in sequences.iter().enumerate() {
            let missing_blocks = sequence
                .num_blocks()
                .checked_sub(sequence.block_table.len())
                .ok_or_else(|| {
                    BlockPoolError(format!(
                        "sequence {} has more allocated than required blocks",
                        sequence.seq_id
                    ))
                })?;
            if missing_blocks > 1 {
                return Err(BlockPoolError(format!(
                    "sequence {} is missing {missing_blocks} blocks before append",
                    sequence.seq_id
                )));
            }
            if missing_blocks == 1 {
                append_indices.push(index);
            }
        }

        if self.free_block_ids.len() < append_indices.len() {
            return Err(BlockPoolError("no free blocks available".to_string()));
        }
        let candidate_block_ids = self
            .free_block_ids
            .iter()
            .take(append_indices.len())
            .copied()
            .collect::<Vec<_>>();
        let mut distinct_block_ids = HashSet::with_capacity(candidate_block_ids.len());
        for block_id in candidate_block_ids {
            if !distinct_block_ids.insert(block_id) {
                return Err(BlockPoolError(format!(
                    "free block {block_id} has inconsistent ownership"
                )));
            }
            self.validate_free_block(block_id)?;
        }

        for sequence_index in append_indices {
            let block_id = self.allocate_block_private()?;
            sequences[sequence_index].block_table.push(block_id);
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
        let (mut h, mut identity) = if start > 0 {
            let prefix = &self.blocks[seq.block_table[start - 1]];
            (prefix.hash, prefix.prefix_identity.clone())
        } else {
            (-1, None)
        };

        for i in start..end {
            let block_id = seq.block_table[i];
            let token_ids = seq.block(i).to_vec();
            h = Self::compute_hash(&token_ids, h);
            let next_identity = PrefixCacheIdentity::extend(identity, token_ids.clone());
            let old_hash = self.blocks[block_id].hash;
            if old_hash != -1 && old_hash != h {
                self.remove_cache_entry(old_hash, block_id);
            }
            let block = &mut self.blocks[block_id];
            block.update(h, token_ids, next_identity.clone());
            self.index_cache_entry(h, block_id);
            identity = Some(next_identity);
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

    fn index_test_prefix(
        pool: &mut BlockPool,
        block_id: usize,
        hash: i64,
        prefix_blocks: &[Vec<u32>],
    ) {
        let mut identity = None;
        for token_ids in prefix_blocks {
            identity = Some(PrefixCacheIdentity::extend(identity, token_ids.clone()));
        }
        let current_tokens = prefix_blocks.last().unwrap().clone();
        pool.blocks[block_id].update(hash, current_tokens, identity.unwrap());
        pool.index_cache_entry(hash, block_id);
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
            index_test_prefix(&mut pool, b1, h1, std::slice::from_ref(&tokens_b1));
            index_test_prefix(&mut pool, b2, h2, &[tokens_b1, tokens_b2]);

            // Sequence with 3 full blocks. Should find 2 cached, need 1 new.
            let tokens_seq: Vec<u32> = (0..(3 * 256) as u32).collect();
            let seq = make_seq(tokens_seq);
            let result = pool.can_allocate(&seq);
            assert_eq!(result, Some(2));
        }

        #[test]
        fn hash_collision_with_same_block_tokens_does_not_cross_prefix_chains() {
            let mut pool = BlockPool::new(12, 256);
            let first_prefix = (0..256).collect::<Vec<u32>>();
            let different_prefix = (2_000..2_256).collect::<Vec<u32>>();
            let shared_second_block = (1_000..1_256).collect::<Vec<u32>>();

            let mut correct_tokens = first_prefix.clone();
            correct_tokens.extend_from_slice(&shared_second_block);
            correct_tokens.push(9);
            let mut correct_owner = make_seq(correct_tokens.clone());
            pool.allocate(&mut correct_owner, 0).unwrap();
            correct_owner.num_scheduled_tokens = 2 * 256;
            pool.hash_blocks(&mut correct_owner);
            let colliding_hash = pool.blocks[correct_owner.block_table[1]].hash;

            let mut wrong_tokens = different_prefix.clone();
            wrong_tokens.extend_from_slice(&shared_second_block);
            wrong_tokens.push(10);
            let mut wrong_owner = make_seq(wrong_tokens);
            pool.allocate(&mut wrong_owner, 0).unwrap();
            let wrong_second_block = wrong_owner.block_table[1];
            pool.hash_to_block_id.remove(&colliding_hash);
            index_test_prefix(
                &mut pool,
                wrong_second_block,
                colliding_hash,
                &[different_prefix, shared_second_block],
            );

            let target = make_seq(correct_tokens);

            assert_eq!(
                pool.can_allocate(&target),
                Some(1),
                "a matching hash and current block are insufficient without the complete prefix chain"
            );
        }

        #[test]
        fn hash_bucket_keeps_colliding_candidates_and_selects_the_exact_chain() {
            let mut pool = BlockPool::new(12, 256);
            let first_prefix = (0..256).collect::<Vec<u32>>();
            let different_prefix = (2_000..2_256).collect::<Vec<u32>>();
            let shared_second_block = (1_000..1_256).collect::<Vec<u32>>();

            let mut correct_tokens = first_prefix;
            correct_tokens.extend_from_slice(&shared_second_block);
            correct_tokens.push(9);
            let mut correct_owner = make_seq(correct_tokens.clone());
            pool.allocate(&mut correct_owner, 0).unwrap();
            correct_owner.num_scheduled_tokens = 2 * 256;
            pool.hash_blocks(&mut correct_owner);
            let correct_second_block = correct_owner.block_table[1];
            let colliding_hash = pool.blocks[correct_second_block].hash;

            let mut wrong_tokens = different_prefix.clone();
            wrong_tokens.extend_from_slice(&shared_second_block);
            wrong_tokens.push(10);
            let mut wrong_owner = make_seq(wrong_tokens);
            pool.allocate(&mut wrong_owner, 0).unwrap();
            let wrong_second_block = wrong_owner.block_table[1];
            index_test_prefix(
                &mut pool,
                wrong_second_block,
                colliding_hash,
                &[different_prefix, shared_second_block],
            );
            assert_eq!(pool.hash_to_block_id[&colliding_hash].len(), 2);

            let mut target = make_seq(correct_tokens);
            let cached = pool.can_allocate(&target).unwrap();
            assert_eq!(cached, 2);
            pool.allocate(&mut target, cached).unwrap();

            assert_eq!(target.block_table[1], correct_second_block);
            assert_ne!(target.block_table[1], wrong_second_block);
            assert_eq!(pool.blocks[correct_second_block].ref_count, 2);
            assert_eq!(pool.blocks[wrong_second_block].ref_count, 1);
        }
    }

    mod allocate {
        use super::*;

        #[test]
        fn capacity_failure_leaves_pool_and_sequence_unchanged() {
            let mut pool = BlockPool::new(1, 256);
            let mut seq = make_seq((0..257).collect());

            let error = pool.allocate(&mut seq, 0).unwrap_err();

            assert_eq!(
                error,
                BlockPoolError("no free blocks available".to_string())
            );
            assert_eq!(pool.num_free_blocks(), 1);
            assert!(pool.used_block_ids.is_empty());
            assert_eq!(pool.blocks[0].ref_count, 0);
            assert!(seq.block_table.is_empty());
            assert_eq!(seq.num_cached_tokens, 0);
        }

        #[test]
        fn cached_lookup_failure_rolls_back_shared_ownership() {
            let mut pool = BlockPool::new(4, 256);
            let prefix = (0..256).collect::<Vec<u32>>();
            let mut cached_owner = make_seq(prefix.clone());
            pool.allocate(&mut cached_owner, 0).unwrap();
            let cached_block = cached_owner.block_table[0];
            let prefix_hash = BlockPool::compute_hash(&prefix, -1);
            index_test_prefix(
                &mut pool,
                cached_block,
                prefix_hash,
                std::slice::from_ref(&prefix),
            );

            let mut target = make_seq((0..513).collect());
            let free_before = pool.num_free_blocks();
            let error = pool.allocate(&mut target, 2).unwrap_err();

            assert_eq!(
                error,
                BlockPoolError("cached block hash not found".to_string())
            );
            assert_eq!(pool.num_free_blocks(), free_before);
            assert_eq!(pool.blocks[cached_block].ref_count, 1);
            assert!(target.block_table.is_empty());
            assert_eq!(target.num_cached_tokens, 0);
        }

        #[test]
        fn invalid_later_free_block_does_not_partially_allocate() {
            let mut pool = BlockPool::new(3, 256);
            pool.free_block_ids = VecDeque::from([0, usize::MAX, 2]);
            let mut seq = make_seq((0..257).collect());

            let error = pool.allocate(&mut seq, 0).unwrap_err();

            assert_eq!(
                error,
                BlockPoolError(format!("free block id {} is out of range", usize::MAX))
            );
            assert_eq!(pool.free_block_ids, VecDeque::from([0, usize::MAX, 2]));
            assert!(pool.used_block_ids.is_empty());
            assert!(pool.blocks.iter().all(|block| block.ref_count == 0));
            assert!(seq.block_table.is_empty());
            assert_eq!(seq.num_cached_tokens, 0);
        }

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
            index_test_prefix(&mut pool, block_id, h, std::slice::from_ref(&prefix_tokens));
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
        fn validation_failure_does_not_partially_release_later_blocks() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq((0..257).collect());
            pool.allocate(&mut seq, 0).unwrap();
            let first_block = seq.block_table[0];
            pool.blocks[first_block].ref_count = 0;
            let free_before = pool.free_block_ids.clone();
            let used_before = pool.used_block_ids.clone();
            let ref_counts_before = pool
                .blocks
                .iter()
                .map(|block| block.ref_count)
                .collect::<Vec<_>>();
            let block_table_before = seq.block_table.clone();

            let error = pool.deallocate(&mut seq).unwrap_err();

            assert_eq!(
                error,
                BlockPoolError(format!(
                    "block {first_block} already has ref_count 0 (double-free?)"
                ))
            );
            assert_eq!(pool.free_block_ids, free_before);
            assert_eq!(pool.used_block_ids, used_before);
            assert_eq!(
                pool.blocks
                    .iter()
                    .map(|block| block.ref_count)
                    .collect::<Vec<_>>(),
                ref_counts_before
            );
            assert_eq!(seq.block_table, block_table_before);
            assert_eq!(seq.num_cached_tokens, 0);
        }

        #[test]
        fn batch_validation_failure_preserves_shared_and_private_refcounts() {
            let mut pool = BlockPool::new(8, 256);
            let mut first_tokens = (0..256).collect::<Vec<u32>>();
            first_tokens.push(9);
            let mut first = make_seq(first_tokens.clone());
            pool.allocate(&mut first, 0).unwrap();
            first.num_scheduled_tokens = 256;
            pool.hash_blocks(&mut first);

            let mut second = make_seq(first_tokens);
            let cached = pool.can_allocate(&second).unwrap();
            assert_eq!(cached, 1);
            pool.allocate(&mut second, cached).unwrap();
            let shared_block = first.block_table[0];
            let invalid_private_block = second.block_table[1];
            assert_eq!(pool.blocks[shared_block].ref_count, 2);
            pool.blocks[invalid_private_block].ref_count = 0;
            let free_before = pool.free_block_ids.clone();
            let used_before = pool.used_block_ids.clone();
            let ref_counts_before = pool
                .blocks
                .iter()
                .map(|block| block.ref_count)
                .collect::<Vec<_>>();
            let mut sequences = [first, second];
            let tables_before = sequences
                .iter()
                .map(|sequence| sequence.block_table.clone())
                .collect::<Vec<_>>();

            let error = pool.deallocate_batch(&mut sequences).unwrap_err();

            assert_eq!(
                error,
                BlockPoolError(format!(
                    "block {invalid_private_block} already has ref_count 0 (double-free?)"
                ))
            );
            assert_eq!(pool.free_block_ids, free_before);
            assert_eq!(pool.used_block_ids, used_before);
            assert_eq!(pool.blocks[shared_block].ref_count, 2);
            assert_eq!(
                pool.blocks
                    .iter()
                    .map(|block| block.ref_count)
                    .collect::<Vec<_>>(),
                ref_counts_before
            );
            assert_eq!(
                sequences
                    .iter()
                    .map(|sequence| sequence.block_table.clone())
                    .collect::<Vec<_>>(),
                tables_before
            );
        }

        #[test]
        fn batch_release_decrements_a_shared_prefix_once_per_owner() {
            let mut pool = BlockPool::new(8, 256);
            let mut first_tokens = (0..256).collect::<Vec<u32>>();
            first_tokens.push(9);
            let mut first = make_seq(first_tokens.clone());
            pool.allocate(&mut first, 0).unwrap();
            first.num_scheduled_tokens = 256;
            pool.hash_blocks(&mut first);

            let mut second = make_seq(first_tokens);
            let cached = pool.can_allocate(&second).unwrap();
            pool.allocate(&mut second, cached).unwrap();
            let shared_block = first.block_table[0];
            assert_eq!(pool.blocks[shared_block].ref_count, 2);
            let mut sequences = [first, second];

            pool.deallocate_batch(&mut sequences).unwrap();

            assert_eq!(pool.num_free_blocks(), 8);
            assert_eq!(pool.blocks[shared_block].ref_count, 0);
            assert_eq!(
                pool.free_block_ids
                    .iter()
                    .filter(|&&block_id| block_id == shared_block)
                    .count(),
                1
            );
            assert!(sequences
                .iter()
                .all(|sequence| sequence.block_table.is_empty()));
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
            index_test_prefix(&mut pool, block_id, h, std::slice::from_ref(&prefix_tokens));

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
            index_test_prefix(&mut pool, block_id, h, std::slice::from_ref(&prefix_tokens));

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
            let mut pool = BlockPool::new(2, 256);
            let mut seq = make_seq((0..256).collect());
            pool.allocate(&mut seq, 0).unwrap();
            // The prompt's one required block is already reserved.
            assert!(pool.can_append(&seq));
            // Appending token 257 requires the one remaining free block.
            seq.append_token(42);
            assert_eq!(seq.num_tokens, 257);
            assert!(pool.can_append(&seq));
        }

        #[test]
        fn returns_false_when_new_block_needed_but_no_free() {
            // Pool with 1 block, fully used.
            let mut pool = BlockPool::new(1, 256);
            // Allocate the only block to a seq.
            let mut seq = make_seq((0..256).collect());
            pool.allocate(&mut seq, 0).unwrap();
            seq.append_token(42);
            assert_eq!(pool.num_free_blocks(), 0);
            // Token 257 needs a second block, but none is free.
            assert!(!pool.can_append(&seq));
        }
    }

    mod may_append {
        use super::*;

        #[test]
        fn later_failure_leaves_every_sequence_and_block_unmodified() {
            let mut pool = BlockPool::new(4, 256);
            let mut first = make_seq((0..256).collect());
            let mut second = make_seq((1_000..1_256).collect());
            pool.allocate(&mut first, 0).unwrap();
            pool.allocate(&mut second, 0).unwrap();
            first.append_token(42);
            second.append_token(43);
            let first_blocks = first.block_table.clone();
            let second_blocks = second.block_table.clone();
            pool.free_block_ids = VecDeque::from([2, usize::MAX]);
            let mut sequences = [first, second];

            let error = pool.may_append_batch(&mut sequences).unwrap_err();

            assert_eq!(
                error,
                BlockPoolError(format!("free block id {} is out of range", usize::MAX))
            );
            assert_eq!(sequences[0].block_table, first_blocks);
            assert_eq!(sequences[1].block_table, second_blocks);
            assert_eq!(pool.free_block_ids, VecDeque::from([2, usize::MAX]));
            assert_eq!(pool.used_block_ids.len(), 2);
        }

        #[test]
        fn does_not_duplicate_an_already_reserved_boundary_block() {
            let mut pool = BlockPool::new(5, 256);
            let mut seq = make_seq((0..257).collect()); // 257 tokens → 2 blocks
                                                        // Allocate the 2 blocks.
            pool.allocate(&mut seq, 0).unwrap();
            assert_eq!(seq.block_table.len(), 2);
            let initial_free = pool.num_free_blocks();

            // Both blocks are already reserved, so the batch transaction is a no-op.
            pool.may_append(&mut seq).unwrap();
            assert_eq!(seq.block_table.len(), 2);
            assert_eq!(pool.num_free_blocks(), initial_free);
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
