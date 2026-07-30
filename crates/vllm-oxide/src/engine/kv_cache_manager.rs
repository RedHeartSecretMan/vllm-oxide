//! Scheduler-facing seam — `KvCacheManager` (V1's `kv_cache_manager.py` role).
//!
//! The **only** scheduler-facing seam over `BlockPool` + physical
//! `PagedKVCache`. Scheduler code never imports `BlockPool` or
//! `attention::PagedKVCache` directly. Owns the mapping from logical
//! block tables to physical block ids and to paged-cache slot indices.
//!
//! # Design: deliberate structural adapter
//!
//! This module is **not** deep (see ADR-0004 M3). The value is
//! information-hiding, not behavioural abstraction: 6 of its public
//! methods (`can_allocate`, `allocate`, `deallocate`, `can_append`,
//! `may_append`, `hash_blocks`) are one-line delegations to
//! `BlockPool`; `num_free_blocks` and `block_size` are trivial
//! accessors. `compute_slot_mapping` is the sole logic-carrying method
//! (~20 LOC: logical block-table index → physical slot via
//! `block_id * block_size + intra_offset`, `-1` sentinel for
//! out-of-range).
//!
//! The deletion test: compute vanishes (it is all pass-through), but
//! every direct `BlockPool` import in `scheduler.rs` reappears — the
//! module earns its keep on what it hides, not what it computes.
//! Future readers: do **not** deepen by moving scheduler logic in,
//! and do **not** delete it for being shallow.
//!
//! # V1 three-layer split (ADR-0004)
//!
//! KvCacheManager is the middle seam: the Scheduler (#21) calls
//! `mgr.can_allocate()`, `mgr.allocate()`, etc., and the public API is
//! the only place that names `BlockPool` or `PagedKVCache` in signatures
//! (`PagedKVCache` is unavoidable in the constructor; `BlockPool` is
//! entirely hidden).

use std::sync::{Arc, Mutex};

use crate::attention::PagedKVCache;
use crate::engine::block_pool::{BlockPool, BlockPoolError};
use crate::engine::sequence::Sequence;

/// Scheduler-facing seam over the block pool and physical paged KV cache.
///
/// Constructed by `EngineCore` (#21) with the shared `PagedKVCache` and
/// the pool size. All block-allocation logic is forwarded to the inner
/// `BlockPool`; `compute_slot_mapping` is the bridge from logical
/// block tables to cache slot indices.
///
/// **Adapter, not computational module.** The value of this module is
/// what the Scheduler cannot see — `BlockPool`, `BlockPoolError`,
/// physical `PagedKVCache` internals — not behavioural depth. Six of
/// its methods are one-line delegations by design;
/// `compute_slot_mapping` is the sole behavioural bridge (logical
/// block table → physical slot indices). Thinness is the design, not
/// debt.
pub struct KvCacheManager {
    pub(crate) block_pool: BlockPool,
    paged_kv: Arc<Mutex<PagedKVCache>>,
    block_size: usize,
}

impl KvCacheManager {
    /// Construct a new `KvCacheManager` with the given pool size and
    /// shared paged KV cache.
    pub fn new(num_blocks: usize, block_size: usize, paged_kv: Arc<Mutex<PagedKVCache>>) -> Self {
        let block_pool = BlockPool::new(num_blocks, block_size);
        Self {
            block_pool,
            paged_kv,
            block_size,
        }
    }

    /// Forwarded: check whether a sequence can be allocated.
    pub fn can_allocate(&self, seq: &Sequence) -> Option<usize> {
        self.block_pool.can_allocate(seq)
    }

    /// Forwarded: allocate blocks for a sequence.
    pub fn allocate(
        &mut self,
        seq: &mut Sequence,
        num_cached_blocks: usize,
    ) -> Result<(), BlockPoolError> {
        self.block_pool.allocate(seq, num_cached_blocks)
    }

    /// Forwarded: deallocate all blocks owned by a sequence.
    pub fn deallocate(&mut self, seq: &mut Sequence) -> Result<(), BlockPoolError> {
        self.block_pool.deallocate(seq)
    }

    /// Forwarded: check whether the pool has room for a decode append.
    pub fn can_append(&self, seq: &Sequence) -> bool {
        self.block_pool.can_append(seq)
    }

    /// Forwarded: allocate a block if the next append crosses a boundary.
    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<(), BlockPoolError> {
        self.block_pool.may_append(seq)
    }

    /// Forwarded: hash filled blocks since the last call.
    pub fn hash_blocks(&mut self, seq: &mut Sequence) {
        self.block_pool.hash_blocks(seq);
    }

    /// Compute the slot mapping for a range of tokens in a sequence.
    ///
    /// For tokens `[token_offset .. token_offset + num_tokens)` in `seq`,
    /// slot_mapping[i] = `block_table[(token_offset + i) / block_size] *
    /// block_size + (token_offset + i) % block_size`. This is what
    /// `reshape_and_cache` consumes — each slot is the flat index into
    /// the physical KV cache's block-slot array.
    ///
    /// Returns `Vec<i64>` to match `AttnMetadata::slot_mapping` type.
    pub fn compute_slot_mapping(
        &self,
        seq: &Sequence,
        token_offset: usize,
        num_tokens: usize,
    ) -> Vec<i64> {
        let mut mapping = Vec::with_capacity(num_tokens);
        for i in 0..num_tokens {
            let abs_idx = token_offset + i;
            let block_idx = abs_idx / self.block_size;
            let intra_offset = abs_idx % self.block_size;
            if block_idx < seq.block_table.len() {
                let block_id = seq.block_table[block_idx];
                mapping.push((block_id * self.block_size + intra_offset) as i64);
            } else {
                // Invalid block index: slot mapping is -1 (reserved sentinel).
                mapping.push(-1_i64);
            }
        }
        mapping
    }

    /// The block size (forwarded from the inner pool).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of free (unused) blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.block_pool.num_free_blocks()
    }
}

impl std::fmt::Debug for KvCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCacheManager")
            .field("block_pool", &self.block_pool)
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::identity_op, clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Create a minimal `PagedKVCache` for tests. CPU-only, tiny shape.
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

    mod construction {
        use super::*;

        #[test]
        fn new_constructs_block_pool_internally() {
            let mgr = KvCacheManager::new(42, 256, fake_cache());
            assert_eq!(mgr.num_free_blocks(), 42);
            assert_eq!(mgr.block_size(), 256);
        }
    }

    mod forwarding {
        use super::*;

        #[test]
        fn can_allocate_and_deallocate_forward() {
            let cache = fake_cache();
            let mut mgr = KvCacheManager::new(10, 256, cache);
            let mut seq = make_seq((0..(3 * 256 + 1) as u32).collect());
            let nc = mgr.can_allocate(&seq).unwrap();
            assert_eq!(nc, 0);
            mgr.allocate(&mut seq, 0).unwrap();
            assert_eq!(seq.block_table.len(), 4);
            assert_eq!(mgr.num_free_blocks(), 10 - 4);
            mgr.deallocate(&mut seq).unwrap();
            assert_eq!(mgr.num_free_blocks(), 10);
        }

        #[test]
        fn can_append_and_may_append_forward() {
            let cache = fake_cache();
            let mut mgr = KvCacheManager::new(10, 256, cache);
            let mut seq = make_seq((0..256).collect());
            mgr.allocate(&mut seq, 0).unwrap();
            // 256 tokens → num_tokens % 256 == 0 → no new block needed.
            assert!(mgr.can_append(&seq));
            let blocks_before = seq.block_table.len();
            mgr.may_append(&mut seq).unwrap();
            assert_eq!(seq.block_table.len(), blocks_before);
            // After append_token: num_tokens=257 → 257%256=1 → needs new block.
            seq.append_token(42);
            assert!(mgr.can_append(&seq));
            mgr.may_append(&mut seq).unwrap();
            assert_eq!(seq.block_table.len(), blocks_before + 1);
        }

        #[test]
        fn hash_blocks_forwards() {
            let cache = fake_cache();
            let mut mgr = KvCacheManager::new(20, 256, cache);
            let mut seq = make_seq((0..(3 * 256) as u32).collect());
            mgr.allocate(&mut seq, 0).unwrap();
            seq.num_cached_tokens = 0;
            seq.num_scheduled_tokens = 3 * 256;
            mgr.hash_blocks(&mut seq);
            for i in 0..3 {
                assert_ne!(
                    mgr.block_pool.blocks[seq.block_table[i]].hash, -1,
                    "block {i} should be hashed"
                );
            }
        }
    }

    mod compute_slot_mapping {
        use super::*;

        #[test]
        fn identity_single_block() {
            let cache = fake_cache();
            let mgr = KvCacheManager::new(10, 256, cache);
            let mut seq = make_seq((0..10).collect());
            // Manually set a block table (simulating allocation).
            seq.block_table = vec![42];
            let slots = mgr.compute_slot_mapping(&seq, 0, 3);
            assert_eq!(slots, vec![42 * 256 + 0, 42 * 256 + 1, 42 * 256 + 2]);
        }

        #[test]
        fn crosses_block_boundary() {
            let cache = fake_cache();
            let mgr = KvCacheManager::new(10, 256, cache);
            let mut seq = make_seq((0..300).collect());
            seq.block_table = vec![10, 20];
            // offset=254, num_tokens=4:
            // token 254 → block 0 (10*256+254)
            // token 255 → block 0 (10*256+255)
            // token 256 → block 1 (20*256+0)
            // token 257 → block 1 (20*256+1)
            let slots = mgr.compute_slot_mapping(&seq, 254, 4);
            assert_eq!(
                slots,
                vec![10 * 256 + 254, 10 * 256 + 255, 20 * 256 + 0, 20 * 256 + 1,]
            );
        }

        #[test]
        fn full_range_multi_block() {
            let cache = fake_cache();
            let mgr = KvCacheManager::new(10, 256, cache);
            let mut seq = make_seq((0..(3 * 256 + 50) as u32).collect());
            seq.block_table = vec![5, 6, 7, 8];
            // offset=0, num_tokens=3*256+50 — all tokens
            let slots = mgr.compute_slot_mapping(&seq, 0, 3 * 256 + 50);
            assert_eq!(slots.len(), 3 * 256 + 50);
            for i in 0..3 * 256 + 50 {
                let block_idx = i / 256;
                let intra = i % 256;
                let expected = (seq.block_table[block_idx] * 256 + intra) as i64;
                assert_eq!(slots[i], expected, "slot {i} mismatch");
            }
        }

        #[test]
        fn empty_range() {
            let cache = fake_cache();
            let mgr = KvCacheManager::new(10, 256, cache);
            let seq = make_seq(vec![1u32, 2, 3]);
            let slots = mgr.compute_slot_mapping(&seq, 0, 0);
            assert!(slots.is_empty());
        }

        #[test]
        fn out_of_range_block_returns_minus_one() {
            let cache = fake_cache();
            let mgr = KvCacheManager::new(10, 256, cache);
            let mut seq = make_seq((0..10).collect());
            seq.block_table = vec![1];
            // offset=256, num_tokens=1 → block_idx=1 which is out of range.
            let slots = mgr.compute_slot_mapping(&seq, 256, 1);
            assert_eq!(slots, vec![-1]);
        }
    }

    mod num_free_blocks {
        use super::*;

        #[test]
        fn decreases_on_allocate_increases_on_deallocate() {
            let cache = fake_cache();
            let mut mgr = KvCacheManager::new(10, 256, cache);
            let mut seq = make_seq((0..257).collect());
            assert_eq!(mgr.num_free_blocks(), 10);
            mgr.allocate(&mut seq, 0).unwrap();
            assert_eq!(mgr.num_free_blocks(), 8); // 2 blocks used
            mgr.deallocate(&mut seq).unwrap();
            assert_eq!(mgr.num_free_blocks(), 10);
        }
    }
}
