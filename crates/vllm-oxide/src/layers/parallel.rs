//! `ParallelStyle` trait + style tags + `TpConfig` + `ShardId` (ADR-0001 / ADR-0002).
//!
//! The TP (tensor-parallel) seam: each style tag is a zero-sized marker encoding
//! which projection this `Linear<P>` IS (QKV-fused / gate-up-fused / row-parallel).
//! The trait carries per-style constants (`SHARD_DIM`, `num_shards`) and the
//! `slice_for_rank` weight-slicing hook. Execution is single-GPU: the hook
//! returns `Cow::Borrowed`, with rank slicing and collective reduction deferred.

use candle_core::{Result, Tensor};
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardId {
    Q,
    K,
    V,
    Index(usize),
}

/// Type-level marker for a parallel-style tag.
///
/// Three projection style tags implement this: [`QkvMerged`], [`GateUpMerged`],
/// [`Row`]. Plain `Column` is deferred (YAGNI — no Qwen3 consumer). Each impl
/// encodes static projection geometry:
///
/// - [`ParallelStyle::SHARD_DIM`] — axis along which the fused weight shards
///   concatenate (and along which TP splits). `0` for column-parallel styles;
///   `1` for `Row`.
/// - [`ParallelStyle::num_shards`] — how many sub-projections the style fuses.
///   `Row=1`, `GateUpMerged=2`, `QkvMerged=3`.
///
/// `slice_for_rank` is the weight-loader TP seam (ADR-0002). It uses the
/// identity default (`Cow::Borrowed`, zero-cost), suitable for hardcoded TP=1.
/// A future implementation would need GQA replication for `QkvMerged`, dim-0
/// slicing for `GateUpMerged`, and dim-1 slicing plus reduction for `Row`.
/// The seam lives here, on the trait, so model code never branches on rank —
/// adding TP>1 means new trait impls, not new model code.
pub trait ParallelStyle: Send + Sync + 'static {
    /// Axis along which the style concatenates sub-projection weights (and
    /// along which TP would split them). Column-parallel = 0; row-parallel = 1.
    const SHARD_DIM: usize;

    /// How many sub-projections this style fuses into one weight tensor.
    fn num_shards() -> usize;

    /// Whether bias should be fused into the matmul on this rank. Column-parallel
    /// styles always fuse (their bias slice is owned by this rank). Row-parallel
    /// styles fuse only on rank 0 at TP>1 — bias is replicated, so summing it
    /// across `world_size` ranks via all-reduce would multiply it. Currently always
    /// `true` (single-rank), so [`crate::layers::linear::Linear::forward`]
    /// unconditionally adds bias when present.
    ///
    /// Rank-aware bias handling is deferred with tensor-parallel execution.
    fn should_fuse_bias(_tp: &TpConfig) -> bool {
        true
    }

    /// Post-matmul output reduction across TP ranks. Column-parallel styles
    /// return the input unchanged — their partial-in-output result is correct
    /// for the rank's owned output slice; the caller concatenates slices to
    /// reconstruct the full output dim. Row-parallel styles all-reduce (sum)
    /// partial-in-input results into the complete output at TP>1.
    /// Current implementation: identity (single rank, no NCCL comms).
    ///
    /// Row all-reduce requires a future tensor-parallel implementation.
    fn reduce_output(output: Tensor, _tp: &TpConfig) -> Result<Tensor> {
        Ok(output)
    }

    /// TP-seam weight slicing. Currently identity (`Cow::Borrowed`, zero clone);
    /// per-rank slicing is deferred with tensor-parallel execution.
    ///
    /// `rank` / `world_size` pass directly — weight-load math has no need for
    /// NCCL comms (comms are for runtime all-reduce, not weight loading).
    /// `shard_id` carries the sub-projection identity for styles that fuse
    /// multiple shards (Q/K/V for `QkvMerged`).
    fn slice_for_rank<'a>(
        full: &'a Tensor,
        _rank: usize,
        _world_size: usize,
        _shard_id: Option<ShardId>,
    ) -> Result<Cow<'a, Tensor>> {
        Ok(Cow::Borrowed(full))
    }
}

/// QKV-fused column-parallel projection. Weight layout along dim 0: `[Q | K | V]`.
#[derive(Debug)]
pub struct QkvMerged;

/// Gate/up-fused column-parallel projection (SwiGLU MLP). Weight layout: `[gate | up]`.
#[derive(Debug)]
pub struct GateUpMerged;

/// Row-parallel projection. Sharded along the input dim (dim 1); bias (when
/// present) is replicated, fused into the matmul only on rank 0 at TP>1.
#[derive(Debug)]
pub struct Row;

impl ParallelStyle for QkvMerged {
    const SHARD_DIM: usize = 0;
    fn num_shards() -> usize {
        3
    }
}

impl ParallelStyle for GateUpMerged {
    const SHARD_DIM: usize = 0;
    fn num_shards() -> usize {
        2
    }
}

impl ParallelStyle for Row {
    const SHARD_DIM: usize = 1;
    fn num_shards() -> usize {
        1
    }
}

/// Internal tensor-parallel feasibility seam. Execution supports
/// [`TpConfig::Single`] only; [`TpConfig::Sharded`] is non-constructible.
///
/// `Sharded`'s inner type is crate-private and cannot be built outside this
/// internal module graph. Runtime NCCL support is deferred.
///
/// Model code does not read `TpConfig` at runtime; the field remains
/// only as an internal feasibility anchor.
#[derive(Debug, Clone)]
pub enum TpConfig {
    /// Single-GPU (TP=1). The only constructible variant.
    Single,
    /// Future tensor-parallel feasibility shape; execution is unsupported.
    Sharded(ShardedParams),
}

#[derive(Debug, Clone)]
pub(crate) struct ShardedParams {
    rank: usize,
    world_size: usize,
}

impl TpConfig {
    /// Returns the supported single-GPU configuration [`TpConfig::Single`].
    pub const fn single() -> Self {
        TpConfig::Single
    }
}

impl Default for TpConfig {
    fn default() -> Self {
        TpConfig::single()
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    mod shard_id {
        use super::*;

        #[test]
        fn q_and_k_distinct() {
            assert_ne!(ShardId::Q, ShardId::K);
            assert_ne!(ShardId::K, ShardId::V);
            assert_ne!(ShardId::Q, ShardId::V);
        }

        #[test]
        fn same_index_equal() {
            assert_eq!(ShardId::Index(0), ShardId::Index(0));
            assert_eq!(ShardId::Index(7), ShardId::Index(7));
            assert_ne!(ShardId::Index(0), ShardId::Index(1));
        }

        #[test]
        fn q_k_v_distinct_from_index() {
            assert_ne!(ShardId::Q, ShardId::Index(0));
            assert_ne!(ShardId::Index(0), ShardId::Q);
        }
    }

    mod style_tags {
        use super::*;

        #[test]
        fn qkv_merged_is_column_parallel_with_three_shards() {
            assert_eq!(<QkvMerged as ParallelStyle>::SHARD_DIM, 0);
            assert_eq!(QkvMerged::num_shards(), 3);
        }

        #[test]
        fn gate_up_merged_is_column_parallel_with_two_shards() {
            assert_eq!(<GateUpMerged as ParallelStyle>::SHARD_DIM, 0);
            assert_eq!(GateUpMerged::num_shards(), 2);
        }

        #[test]
        fn row_is_row_parallel_with_one_shard() {
            assert_eq!(<Row as ParallelStyle>::SHARD_DIM, 1);
            assert_eq!(Row::num_shards(), 1);
        }
    }

    mod tp_config {
        use super::*;

        #[test]
        fn single_is_constructible_via_named_ctor() {
            let via_ctor: TpConfig = TpConfig::single();
            assert!(matches!(via_ctor, TpConfig::Single));
        }

        #[test]
        fn default_is_single() {
            let via_default: TpConfig = Default::default();
            assert!(matches!(via_default, TpConfig::Single));
        }
    }

    mod tp_seam_defaults {
        use super::*;

        #[test]
        fn should_fuse_bias_defaults_true_under_single_tp() {
            let tp = TpConfig::single();
            assert!(QkvMerged::should_fuse_bias(&tp));
            assert!(GateUpMerged::should_fuse_bias(&tp));
            assert!(Row::should_fuse_bias(&tp));
        }

        #[test]
        fn reduce_output_defaults_identity_under_single_tp() {
            let dev = candle_core::Device::Cpu;
            let t = candle_core::Tensor::zeros((2, 2), candle_core::DType::F32, &dev).unwrap();
            let tp = TpConfig::single();
            let out_q = QkvMerged::reduce_output(t.clone(), &tp).unwrap();
            let out_r = Row::reduce_output(t.clone(), &tp).unwrap();
            assert_eq!(out_q.shape().dims(), [2, 2]);
            assert_eq!(out_r.shape().dims(), [2, 2]);
        }
    }

    mod slice_for_rank_identity {
        use super::*;

        #[test]
        fn qkv_merged_returns_borrowed_tensor_unchanged_at_tp1() {
            let dev = candle_core::Device::Cpu;
            let t = candle_core::Tensor::zeros((4, 4), candle_core::DType::F32, &dev).unwrap();
            let out = QkvMerged::slice_for_rank(&t, 0, 1, None).unwrap();
            assert!(matches!(out, Cow::Borrowed(_)));
        }

        #[test]
        fn row_returns_borrowed_tensor_unchanged_at_tp1() {
            let dev = candle_core::Device::Cpu;
            let t = candle_core::Tensor::zeros((8, 8), candle_core::DType::F32, &dev).unwrap();
            let out = Row::slice_for_rank(&t, 0, 1, Some(ShardId::Index(0))).unwrap();
            assert!(matches!(out, Cow::Borrowed(_)));
        }

        #[test]
        fn gate_up_merged_ignores_shard_id_at_tp1() {
            let dev = candle_core::Device::Cpu;
            let t = candle_core::Tensor::zeros((2, 2), candle_core::DType::F32, &dev).unwrap();
            let with_none = GateUpMerged::slice_for_rank(&t, 0, 1, None).unwrap();
            let with_q = GateUpMerged::slice_for_rank(&t, 0, 1, Some(ShardId::Q)).unwrap();
            assert!(matches!(with_none, Cow::Borrowed(_)));
            assert!(matches!(with_q, Cow::Borrowed(_)));
        }
    }
}
