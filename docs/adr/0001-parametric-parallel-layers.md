# Parametric parallel layers

`Linear<P: ParallelStyle>` identifies a weight tensor's projection through a
type-level tag, without runtime branching. v0.1 has three tags:
`QkvMerged` (3 shards, concat dim 0), `GateUpMerged` (2 shards, concat dim 0),
and `Row` (1 shard, dim 1). Plain `Column` is deferred because Qwen3 has no
consumer for it.

**Decision**: Each style tag is a zero-sized marker implementing a shared
`ParallelStyle` trait with static constants (`SHARD_DIM`, `num_shards`) and
hooks (`slice_for_rank`, `reduce_output`, `should_fuse_bias`). The tag owns
QKV/gate-up fusion at load time through `Linear::from_vb`. The
`x @ W^T + b` forward operation is independent of the style.

**R2**: `PhantomData<P>` adds a marker to the type definition. This is an
accepted cost of checking at compile time that the model's declared projection
style matches the layer's expected layout.

**TP seam (v0.1)**: `slice_for_rank` ships the identity default
(`Cow::Borrowed`, zero-cost). `reduce_output` is identity (no NCCL). v0.2
keeps this as a feasibility seam without runtime TP/NCCL support; any real
per-style sharding implementation is deferred beyond v0.2.0 (ADR-0006).

**Considered Options**:
- Runtime enum dispatch (rejected): per-forward match on style incurs a branch
  in the hot path; the typestate pattern eliminates it with zero runtime cost.
- Separate types per style (rejected): `QkvLinear`, `RowLinear`, etc. would
  duplicate forward logic across 3+ structs with no shared generic.

**Consequences**:
- Model code declares types such as `Linear<QkvMerged>`; the compiler checks
  that the style and layout agree.
- Adding TP>1 means new trait impls (or overrides), not new model code.
- `layers/` is a leaf: no internal crate dependencies.

**Status**: accepted
