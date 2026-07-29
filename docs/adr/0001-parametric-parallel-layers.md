# Parametric parallel layers

`Linear<P: ParallelStyle>` uses a type-level style tag to encode which
projection a weight tensor IS, without runtime branching. Three v0.1 tags:
`QkvMerged` (3 shards, concat dim 0), `GateUpMerged` (2 shards, concat dim 0),
`Row` (1 shard, dim 1). Plain `Column` is deferred (YAGNI — no Qwen3 consumer).

**Decision**: Each style tag is a zero-sized marker implementing a shared
`ParallelStyle` trait with static constants (`SHARD_DIM`, `num_shards`) and
hooks (`slice_for_rank`, `reduce_output`, `should_fuse_bias`). The tag owns
QKV/gate-up fusion at load time via `Linear::from_vb` — `x @ W^T + b` forward
is style-agnostic.

**R2**: `PhantomData<P>` is the documented cosmetic cost of the typestate
pattern — accepted in exchange for compile-time checking that the model code's
declared projection style matches the layer's expected layout.

**TP seam (v0.1)**: `slice_for_rank` ships the identity default
(`Cow::Borrowed`, zero-cost). `reduce_output` is identity (no NCCL). v0.2
overrides per style with real GQA math for `QkvMerged`, 2-way split for
`GateUpMerged`, dim-1 narrow for `Row`.

**Considered Options**:
- Runtime enum dispatch (rejected): per-forward match on style incurs a branch
  in the hot path; the typestate pattern eliminates it with zero runtime cost.
- Separate types per style (rejected): `QkvLinear`, `RowLinear`, etc. would
  duplicate forward logic across 3+ structs with no shared generic.

**Consequences**:
- Model code declares `Linear<QkvMerged>` etc. — compiler enforces style/layout
  agreement.
- Adding TP>1 means new trait impls (or overrides), not new model code.
- `layers/` is a leaf: no internal crate dependencies.

**Status**: accepted
