# Weight loader seam

The boundary between model-specific configuration and model-agnostic layers is
`LinearSpec`, a neutral geometry struct (`{ in_features, out_features_per_shard,
bias }`) produced by model code and consumed by `Linear<P>`.

**Decision**: The Weight loader resolves a `Source` exactly once into an
immutable `ResolvedModel` before model construction. That value binds the
canonical local root or exact Hub commit, configuration, tokenizer, weight
paths, requested dtype, and special-token metadata. Model factories consume it
without following a revision or rediscovering artifacts (ADR-0007).

Model code still unpacks architecture configuration (e.g. `Qwen3Config`) into
`LinearSpec`. `Linear<P>` never imports architecture-specific types, and the
Weight loader performs no model-specific fusion or rank slicing. All fusion
lives in `Linear::<P>::from_vb` (ADR-0001 style tags).

**HF name mapping**: Checkpoint tensor names match the model's expected names
1:1, so no remap table is needed. `from_vb` for `QkvMerged` reads `q_proj.weight` /
`k_proj.weight` / `v_proj.weight` and `Tensor::cat`s along dim 0.

**Source resolution**:
- `Source::Local`: fallback chain `model.safetensors.index.json` → single
  `model.safetensors` → glob `*.safetensors`, all under one canonical root.
- `Source::Hub`: hf-hub 0.5 resolves a branch or tag to one immutable commit
  before fetching configuration, tokenizer, or weights.
- `HF_HUB_OFFLINE=1`: the cached ref is resolved to one
  `snapshots/<commit>` root before artifacts are read.

**Lazy mmap**: Mapping the safetensors files avoids eagerly copying all tensor
data. Header parsing still scales with metadata, and `vb.get(..)` materializes
requested tensors, including device transfer and dtype conversion when needed.
Constructing the full model therefore still reads and allocates its weights.

**`slice_for_rank` seam** (v0.1 identity): The weight-loader TP hook lives on
the `ParallelStyle` trait (ADR-0001), not the loader. v0.1 returns
`Cow::Borrowed`; v0.2.0 retains that identity-only feasibility seam because
runtime TP/NCCL is out of scope. A later release may add per-style rank-slicing
math without making model code branch on rank (ADR-0006).

**`unsafe` boundary**: At T15, this was the only `vllm_oxide` module calling
unsafe. Its call to `ShardedSafeTensors::var_builder` inherits the safety
requirements of `memmap2::MmapOptions`. The documented justification accepts
the same mapping risk present in Candle, mistral.rs, and Hugging Face tooling.

**Consequences**:
- `layers/` stays model-agnostic. Check this by confirming that its module tree
  contains no `use crate::models::*` import.
- Adding a new architecture requires zero changes to the loader.
- `out_features_per_shard` semantics are per-style (documented in `LinearSpec`).

**Status**: accepted
