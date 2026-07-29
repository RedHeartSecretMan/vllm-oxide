# Weight loader seam

The boundary between model-specific configuration and model-agnostic layers is
`LinearSpec` — a neutral geometry struct (`{ in_features, out_features_per_shard,
bias }`) that model code produces and `Linear<P>` consumes.

**Decision**: Model code unpacks its own config (e.g. `Qwen3Config`) into
`LinearSpec`. `Linear<P>` never imports architecture-specific types. The loader
(`load_weights`) resolves a `Source` to safetensors shards and returns a candle
`ShardedVarBuilder`; it performs NO model-specific work — no QKV fusion, no
gate/up fusion, no per-rank slicing. All fusion lives in `Linear::<P>::from_vb`
(ADR-0001 style tags).

**HF name mapping**: Checkpoint tensor names map 1:1 with what the model
expects. No remap table. `from_vb` for `QkvMerged` reads `q_proj.weight` /
`k_proj.weight` / `v_proj.weight` and `Tensor::cat`s along dim 0.

**Source resolution**:
- `Source::Local`: fallback chain `model.safetensors.index.json` → single
  `model.safetensors` → glob `*.safetensors`.
- `Source::Hub`: hf-hub 0.5 sync API. `HF_HUB_OFFLINE=1` switches to local
  `Cache` lookup for air-gapped / CI runs.

**Lazy mmap**: Tensors are not materialised until `vb.get(..)` is called.
Loading a multi-GB checkpoint is O(1); only touched tensors cost memory.

**`slice_for_rank` seam** (v0.1 identity): The weight-loader TP hook lives on
the `ParallelStyle` trait (ADR-0001), not the loader. v0.1 returns
`Cow::Borrowed`; v0.2 overrides per style with rank-slicing math. The seam
lives on the trait so model code never branches on rank.

**`unsafe` boundary**: This module is the only `vllm_oxide` module calling
unsafe at T15. The single call site is `ShardedSafeTensors::var_builder`,
whose unsafe is inherited from `memmap2::MmapOptions`. Accepted with documented
justification (same risk as upstream candle / mistral.rs / HF tooling).

**Consequences**:
- `layers/` stays fully model-agnostic — verified by the absence of any
  `use crate::models::*` import in the layers module tree.
- Adding a new architecture requires zero changes to the loader.
- `out_features_per_shard` semantics are per-style (documented in `LinearSpec`).

**Status**: accepted
