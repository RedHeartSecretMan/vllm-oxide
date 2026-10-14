# Model registry and RoPE

Two decisions governing the `models/` module.

## Registry: additive-only architecture registration

**Decision**: Each architecture has one module and registers itself with
`inventory::submit!`. Its single `ModelEntry` factory consumes an immutable
`ResolvedModel` owned by the Weight loader and produces the model plus shared
attention state. Registry lookup remains a pure architecture query and never
resolves sources or fetches artifacts (ADR-0007). The `CausalLM` trait
(`forward`, `compute_logits`, `vocab_size`, `device`) remains the
engine-facing contract. Adding an architecture is purely additive: new file +
`mod xxx;` line in `models/mod.rs`, zero edits to `registry.rs` or any
existing model file.

**Consequences**:
- The registry maps HF architecture strings (e.g. `"Qwen3ForCausalLM"`) to
  factories consuming `&ResolvedModel`.
- `models/` depends on `layers/` + `attention/` + `loader/`; nothing depends
  on `models/` except `llm.rs` (the composition root).
- v0.1 compiles architectures into the binary; it has no dynamic plugin loading.

## RoPE: cos/sin cache with position lookup

**Decision**: Implement `inv_freq`, `cos_sin_cache`, and half-rotation using
the pattern in nano-vllm's `rotary_embedding.py`. The cache has shape
`(max_position_embeddings, rotary_dim)`. Forward uses per-token `positions`,
supporting non-sequential decode positions such as `[15, 42, 7]` as well as
prefill's `[0..seq_len)`.

**R5**: No scaling knob exposed. Qwen3 ships `rope_theta = 1_000_000` with no
`rope_scaling`. Scaling variants (`linear`, `dynamic`, `yarn`) remain
outside v0.2.0 and land only when a later supported model requires them
(ADR-0006).

**Why not candle-nn**: `candle_nn::rotary_emb::rope` (free function) consumes
cos/sin shaped for sequential `[0..seq_len)` positions. Decode instead needs
a position for each sequence. The nano-vllm cache-lookup
pattern is the V1 parity path.

**Considered Options**:
- candle-nn free functions (rejected): wrong positional convention for decode.
- Upstream a struct to candle-nn (deferred): would require coordination with
  upstream; revisit after v0.2.0 when supported scope requires it.

**Consequences**:
- `layers/rope.rs` is self-contained (no candle-nn RoPE dependency).
- `models/` module is a leaf consumer of `layers/`.

**Status**: accepted
