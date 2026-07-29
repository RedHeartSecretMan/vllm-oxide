# Model registry and RoPE

Two decisions governing the `models/` module.

## Registry: additive-only architecture registration

**Decision**: Each architecture is one module + one `inventory::submit!`
self-registration. The `CausalLM` trait (`forward`, `compute_logits`,
`vocab_size`, `device`) is the engine-facing contract. Adding an architecture
is purely additive: new file + `mod xxx;` line in `models/mod.rs`, zero edits
to `registry.rs` or any existing model file.

**Consequences**:
- The registry maps HF architecture strings (e.g. `"Qwen3ForCausalLM"`) to
  factory functions producing `Box<dyn CausalLM>`.
- `models/` depends on `layers/` + `attention/` + `loader/`; nothing depends
  on `models/` except `llm.rs` (the composition root).
- No plugin dynamic loading in v0.1 — all architectures are compiled in.

## RoPE: cos/sin cache with position lookup

**Decision**: Hand-roll `inv_freq` + `cos_sin_cache` + half-rotation following
the nano-vllm `rotary_embedding.py` pattern. The cache has shape
`(max_position_embeddings, rotary_dim)` and forward indexes by per-token
`positions` — supporting non-sequential decode positions (e.g. `[15, 42, 7]`),
not just prefill's `[0..seq_len)`.

**R5**: No scaling knob exposed. Qwen3 ships `rope_theta = 1_000_000` with no
`rope_scaling`. Scaling variants (`linear`, `dynamic`, `yarn`) land in v0.2
if a supported model requires them.

**Why not candle-nn**: `candle_nn::rotary_emb::rope` (free function) consumes
cos/sin shaped against a sequential `[0..seq_len)` convention — incompatible
with decode where positions are per-sequence. The nano-vllm cache-lookup
pattern is the V1 parity path.

**Considered Options**:
- candle-nn free functions (rejected): wrong positional convention for decode.
- Upstream a struct to candle-nn (deferred): would require coordination with
  upstream; revisit before v0.2 (see TODO in rope.rs).

**Consequences**:
- `layers/rope.rs` is self-contained (no candle-nn RoPE dependency).
- `models/` module is a leaf consumer of `layers/`.

**Status**: accepted
