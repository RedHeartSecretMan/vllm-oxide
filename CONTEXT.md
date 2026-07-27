# vllm-oxide

A Rust port of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) trending toward vLLM's V1 architecture: a single-GPU, Qwen3-focused, offline LLM inference engine for v0.1.

## Language

**Column-parallel (projection)**:
A linear projection whose weight is sharded along the output dimension (dim 0). The matmul produces a complete-in-input, partial-in-output result; reconstruction across TP ranks is by all-gather (concatenation), not reduce.
_Avoid_: output-sharded, column-shard.

**Row-parallel (projection)**:
A linear projection whose weight is sharded along the input dimension (dim 1). The matmul produces a partial sum (partial-in-input, complete-in-output); reconstruction across TP ranks is by all-reduce (summation). Bias is fused into the matmul only on rank 0 to avoid multiplication under all-reduce.
_Avoid_: input-sharded, row-shard.

**Fused projection**:
Multiple sub-projections (Q/K/V for attention, or gate/up for SwiGLU MLP) concatenated into a single weight matrix, computed in one matmul, then split by the caller. At TP=1 the layout is `[Q | K | V]` or `[gate | up]` along dim 0.
_Avoid_: packed projection, merged matmul.

**TP seam**:
The abstraction boundary where future tensor-parallelism wiring (NCCL communicator, all-reduce, all-gather) attaches without rewriting model code. In v0.1 the seam lives in the `ParallelStyle` trait + `TpConfig` enum; `TpConfig::Single` is the zero-cost identity path, `TpConfig::Sharded` is the named-but-unimplemented v0.2 contract.
_Avoid_: TP interface, parallelism hook.

**Paged attention**:
An attention computation that reads K/V from a paged block cache (fixed-size blocks, `block_size=256`) rather than contiguous per-sequence buffers. Prefill uses unpaged `flash_attn_varlen`; decode uses paged `flash_attn_varlen_paged_windowed`.
_Avoid_: paged KV cache attention, block attention.

**CausalLM (trait)**:
The engine-facing model contract in v0.1 — `forward(&mut self, input_ids, positions) -> hidden_states` + `compute_logits(hidden) -> logits` + `vocab_size()` + `device()`. Returned by the registry as `Box<dyn CausalLM>`; EngineCore holds one stable type regardless of architecture. `forward` returns hidden_states (not logits) so prefill can skip lm_head projection on non-last tokens; `compute_logits` is called separately when sampling. v0.2 adds new tasks via new independent traits (`SequenceClassifier`, `Embedder`), not by overloading `CausalLM`.
_Avoid_: model interface, Model trait, CausalLM struct.

**Model registry (inventory)**:
The map from HF architecture strings (`"Qwen3ForCausalLM"`) to factory functions producing `Box<dyn CausalLM>`. Implemented as an `inventory`-distributed static registry: each model file self-registers via `inventory::submit! { ModelEntry { arch, factory } }`, and `registry.rs` is a pure query function with no model-specific knowledge. Keyed off `config.json["architectures"][0]`. Adding an architecture is purely additive (new file + `mod xxx;`), zero edits to existing files.
_Avoid_: model loader (loader is T7's `load_weights`), dispatcher, factory table.

**LinearSpec**:
The neutral geometry parameter struct that `Linear<P>::from_vb` consumes — `{ in_features, out_features_per_shard, bias }`. Model code unpacks its own `Config` (e.g. `Qwen3Config`) into `LinearSpec`. Closes the ADR-0002 seam: `Linear<P>` (in shared `layers/`) stays fully model-agnostic — it never imports `models::qwen3::Qwen3Config` or any architecture-specific type.
_Avoid_: layer config, linear config, projection shape.

## Engine (v0.1 — single-GPU, offline)

**EngineCore**:
The in-process engine that collapses V1's `ModelRunner` (ADR-0004 micro-decision). Owns `Scheduler`, `BlockPool`, `KVCacheManager`, `Box<dyn CausalLM>`, `Sampler`, `Arc<Mutex<PagedKVCache>>`. `step()` runs the full loop: scheduler → tensor prep → `model.forward()` → sampler → KV update. No async, no ZMQ, no CUDA graph capture in v0.1.
_Avoid_: model runner (collapsed into EngineCore for single-GPU scope).

**BlockPool**:
Owns physical `Block`s (`block_id`, `ref_count`, `hash`, `token_ids`), the free-list deque, the used-set, and the prefix-cache hashtable (`hash_to_block_id`). Mirrors nano-vllm's xxhash-chained prefix-cache algorithm. CoW semantics for shared prefix blocks. `kvcache_block_size = 256` (constant).
_Avoid_: block manager (V0 term; V1 split BlockPool from KVCacheManager).

**KVCacheManager**:
The **only** Scheduler-facing seam over `BlockPool` + the physical `PagedKVCache`. The Scheduler never imports `BlockPool` or `attention::PagedKVCache` directly. Owns the mapping from logical block tables to physical block ids and paged-cache slot indices.
_Avoid_: KV cache adapter, block-to-slot mapper.

**Scheduler**:
Token-level scheduling with `waiting`/`running` deques. One `schedule()` step is either pure-prefill or pure-decode. Chunked prefill applies only to the first sequence in a step. Preemption is recompute-only (deallocate + requeue front of `waiting`). `postprocess()` updates block hashes, advances `num_cached_tokens`, and finalises sequences on EOS or `max_tokens`.
_Avoid_: batch scheduler, request scheduler.

**Sequence / SequenceGroup**:
V1 data model for tracking individual requests. `SequenceStatus { Waiting, Running, Finished }`. Per-sequence: `block_table`, `num_tokens`, `num_cached_tokens`, `num_scheduled_tokens`, `is_prefill`, `last_token`, plus attached sampling scalars. `SequenceGroup` is a thin 1:1 wrapper (n>1 sampling deferred to v0.2).
_Avoid_: request, prompt context, generation state.

**PagedKVCache**:
The physical GPU buffer shaped `[2, num_layers, num_blocks, 256, num_kv_heads, head_dim]`. Held as `Arc<Mutex<PagedKVCache>>` and shared between `EngineCore` and every attention layer. `reshape_and_cache` writes per-step K/V into the paged cache.
_Avoid_: block cache buffer, GPU cache pool.

**AttnMetadata**:
Flash-attention metadata carrying `cu_seqlens_q`/`cu_seqlens_k` (flash-attn cumulative convention, NOT vLLM `context_lens` convention — T10 finding honoured). The conversion from scheduler convention lives in a `build_paged_metadata` helper inside the attention backend.
_Avoid_: attention context, batch metadata.

## API surface

**LLM::generate**:
The single public function — `generate(&mut self, prompts: &[Prompt], sampling_params: &[SamplingParams]) -> Result<Vec<RequestOutput>>`. Internally loops `step()` until `scheduler.is_finished()`, then detokenizes. Mirrors nano-vllm `LLM.generate`.
_Avoid_: run, infer, complete, __call__.

**Prompt**:
Input enum: `Text(String)` for natural-language prompts, `TokenIds(Vec<u32>)` for pre-tokenized fixtures. Both are accepted in the same batch.
_Avoid_: input, query, user message.

**RequestOutput**:
Per-request result: `{ seq_id: usize, text: String, token_ids: Vec<u32> }`. Both decoded text and raw token IDs are always provided so callers can post-process tokens without re-tokenizing.
_Avoid_: generation result, completion output.

**SamplingParams**:
Per-prompt sampling configuration: `temperature` (0 ≡ greedy), `top_k`, `top_p`, `max_tokens`, `ignore_eos`, `presence_penalty`/`frequency_penalty`/`repetition_penalty`. Lives in `sampler.rs` (M1 placement per ADR-0004).
_Avoid_: generation config, decode params, sampling config.

**EngineOptions**:
Construction-time configuration for `LLM::new`: `max_num_batched_tokens` (default 16384), `max_num_seqs` (512), `max_model_len`, `gpu_memory_utilization` (0.9), `enforce_eager` (always true in v0.1), `dtype` override. Mirrors nano-vllm's `Config`.
_Avoid_: engine config, runtime options.

## Correctness

**Golden fixture**:
A content-addressed `.safetensors` file produced by the Python harness (`tools/golden-gen/`) running the oracle triangle (HF Transformers / nano-vllm / vLLM V1) on fixed prompts. Used as ground truth for L1 (token-sequence exact match) and L2 (logits tensor comparison) validation of the Rust engine. Stored as GitHub Release assets, NOT in git mainline.
_Avoid_: reference output, expected output, snapshot, oracle output.

**Oracle triangle**:
The three reference implementations cross-validated to produce golden fixtures and calibrate numerical tolerances. If two oracles disagree beyond the calibrated threshold, the disagreement is recorded as a known deviation rather than trusting one blindly.
_Avoid_: reference models, ground-truth engines.

**Release gate vs CI gate**:
CI (every push, CPU-only) runs property tests → "CI green". Release gate (pre-release, manual, GPU) runs golden comparison → "numerically validated". These are explicitly different — the repo README must document that CI green ≠ validated.
_Avoid_: CI vs release check.
