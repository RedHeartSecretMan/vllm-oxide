# vllm-oxide

A Rust engine for single-GPU, synchronous, offline Qwen3 inference, drawing on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [vLLM](https://github.com/vllm-project/vllm).

This vocabulary describes the historical v0.1.0 source. Future work mentioned below is historical design intent, not a guarantee of v0.2.0 features. See the [architecture overview](docs/architecture.md), [validation limits](docs/validation/README.md) and [documentation maintenance record](docs/version-doc-update.md).

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
The abstraction boundary where future tensor-parallelism wiring (NCCL communicator, all-reduce, all-gather) attaches without rewriting model code. In v0.1 the seam lives in the `ParallelStyle` trait + `TpConfig` enum; `TpConfig::Single` is the zero-cost identity path, `TpConfig::Sharded` is named but unimplemented; future TP wiring was a historical proposal.
_Avoid_: TP interface, parallelism hook.

**Paged attention**:
An attention computation that reads K/V from a paged block cache (fixed-size blocks, `block_size=256`) rather than contiguous per-sequence buffers. Prefill uses unpaged `flash_attn_varlen`; decode uses paged `flash_attn_varlen_paged_windowed`.
_Avoid_: paged KV cache attention, block attention.

**CausalLM (trait)**:
The engine-facing model contract in v0.1 — `forward(&mut self, input_ids, positions) -> hidden_states` + `compute_logits(hidden) -> logits` + `vocab_size()` + `device()`. Returned by the registry as `Box<dyn CausalLM>`; EngineCore holds one stable type regardless of architecture. `forward` returns hidden_states (not logits) so prefill can skip lm_head projection on non-last tokens; `compute_logits` is called separately when sampling. A historical proposal would add new tasks through independent traits (`SequenceClassifier`, `Embedder`), not by overloading `CausalLM`; those tasks are not implemented here.
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

**KvCacheManager**:
The scheduler-facing seam over `BlockPool` and the physical `PagedKVCache`. The Scheduler never imports `BlockPool` or `attention::PagedKVCache` directly. Owns the mapping from logical block tables to physical block ids and paged-cache slot indices.
_Avoid_: KV cache adapter, block-to-slot mapper.

**Scheduler**:
Token-level scheduling with `waiting`/`running` deques. One `schedule()` step is either pure-prefill or pure-decode. Chunked prefill applies only to the first sequence in a step. Preemption is recompute-only (deallocate + requeue front of `waiting`). `postprocess()` updates block hashes, advances `num_cached_tokens`, and finalises sequences on EOS or `max_tokens`.
_Avoid_: batch scheduler, request scheduler.

**Sequence / SequenceGroup**:
V1 data model for tracking individual requests. `SequenceStatus { Waiting, Running, Finished }`. Per-sequence: `block_table`, `num_tokens`, `num_cached_tokens`, `num_scheduled_tokens`, `is_prefill`, `last_token`, plus attached sampling scalars. `SequenceGroup` is a thin 1:1 wrapper (n>1 sampling unimplemented in this version).
_Avoid_: request, prompt context, generation state.

**PagedKVCache**:
The physical GPU buffer shaped `[2, num_layers, num_blocks, 256, num_kv_heads, head_dim]`. Held as `Arc<Mutex<PagedKVCache>>` and shared between `EngineCore` and every attention layer. `reshape_and_cache` writes per-step K/V into the paged cache.
_Avoid_: block cache buffer, GPU cache pool.

**AttnMetadata**:
Flash-attention metadata carrying `cu_seqlens_q`/`cu_seqlens_k` (flash-attn cumulative convention, NOT vLLM `context_lens` convention — T10 finding honoured). The conversion from scheduler convention lives in a `build_paged_metadata` helper inside the attention backend.
_Avoid_: attention context, batch metadata.

## API surface

**LLM::generate**:
The intended public generation entrypoint, `generate(&mut self, prompts: &[Prompt], sampling_params: &[SamplingParams]) -> Result<Vec<RequestOutput>>`. Internally loops `step()` until `scheduler.is_finished()`, then detokenizes. Mirrors nano-vllm `LLM.generate`; this historical crate also exposes internal and diagnostic APIs.
_Avoid_: run, infer, complete, __call__.

**Prompt**:
Input enum: `Text(String)` for natural-language prompts, `TokenIds(Vec<u32>)` for pre-tokenized fixtures. Both are accepted in the same batch.
_Avoid_: input, query, user message.

**RequestOutput**:
Per-request result: `{ seq_id: usize, text: String, token_ids: Vec<u32>, finished: bool }`. Both decoded text and raw token IDs are always provided so callers can post-process tokens without re-tokenizing.
_Avoid_: generation result, completion output.

**SamplingParams**:
Per-prompt sampling configuration: `temperature` (0 ≡ greedy), `top_k`, `top_p`, `max_tokens`, `ignore_eos`, `presence_penalty`/`frequency_penalty`/`repetition_penalty`. Lives in `sampler.rs` (M1 placement per ADR-0004).
_Avoid_: generation config, decode params, sampling config.

**EngineOptions**:
Construction-time configuration for `LLM::new`: `max_num_batched_tokens` (default 16384), `max_num_seqs` (512), `max_model_len`, `gpu_memory_utilization` (0.9), `enforce_eager` (always true in v0.1), `dtype` override. Mirrors nano-vllm's `Config`.
_Avoid_: engine config, runtime options.

## Correctness

**Golden fixture**:
A content-addressed `.safetensors` file produced by the Python harness (`tools/golden-gen/`) running two oracle engines (transformers + vLLM) on fixed prompts. Used for L1 (token-sequence exact match) and L2 (logits tensor comparison) validation of the Rust engine. Stored as GitHub Release assets, NOT in git mainline.
_Avoid_: reference output, expected output, snapshot, oracle output.

**Reference oracle**:
The oracle used as the correctness target — `transformers` with `output_logits=True` and `attn_implementation=sdpa`. vllm-oxide's L1 and L2 comparisons run against this oracle's output. It is not a "ground truth" (Qwen3-0.6B weights are BF16; BF16 computation is non-associative), but it is the single consistent reference point we measure against.
_Avoid_: ground truth, canonical engine, expected engine.

**Baseline oracle**:
The oracle used to calibrate numerical tolerances — `vLLM` (BF16, same dtype path as vllm-oxide). The maximum per-element |transformers - vLLM| across all canonical prompts × 2.0 gives the `atol` for L2 comparison. vLLM's token output is also used to determine which L1 positions are inherently non-deterministic under BF16 (skip map).
_Avoid_: secondary oracle, calibration oracle.

**atol calibration**:
The process of measuring `max(|transformers_logits - vllm_logits| per-element)` across all 5 canonical prompts, then multiplying by 2.0 to produce a global `atol` for L2 comparison. Runs as a separate `golden-gen calibrate` step after fixture generation. No rtol is used — the logit values of interest are large (top-k candidates), and rtol is unstable for near-zero tail logits.
_Avoid_: tolerance derivation, threshold computation.

**Near-tie skip (L1)**:
When vllm-oxide's argmax token differs from the reference, the maintained helper requires the actual token to be the candidate argmax and the expected token to belong to its top two, including ties. A top-two gap strictly below ε is a legacy near-tie exclusion; the default ε remains atol × 2.0. The original helper failed to recognize the reference runner-up. Exclusions remain separate from exact matches; see [validation](docs/validation/README.md). Used for canonical prompts with full logits.
_Avoid_: epsilon skip, close-call skip.

**Skip map (L1 regression)**:
For regression prompts (no full logits), calibration records positions where vLLM's token also differs from the reference. The historical L1 comparator excludes those positions. Oracle disagreement alone does not establish nondeterminism or rule out an implementation defect; exclusions are recorded separately from passing comparisons.
_Avoid_: exclusion set, known-mismatch list.

**Same-prefix comparison (L2)**:
The maintained `compare_l2` compares complete shared-history rows through the first divergent token, including that row, then excludes the later suffix. The original implementation instead filtered by current-token equality and could compare different histories; see [validation](docs/validation/README.md).
_Avoid_: prefix-aware L2, context-aware comparison.

**Release gate vs CI gate**:
CPU tests check project behavior without running GPU inference. The historical source archive has no GitHub CI workflow. Manual GPU fixture comparisons are a separate operation with the coverage/failure limitations documented in [validation](docs/validation/README.md). Passing CPU checks or a partial historical report does not establish complete numerical acceptance. These are explicitly different — the repo README must document that CI green ≠ validated.
_Avoid_: CI vs release check.
