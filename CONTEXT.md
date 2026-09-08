# vllm-oxide

A Rust port of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) trending toward vLLM's V1 architecture: a single-GPU, Qwen3 offline LLM inference engine with a correctness-first v0.2.0 contract.

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
The abstraction boundary where future tensor-parallelism wiring can attach without rewriting model code. `TpConfig::Single` is the supported identity path; `TpConfig::Sharded` names a feasibility seam, not supported v0.2.0 TP/NCCL behavior.
_Avoid_: TP interface, parallelism hook.

**Paged attention**:
An attention computation that reads K/V from a paged block cache (fixed-size blocks, `block_size=256`) rather than contiguous per-sequence buffers. Prefill uses unpaged `flash_attn_varlen`; decode uses paged `flash_attn_varlen_paged_windowed`.
_Avoid_: paged KV cache attention, block attention.

**CausalLM (trait)**:
The engine-facing contract for causal text generation. New task families use independent traits rather than overloading `CausalLM`; v0.2.0 remains limited to causal generation.
_Avoid_: model interface, Model trait, CausalLM struct.

**Model registry (inventory)**:
The map from HF architecture strings (`"Qwen3ForCausalLM"`) to factory functions producing `Box<dyn CausalLM>`. Implemented as an `inventory`-distributed static registry: each model file self-registers via `inventory::submit! { ModelEntry { arch, factory } }`, and `registry.rs` is a pure query function with no model-specific knowledge. Keyed off `config.json["architectures"][0]`. Adding an architecture is purely additive (new file + `mod xxx;`), zero edits to existing files.
_Avoid_: dispatcher, factory table.

**Weight loader**:
The internal component that resolves checkpoint artifacts and exposes their weights for model construction. The `model-loader` wording in v0.2 tracker prose names this component, not the `Model registry`.
_Avoid_: model registry, dispatcher.

**ResolvedModel**:
The immutable construction identity that binds one source revision, configuration, tokenizer, weight set, dtype, and special-token contract for an `LLM` lifetime.
_Avoid_: resolved source, model snapshot, artifact bundle.

**LinearSpec**:
The neutral geometry parameter struct that `Linear<P>::from_vb` consumes — `{ in_features, out_features_per_shard, bias }`. Model code unpacks its own `Config` (e.g. `Qwen3Config`) into `LinearSpec`. Closes the ADR-0002 seam: `Linear<P>` (in shared `layers/`) stays fully model-agnostic — it never imports `models::qwen3::Qwen3Config` or any architecture-specific type.
_Avoid_: layer config, linear config, projection shape.

## Engine (single-GPU, offline)

**EngineCore**:
The in-process, single-GPU execution engine for one `StepPlan`. It returns one `StepResult`; lifecycle and scheduling decisions remain owned by the `Scheduler`.
_Avoid_: model runner (collapsed into EngineCore for single-GPU scope).

**BlockPool**:
Owns physical `Block`s (`block_id`, `ref_count`, `hash`, `token_ids`), the free-list deque, the used-set, and the prefix-cache hashtable (`hash_to_block_id`). Mirrors nano-vllm's xxhash-chained prefix-cache algorithm. CoW semantics for shared prefix blocks. `kvcache_block_size = 256` (constant).
_Avoid_: block manager (V0 term; V1 split BlockPool from KVCacheManager).

**KVCacheManager**:
Deliberate information-hiding adapter at the scheduler-facing seam.
The **only** Scheduler-facing seam over `BlockPool` + the physical
`PagedKVCache`. The Scheduler never imports `BlockPool` or
`attention::PagedKVCache` directly. Its value is what the Scheduler
cannot see — `BlockPool`, `BlockPoolError`, physical `PagedKVCache`
internals — not behavioural depth: 6 methods are one-line delegations
by design; `compute_slot_mapping` is the sole logic-carrying bridge.
Owns the mapping from logical block tables to physical block ids and
paged-cache slot indices.
_Avoid_: block-to-slot mapper (undersells the seam — the value is what
it hides, not what it maps).

**Scheduler**:
The sole owner of request lifecycle, admission, fairness, token and cache budgets, preemption, progress detection, and application of `StepResult`. It expresses executable work as immutable `StepPlan` values.
_Avoid_: batch scheduler, request scheduler.

**StepPlan**:
The immutable unit of schedulable model work, naming the participating requests, exact token ranges, causal positions, cache mappings, token budget, phase, and sampling permission.
_Avoid_: schedule output, execution batch.

**StepResult**:
The result corresponding to one `StepPlan`, applied exactly once by the `Scheduler` to advance request, cache, and completion state.
_Avoid_: postprocess output, step output.

**Sequence**:
The scheduler-owned lifecycle state for one request. Multi-candidate (`n > 1`) sampling remains deferred beyond v0.2.0 and would reintroduce grouping only when that capability exists.
_Avoid_: SequenceGroup (absorbed), request wrapper.

**Request identity**:
The stable public identifier assigned to one accepted prompt. It is independent of both the scheduler's internal sequence identity and the prompt's position in one input batch.
_Avoid_: sequence id, batch index.

**PagedKVCache**:
The physical GPU buffer shaped `[2, num_layers, num_blocks, 256, num_kv_heads, head_dim]`. Held as `Arc<Mutex<PagedKVCache>>` and shared between `EngineCore` and every attention layer. `reshape_and_cache` writes per-step K/V into the paged cache.
_Avoid_: block cache buffer, GPU cache pool.

**AttnMetadata**:
The logical attention description for one scheduled step, including cumulative lengths, slot mappings, and block tables. It is prepared once into an `AttentionContext`, not rebuilt independently by each layer.
_Avoid_: batch metadata.

**AttentionContext**:
The device-ready attention state shared across all model layers participating in one `StepPlan`. It owns the prepared form of `AttnMetadata` for that step.
_Avoid_: attention bundle, per-layer metadata.

## API surface

**Public generation contract**:
The supported v0.2.0 crate-root surface: `LLM`, `EngineOptions`, `Prompt`, `SamplingParams`, `RequestOutput`, and `Source`. Construction and offline generation are the public boundary; engine, model, cache, attention, loader, registry, and sampler mechanics remain internal.
_Avoid_: public engine API, public model API, compatibility exports.

**LLM::generate**:
The authoritative supported generation method — `generate(&mut self, prompts: &[Prompt], sampling_params: &[SamplingParams]) -> Result<Vec<RequestOutput>>`. `LLM::new` is the composition-root constructor; raw logits and internal execution controls are not part of this contract.
_Avoid_: run, infer, complete, __call__.

**Prompt**:
Input enum: `Text(String)` for natural-language prompts, `TokenIds(Vec<u32>)` for pre-tokenized fixtures. Both are accepted in the same batch.
_Avoid_: input, query, user message.

**RequestOutput**:
Per-request result: `{ request_id: usize, text: String, token_ids: Vec<u32>, finished: bool }`. The enclosing result vector preserves input-prompt order, while `request_id` remains stable independently of that call-local position.
_Avoid_: generation result, completion output.

**SamplingParams**:
The validated per-request token-selection and completion-stopping policy. One complete value remains associated with the stable request identity for the request lifetime rather than being reconstructed from scheduler scalars.
_Avoid_: generation config, decode params, sampling config.

**EngineOptions**:
Construction-time configuration for `LLM::new`: `max_num_batched_tokens` (default 16384), `max_num_seqs` (512), `max_model_len`, `gpu_memory_utilization` (0.9), `enforce_eager` (always true in v0.1), `dtype` override. Mirrors nano-vllm's `Config`.
_Avoid_: engine config, runtime options.

## Correctness

**Golden fixture**:
A content-addressed model-output record for a fixed input and oracle identity under one validation protocol. It supplies numerical and token evidence through the `Golden asset bundle`, rather than acting as an unversioned expected answer.
_Avoid_: reference output, expected output, snapshot, oracle output.

**Golden asset bundle**:
The immutable release pair containing one standalone manifest and one compressed archive of every declared `Golden fixture`. It is the only published transport for a golden version; individual fixture release assets are outside the contract.
_Avoid_: fixture pack, golden package, per-fixture assets.

**Reference oracle**:
The authoritative correctness target: Transformers BF16 with `output_logits=True` and `attn_implementation=sdpa`. A reference-oracle failure cannot be overridden by baseline evidence.
_Avoid_: ground truth, canonical engine, expected engine.

**Baseline oracle**:
The vLLM BF16 implementation whose distance from the `Reference oracle` defines the engineering comparison baseline. Its paired numerical results may supply an additional budgeted gate, but cannot waive an absolute reference or behavior failure.
_Avoid_: secondary oracle, calibration oracle.

**Tolerance policy**:
The approved, versioned operator and model error budgets for a fixed dtype, kernel scope and corpus. It separates absolute reference compatibility, paired baseline comparison and reference-relative token preference.
_Avoid_: hidden tolerance, pass-until-green threshold.

**Calibration observation**:
A non-accepting measurement of same-prefix numerical distributions and diagnostic evidence before the corresponding budgets are approved. Observation completion is not release acceptance.
_Avoid_: calibration pass, auto-tuned tolerance, baseline acceptance.

**Performance baseline**:
The reproducible, evidence-only measurements for fixed single-request and batch workloads: prefill throughput, decode throughput, time to first token, inter-token latency, and peak device memory. It is not a cross-hardware latency promise.
_Avoid_: benchmark gate, performance SLA, vLLM parity claim.

**Golden release evidence report**:
The committed Markdown record that binds the exact environment, kernels, corpus, approved tolerance, comparison totals, performance samples, limitations, and two release-asset hashes for `goldens-v0.2`. It is release evidence, not a third release asset.
_Avoid_: release manifest, benchmark artifact, extra golden asset.

**Operator verification (L0)**:
The validation of individual numerical operators and discrete execution-state invariants under controlled inputs and independent references. It is a verification tier, not a Transformer layer index.
_Avoid_: layer-zero model trace, whole-model accuracy pass.

**Model numerical verification (L1)**:
The comparison of complete model predictions conditioned on identical token histories under the approved absolute and paired-baseline budgets. It includes logit and probability-distribution evidence, not an L1 norm.
_Avoid_: token-only L1, full-model proof from one layer.

**Decoding and behavior verification (L2)**:
The validation of predicted token choices and public generation outcomes under the approved reference-choice and engine-behavior contracts. It is not an L2 norm or an unversioned promise of identical cross-kernel strings.
_Avoid_: logits-only L2, token match as complete numerical proof.

**Near-tie classification (L2)**:
A differing token choice evaluated against the reference's preference under the versioned `Tolerance policy`. It is an explicit result, never a skipped comparison or a classification borrowed from the baseline oracle.
_Avoid_: near-tie skip, epsilon skip, close-call skip.

**Same-prefix comparison**:
A numerical comparison whose corresponding predictions have identical causal token histories. Different-history outputs are not comparable rows, whether the histories arose from free generation or controlled replay.
_Avoid_: prefix-aware L2, context-aware comparison.

**Numerical case**:
One registered request member with its own fixed prediction history and numerical verdict. Batch peers are separate cases even when they share an execution.
_Avoid_: batch-average acceptance unit, token row as independent case.

**Execution group**:
A registered set of request members executed together in one generation call. It captures shared execution behavior without replacing the individual numerical cases.
_Avoid_: sequential calls labeled as batch, pooled numerical verdict.

**Fixed-prefix replay**:
A run whose continuation token stream is fixed independently of each implementation's predicted choices. It preserves shared histories while exercising the actual incremental engine path.
_Avoid_: forced prediction, repeated prefill as decode.

**Predicted token**:
The choice produced from an implementation's unmodified logits and declared sampling policy. In fixed-prefix replay it is distinct from the token used to advance the history.
_Avoid_: forced token as model output evidence.

**Advance token**:
The frozen continuation token appended to the logical history during fixed-prefix replay. It names the controlled history, not the model's original preference.
_Avoid_: predicted token, raw argmax.

**Release gate vs CI gate**:
CI (every push, CPU-only) runs property tests → "CI green". Release gate (pre-release, manual, GPU) runs golden comparison → "numerically validated". These are explicitly different — the repo README must document that CI green ≠ validated.
_Avoid_: CI vs release check.
