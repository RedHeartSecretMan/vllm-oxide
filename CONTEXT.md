# vllm-oxide

A large language model inference engine written in Rust, drawing on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [vLLM](https://github.com/vllm-project/vllm). The v0.2.0 implementation supports synchronous, offline Qwen3 inference on one CUDA GPU, validated with Qwen3-0.6B/BF16. This file defines repository vocabulary; start with the [architecture guide](docs/architecture.md) for a walkthrough and the [release notes](docs/releases/v0.2.0.en.md) for shipped behavior.

## Language

**Column-parallel (projection)**:
A linear projection whose weight is sharded along the output dimension (dim 0). The matmul produces a complete-in-input, partial-in-output result; reconstruction across TP ranks is by all-gather (concatenation), not reduce.
_Avoid_: output-sharded, column-shard.

**Row-parallel (projection)**:
A linear projection whose weight is sharded along the input dimension (dim 1). The matmul produces a partial sum (partial-in-input, complete-in-output); reconstruction across TP ranks is by all-reduce (summation). Bias is fused into the matmul only on rank 0 to avoid multiplication under all-reduce.
_Avoid_: input-sharded, row-shard.

**Fused projection**:
Multiple sub-projections (Q/K/V for attention, or gate/up for SwiGLU MLP) stored together in one weight matrix. Packing describes storage; each sub-projection may execute independently. At TP=1 the layout is `[Q | K | V]` or `[gate | up]` along dim 0.
_Avoid_: packed projection, merged matmul.

**TP seam**:
The abstraction boundary where future tensor-parallelism wiring can attach without rewriting model code. `TpConfig::Single` is the supported identity path; `TpConfig::Sharded` names a feasibility seam, not supported v0.2.0 TP/NCCL behavior.
_Avoid_: TP interface, parallelism hook.

**Paged attention**:
An attention computation that reads K/V from a paged block cache rather than contiguous per-sequence buffers. Continued prefill and decode use the complete cached key/value history, while initial prefill can use its projections directly.
_Avoid_: paged KV cache attention, block attention.

**CausalLM (trait)**:
The engine-facing contract for causal text generation. New task families use independent traits rather than overloading `CausalLM`; v0.2.0 remains limited to causal generation.
_Avoid_: model interface, Model trait, CausalLM struct.

**Model registry (inventory)**:
The map from HF architecture strings (`"Qwen3ForCausalLM"`) to factory functions producing `Box<dyn CausalLM>`. Implemented as an `inventory`-distributed static registry: each model file self-registers via `inventory::submit! { ModelEntry { arch, factory } }`, and `registry.rs` is a pure query function with no model-specific knowledge. Keyed off `config.json["architectures"][0]`. Adding an architecture requires a new model file and a `mod xxx;` declaration; existing model implementations and registry lookup stay unchanged.
_Avoid_: dispatcher, factory table.

**Weight loader**:
The internal component that resolves checkpoint artifacts and exposes their weights for model construction. The `model-loader` wording in v0.2 tracker prose names this component, not the `Model registry`.
_Avoid_: model registry, dispatcher.

**ResolvedModel**:
The immutable construction identity that binds one source revision, configuration, tokenizer, weight set, dtype, and special-token contract for an `LLM` lifetime.
_Avoid_: resolved source, model snapshot, artifact bundle.

**LinearSpec**:
The neutral geometry struct consumed by `Linear<P>::from_vb`: `{ in_features, out_features_per_shard, bias }`. Model code converts its own `Config`, such as `Qwen3Config`, into `LinearSpec`. This implements the ADR-0002 boundary: `Linear<P>` in shared `layers/` is model-agnostic and imports neither `models::qwen3::Qwen3Config` nor any other architecture-specific type.
_Avoid_: layer config, linear config, projection shape.

## Engine (single-GPU, offline)

**EngineCore**:
The in-process, single-GPU execution engine for one `StepPlan`. It returns one `StepResult`; lifecycle and scheduling decisions remain owned by the `Scheduler`.
_Avoid_: model runner (collapsed into EngineCore for single-GPU scope).

**BlockPool**:
Owns physical `Block`s (`block_id`, `ref_count`, `hash`, `token_ids`), the free-list deque, the used-set, and the prefix-cache hashtable (`hash_to_block_id`). Mirrors nano-vllm's xxhash-chained prefix-cache algorithm. CoW semantics for shared prefix blocks. `kvcache_block_size = 256` (constant).
_Avoid_: block manager (V0 term; V1 split BlockPool from KVCacheManager).

**KvCacheManager**:
The scheduler-facing boundary over `BlockPool` and the physical `PagedKVCache`.
It owns allocation/release operations and translates logical block tables into
physical cache slots. The Scheduler uses this boundary without importing
`BlockPool` or `attention::PagedKVCache` directly. Older design records spell
the conceptual name `KVCacheManager`; the Rust type is `KvCacheManager`.
_Avoid_: block-to-slot mapper (omits allocation and ownership responsibilities).

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
The public generation method: `generate(&mut self, prompts: &[Prompt], sampling_params: &[SamplingParams]) -> Result<Vec<RequestOutput>>`. `LLM::new` is the composition-root constructor; raw logits and internal execution controls are not part of this contract.
_Avoid_: run, infer, complete, __call__.

**Prompt**:
Input enum: `Text(String)` for natural-language prompts and `TokenIds(Vec<u32>)` for pre-tokenized prompts. Both are accepted in the same batch.
_Avoid_: input, query, user message.

**RequestOutput**:
Per-request result: `{ request_id: usize, text: String, token_ids: Vec<u32>, finished: bool }`. The enclosing result vector preserves input-prompt order, while `request_id` remains stable independently of that call-local position.
_Avoid_: generation result, completion output.

**SamplingParams**:
The validated per-request token-selection and completion-stopping policy. One complete value remains associated with the stable request identity for the request lifetime rather than being reconstructed from scheduler scalars.
_Avoid_: generation config, decode params, sampling config.

**EngineOptions**:
Construction-time configuration for `LLM::new`: a shared prefill/decode token budget (`max_num_batched_tokens`, default 16384), concurrency (`max_num_seqs`, 512), context length (`max_model_len`, 4096), KV allocation fraction (`gpu_memory_utilization`, 0.9), eager execution (`enforce_eager` must be true), and a checkpoint dtype override.
_Avoid_: engine config, runtime options.

## Correctness

**Golden fixture**:
A content-addressed model-output record for a fixed input and oracle identity under one validation protocol. It supplies numerical and token evidence through the `Golden asset bundle`, rather than acting as an unversioned expected answer.
_Avoid_: reference output, expected output, snapshot, oracle output.

**Golden asset bundle**:
The optional immutable transport pair containing one standalone `Golden release manifest` and one compressed archive of the complete declared evidence. Individual fixture assets are outside this transport contract. The v0.2.0 product release keeps raw evidence local and publishes source archives; it does not publish a golden bundle or claim completion of the separate bundle-consumer gate.
_Avoid_: fixture pack, golden package, per-fixture assets.

**Golden release manifest**:
The versioned inventory binding a golden version to its archive, original evidence identities and validation entrypoints. It describes published transport, not the result of a measurement stage.
_Avoid_: execution manifest, second manifest wrapper.

**Execution evidence manifest**:
The source-bound record of one validation stage and its original evidence dependencies. It remains part of the evidence even when a release transports several stages together.
_Avoid_: release manifest, asset inventory.

**Reference oracle**:
The authoritative correctness target: Transformers BF16 with `output_logits=True` and `attn_implementation=sdpa`. Its approved arithmetic profile fixes intermediate precision; a different profile requires new calibration and scope approval. A reference-oracle failure cannot be overridden by baseline evidence.
_Avoid_: ground truth, canonical engine, expected engine.

**Baseline oracle**:
The vLLM BF16 implementation whose distance from the `Reference oracle` defines the engineering comparison baseline. Its paired numerical results may supply an additional budgeted gate, but cannot waive an absolute reference or behavior failure.
_Avoid_: secondary oracle, calibration oracle.

**Tolerance policy**:
The approved, versioned operator and model error budgets for a fixed dtype, kernel scope and corpus. It separates absolute reference compatibility, paired baseline comparison and reference-relative token preference.
_Avoid_: hidden tolerance, pass-until-green threshold.

**Calibration observation**:
A non-accepting measurement of same-prefix numerical distributions and diagnostic evidence before the corresponding scope and budget bindings are approved. Observation completion is not release acceptance.
_Avoid_: calibration pass, auto-tuned tolerance, baseline acceptance.

**Independent confirmation**:
A predeclared set of previously unobserved cases used after a failure has informed a repair. It supplements every observed regression obligation; repeating the failed cases does not restore their independence.
_Avoid_: renamed holdout, optional confirmation report, repeated blind test.

**Performance baseline**:
The reproducible, evidence-only measurements for fixed single-request and batch workloads: prefill throughput, decode throughput, time to first token, inter-token latency, and peak device memory. It is not a cross-hardware latency promise.
_Avoid_: benchmark gate, performance SLA, vLLM parity claim.

**Golden release evidence report**:
The committed Markdown summary of actual validation sources, environment, kernels, corpus, budgets, comparison totals, performance samples and limitations. Original raw evidence retains its identities locally; any separately published transport records its own archive and manifest hashes.
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

**Confirmation cohort**:
A frozen group of confirmation cases assigned a role for one Definition Checkpoint. Retained cohorts remain mandatory regressions; one independently frozen cohort supplies the new unobserved holdout. Every case keeps its original split, input and owner identity, and every cohort must pass. The independent cohort must share no reachable prediction history with any prior case or cohort.
_Avoid_: acceptance reset, replacement prompts, renamed historical split.

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
CPU CI runs project checks on every push. Manual GPU release checks validate model results through golden comparison. The README must distinguish passing CI from completed numerical validation.
_Avoid_: CI vs release check.
