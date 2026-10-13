# Architecture — v0.1.0

[Documentation](README.md) · [Domain vocabulary](../CONTEXT.md) · [Validation](validation/README.md)

The historical engine is a synchronous, single-process, single-GPU Qwen3 stack. It uses pinned Candle tensors and FlashAttention APIs, plus project CUDA kernels. Architectural references are [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [vLLM](https://github.com/vllm-project/vllm).

## Request flow

`LLM::new` resolves model files, tokenizer, architecture and dtype, creates the model/device, runs warmup, then sizes and shares a paged KV cache. `LLM::generate` tokenizes prompts, adds requests, loops synchronous engine steps, and decodes completed tokens.

| Module | Responsibility |
|---|---|
| `llm` | Composition root, input conversion, generation loop and device/memory setup |
| `config` / `loader` | Local/Hub sources, config dtype and safetensors loading |
| `models` | Architecture registry and Qwen3 causal model |
| `layers` | Shared linear, RMSNorm, RoPE and activation operations |
| `attention` / `kernels` | Attention metadata, paged physical cache, cache writes and CUDA kernels |
| `engine` | Sequences, block ownership, KV allocation, scheduling and step execution |
| `sampler` | Greedy and stochastic sampling with configured filters/penalties |

The scheduler operates through `KvCacheManager` and tracks waiting/running sequences. A scheduled step is prefill or decode; preemption deallocates and recomputes rather than swapping to CPU. The fixed cache block size is 256 tokens. The physical paged buffer layout is `[2, layers, blocks, 256, kv_heads, head_dim]`, shared between attention layers and the engine.

## Historical API and seams

This source publicly re-exports more internals than v0.2.0, including scheduler/cache/model utilities. The intended user-facing entry is `LLM::new` / `LLM::generate`, with `Source`, `EngineOptions`, `Prompt`, `SamplingParams` and `RequestOutput`. `RequestOutput` uses `seq_id`; the modern `request_id` name and repeated-generation guarantees do not apply to this archive.

The model-facing `CausalLM` separates `forward` hidden states from `compute_logits`. Projection helpers preserve `ParallelStyle` and `TpConfig` seams, but only the single-rank path is implemented. References to future TP/NCCL, CUDA Graphs, model introspection, embeddings or classifiers in historical comments are plans, not available features or current-release promises.

The v0.2.0 internal `StepPlan` / `StepResult` execution contract is absent from v0.1.0. This version uses its original scheduling data and `EngineCore` step loop; documentation modernization does not backport newer engine behavior.

## Validation and experiments

The [generator](../tools/golden-gen/README.md) writes schema-1 Transformers/vLLM fixtures. The [Rust tool](../crates/vllm-oxide-test/README.md) performs historical L1/L2 comparisons with known coverage and skip limitations. L3 is a stub. There is no modern layered schema-5 evaluator or validated performance/memory sampler in this archive.

The [research prototype](../research/prototype-candle-stack/README.md) separately checks tensor-to-CUDA FFI and a paged FlashAttention call. It loads no Qwen3 weights and does not validate engine numerical correctness. The [ADR](adr/0005-golden-generation-correctness-strategy.md) records the original golden strategy; current implementation gaps are explained in [validation](validation/README.md).
