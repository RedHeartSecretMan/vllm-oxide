# vllm-oxide v0.1.0

**English** | [简体中文](README.zh-CN.md)

[Release](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.1.0) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) · [Support](SUPPORT.md)

vllm-oxide is a Rust engine for **single-GPU, synchronous, offline Qwen3 inference**, built on [Candle](https://github.com/huggingface/candle) and project-owned CUDA kernels. Its architecture draws on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [vLLM](https://github.com/vllm-project/vllm).

This is the **historical experimental v0.1.0 source release**. New users should start with [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0) and its [getting-started guide](https://github.com/RedHeartSecretMan/vllm-oxide/blob/v0.2.0/docs/getting-started.md). v0.1.0 recorded token differences in four of eight evaluated groups and skipped comparisons; its original results remain unchanged.

## What this version contains

- A synchronous `LLM::new` / `LLM::generate` Rust API and a thin offline CLI.
- Single-GPU Qwen3 model loading, continuous batching, a paged KV cache, prefix caching and recompute preemption.
- Greedy, top-k and top-p sampling, with per-prompt settings in the Rust API.
- Transformers/vLLM fixture generation and a historical Rust comparison tool.

The recorded model was **Qwen3-0.6B/BF16 on RTX 4080/WSL2**. Serving, async/streaming APIs, multi-GPU/TP/NCCL, other model families, quantization, LoRA, speculative decoding and CUDA Graphs are outside this version's supported scope. CPU builds support development and tests, not CPU inference.

## Get started

From the repository root, with CUDA Toolkit, a compatible driver and an SM 8.9+ NVIDIA GPU:

```bash
cargo build --locked --release -p vllm_oxide_cli --features vllm_oxide/cuda
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
target/release/vllm_oxide_cli --model "$QWEN3_MODEL_DIR" \
    --max-tokens 16 "The meaning of life is"
```

The historical CLI has **no `cuda` feature of its own**; the qualified `vllm_oxide/cuda` flag enables its library dependency. See the [version-specific guide](docs/getting-started.md) for model files, flags, Rust API and troubleshooting. `Cargo.toml` declares Rust 1.75; that historical minimum is not an independently verified compatibility guarantee.

## Documentation and development

| Task | Start here |
|---|---|
| Use the historical source | [Getting started](docs/getting-started.md) |
| Understand the engine and API | [Architecture](docs/architecture.md) and [domain vocabulary](CONTEXT.md) |
| Contribute or run CPU checks | [Contributing](CONTRIBUTING.md) and [development](docs/development.md) |
| Inspect fixtures and comparisons | [Tools](tools/README.md), [golden-gen](tools/golden-gen/README.md), [Rust comparison tool](crates/vllm-oxide-test/README.md) |
| Explore the stack experiment | [Research](research/README.md) |
| Inspect release scope and original assets | [Release notes](docs/releases/v0.1.0.en.md) |
| Understand the source documentation refresh | [Documentation maintenance](docs/version-doc-update.md) |

CPU test success does not establish GPU numerical correctness. The [validation guide](docs/validation/README.md) preserves the actual historical failures, comparison coverage limitations and the absence of a validated performance baseline.

The workspace declares [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) licensing. Please follow the [security policy](SECURITY.md) when reporting vulnerabilities.
