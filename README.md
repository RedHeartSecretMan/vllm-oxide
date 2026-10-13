# vllm-oxide

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

vllm-oxide is a Rust engine for **single-GPU, synchronous, offline Qwen3 inference**, built on [Candle](https://github.com/huggingface/candle) and project-owned CUDA kernels. It draws on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and vLLM's architecture, with continuous batching, prefix caching, a paged KV cache, and recompute preemption. This project is maintained independently of upstream vLLM.

**v0.2.0 is available as source.** The validated workload is **Qwen3-0.6B in BF16** on one CUDA GPU. See the [release notes](docs/releases/v0.2.0.en.md), [validation summary](docs/releases/goldens-v0.2.md), and [GitHub Release](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0).

[Getting started](docs/getting-started.md) · [Documentation](docs/README.md) · [Upgrade from v0.1](docs/migration-v0.2.0.md) · [Contributing](CONTRIBUTING.md)

## Quick start

GPU inference needs Linux or WSL2, Rust **1.94 or newer**, an NVIDIA GPU with compute capability **8.9 or newer**, a working NVIDIA driver, and a CUDA Toolkit containing `nvcc`. The release was tested on RTX 4080 with driver 595.71 and Toolkit 13.2.51; other hardware and toolkit combinations have not received the same release validation.

Clone the released source and run the provided example with the exact model revision used for validation:

```bash
git clone --branch v0.2.0 --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
cd vllm-oxide
cargo run --locked --release -p vllm_oxide --example generate_qwen3 --features cuda -- \
    hub:Qwen/Qwen3-0.6B@7e4ae267688d671ddfca3122e4528ee980cf3234
```

The first run builds the CUDA backend and downloads model files. The example generates a continuation of “The meaning of life is”. Later runs reuse downloaded files. A local model directory can be passed instead of the `hub:` argument.

To choose your own prompt, use the CLI with a local checkpoint:

```bash
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
cargo run --locked --release -p vllm_oxide_cli --features cuda -- \
    --model "$QWEN3_MODEL_DIR" --max-tokens 2 "The capital of France is"
```

The CLI also accepts a Hub repo ID, such as `Qwen/Qwen3-0.6B`, but does not expose a revision flag. See [getting started](docs/getting-started.md) for CLI options, local checkpoints, and offline use.

## Use from Rust

Pin the source release in your `Cargo.toml`:

```toml
[dependencies]
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", tag = "v0.2.0", features = ["cuda"] }
anyhow = "1"
```

```rust
use vllm_oxide::{EngineOptions, Prompt, SamplingParams, Source, LLM};

fn main() -> anyhow::Result<()> {
    let mut llm = LLM::new(
        Source::Hub {
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: Some("7e4ae267688d671ddfca3122e4528ee980cf3234".into()),
        },
        EngineOptions::default(),
    )?;
    let outputs = llm.generate(
        &[Prompt::Text("The capital of France is".into())],
        &[SamplingParams {
            max_tokens: 2,
            ..Default::default()
        }],
    )?;
    println!("{}", outputs[0].text);
    Ok(())
}
```

The supported public types are `LLM`, `EngineOptions`, `Prompt`, `SamplingParams`, `RequestOutput`, and `Source`. `generate` accepts one sampling policy per prompt and returns outputs in input order with stable request IDs. Text and token-ID prompts can share a batch. See the [usage guide](docs/getting-started.md#rust-api) and [v0.2 migration guide](docs/migration-v0.2.0.md) for details.

## Scope and performance

v0.2.0 provides an in-process generation API and a thin CLI. Serving APIs, async streaming, multiple GPUs/TP/NCCL, other model families, quantization, LoRA, speculative decoding, and CUDA Graphs are outside this release.

The implementation prioritizes numerical correctness. In the fixed RTX 4080/WSL2 release workload, median decode throughput was **29.7601 token/s for one request** and **115.8207 token/s across four requests**. These are reproducible baseline observations, not cross-hardware guarantees or a claim of vLLM performance parity. Default cache sizing also reserves a substantial share of free GPU memory. See the [validation summary](docs/releases/goldens-v0.2.md) for workloads, memory measurements, and limitations.

## Development and validation

CPU development does not require CUDA:

```bash
cargo build --locked --workspace
cargo test --locked --workspace
```

A CPU build supports development and tests; **actual inference requires `--features cuda` and a GPU**. Automatic GitHub CI checks compilation, formatting, Clippy, Rust tests, Python CPU validation tooling, and dependencies. Numerical accuracy, CUDA API/CLI behavior, and performance are validated separately on a local GPU. CI success alone does not establish model accuracy.

For contribution setup and complete checks, read [CONTRIBUTING.md](CONTRIBUTING.md). See [SUPPORT.md](SUPPORT.md) for usage questions, use [GitHub Issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues) for bugs and proposals, and follow [SECURITY.md](SECURITY.md) for security reporting. Architecture decisions and domain vocabulary are linked from the [documentation index](docs/README.md).

## License

[Apache-2.0](LICENSE).
