# vllm-oxide

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/RedHeartSecretMan/vllm-oxide)](https://github.com/RedHeartSecretMan/vllm-oxide/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

vllm-oxide is a Rust engine for **single-GPU, synchronous, offline Qwen3 inference**, built on [Candle](https://github.com/huggingface/candle) and project-owned CUDA kernels. It draws on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [vLLM](https://github.com/vllm-project/vllm)'s architecture, with continuous batching, prefix caching, a paged KV cache, and recompute preemption.

**This checkout is the historical experimental prerelease v0.2.0.rc.1.** It targets **Qwen3-0.6B in BF16** on one CUDA GPU. The original numerical run recorded **48/66 PASS and 18 FAIL**, with full performance and Rust-minimum validation incomplete. New users should use the final [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0). See the [release notes](docs/releases/v0.2.0-rc.1.en.md), [historical validation](docs/validation/README.md), and [GitHub Release](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0.rc.1).

[Getting started](docs/getting-started.md) · [Documentation](docs/README.md) · [Architecture](docs/architecture.md) · [Releases](docs/releases/README.md) · [Contributing](CONTRIBUTING.md)

## What you can build

Embed the engine in a Rust application, submit a batch of text or token-ID prompts, and receive completed text and token IDs. Reuse one `LLM` across calls to retain its model and cache allocation. The CLI offers the same generation path for one prompt at a time.

| Area | RC1 implementation |
|---|---|
| Model and execution | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), BF16, one CUDA GPU, synchronous offline generation |
| Request handling | Continuous batching within a generation call, chunked prefill, stable request IDs, input-order results |
| KV cache | Paged storage, shared-prefix caching, copy-on-write, recompute preemption |
| Token selection | Greedy, temperature, top-k, top-p, presence/frequency/repetition penalties, EOS and token-limit stopping |
| Distribution | Tagged source, a Rust library, and a CLI; no prebuilt binary or published crates.io package |

Serving, async streaming, multi-GPU/TP/NCCL, other model families, quantization, LoRA, speculative decoding, and CUDA Graphs are outside this release. Architecture inspiration does not imply API compatibility or feature parity with upstream vLLM.

## Quick start

GPU inference needs Linux or WSL2, the recorded Rust **1.94.0** toolchain, an NVIDIA GPU with compute capability **8.9 or newer**, a working NVIDIA driver, and a CUDA Toolkit containing `nvcc`. RC results were recorded on RTX 4080/WSL2. Use a compatible CUDA driver and full Toolkit containing `nvcc`; the maintained source now pins Rust 1.94.0. Original RC measurements remain separate from the maintenance checks.

Clone the released source and run the provided example with the exact model revision used for validation:

```bash
git clone --branch v0.2.0.rc.1 --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
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
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", tag = "v0.2.0.rc.1", features = ["cuda"] }
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

## Validation and performance

RC1 remains an experimental historical candidate: 487 capture items, 153 groups, seven operator profiles and 22 public behavior checks were checked, but **18 numerical cases failed**. No complete performance result or production accuracy guarantee is claimed. Read the [original result and provenance](docs/releases/v0.2.0-rc.1.en.md#validation-and-performance). This documentation update keeps implementation, numerical budgets, evidence identities and verdicts unchanged; see the [maintenance record](docs/version-doc-update.md).

## Development and validation

CPU development does not require CUDA:

```bash
cargo build --locked --workspace
cargo test --locked --workspace
```

A CPU build supports development and tests; **actual inference requires `--features cuda` and a GPU**. This checkout's automatic GitHub CI checks formatting, Clippy, Rust tests and dependencies on CPU. Python tooling has its own local CPU tests. Current `main` has newer CI; this source snapshot does not claim those later jobs. Numerical accuracy, CUDA API/CLI behavior, and performance are validated separately on a local GPU. CI success alone does not establish model accuracy.

For contribution setup and complete checks, read [CONTRIBUTING.md](CONTRIBUTING.md) and the [development guide](docs/development.md). The [architecture overview](docs/architecture.md) maps a generation call to the current implementation, and the [validation guide](docs/validation/README.md) separates routine CI from real-model checks.

## Community and project information

| Looking for | Start here |
|---|---|
| Usage help, bugs, or a feature proposal | [Support](SUPPORT.md) and [GitHub Issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues) |
| A contribution | [Contributing](CONTRIBUTING.md) and [open pull requests](https://github.com/RedHeartSecretMan/vllm-oxide/pulls) |
| Community expectations | [Community standards](CODE_OF_CONDUCT.md) |
| A private security report | [Security policy](SECURITY.md) |
| Version history or an upgrade | [Releases](docs/releases/README.md), [changelog](CHANGELOG.md), and [v0.2 migration](docs/migration-v0.2.0.md) |
| Implementation decisions and experiments | [Architecture](docs/architecture.md), [ADRs](docs/adr/), and [research](research/README.md) |

## License

[Apache-2.0](LICENSE).
