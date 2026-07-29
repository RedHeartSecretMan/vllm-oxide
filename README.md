# vllm-oxide

**English** | [简体中文](README.zh-CN.md)

[![CI][ci-badge]][ci-url]
[![License: Apache-2.0][license-badge]][license-url]

[ci-badge]: https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml
[license-badge]: https://img.shields.io/badge/license-Apache--2.0-blue.svg
[license-url]: LICENSE

A Rust port of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) trending toward vLLM's V1 architecture.

- **Single-GPU, offline inference** — no server, no async. Continuous batching, prefix caching, paged KV cache, and recompute-only preemption in a synchronous engine.
- **Architecture-agnostic model registry** — Qwen3-first today; adding a new architecture is one file plus one `mod` declaration, zero changes to existing code.
- **Two-tier correctness** — CI property tests catch regressions fast; a GPU release gate with golden fixtures validates numerical output against a transformers oracle.

[Architecture](#architecture-overview) | [Quick Start](#quick-start) | [Testing](#testing) | [Contributing](#contributing)

## Table of Contents

- [What is this?](#what-is-this)
- [Architecture overview](#architecture-overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Library usage](#library-usage)
- [Build features](#build-features)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Minimum Supported Rust Version (MSRV)](#minimum-supported-rust-version-msrv)
- [Security](#security)
- [License](#license)

---

## What is this?

vllm-oxide brings LLM inference to the Rust ecosystem. Built on [candle](https://github.com/huggingface/candle) (CUDA kernels, safe tensor ops) and flash-attention (paged attention kernels), it provides a synchronous, in-process engine with:

- Continuous batching and prefix caching
- Paged KV cache (`block_size = 256`)
- Recompute-only preemption

v0.1 targets **single-GPU, offline inference** (no server, no async). The engine is Qwen3-first but the model registry is architecture-agnostic: adding a new architecture means adding one file plus one `mod` declaration, with zero changes to existing code.

### Project goals

- A drop-in replacement for Python inference engines in production — correct first, then fast.
- Architecture decisions recorded as ADRs (`docs/adr/`), domain vocabulary documented in `CONTEXT.md`.

## Architecture overview

```
┌──────────────────────────────────────────────────────┐
│                   LLM (composition root)              │
│  LLM::new(source, opts) → LLM::generate(prompts, …)  │
└────────────────────┬─────────────────────────────────┘
                     │ owns
┌────────────────────▼─────────────────────────────────┐
│                    EngineCore                          │
│  Scheduler → Blocks → KVCacheManager → PagedKVCache  │
│  ↓                                                     │
│  model.forward() → compute_logits() → Sampler         │
│  ↓                                                     │
│  detokenize → RequestOutput                            │
└────────────────────────────────────────────────────────┘
```

The engine runs a synchronous `step()` loop: schedule tokens, prepare tensors, run the model forward pass (hidden states), compute logits from the last-token hidden state, sample the next token, update the KV cache, and repeat until all sequences finish.

Key design decisions (see `CONTEXT.md` for the full vocabulary):

- **Paged attention**: K/V cache stored in fixed-size blocks (`block_size = 256`). Prefill uses unpaged `flash_attn_varlen`; decode uses paged `flash_attn_varlen_paged_windowed`.
- **Prefix caching**: Chained XXH64 hash table in `BlockPool` deduplicates common prompt prefixes across requests (CoW semantics).
- **TP seam**: The `ParallelStyle` trait + `TpConfig` enum make future tensor-parallelism wiring additive. v0.1 uses `TpConfig::Single` exclusively.
- **CausalLM trait**: Engine-facing model contract — `forward(&mut self, input_ids, positions) -> hidden_states` + `compute_logits(hidden) -> logits`. A model registry (inventory-based) maps HF architecture strings to factory functions producing `Box<dyn CausalLM>`.

## Requirements

### Hardware

- **CPU-only** (tests, development): any x86-64 or aarch64 machine. No GPU needed.
- **Inference / release gate** (with `--features cuda`):
  - NVIDIA GPU with compute capability **sm_89** or higher (Ada Lovelace RTX 40-series, Hopper H100/H200, or newer).
  - At least 8 GB of GPU memory recommended for Qwen3-0.6B.
  - CUDA driver installed (tested with CUDA 12.x and 13.2).

### Toolchain

- **Rust**: edition 2021, rust-version 1.75+ (as declared in [workspace.package]).
- **System**: Linux (the only platform with NVIDIA CUDA support). No Windows or macOS GPU support in v0.1.

## Quick Start

### Build

```bash
# CPU-only build (tests, development iteration)
cargo build

# Production build with CUDA backend
cargo build --features cuda --release
```

### Run the CLI

The thin CLI (`crates/vllm-oxide-cli`) accepts a model source and an optional prompt:

```bash
cargo run --release -p vllm_oxide_cli --features cuda -- \
    --model Qwen/Qwen3-0.6B \
    "The meaning of life is"
```

If no prompt is given on the command line, the CLI reads from stdin:

```bash
echo "The meaning of life is" | \
    cargo run --release -p vllm_oxide_cli --features cuda -- \
        --model Qwen/Qwen3-0.6B
```

#### CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model` | Local checkpoint directory _or_ HuggingFace Hub repo id (e.g. `Qwen/Qwen3-0.6B`). Existing directories resolve to local checkpoints; everything else resolves to the Hub. | (required) |
| `prompt` (positional) | Prompt text. Reads from stdin when not provided. | stdin |
| `--temperature` | Sampling temperature. `0` = greedy (deterministic). | `0` |
| `--top-k` | Top-k sampling: keep only the `k` highest-logit tokens. | `None` (disabled) |
| `--top-p` | Top-p (nucleus) sampling: keep smallest token set with cumulative probability >= `p`. | `None` (disabled) |
| `--max-tokens` | Maximum tokens to generate. | `16` |

### Running examples

```bash
# Load and inspect a Qwen3 checkpoint (weights, dtype, shards)
cargo run --release --example load_qwen3 --features cuda -- hub:Qwen/Qwen3-0.6B

# Run a forward pass on dummy input
cargo run --release --example forward_qwen3 --features cuda -- hub:Qwen/Qwen3-0.6B
```

The examples accept `hub:<repo>` and `hub:<repo>@<revision>` URLs, or a local directory path.

## Library usage

Add `vllm_oxide` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", features = ["cuda"] }
```

The main API is `LLM::generate`, which accepts batched prompts and per-prompt sampling parameters:

```rust
use vllm_oxide::{LLM, Prompt, SamplingParams, EngineOptions, Source};

fn main() -> anyhow::Result<()> {
    // Build the engine from a HuggingFace Hub repo.
    let mut llm = LLM::new(
        Source::Hub {
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: None,
        },
        EngineOptions::default(),
    )?;

    // Run inference on a batch of prompts.
    let outputs = llm.generate(
        &[
            Prompt::Text("The meaning of life is".into()),
            Prompt::Text("Once upon a time".into()),
        ],
        &[
            SamplingParams {
                max_tokens: 64,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                max_tokens: 32,
                temperature: 0.0, // greedy
                ..Default::default()
            },
        ],
    )?;

    for output in outputs {
        println!("[{}] {} (finished: {})", output.seq_id, output.text, output.finished);
    }

    Ok(())
}
```

### Key types

| Type | Description |
|------|-------------|
| `LLM` | Composition root. Constructed via `LLM::new(source, options)`, invoked via `LLM::generate(prompts, params)`. |
| `Prompt` | Input enum: `Text(String)` for natural-language prompts, `TokenIds(Vec<u32>)` for pre-tokenized fixtures. Both are accepted in the same batch. |
| `SamplingParams` | Per-prompt configuration: `temperature`, `top_k`, `top_p`, `max_tokens`, `ignore_eos`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`. Default is greedy (temperature=0). |
| `RequestOutput` | Per-request result: `{ seq_id, token_ids, text, finished }`. Both decoded text and raw token IDs are always provided. |
| `EngineOptions` | Construction-time config: `max_num_batched_tokens` (default 16384), `max_num_seqs` (512), `max_model_len`, `gpu_memory_utilization` (0.9), `enforce_eager` (always true in v0.1), `dtype` override. |
| `Source` | Weight source: `Source::Local(PathBuf)` for a local directory, or `Source::Hub { repo, revision }` for HuggingFace Hub. |

## Build features

vllm-oxide uses a `cuda` feature gate to separate CPU-only development from GPU inference:

```toml
[features]
default = []        # CPU-only — tests and dev iteration run without CUDA.
cuda = ["dep:candle-flash-attn", "candle-core/cuda"]  # Production backend.
```

Default is CPU-only so `cargo test` runs on CI without a GPU. Production callers (the CLI, the engine) pass `--features cuda`.

## Testing

vllm-oxide has two distinct test tiers with different guarantees:

### Tier 1: CI gate (every push, CPU-only)

```bash
# Unit tests, property tests — no GPU required
cargo test
```

Covers `EngineOptions` defaults, `Prompt` variants, `SamplingParams` validation, config parsing, `Source` classification, and CLI argument parsing.

### Tier 2: Release gate (manual, GPU)

The release gate validates the Rust engine's numerical output against golden fixtures. It requires a GPU (sm_89+), model weights, and a golden fixture archive.

```bash
# L1 token-sequence exact match + L2 logits comparison
cargo run --release -p vllm_oxide_test --features cuda -- \
    --model-path /path/to/Qwen3-0.6B \
    --release-tag goldens-v0.1
```

**What it checks:**

| Layer | What | How |
|-------|------|-----|
| **L1** | Greedy token-sequence exact match | Compares generated token IDs against golden token IDs, position by position. Positions where the top-2 logit gap is within epsilon (2x calibrated atol) are skipped as BF16 precision artifacts. |
| **L2** | Per-step logits tensor comparison | Runs `LLM::generate_logits` and compares raw pre-sampling logits against golden logits using calibrated absolute tolerance (`atol`). Only compares steps where the token sequence matches (same-prefix comparison). |
| **L3** | Per-layer activations (debug) | Skeleton in v0.1. |

Golden fixtures are produced by `tools/golden-gen/` (Python), which runs two oracle engines:

- **Reference oracle**: transformers (BF16, `output_logits=True`, `attn_implementation=flash_attention_2`)
- **Baseline oracle**: vLLM (BF16, calibrates acceptable numerical drift)

Tolerance: `atol = max(|transformers - vllm|, across all canonical prompts) x 2.0`.

Fixtures are stored as GitHub Release assets (tag: `goldens-v0.1`), not in git. See [ADR-0005](docs/adr/0005-golden-generation-correctness-strategy.md) for the full strategy.

### CI green vs numerically validated

| | CI gate | Release gate |
|---|---|---|
| **When** | Every push | Manual, before tagging |
| **Where** | CPU-only | GPU (sm_89+) |
| **What** | Property tests | Golden comparison vs transformers oracle |
| **Proves** | Compiles + types correct | Numerically correct within tolerance |

## Documentation

- **[CONTEXT.md](CONTEXT.md)** — Domain vocabulary and ubiquitous language. Every term used in the codebase (`CausalLM`, `BlockPool`, `PagedKVCache`, `EngineCore`, `Prompt`, `SamplingParams`, etc.) is defined here with "Avoid" notes for synonyms that should not be used.
- **[docs/adr/](docs/adr/)** — Architecture Decision Records (5 ADRs): parametric parallel layers, weight loader seam, model registry + RoPE, engine dependency DAG, and golden generation correctness strategy.
- **Crate source** — Each module carries a doc comment that explains its role and the ADR-0004 dependency DAG. The `lib.rs` doc comment is the best starting point.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for branch conventions, commit format, and the CI pipeline before opening a pull request.

## Minimum Supported Rust Version (MSRV)

The current MSRV is **1.75** (declared in `[workspace.package]`). We follow a rolling policy: the MSRV may increase in a minor release, but only to a Rust version that has been stable for at least 6 months.

## Security

To report a security vulnerability, please use [GitHub Security Advisories](https://github.com/RedHeartSecretMan/vllm-oxide/security/advisories/new). Do **not** open a public issue for security reports.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
