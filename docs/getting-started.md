# Getting started — v0.1.0

[Documentation](README.md) · [中文介绍](../README.zh-CN.md) · [Current v0.2.0 usage](https://github.com/RedHeartSecretMan/vllm-oxide/blob/v0.2.0/docs/getting-started.md)

This guide reproduces the historical single-GPU offline engine. v0.1.0 has known numerical differences; new applications should use v0.2.0. These commands were checked against the archived source, not rerun as new GPU acceptance.

## Requirements and source

Inference needs Linux (including WSL2), one SM 8.9+ NVIDIA GPU, a compatible CUDA driver, CUDA Toolkit with `nvcc`, a host C/C++ compiler and enough free memory for weights, activations and KV cache. Historical measurement used Qwen3-0.6B/BF16 on RTX 4080/WSL2. No generic minimum VRAM, performance guarantee, prebuilt binary or model weights are supplied.

The maintained source requires Rust 1.94 and pins Rust 1.94.0. The original Rust 1.75 claim was not verified and is incompatible with the locked dependencies, which require at least Rust 1.88. The source lockfile fixes dependency selection; Candle is pinned to `27f20fea993c81ea6d32ce44018f42b68466525e`. CPU development does not require CUDA and does not enable CPU inference.

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
cd vllm-oxide
cargo build --locked --release -p vllm_oxide_cli --features vllm_oxide/cuda
```

The executable is `target/release/vllm_oxide_cli`. The CLI package has no feature table; `vllm_oxide/cuda` explicitly enables the library's CUDA feature. The v0.2.0 CLI's unqualified `--features cuda` instruction must not be copied to this version.

## Model sources

A complete local directory contains `config.json`, `tokenizer.json` and safetensors weights. For sharded weights, include the index and every referenced shard. The historical fixture model revision is `7e4ae267688d671ddfca3122e4528ee980cf3234` of `Qwen/Qwen3-0.6B`.

```bash
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
target/release/vllm_oxide_cli --model "$QWEN3_MODEL_DIR" \
    --temperature 0 --max-tokens 16 "The meaning of life is"
```

An existing directory is loaded locally. Every other `--model` value is interpreted as a Hub repo ID and follows its default revision. The CLI has no revision flag and does not parse `hub:` or `@revision` syntax. Use a local checkpoint or `Source::Hub { revision: Some(...) }` in the Rust API to pin it.

`HF_HUB_OFFLINE=1` disables Hub fetching; all required files for the selected revision must already be cached. Build dependencies and obtain model files before disconnecting. `Source::Local` selects a complete local checkpoint directly.

## CLI

```bash
target/release/vllm_oxide_cli --help
printf '%s' 'The meaning of life is' | \
    target/release/vllm_oxide_cli --model "$QWEN3_MODEL_DIR" --max-tokens 16
```

| Argument | Meaning | Default |
|---|---|---|
| `-m`, `--model` | Existing directory or Hub repo ID | Required |
| Positional prompt | Raw completion text | Read all stdin |
| `--temperature` | `0` selects greedy generation | `0` |
| `--top-k` | Keep the highest-logit `k` candidates | Disabled |
| `--top-p` | Nucleus probability threshold | Disabled |
| `--max-tokens` | Maximum completion tokens | `16` |

The CLI rejects negative or NaN temperature, zero top-k/max-tokens and non-finite or out-of-range top-p during argument parsing, before loading a model; positive-infinity temperature remains accepted. It submits one prompt, prints continuation text and propagates construction/generation errors. It does not apply a chat template. Batch input, token-ID prompts, penalties, `ignore_eos`, engine options, dtype and GPU selection have no CLI switches.

## Rust API

For historical investigation, the library dependency uses the v0.1.0 tag and its own CUDA feature:

```toml
[dependencies]
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", tag = "v0.1.0", features = ["cuda"] }
anyhow = "1"
```

```rust
use vllm_oxide::{EngineOptions, LLM, Prompt, SamplingParams, Source};

fn main() -> anyhow::Result<()> {
    let mut llm = LLM::new(
        Source::Hub {
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: Some("7e4ae267688d671ddfca3122e4528ee980cf3234".into()),
        },
        EngineOptions::default(),
    )?;
    let output = llm.generate(
        &[Prompt::Text("The meaning of life is".into())],
        &[SamplingParams { max_tokens: 16, ..SamplingParams::default() }],
    )?;
    for request in output {
        println!("{}: {}", request.seq_id, request.text);
    }
    Ok(())
}
```

This version's output field is `seq_id`, not v0.2.0's `request_id`. Supply one `SamplingParams` for every prompt. `Prompt::Text` and `Prompt::TokenIds` are available; outputs contain decoded text and raw completion token IDs. The archived `LLM::generate` has known lifecycle limits, so this example deliberately demonstrates a single call rather than promising the current version's repeated-call guarantees.

Default engine settings are 16,384 batched tokens, 512 sequences, 4,096 context tokens, `gpu_memory_utilization=0.9`, `enforce_eager=true`, and checkpoint dtype. In this source, memory utilization sizes the KV pool from available device memory after warmup; it is not a peak-memory cap. `enforce_eager=false` is rejected; CUDA Graphs are unsupported. Token/concurrency budgets and context length must be positive, and the memory fraction must be finite and in `(0, 1]`.

## Diagnostic examples and common errors

The archive supplies `load_qwen3` (weight inspection) and `forward_qwen3` (dummy forward-pass inspection). Neither is a supported generation example or numerical acceptance check:

```bash
cargo run --locked -p vllm_oxide --example load_qwen3 -- \
    hub:Qwen/Qwen3-0.6B@7e4ae267688d671ddfca3122e4528ee980cf3234
cargo run --locked --release -p vllm_oxide --example forward_qwen3 --features cuda -- \
    "$QWEN3_MODEL_DIR"
```

For missing CUDA, check the driver, Toolkit and enabled library feature. An SM below 8.9 is rejected by `LLM::new`. A missing local directory is treated as a Hub ID by the CLI, so use an absolute existing path. An incomplete offline Hub cache must be populated for the exact revision. Out-of-memory errors require leaving more free device memory or reducing workload/configuration; this version has no validated memory envelope.

Historical fixture comparisons and their gaps are documented in [validation](validation/README.md). Upgrade instructions and API changes are in the [current migration guide](https://github.com/RedHeartSecretMan/vllm-oxide/blob/v0.2.0/docs/migration-v0.2.0.md).
