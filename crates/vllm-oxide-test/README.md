# vllm-oxide-test — Golden Comparison Crate

> **⚠️ Release gate — NOT a CI gate.**
>
> This crate validates the Rust inference engine against golden fixtures on
> a GPU. It is a **manual, pre-release action**. CI green (CPU property tests)
> does **NOT** imply numerical correctness. See [Release Gate vs CI Gate](#release-gate-vs-ci-gate).

## Overview

The golden comparison crate (`crates/vllm-oxide-test/`) implements three
layers of comparison against golden fixtures produced by the Python golden
generator (`tools/golden-gen/`):

| Layer | What it compares | When |
|-------|-----------------|------|
| **L1** | Greedy token-sequence exact match | Always |
| **L2** | Per-step logits tensor comparison (atol+rtol) | Always |
| **L3** | Per-layer activations (debug only) | `--debug` flag |

## Usage

### Prerequisites

- Linux with NVIDIA GPU (sm_89+, e.g., RTX 40-series, A10, H100)
- Rust toolchain (1.75+)
- Model weights for Qwen3-0.6B (local directory or HF Hub)
- Golden fixtures (local directory or GitHub Release)

### Run from local golden fixtures

```bash
cargo run --release -p vllm_oxide_test -- \
    --model-path /path/to/Qwen3-0.6B \
    --manifest /path/to/goldens/manifest.json
```

### Download goldens from GitHub Release and run

```bash
cargo run --release -p vllm_oxide_test -- \
    --model-path /path/to/Qwen3-0.6B \
    --release-tag goldens-v0.1 \
    --cache-dir /tmp/vllm-oxide-goldens
```

### Options

| Flag | Description |
|------|-------------|
| `--model-path PATH` | Model directory (config.json + tokenizer.json + weights) |
| `--manifest PATH` | Local manifest.json + fixture directory |
| `--release-tag TAG` | GitHub Release tag to download goldens from |
| `--repo OWNER/REPO` | GitHub repo (default: `RedHeartSecretMan/vllm-oxide`) |
| `--cache-dir PATH` | Cache directory for downloaded goldens (default: `/tmp/vllm-oxide-goldens`) |
| `--epsilon VALUE` | Override near-tie ε for L1 (default: 2× manifest atol) |
| `--debug` | Enable L3 per-layer activations comparison |
| `--json` | Output results as JSON |
| `--l1-only` | Only run L1 comparison |
| `--l2-only` | Only run L2 comparison |

## How it works

### Manifest

Golden fixtures are described by a `manifest.json` (produced by
`tools/golden-gen/`). The manifest records:

- **Provenance**: model ID, revision, architecture, dtype
- **Tolerances**: calibrated `atol` and `rtol` from oracle cross-validation
- **Known deviations**: documented disagreements between oracle implementations
- **Fixtures**: per-file metadata including SHA-256 hashes

### L1: Token-sequence exact match

Drives the engine via `LLM::generate` (greedy, temperature=0). Compares
generated token IDs against golden token IDs position-by-position.

**Near-tie skipping**: positions where the top-2 logit gap < ε (default:
2× calibrated atol) are skipped — these are inherently non-deterministic
under BF16/FP16 precision.

### L2: Logits tensor comparison

Drives the engine via `LLM::generate_logits` and compares the raw pre-sampling
logits `[n, vocab_size]` against golden logits using:

```
|actual - expected| <= atol + rtol × |expected|
```

Tolerances are calibrated from oracle-vs-oracle divergence (×2 safety factor).

### L3: Per-layer activations (debug)

Skeleton in v0.1. When model introspection lands in v0.2, this will compare
per-layer hidden states to localise divergence sources.

## Release Gate vs CI Gate

| | CI (every push) | Release gate (pre-release) |
|---|---|---|
| **When** | Every push to any branch | Manual, before tagging a release |
| **Where** | CPU-only (`cargo test`) | GPU (requires sm_89+) |
| **What** | Property tests | Numerical validation against goldens |
| **Result** | "CI green" | "Numerically validated" |

**CI green ≠ validated.** The repository README documents this distinction
to prevent users from mistaking passing property tests for numerical
correctness.

## Directory Structure

```
crates/vllm-oxide-test/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs         # Public API
    ├── main.rs        # CLI entrypoint
    ├── types.rs       # Manifest schema types (matches Python schema.py)
    ├── manifest.rs    # Manifest parsing + fixture loading
    ├── download.rs    # GitHub Release asset download + SHA-256 verification
    ├── l1.rs          # L1: token-sequence exact match with near-tie skipping
    ├── l2.rs          # L2: logits tensor comparison (atol+rtol)
    ├── l3.rs          # L3: per-layer activations (debug-only, skeleton)
    └── report.rs      # Comparison report generation
```

## Testing

```bash
# Run unit tests (CPU-only, no GPU required)
cargo test -p vllm_oxide_test

# Run full workspace tests
cargo test
```
