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
| **L1** | Greedy token reference match or explicit candidate-logit near tie | Always |
| **L2** | Same-prefix logits tensor comparison (absolute tolerance) | Always |
| **L3** | Per-layer activations (debug only) | `--debug` flag |

Before any GPU comparison, a CPU preflight matches the schema-v3 manifest to
all canonical, flattened batch, and regression prompts; verifies asset names,
SHA-256 digests, tensor sets, dtypes, and shapes; and classifies the two oracle
roles. Only Transformers reference artifacts feed correctness comparisons.
vLLM artifacts count only after the manifest records successful calibration.

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
| `--debug` | Enable L3 per-layer activations comparison |
| `--json` | Output results as JSON |
| `--l1-only` | Only run L1 comparison |
| `--l2-only` | Only run L2 comparison |

## How it works

### Manifest

Golden fixtures are described by a `manifest.json` (produced by
`tools/golden-gen/`). The manifest records:

- **Provenance**: model ID, revision, architecture, dtype
- **Expected fixtures**: family, immutable model identity, oracle role, and
  required comparison for every artifact
- **Tolerance policy**: an explicit version and dtype/kernel scope, L1
  candidate-gap threshold, L2 absolute tolerance, rationale, and evidence
- **Baseline calibration**: observed oracle differences and methodology,
  reported separately from reference correctness
- **Fixtures**: per-file metadata including SHA-256 hashes

The report includes exact `expected`, `discovered`, `generated`, `compared`,
`missing`, `unexpected`, `skipped`, and `failed` totals. Release acceptance is
fail-closed: the set must be non-empty, every expected fixture must reach its
declared comparison, and missing/unexpected/skipped/failed must all be zero.
Malformed manifests, unmatched identifiers, unsupported fixture shapes, and
empty comparison sets exit non-zero. `--l1-only` or `--l2-only` is exploratory
when it omits a declared comparison and therefore cannot satisfy the release
gate.

### L1: Token-sequence exact match

Drives the engine via `LLM::generate` (greedy, temperature=0). Compares
generated token IDs against golden token IDs position-by-position.

At the first token mismatch, L1 compares the expected and actual candidate
logits from that same-prefix row. The mismatch is accepted only when their
absolute gap satisfies `tolerance_policy.l1_near_tie_max_abs_logit_gap`.
The near tie remains an explicit classification, and later token positions are
excluded because their causal histories differ.

### L2: Logits tensor comparison

Drives the engine via `LLM::generate_logits` and compares the raw pre-sampling
logits `[n, vocab_size]` against golden logits using:

```
|actual - expected| <= tolerance_policy.l2_atol
```

The divergence row is still comparable because it was produced from the shared
prefix. L2 stops immediately after that token is selected and excludes every
later row from aggregate metrics.

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
    ├── lifecycle.rs   # Discovery, asset preflight, and exact coverage accounting
    ├── download.rs    # GitHub Release asset download + SHA-256 verification
    ├── l1.rs          # L1: token reference match + explicit near-tie classification
    ├── l2.rs          # L2: same-prefix logits comparison (absolute tolerance)
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
