# vllm-oxide-test: RC comparison tools

This document describes the **historical schema-v4 comparator** and its original
L1=token/L2=logits numbering. It cannot satisfy new release acceptance under
[ADR-0015](../../docs/adr/0015-layered-accuracy-validation.md). The current
[layered workflow](../../tools/golden-gen/README.md#layered-accuracy-protocol-new-workflow)
uses L0 operators, L1 model numerics and L2 decoding/public behavior. Historical
artifacts retain their original verdicts; the old shell authoritative/publication
entrypoints stay disabled. The new schema5 adapter is documented in the
[release stages](../../tools/golden-gen/README.md#schema5-release-stages).
Its CPU-tested transport does not imply GPU numerical acceptance.

Run golden model comparisons manually on a GPU before release. CPU CI tests
code and tooling behavior; numerical correctness requires the separate GPU
checks. See [Release Gate vs CI Gate](#release-gate-vs-ci-gate).

## Overview

The comparison crate (`crates/vllm-oxide-test/`) reads fixtures from the Python
generator (`tools/golden-gen/`) and supports L1 and L2 comparisons. The L3
entrypoint remains unavailable:

| Layer | What it compares | When |
|-------|-----------------|------|
| **L1** | Greedy token reference match or explicit candidate-logit near tie | Always |
| **L2** | Same-prefix logits tensor comparison (absolute tolerance) | Always |
| **L3** | Unavailable; requesting it returns an error | `--debug` is rejected |

Before any GPU comparison, a CPU preflight matches the schema-v4 manifest to
all canonical, flattened batch, and regression prompts; verifies asset names,
SHA-256 digests, tensor sets, dtypes, and shapes; and classifies the two oracle
roles. Only Transformers reference artifacts feed correctness comparisons.
vLLM artifacts count only after the manifest records successful calibration.

## Usage

### Prerequisites

- Linux or WSL2 with the recorded RTX 4080 / SM 8.9 reproduction profile; source-bound validation uses its exact pinned environment.
- The maintained manifest and toolchain require Rust 1.94.0. The original 1.75 minimum was not validated and did not satisfy the locked dependencies.
- Complete local Qwen3-0.6B weights; `--model-path` does not accept a Hub repository ID
- Golden fixtures (local directory or GitHub Release)

A source archive without `.git` can compile CPU tools. Its build receipt is
`unavailable-without-git`, so formal capture, observation and performance still
require a reviewed Git checkout and a matching source-bound binary. A successful
archive build is not an authoritative source receipt.

### Run the authoritative comparator

The former `tools/validate-release.sh authoritative <run-root>` entrypoint is
disabled. Use the direct commands below for historical diagnostics, with a
Definition-approved observation and a second set of candidate captures. These
commands do not authorize a layered release:

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
cargo run --locked --release -p vllm_oxide_test --features cuda --bin vllm-oxide-test -- \
    --mode authoritative \
    --repo-root /path/to/reviewed-checkout \
    --measurement-commit "$VLLM_OXIDE_MEASUREMENT_COMMIT" \
    --measurement-tree "$VLLM_OXIDE_MEASUREMENT_TREE" \
    --approved-observation docs/releases/goldens-v0.2-calibration-observation.json \
    --model-path /path/to/Qwen3-0.6B \
    --manifest /path/to/goldens/manifest.json \
    --capture-dir /tmp/fresh-candidate-captures
```

### Download goldens from GitHub Release and run

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
cargo run --locked --release -p vllm_oxide_test --features cuda --bin vllm-oxide-test -- \
    --mode authoritative \
    --repo-root /path/to/reviewed-checkout \
    --measurement-commit "$VLLM_OXIDE_MEASUREMENT_COMMIT" \
    --measurement-tree "$VLLM_OXIDE_MEASUREMENT_TREE" \
    --approved-observation docs/releases/goldens-v0.2-calibration-observation.json \
    --model-path /path/to/Qwen3-0.6B \
    --release-tag goldens-v0.2 \
    --cache-dir /tmp/vllm-oxide-goldens \
    --capture-dir /tmp/fresh-downloaded-candidate-captures
```

### Options

| Flag | Description |
|------|-------------|
| `--mode authoritative` | Historical comparator mode; it does not accept a layered release |
| `--repo-root PATH` | Clean reviewed source checkout, matching the binary build fingerprint |
| `--measurement-commit SHA` / `--measurement-tree SHA` | Frozen measured source identities |
| `--approved-observation PATH` | Definition-tracked observation that unlocks holdout access |
| `--model-path PATH` | Model directory (config.json + tokenizer.json + weights) |
| `--manifest PATH` | Local manifest.json + fixture directory |
| `--release-tag TAG` | GitHub Release tag to download goldens from |
| `--repo OWNER/REPO` | GitHub repo (default: `RedHeartSecretMan/vllm-oxide`) |
| `--cache-dir PATH` | Cache directory for downloaded goldens (default: `/tmp/vllm-oxide-goldens`) |
| `--capture-dir PATH` | Fresh private destination for replay-audited candidate captures |
| `--debug` | Rejected: L3 activation comparison is not implemented |
| `--json` | Output results as JSON |
| `--l1-only` | Only run L1 comparison |
| `--l2-only` | Only run L2 comparison |

## How it works

### Manifest

Golden fixtures are described by a `manifest.json` (produced by
`tools/golden-gen/`). The manifest records:

- **Provenance**: exact model/tokenizer revisions and hashes, architecture, dtype
- **Runtime/kernel identity**: locked Python wheels, CUDA/Rust/driver/GPU/OS
  identity, deterministic process settings, and all three kernel paths
- **Expected fixtures**: family, immutable model identity, oracle role, and
  required comparison for every artifact
- **Tolerance policy**: an explicit version and dtype/kernel scope, L1
  candidate-gap threshold, L2 absolute tolerance, rationale, and evidence
- **Baseline calibration**: observed oracle differences and methodology,
  reported separately from reference correctness
- **Fixtures**: per-file metadata including SHA-256 hashes
- **Asset identity**: schema `4`, product `v0.2.0`, golden tag
  `goldens-v0.2`, and the SHA-256 identity of `goldens-v0.2.tar.gz`

The historical golden contract requires exactly `manifest.json` and `goldens-v0.2.tar.gz`; it does
not contain individual fixture assets. The downloader requires the requested
tag to equal the manifest golden version, verifies the complete archive before
exposure, and installs to
`<cache-dir>/goldens-v0.2/<archive-sha256>`. A complete existing immutable
install is reused. An incomplete or modified existing install is rejected and
never overwritten.

Archive paths and types are checked from raw USTAR headers. Absolute, nested,
`.`/`..`, backslash, non-ASCII, duplicate, undeclared, link, directory, device,
FIFO, non-USTAR, oversized, malformed, and checksum-mismatched entries fail
closed. Extraction occurs in a unique sibling staging directory; only a fully
verified fixture set and standalone manifest become visible through one rename.
Failures clean only their unpublished staging directory, preserving older
installs. Concurrent installers verify and reuse the winning rename.
The Linux installer uses `renameat2(RENAME_NOREPLACE)` through safe `rustix`
bindings; unsupported kernels/filesystems fail closed rather than falling back
to a racy replacement. Staging and final paths share the same version directory
and therefore the same filesystem.

The report includes exact `expected`, `discovered`, `generated`, `compared`,
`missing`, `unexpected`, `skipped`, and `failed` totals. Release acceptance is
fail-closed: the set must be non-empty, every expected fixture must reach its
declared comparison, and missing/unexpected/skipped/failed must all be zero.
Malformed manifests, unmatched identifiers, unsupported fixture shapes, and
empty comparison sets exit non-zero. L2 refuses empty/inconsistent tensor and
token geometry or invalid tolerances; non-finite compared values cannot pass. `--l1-only` or `--l2-only` is exploratory
when it omits a declared comparison and therefore cannot satisfy the release
gate. Before this comparator can run, its approval gate independently verifies
the observation SHA, mechanical proposal, runtime/kernel identity, and sealed
holdout list.

### L1: Reference-token match or explicit near tie

Drives the engine via `LLM::generate` (greedy, temperature=0). Compares
generated token IDs against golden token IDs position-by-position.

At the first token mismatch, L1 compares the expected and actual candidate
logits from that same-prefix row. The mismatch is accepted only when their
absolute gap satisfies `tolerance_policy.l1_near_tie_max_abs_logit_gap`.
The near tie remains an explicit classification, and later token positions are
excluded because their causal histories differ.

### L2: Logits tensor comparison

Drives the engine through the same supported `LLM::generate` method as normal
users. The workspace-only, default-off `internal-golden` feature captures raw
pre-sampling logits into one caller-private JSONL artifact only when all three
reserved process settings are present. The feature adds no public Rust method,
module, trait, or type and is unsupported outside release tooling.

The producer streams one vocabulary row at a time through private staging,
self-validates the complete artifact, and publishes it with atomic NOREPLACE.
The consumer verifies call identity, input position, stable request identity,
selected token, zero-based completion step, request-major row order, shape,
dtype, completeness, and agreement with `RequestOutput` before comparing raw
logits `[n, vocab_size]` using:

```
|actual - expected| <= tolerance_policy.l2_atol
```

The divergence row is still comparable because it was produced from the shared
prefix. L2 stops immediately after that token is selected and excludes every
later row from aggregate metrics.

Absent capture configuration performs no diagnostic I/O. Partial or invalid
configuration, a reused or symlink destination, capture/serialization/write
failure, malformed or stale rows, and publication collision all fail the
generation call without exposing a partial destination. Full-logit device-to-
host transfer occurs only for this explicitly configured diagnostic path.

### L3: Per-layer activations (debug)

Per-layer comparison is not implemented. The maintained shared comparator
returns an error, so the CLI rejects `--debug` rather than printing a synthetic
PASS. A future implementation would need its own validated contract.

## Release gate vs CI gate

| | CI (every push) | Release gate (pre-release) |
|---|---|---|
| **When** | Pull requests and pushes to `main` | Manual, before tagging a release |
| **Where** | CPU-only (`cargo test`) | GPU (requires sm_89+) |
| **What** | Property tests | Numerical validation against goldens |
| **Result** | "CI green" | "Numerically validated" |

Passing CPU checks establishes code and tooling behavior. Model numerical
results require the separate GPU comparison described above.

## Directory structure

```
crates/vllm-oxide-test/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs         # Workspace validation helpers; unsupported application API
    ├── main.rs        # CLI entrypoint
    ├── capture.rs     # private fail-closed raw-logit artifact consumer
    ├── types.rs       # Manifest schema types (matches Python schema.py)
    ├── manifest.rs    # Manifest parsing + fixture loading
    ├── lifecycle.rs   # Discovery, asset preflight, and exact coverage accounting
    ├── download.rs    # exact-two download + verified atomic archive installation
    ├── l1.rs          # L1: token reference match + explicit near-tie classification
    ├── l2.rs          # L2: same-prefix logits comparison (absolute tolerance)
    ├── l3.rs          # Explicit error for unavailable per-layer comparison
    └── report.rs      # Comparison report generation
```

## Testing

```bash
# Run unit tests (CPU-only, no GPU required)
cargo test --locked -p vllm_oxide_test

# Run full workspace tests
cargo test --locked --workspace
```
