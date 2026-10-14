# vllm-oxide-test — Validation tools

This workspace-only crate provides the Rust side of vllm-oxide validation:
CUDA capture and performance producers, a CPU bundle transport reader, and the
retained historical comparator. It is not a published crate or a supported
application API. Normal applications use `LLM::generate` from `vllm_oxide`.

The current protocol is **`layered-accuracy-v1`**: L0 verifies operators and state
invariants, L1 compares model numerics on the same frozen token history, and L2
checks decoding and public behavior. These names differ from the historical
schema-v4 comparator's L1=token/L2=logits numbering.

Start with the [validation guide](../../docs/validation/README.md) for the
validation map, [development guide](../../docs/development.md) for CPU setup,
and [v0.2.0 validation summary](../../docs/releases/goldens-v0.2.md) for actual
release results. The [Python tooling guide](../../tools/golden-gen/README.md)
coordinates the complete source-bound workflow.

## Release gate vs CI gate

| Check | Where it runs | What it establishes |
|---|---|---|
| Rust unit, integration and documentation tests; format and Clippy | Ordinary CPU GitHub CI and local development | Code and validation-tool behavior |
| Rust/Python bundle reader and adapter tests | Ordinary CPU GitHub CI | Transport, provenance and failure handling on test evidence |
| L0/L1/L2 model validation, public API/CLI smoke and benchmarks | A local CUDA machine | Actual operator, model, behavior and performance results |

CI runs on pull requests, pushes to `main`, and manual dispatch. It requires no
GPU. Passing CPU or synthetic-fixture tests does not establish model accuracy.
A GitHub GPU runner is optional; local Linux or WSL2 GPU verification can supply
the actual release evidence. The manual CUDA smoke workflow is separate from
ordinary CI.

## Prerequisites

- Rust **1.94 or newer**; this checkout pins Rust 1.94.0.
- For CUDA producers: Linux, including WSL2, one NVIDIA GPU with **SM 8.9 or
  newer**, a compatible driver and CUDA Toolkit including `nvcc`. RTX 4080 is
  the recorded release GPU; the requirement is a compute capability, not a list
  of untested hardware.
- For model checks: the complete, pinned Qwen3-0.6B BF16 checkpoint selected by
  the validation registry. Tools verify the config, tokenizer and weight hashes.
- For evidence producers: a clean reviewed checkout, its exact commit/tree and
  matching binary fingerprint, deterministic process settings, and fresh output
  paths outside the source checkout. Preserve the full original evidence locally.

CPU tests and the transport reader require neither CUDA nor model weights.
A source archive without `.git` can compile the CPU tools, but its build receipt
is `unavailable-without-git`. Formal capture, observation and performance
validation still require the reviewed Git checkout and a matching source-bound
binary; an archive build cannot provide those source identities.
See the [GPU smoke guide](../../docs/validation/public-gpu-smoke.md) and the
[Python setup guide](../../tools/golden-gen/README.md#setup-and-cpu-tests) for
runtime preparation and resource guards.

## Executables

| Binary | Purpose | GPU |
|---|---|---|
| `vllm-oxide-fixed-prefix` | Current L1 fixed-history captures, unforced controls, L0 operator suites and L2 standalone behavior cases | Yes |
| `vllm-oxide-benchmark` | Fixed single-request and four-request release workloads with synchronized timing and raw memory telemetry | Yes |
| `vllm-oxide-verify-bundle` | Verify and install a local manifest/archive pair without loading a model | No |
| `vllm-oxide-observe` | Historical schema-v4 calibration observation | Yes |
| `vllm-oxide-test` | Historical schema-v4 token/logit comparator | Yes |

Build the CPU reader from the repository root:

```bash
cargo build --locked -p vllm_oxide_test --bin vllm-oxide-verify-bundle
```

On a prepared CUDA host, build the current producers separately:

```bash
CUDA_COMPUTE_CAP=89 cargo build --locked --release -p vllm_oxide_test \
    --features cuda --bin vllm-oxide-fixed-prefix --bin vllm-oxide-benchmark
```

`CUDA_COMPUTE_CAP=89` selects the validated SM 8.9 target. Use the appropriate
compute capability for another supported GPU. Building a producer does not run
model validation or create acceptance evidence.

## Current capture workflow

Use [`golden_gen.layered_cli`](../../tools/golden-gen/README.md#layered-accuracy-protocol-new-workflow)
for supervised collection, replay, manifest assembly and independent evaluation.
It builds plans from the tracked case registry and records source, binary,
worker and guard identities. Collection uses a previously built candidate;
it does not install dependencies or invoke Cargo.

The underlying `vllm-oxide-fixed-prefix` flags are:

| Flags | Meaning |
|---|---|
| `--model-path`, `--repo-root` | Local checkpoint and reviewed producer checkout |
| `--measurement-commit`, `--measurement-tree` | Exact producer commit/tree, checked against the build fingerprint |
| `--options`, `--output` | Registry-derived engine options JSON and fresh capture destination |
| `--plan` | Fixed-prefix replay plan; records predictions for the frozen history |
| `--setup-plan` | Repeatable setup plans that establish prefix/cache/pressure state |
| `--control` | Advance using the candidate's own greedy predictions |
| `--operator-plan` | Run the L0 operator suite; conflicts with replay/control/setup/behavior modes |
| `--behavior-plan` | Run a standalone L2 public behavior case; conflicts with replay/control/setup/operator modes |
| `--layer-trace-dir` | Nonaccepting single-request diagnostics, requiring `--plan` and a fresh directory |

A fixed plan records exactly T predictions for `prompt + continuation[:t]`.
The predicted token and the token used to advance history are separate fields.
Forced replay alone does not prove free decoding or public behavior; those
checks require their own declared controls and public return values.

The internal capture feature does not expand the public Rust API. Private
capture settings are rejected when stale or incomplete, and invalid plans,
identity mismatches or write failures fail the operation. Layer tracing is
bounded diagnostic output, not an additional acceptance layer.

## Performance producer

`vllm-oxide-benchmark` accepts `--model-path`, `--prompts-dir`, `--output`,
`--repo-root`, `--measurement-commit` and `--measurement-tree`. The prompts
must be the reviewed checkout's `tools/golden-gen/prompts` directory.

The fixed workloads are `canonical_04` (one request) and `canonical_05` (four
requests). Each discards one warmup and retains three measured repetitions,
using greedy decoding and 64 completion tokens. Reports retain prefill/decode
throughput, TTFT, inter-token latency and peak device memory, with original
per-repetition telemetry. The current `direct-nvml-v1` monitor targets 5 ms
samples and keeps the strict 50 ms completion-gap and API-span limits.

The [schema5 release adapter](../../tools/golden-gen/README.md#schema5-release-stages)
checks complete telemetry and source bindings. The [release summary](../../docs/releases/goldens-v0.2.md)
records the actual tested hardware and results. These figures are a local
baseline, not a cross-hardware throughput guarantee or vLLM parity claim.

## CPU bundle transport

For an existing schema5 local evidence bundle and a non-existing cache directory:

```bash
target/debug/vllm-oxide-verify-bundle --layered-transport \
    --bundle-dir /absolute/path/to/local-bundle \
    --cache-dir /absolute/path/to/fresh-verified-cache
```

The bundle contains exactly `manifest.json` and `goldens-v0.2.tar.gz`. The Rust
reader checks bounded archive structure, paths, hashes, inventory and immutable
installation. Its schema5 output deliberately reports `accepting=false`:
transport verification alone does not reproduce numerical or performance
acceptance. Python `release_cli verify` additionally reconstructs and evaluates
the declared raw evidence. Omitting `--layered-transport` selects the historical
schema-v4 reader.

Never use the released product's automatic GitHub source archive as a golden
bundle. Complete evidence stays on durable local storage; golden publication
is an independently authorized operation. The v0.2.0 product summary does not
claim that the old combined schema5 bundle's complete consumer acceptance passed.

## Testing

From the repository root:

```bash
cargo test --locked -p vllm_oxide_test
cargo test --locked --workspace
cargo test --locked --workspace --features internal-golden
```

These are CPU checks. Complete Python adapter tests also need the prepared
worker interpreter and CPU Rust reader; see the
[Python testing instructions](../../tools/golden-gen/README.md#setup-and-cpu-tests).
Use the [public GPU smoke entrypoint](../../docs/validation/public-gpu-smoke.md)
for actual warmup, repeated API calls, mixed input batches and the Quick Start
CLI. A smoke PASS covers those public behaviors, not full model numerics.

## Historical schema-v4 comparator

This section preserves the old comparator contract for interpreting existing
artifacts. It cannot satisfy new `layered-accuracy-v1` acceptance, and the old
`tools/validate-release.sh authoritative` and `publish` entrypoints remain
disabled. Historical verdicts retain their original source, numbering and
budgets; they are not converted into current PASS results.

### Usage

The retained binary can run a schema-v4 diagnostic with all original identity
and approval inputs. These flags do not grant access to the current protocol:

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
cargo run --locked --release -p vllm_oxide_test --features cuda \
    --bin vllm-oxide-test -- \
    --mode authoritative \
    --repo-root /absolute/path/to/reviewed-checkout \
    --measurement-commit "$VLLM_OXIDE_MEASUREMENT_COMMIT" \
    --measurement-tree "$VLLM_OXIDE_MEASUREMENT_TREE" \
    --approved-observation docs/releases/goldens-v0.2-calibration-observation.json \
    --model-path /absolute/path/to/Qwen3-0.6B \
    --manifest /absolute/path/to/legacy-goldens/manifest.json \
    --capture-dir /absolute/path/to/fresh-legacy-captures
```

### Options

| Flag | Historical meaning |
|---|---|
| `--mode authoritative` | Only comparator mode; requires the original schema-v4 approval gate |
| `--repo-root`, `--measurement-commit`, `--measurement-tree` | Clean source identity and binary fingerprint |
| `--approved-observation` | Tracked observation that unlocks historical holdout access |
| `--model-path` | Model directory |
| `--manifest` | Local schema-v4 manifest and fixtures |
| `--release-tag`, `--repo`, `--cache-dir` | Alternative historical golden download source; repository defaults to `RedHeartSecretMan/vllm-oxide` |
| `--capture-dir` | Fresh private directory for replay-audited candidate captures |
| `--prompts-dir` | Must resolve to the reviewed checkout's `tools/golden-gen/prompts` directory |
| `--json` | JSON report on stdout; diagnostics go to stderr |
| `--l1-only`, `--l2-only` | Exploratory partial comparisons; omitted declared coverage cannot pass the old release gate |
| `--debug` | Rejected: legacy L3 per-layer comparison is not implemented |

### Manifest and comparison semantics

Schema-v4 manifests bind model/tokenizer/config/weights, exact runtime and
kernel paths, expected fixtures, file hashes and a reviewed `same-prefix-v1`
tolerance policy. Only Transformers reference fixtures supply correctness
expectations. vLLM fixtures supply baseline calibration, never a second truth
or permission to select thresholds automatically. The
[historical manifest and tensor definitions](../../tools/golden-gen/README.md#manifest-schema)
retain every field.

| Historical layer | Comparison |
|---|---|
| L1 | Greedy token IDs. The first mismatch is an explicit near tie only if the expected/actual candidate-logit gap is within `l1_near_tie_max_abs_logit_gap`. |
| L2 | Same-prefix full FP32 logits with `abs(actual - expected) <= l2_atol`; include the first divergence row and exclude later rows with different histories. |
| L3 | Unavailable; the CLI returns an error rather than a placeholder PASS. |

CPU preflight verifies canonical, flattened batch and regression prompt
inventories, names, tensor shapes/dtypes and hashes. Reports retain exact
`expected`, `discovered`, `generated`, `compared`, `missing`, `unexpected`,
`skipped` and `failed` totals. Empty, malformed, missing, unexpected, skipped or
failed coverage rejects acceptance; a supplied summary cannot replace comparisons.
L2 also refuses empty or inconsistent tensor/token geometry and invalid
tolerances; non-finite compared values cannot pass.

Capture rows bind call, input position, stable request identity, chosen token,
completion step, request-major order and agreement with `RequestOutput`.
Partial/stale captures, symlink or reused destinations, serialization failures
and publication collisions fail without exposing a partial capture.

The historical downloader requires the exact manifest/archive pair, verifies
raw USTAR paths/types and checksums, stages only regular declared files and
installs with atomic `renameat2(RENAME_NOREPLACE)`. Absolute/traversing/link,
duplicate, undeclared or malformed entries fail closed. Existing complete
immutable installs are reused; incomplete or modified installs are rejected,
never overwritten. Unsupported no-replace installation fails rather than
using a racy replacement.

## Source map

| Path | Responsibility |
|---|---|
| `src/bin/fixed_prefix.rs` | Current fixed-history/operator/behavior producer |
| `src/benchmark.rs`, `src/benchmark/` | Workloads, synchronized telemetry and direct NVML monitor |
| `src/bin/verify_bundle.rs`, `src/release_transport.rs` | CPU schema5 transport reader |
| `src/measurement.rs`, `build.rs`, `source_identity.rs` | Clean source, model and binary identity checks |
| `src/main.rs`, `src/driver.rs`, `src/l1.rs`, `src/l2.rs`, `src/l3.rs` | Retained schema-v4 comparator |
| `src/capture.rs`, `src/lifecycle.rs`, `src/download.rs` | Historical capture, coverage and safe asset installation |
| `tests/` | CPU transport/integration tests and explicit CUDA public smoke |
