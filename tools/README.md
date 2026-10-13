# Development and validation tools

[Documentation](../docs/README.md) · [Development](../docs/development.md) · [Validation](../docs/validation/README.md)

The engine is built from the Rust workspace under `crates/`. This directory
contains maintainer tooling for validation and historical investigation; it is
not needed by an application that uses `LLM::generate`.

| Tool | Purpose | Environment | Details |
|---|---|---|---|
| `golden-gen` | Layered operators, numerics and behavior; CPU adapters; performance and local evidence transport | CPU for tests/evaluation; a prepared CUDA GPU for model collection | [Package guide](golden-gen/README.md) |
| `golden-gen/scripts/` | Tiny BF16 arithmetic probes and fixed historical layer traces | CPU or CUDA for tiny probes; CUDA for model tracing | [Diagnostic guide](golden-gen/scripts/README.md) |
| `validate-release.sh` | Retained ADR-0012 schema-v4 stage driver | Historical workstation-specific environment | [Limitations](#historical-stage-driver) |
| Rust capture/benchmark/consumer binaries | Candidate captures, fixed performance workloads, CPU transport | CUDA for model runs; CPU for archive transport | [Rust tool guide](../crates/vllm-oxide-test/README.md) |

## Choose an entrypoint

- Ordinary development: the [CPU checks](../docs/development.md).
- Production API/CLI coverage:
  [`golden_gen.public_gpu_smoke`](../docs/validation/public-gpu-smoke.md).
- Correctness: `golden_gen.layered_cli` with the frozen registry, approved
  budgets, and prepared candidate/oracle environments.
- CPU/performance records and local transport: `golden_gen.release_cli`.
  Successful byte transport alone is not model acceptance.
- Historical rounding/layer diagnosis: the scripts, preserving their
  `diagnostic_only=true`, `accepting=false` status.

The package guide records CLI flags, stage inputs/outputs, source-role bindings,
sampling and archive limits. Run help without allocating a GPU:

```bash
uv sync --project tools/golden-gen --locked --extra dev
uv run --project tools/golden-gen --locked --extra dev \
    python -m golden_gen.layered_cli --help
uv run --project tools/golden-gen --locked --extra dev \
    python -m golden_gen.release_cli --help
uv run --project tools/golden-gen --locked --extra dev \
    python -m golden_gen.public_gpu_smoke --help
```

Installation may access the network. Actual prepared GPU collection uses its
required offline/deterministic environment and fresh source-bound output paths.
Run GPU owners serially. Keep weights, captures, logits, guards, telemetry and
archives outside Git; evidence is local unless an upload is explicitly requested.

## Historical stage driver

[`validate-release.sh`](validate-release.sh) preserves the old schema-v4 flow
from [ADR-0012](../docs/adr/0012-goldens-v0.2-calibration-and-performance-protocol.md).
It is not a portable current-release automation wrapper.

| Argument or setting | Actual behavior |
|---|---|
| First positional argument | `env`, `generate`, `calibrate`, `observe`, `authoritative`, `benchmark`, `report`, `bundle`, `verify-local`, `publish`, or `verify` |
| Second positional argument | Fresh canonical absolute child of `/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts/`; traversal and symlink aliases are rejected |
| Third positional argument | Optional complete model directory; defaults to the original author's pinned Qwen3-0.6B cache |
| Build target | Hard-coded original workspace target directory and one Cargo job |
| Progression | Content-bound predecessor markers and clean measurement checkout required |
| Resource checks | At least 16 GiB available host RAM for guarded stages |
| `authoritative` / `publish` | Disabled by initial dispatch; listed for historical context, not usable acceptance/publication commands |
| `observe` | A completed nonaccepting observation deliberately exits `3` |

Print usage with no valid stage; this exits `2`:

```bash
bash tools/validate-release.sh
```

Do not use `env` as generic setup: its paths and marker graph belong to the
historical protocol. Use the Python entrypoints for current work. Do not bypass
the disabled acceptance/publication blocks to claim a new release result.

## Version boundary

`tools/check_docs.py` and `tools/test_check_docs.py` were added afterward on
`main`; they are not bundled here. See the
[documentation revision](../docs/releases/version-doc-update.md).
