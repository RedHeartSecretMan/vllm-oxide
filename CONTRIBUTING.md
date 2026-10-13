# Contributing to vllm-oxide

Contributions to code, tests, documentation and issue reproduction are welcome.
You can develop and run the ordinary CI checks without a GPU. Start with the
[documentation index](docs/README.md) and [current release scope](docs/releases/v0.2.0.md).

## Development Environment

- Rust edition 2021, minimum Rust 1.94.0, pinned in `rust-toolchain.toml`.
  Install `rustfmt` and Clippy; the pinned toolchain includes both.
- CPU development: no NVIDIA GPU, CUDA toolkit or model download is needed for
  the Rust checks below. GitHub Actions runs them on Linux.
- Python tooling: Python 3.12 and uv 0.10.9, matching the CI environment. This
  setup is only needed when changing `tools/golden-gen/` or running its checks.
- CUDA inference: a supported GPU and CUDA development environment are needed
  to build and exercise `--features cuda`. See the [README](README.md) for
  supported hardware and the [GPU smoke guide](docs/validation/public-gpu-smoke.md)
  for the actual model check.

## Workflow

1. Search [existing issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues)
   before opening a bug or proposal. An issue marked `ready-for-agent` has a
   specified implementation scope; the [triage labels](docs/agents/triage-labels.md)
   explain the other statuses.
2. For a ticket, use `feat/t<NN>-<slug>` (for example,
   `feat/t24-vllm-oxide-cli`). Use a descriptive branch for a small contribution.
3. Keep the change focused. Add a regression test for a behavior change and
   update the relevant user documentation when behavior or the public API changes.
4. Run checks relevant to the change and report their actual results. You can
   contribute without running every GPU check yourself; describe any validation
   you could not perform so it can be completed before merging when needed.
5. Open a PR against `main`. Describe the problem, resulting behavior and
   evidence, and link the related issue.

Small documentation fixes do not need an issue or a GPU run. Check examples and
links, and let the ordinary PR CI run automatically.

```bash
python3 -m unittest discover -s tools -p test_check_docs.py
python3 tools/check_docs.py
```

The checker reads Git-tracked Markdown; stage new documentation before checking
its links. The separate [Documentation workflow](.github/workflows/docs.yml)
performs these checks automatically without a GPU or model dependencies.

## Local CPU Checks

Run these from the repository root. They match the Rust PR CI, which also runs
on the latest stable toolchain:

```bash
cargo build --locked --workspace
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --features internal-golden -- -D warnings
cargo test --locked --workspace
cargo test --locked --workspace --features internal-golden
```

`internal-golden` enables internal release-tooling tests; it does not enable CUDA
or require a model. For focused iteration, run the affected tests first and use
the workspace checks before submitting a code change.

The dependency-policy check used by CI is:

```bash
cargo install --locked --version 0.20.2 cargo-deny
cargo deny --log-level warn --manifest-path ./Cargo.toml --all-features check
```

### Python Tooling and Adapter Tests

The complete Python suite uses two locked environments and a Rust CPU transport
reader. The worker environment installs the pinned Torch/vLLM dependencies so
their CPU adapters can be tested; it does not run GPU inference. Its dependency
download is substantial, so Rust-only contributors do not need this setup for
local iteration.

The following Bash commands mirror the Python CI job, including its check that
no adapters or test cases were skipped:

```bash
set -euo pipefail
UV_PROJECT_ENVIRONMENT="$PWD/.venv/golden-dev" \
    uv sync --project tools/golden-gen --locked --extra dev
UV_PROJECT_ENVIRONMENT="$PWD/.venv/golden-worker" \
    uv sync --project tools/golden-gen --locked --all-extras
cargo build --locked -p vllm_oxide_test --bin vllm-oxide-verify-bundle

export GOLDEN_WORKER_TEST_PYTHON="$PWD/.venv/golden-worker/bin/python"
export GOLDEN_TRANSPORT_TEST_BINARY="$PWD/target/debug/vllm-oxide-verify-bundle"
export PYTEST_ADDOPTS=""
export PYTHONNOUSERSITE="1"
.venv/golden-dev/bin/python -m pytest tools/golden-gen/tests -q \
    | tee .venv/python-tests.log
.venv/golden-dev/bin/python - <<'PY'
from pathlib import Path
from golden_gen.release_cpu import _output_ok

_output_ok("python", Path(".venv/python-tests.log").read_text(), "")
PY
.venv/golden-dev/bin/python -m ruff check \
    tools/golden-gen/src tools/golden-gen/tests tools/golden-gen/scripts
.venv/golden-dev/bin/python -m mypy \
    --config-file tools/golden-gen/pyproject.toml \
    tools/golden-gen/src tools/golden-gen/scripts
```

If you set `CARGO_TARGET_DIR`, adjust `GOLDEN_TRANSPORT_TEST_BINARY` to its actual
build location. A passing partial suite is useful development feedback, but
does not replace the complete CI job.

## Code Style

- `rustfmt` with `max_width = 100` (see `rustfmt.toml`)
- Clippy with 12 enabled lints including 4 promoted from `clippy::pedantic` (see `[workspace.lints.clippy]` in `Cargo.toml`)
- `unsafe_code = "deny"` at workspace level; opt-in per module with documented justification

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: ...` — new feature or capability
- `fix: ...` — bug fix
- `chore: ...` — maintenance (deps, config, formatting)
- `docs: ...` — documentation only
- `ci: ...` — CI/CD pipeline changes
- `test: ...` — test additions or fixes

Scope is optional: `feat(engine): ...`, `fix(sampler): ...`

## CI Pipeline

Every PR automatically runs [CPU CI](.github/workflows/ci.yml) on GitHub-hosted
`ubuntu-latest` runners: the Rust 1.94.0/stable matrix, complete Python CPU adapter
tests, Ruff, mypy and dependency auditing. Contributors do not need to configure
a runner or manually trigger these checks. These jobs check executable behavior
and tool quality as well as compilation and formatting.

CPU CI does not measure real CUDA outputs or performance. CUDA, model arithmetic,
sampling or generation changes need the relevant model checks in addition to
CPU CI. Report the source revision, GPU/CUDA environment, model revision,
workload and results of any local GPU check. A documentation-only change does
not require repeating full numerical and performance validation.

The [public GPU smoke guide](docs/validation/public-gpu-smoke.md) covers real
warmup, repeated `LLM::generate`, mixed requests and the exact Quick Start. The
separate [CUDA workflow](.github/workflows/cuda-smoke.yml) is manually dispatched
and requires an already configured GPU runner; it is not part of ordinary PR CI.
Local GPU validation can supply the required evidence, including on WSL. See the
[layered accuracy contract](docs/validation/layered-accuracy-contract.md) and
[published validation summary](docs/releases/goldens-v0.2.md) for numerical and
performance verification. CPU CI success alone is not a model accuracy result.

Keep model weights, raw validation evidence and local working directories out of
the PR. A concise validation summary is sufficient for the review entrypoint;
coordinate access separately if additional evidence is needed.

## Architecture Decisions

Significant design decisions are recorded as ADRs in `docs/adr/`. If your change alters architecture or public API, propose an ADR.

## Domain Vocabulary

Use the vocabulary defined in `CONTEXT.md`. Do not introduce synonyms for established terms (check the "Avoid" notes). Key terms to know: `CausalLM`, `EngineCore`, `BlockPool`, `PagedKVCache`, `Scheduler`, `Prompt`, `SamplingParams`, `RequestOutput`.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
