# Development guide

[Documentation index](README.md) · [Contributing](../CONTRIBUTING.md)

Run the commands below from the repository root. These checks use the source
checkout and its lockfiles. CPU development verifies executable behavior and
tooling without providing CPU inference.

## Environment

| Area | Environment |
|---|---|
| Rust | Edition 2021; minimum supported Rust version 1.94, pinned to 1.94.0 in `rust-toolchain.toml`, with Rustfmt and Clippy |
| Ordinary CI | GitHub-hosted Linux CPU runners; Rust 1.94.0 and current stable |
| Python tooling | Python 3.12 and uv 0.10.9, matching the locked CI setup |
| CUDA inference | A supported NVIDIA GPU, working driver and CUDA development tools; see [getting started](getting-started.md#requirements) |

Rust and documentation checks need no GPU, CUDA Toolkit or model download. The
Python worker environment has a substantial dependency download; prepare it
when working on `tools/golden-gen/`, rather than for an unrelated Rust change.

## Choose the checks for your change

| Change | Local validation |
|---|---|
| Wording or links | Check local links and anchors; verify any changed commands or examples |
| Rust behavior | Relevant regression tests, then the workspace format, Clippy and test checks |
| Python validation tooling | Complete Python CPU adapter suite, Ruff and mypy |
| CUDA kernels, model arithmetic, sampling or generation | Relevant CPU checks plus local GPU/model validation |
| Dependencies | Lockfile update, affected checks and dependency-policy audit |

Automatic CI for this version is defined in [CI](../.github/workflows/ci.yml).
The separate documentation workflow and `tools/check_docs.py` were added after
the original v0.2.0 implementation and are not included in this source snapshot.
Check the workflow file when changing tool versions or commands; it defines
this version's automation.

## Rust CPU checks

```bash
cargo build --locked --workspace
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --features internal-golden -- -D warnings
cargo test --locked --workspace
cargo test --locked --workspace --features internal-golden
```

`internal-golden` enables internal release-tooling tests. It does not enable
CUDA or require a model. Run focused tests while iterating, then use the
workspace checks before submitting a code change. CI runs format, Clippy and
tests on both the pinned minimum toolchain and current stable.

To build local public API documentation:

```bash
cargo doc --locked -p vllm_oxide --no-deps
```

The dependency-policy check used by CI is:

```bash
cargo install --locked --version 0.20.2 cargo-deny
cargo deny --log-level warn --manifest-path ./Cargo.toml --all-features check
```

[`deny.toml`](../deny.toml) defines the vulnerability, license, duplicate-crate
and dependency-source policy. A new advisory can fail CI even when source code
has not changed.

## Python tooling and CPU adapter tests

The complete suite uses two separate locked environments:

- **Development:** pytest, Ruff, mypy and the validation package.
- **Worker:** the locked Torch/vLLM dependencies needed by CPU adapter subprocesses.

The tests also need a Rust CPU transport reader. Installing worker dependencies
does not mean GPU inference runs in CI. The complete CI job rejects skipped
cases, missing adapters and empty test results.

These Bash commands mirror that job:

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

If you set `CARGO_TARGET_DIR`, adjust `GOLDEN_TRANSPORT_TEST_BINARY` to the actual
build location. A focused or partial suite is useful during development; report
it as such. It does not replace the complete CI job.

## Documentation checks

Check whitespace and review the changed links from the source root:

```bash
git diff --check
```

Resolve relative links against their containing document, check that local
files and heading anchors exist, and verify changed command examples against
the code in this version. Maintainers can also apply the
[current main-branch Markdown checker](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/tools/check_docs.py)
from a separate checkout; it is not bundled with this version. External URLs
and runtime behavior of examples need separate review. Keep release-specific
claims tied to the appropriate version and measured environment.

## Code and design conventions

- Use Rustfmt, configured with `max_width = 100` in `rustfmt.toml`.
- Follow the focused Clippy lints in `[workspace.lints.clippy]` in `Cargo.toml`.
- Workspace Rust code denies `unsafe_code`; a necessary module-level exception
  must document its safety boundary.
- Use the domain vocabulary in [CONTEXT.md](../CONTEXT.md), especially the public
  generation contract and engine/cache ownership terms.
- Record significant architecture or public API decisions in
  [ADRs](adr/). Read their status and scope: older proposals and superseded
  decisions are historical context, rather than promises of current behavior.

Use commit prefixes such as `feat`, `fix`, `docs`, `test`, `ci` and `chore`.
For PRs, describe the concrete behavior and compatibility impact; the size of
the explanation should match the change.

## GPU validation and release evidence

CPU CI checks compilation, tests and tooling. It cannot establish CUDA model
accuracy or GPU performance. A documentation-only change does not require
repeating model validation.

The [public GPU smoke guide](validation/public-gpu-smoke.md) covers production
CUDA builds, warmup, repeated `LLM::generate` calls, mixed requests and the exact
Quick Start command. Local GPU validation, including on WSL2, can supply these
results. The separate [CUDA workflow](../.github/workflows/cuda-smoke.yml) is
manual and requires an already configured GPU runner; it is not an ordinary PR
requirement.

For numerical changes, use the [layered accuracy contract](validation/layered-accuracy-contract.md)
and [validation tooling](../tools/golden-gen/README.md). Report the source commit,
model revision, GPU/driver/CUDA environment, configuration, workload and actual
results. The [published validation summary](releases/goldens-v0.2.md) records
the released configuration; it does not certify untested hardware or models.

Keep generated fixtures and full raw evidence local and outside git. Evidence
uploads require an explicit request. For publication or release-description
changes, follow the [maintainer release guide](releases/README.md).
