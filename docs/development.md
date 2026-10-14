# Development — v0.1.0

[Documentation](README.md) · [Contributing](../CONTRIBUTING.md) · [Tools](../tools/README.md)

This guide covers the archived source. New fixes and features belong on current `main`; use the [current development guide](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/docs/development.md) for its toolchain and CI matrix.

## Rust checks

Run from the repository root:

```bash
cargo check --locked --workspace --all-targets
cargo test --locked --workspace
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
```

These CPU checks cover compilation, formatting, lints, and tests without model weights, a CUDA Toolkit, or a GPU. Model output needs separate validation. Use the maintained archive's pinned Rust 1.94.0; the original Rust 1.75 declaration did not match its locked dependencies. This archive has no GitHub CI workflow, so run the checks locally and report their results.

Enable the library CUDA backend only for GPU builds. The historical CLI needs `--features vllm_oxide/cuda`; the core and comparison packages declare `cuda` themselves. The standalone research prototype always enables CUDA and is outside workspace checks.

## Python tooling

`tools/golden-gen/pyproject.toml` and `.python-version` require Python 3.12. From the generator directory:

```bash
cd tools/golden-gen
uv sync --locked --extra dev
uv run --locked --extra dev pytest
uv run --locked --extra dev ruff check .
uv run --locked --extra dev ruff format --check .
uv run --locked --extra dev mypy src
```

The `dev` extra installs test/type/lint tools; GPU oracle packages are an optional `gpu` extra. CPU tests exercise schema, fixture I/O, prompt expansion, calibration, fake-oracle generation and CLI orchestration. They do not run actual Transformers/vLLM inference. Use the locked tool versions when reproducing checks, and record the results of the commands you run.

## Where changes belong

| Area | Location |
|---|---|
| Composition, generation and loading | `crates/vllm-oxide/src/llm.rs`, `config.rs`, `loader.rs` |
| Scheduling and paged-cache ownership | `crates/vllm-oxide/src/engine/` |
| Models, layers, attention and CUDA | `crates/vllm-oxide/src/models/`, `layers/`, `attention/`, `kernels/` |
| Offline command parsing | `crates/vllm-oxide-cli/` |
| Historical Rust comparison | [vllm-oxide-test](../crates/vllm-oxide-test/README.md) |
| Fixture generation | [golden-gen](../tools/golden-gen/README.md) |
| Independent CUDA stack experiment | [Prototype](../research/prototype-candle-stack/README.md) |

Keep generated fixtures, `.venv`, caches, target directories, model weights and raw evidence outside git. Read the [tool artifact guidance](../tools/README.md#artifacts-and-cleanup) before deleting outputs. A documentation correction requires link and command review, not fresh GPU collection. Preserve historical FAIL/skipped/partial results and original evidence identities.

Domain terms live in [CONTEXT.md](../CONTEXT.md); the [architecture guide](architecture.md) separates implemented behavior from historical future plans. Document unsafe CUDA and mmap boundaries near their implementation. Use focused changes and meaningful validation rather than tests that duplicate implementation details.
