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

These are CPU checks and do not require model weights, CUDA Toolkit or a GPU. They check compilation, formatting, lints and tests; they do not validate model output. The archive declares Rust 1.75 but has no pinned toolchain or GitHub CI workflow, so passing current-version CI is not evidence that the original source passes this historical minimum or all modern lint rules. Run and report actual results when changing it.

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

The `dev` extra installs test/type/lint tools; GPU oracle packages are an optional `gpu` extra. CPU tests exercise schema, fixture I/O, prompt expansion, calibration, fake-oracle generation and CLI orchestration. They do not run actual Transformers/vLLM inference. The archived code may not satisfy newer tool versions; this guide describes check commands without inventing new passing results.

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
