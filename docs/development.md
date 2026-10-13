# Development guide — RC source

[Documentation](README.md) · [Contributing](../CONTRIBUTING.md)

Use `main` for new contributions. This guide describes the actual historical `v0.2.0.rc.1` tree when reproducing or inspecting it; later CI jobs and source tools are not included here.

## Environment and source layout

| Component | Role |
|---|---|
| Rust 1.94.0, Rustfmt and Clippy | Recorded RC and maintained source toolchain; Rust 1.94 minimum |
| `crates/vllm-oxide` | Public `LLM` API, internal engine, models and CUDA kernels |
| `crates/vllm-oxide-cli` | One raw-text prompt via the library API |
| `crates/vllm-oxide-test` | Unsupported numerical capture, benchmark and transport executables |
| `tools/golden-gen` | Python 3.12 validation package, locked CPU/GPU dependencies, fixtures and CPU tests |
| `research/prototype-candle-stack` | Separate experimental Cargo project; not a workspace member, run from an isolated copy |
| `docs/adr`, `docs/validation` | Original decisions, frozen inventories, budgets and contracts |

Rust CPU checks need neither CUDA nor model downloads. Installing Torch/vLLM worker dependencies for adapter tests does not imply GPU inference.

## Rust checks

Run from the repository root:

```bash
cargo +1.94.0 build --locked --workspace
cargo +1.94.0 fmt --all --check
cargo +1.94.0 clippy --locked --workspace --all-targets --features internal-golden -- -D warnings
cargo +1.94.0 test --locked --workspace
cargo +1.94.0 test --locked --workspace --features internal-golden
cargo +1.94.0 doc --locked -p vllm_oxide --no-deps
cargo deny --frozen check
```

The historical [GitHub workflow](../.github/workflows/ci.yml) runs current stable format, Clippy, default tests and cargo-deny on `ubuntu-latest`; it has no Python job, Rust-version matrix or documentation checker. The additional local commands above exercise the internal harness as well. Report exactly which commands ran. The nine-gate producer in `golden_gen.release_cpu` also requires a complete prepared dependency cache for its `--offline` commands.

## Python checks

Prepare dependencies in external environments and a CPU Rust consumer; see the [tooling instructions](../tools/golden-gen/README.md#development-setup-and-cpu-checks). Run all adapter tests with `GOLDEN_WORKER_TEST_PYTHON` and `GOLDEN_TRANSPORT_TEST_BINARY` configured. A partial test selection is useful while iterating, but skipped worker or transport checks do not establish complete release tooling verification.

## Select relevant validation

| Change | Validation |
|---|---|
| Markdown or links | Local link/anchor review and shell syntax for changed examples; no GPU rerun |
| Scheduler, cache or input behavior | Focused regression tests, then workspace CPU checks |
| Python evidence tooling | Complete CPU adapters, Ruff and mypy |
| CUDA/model/sampling arithmetic | CPU checks plus relevant real-model CUDA measurements |
| Dependencies | Lockfile consistency and dependency policy |

The modern repository's `tools/check_docs.py` can inspect this historical checkout through `--root`; that checker is not shipped in this RC tree. Validate new documents as well as tracked ones before committing. External URLs and successful runtime execution of examples need separate review.

## Implementation conventions

Use Rustfmt and the workspace Clippy rules. Preserve the ownership vocabulary in [CONTEXT.md](../CONTEXT.md), avoid exposing internal execution types through the public API, and describe necessary unsafe CUDA boundaries with their safety assumptions. Record a material interface or architecture choice in [ADRs](adr/README.md).

Keep bilingual README scope aligned. Describe the concrete trigger, resulting behavior, compatibility impact and actual validation in a PR. Keep generated fixtures, local environments, captures and model weights outside git.

## Local GPU work

Build inference with `--features cuda`, then run the pinned example or CLI from [getting started](getting-started.md). The RC has no later `public_gpu_smoke` Python entrypoint or manual GitHub CUDA workflow. For numerical evidence use its own [layered contract](validation/layered-accuracy-contract.md) and [tool version](../tools/golden-gen/README.md). Do not substitute the final release's source or results.

WSL2 is a usable local CUDA environment. The historical harness records resource and process observations; lack of a GitHub GPU runner is not a product or documentation release blocker. RC accuracy/performance limits remain those recorded in its [release notes](releases/v0.2.0-rc.1.en.md). New raw evidence and fixtures remain local; no upload is part of these instructions.
