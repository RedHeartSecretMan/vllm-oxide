# Contributing to vllm-oxide

Thank you for your interest in contributing. This document covers the development workflow.

## Development Environment

- Rust edition 2021, MSRV 1.94.0 (pinned in `rust-toolchain.toml`)
- CPU-only development: `cargo build` works without CUDA
- CUDA feature: `cargo build --features cuda --release` requires NVIDIA GPU (sm_89+)
- Linux only (CUDA requirement)

## Workflow

1. Pick an issue (look for `ready-for-agent` or `good first issue` labels)
2. Create a branch: `feat/t<NN>-<slug>` (e.g., `feat/t24-vllm-oxide-cli`)
3. Implement with tests
4. Ensure all CI checks pass locally
5. Open a PR against `main`

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

Every PR runs Rust checks on 1.94.0 and stable, complete Python CPU checks, and dependency auditing:

**ci** (ubuntu-latest, Rust 1.94.0/stable matrix):
1. `cargo fmt --check` — formatting
2. `cargo clippy --all-targets -- -D warnings` — lint gate
3. `cargo test --locked --workspace` and `cargo test --locked --workspace --features internal-golden` — unit, integration and doc tests (CPU-only)

**python**: locked development and worker environments from `tools/golden-gen/uv.lock`; all pytest cases (including CPU worker/transport adapters), Ruff and mypy. Missing adapters, skipped cases and empty results fail the gate. Mypy uses the CPU development environment; the separate worker environment supplies pinned Torch/vLLM imports for CPU adapter tests.

The manual [public GPU smoke workflow](docs/validation/public-gpu-smoke.md) exercises real warmup, repeated `LLM::generate`, mixed requests and the exact Quick Start on a CUDA runner.

**deps** (cargo-deny):
4. `cargo deny check` — advisory, license, ban, and source audits

Note: CI green does NOT mean numerically validated. Numerical correctness requires the GPU release gate (see README, Testing section, Tier 2).

## Architecture Decisions

Significant design decisions are recorded as ADRs in `docs/adr/`. If your change alters architecture or public API, propose an ADR.

## Domain Vocabulary

Use the vocabulary defined in `CONTEXT.md`. Do not introduce synonyms for established terms (check the "Avoid" notes). Key terms to know: `CausalLM`, `EngineCore`, `BlockPool`, `PagedKVCache`, `Scheduler`, `Prompt`, `SamplingParams`, `RequestOutput`.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
