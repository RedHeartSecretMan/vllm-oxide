# Research and prototypes

[Project overview](../README.md) · [Documentation](../docs/README.md) · [Architecture](../docs/architecture.md)

This directory keeps exploratory work and references used to develop vllm-oxide. Its prototypes and notes are not the supported Rust API, the production engine, or installation instructions for the current release.

| Material | Purpose | Status |
|---|---|---|
| [Candle stack prototype](prototype-candle-stack/README.md) | Explore Candle/CUDA interoperability and candidate attention-kernel entrypoints | Historical, independent experiment; use the main workspace for the shipped engine |
| [vLLM community practices](vllm-community-practices.md) | Source-backed comparison of user documentation, contribution paths, and release presentation | Repository design reference; upstream rules do not automatically apply here |

The prototype's README records the assumptions and toolchain at the time of the experiment. Its run command, hardware scope, and architecture claims do not override the current [getting-started guide](../docs/getting-started.md) or [release notes](../docs/releases/v0.2.0.en.md).

Accepted architecture choices live in [docs/adr](../docs/adr/); the [architecture overview](../docs/architecture.md) describes their current implementation. Numerical and performance results belong in [validation](../docs/validation/README.md) and [benchmarks](../docs/benchmarks/README.md), with the actual measurement source and environment preserved.
