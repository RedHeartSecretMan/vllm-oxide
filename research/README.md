# Research and prototypes

[Project overview](../README.md) · [Documentation](../docs/README.md) · [Architecture](../docs/architecture.md)

This directory keeps two kinds of development material: a retained executable
stack experiment, and a source-backed repository-design note. Choose the
material according to the question you want to answer. The supported engine and
installation path are described in the [getting-started guide](../docs/getting-started.md).

| Material | Purpose | Status |
|---|---|---|
| [Candle stack prototype](prototype-candle-stack/README.md) | Inspect a raw tensor-pointer/stream CUDA call and a concrete paged-FlashAttention call using synthetic tensors | Retained T10 experiment; its detailed guide explains the standalone build, dependencies, checks and limits |
| [vLLM community practices](vllm-community-practices.md) | Source-backed comparison of user documentation, contribution paths, and release presentation | Repository design reference; upstream rules do not automatically apply here |

## Reproduce or inspect an experiment

The prototype has no model download, CLI flags, or dedicated tests. It is
outside the root Cargo workspace, so its guide uses a fresh external copy for
execution rather than a failing in-tree `cargo run`. Its PASS messages cover a
constant-fill readback and an attention shape/sampled-NaN check; they do not
establish numerical model accuracy or performance. Read its
[source and dependency map](prototype-candle-stack/README.md#source-and-dependency-map)
before rebuilding the pinned CUDA dependencies.

The repository-design note links the official pages consulted and records the
scope and date of its conclusions. Advice about contribution or release
presentation should be checked against the current first-party guides before
applying it to a new release.

For BF16 residual/RMSNorm/RoPE/SiLU probes and fixed historical layer
localization, use the [diagnostic scripts guide](../tools/golden-gen/scripts/README.md).
Those scripts live with the validation tooling because they import its model,
identity and replay helpers; their diagnostic reports remain nonaccepting.

Accepted architecture choices live in [docs/adr](../docs/adr/); the [architecture overview](../docs/architecture.md) describes their current implementation. Numerical and performance results belong in [validation](../docs/validation/README.md) and [benchmarks](../docs/benchmarks/README.md), with the actual measurement source and environment preserved.

Keep experiment logs, tensors, local model files and generated build artifacts
outside tracked source. Record the tested commit, dependencies, command,
environment and exit status alongside an experiment; do not turn an expected
value in source or a historical README statement into a newly measured result.
