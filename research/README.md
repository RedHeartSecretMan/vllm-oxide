# Research and prototypes

[Project overview](../README.md) · [Documentation](../docs/README.md) · [Architecture](../docs/architecture.md)

This directory contains the Candle/CUDA stack experiment, historical accuracy
research, and notes on repository documentation. Use them to investigate the
questions listed below. To build and run the supported engine, follow the
[getting-started guide](../docs/getting-started.md).

| Material | Purpose | Status |
|---|---|---|
| [Candle stack prototype](prototype-candle-stack/README.md) | Inspect a raw tensor-pointer/stream CUDA call and a concrete paged-FlashAttention call using synthetic tensors | Retained T10 experiment; its detailed guide explains the standalone build, dependencies, checks and limits |
| [vLLM community practices](vllm-community-practices.md) | Source-backed comparison of user documentation, contribution paths, and release presentation | Repository design reference; upstream rules do not automatically apply here |
| [Accuracy proposal](inference-accuracy-acceptance-proposal.md) | Preserve the 2026-09-06 design before the adopted layered protocol | Historical draft; current budgets and acceptance come from the linked ADRs and release summary |
| [Accuracy industry review](inference-accuracy-validation-industry-review.md) | Compare fixed primary-source examples and explain their limits | Dated research; no universal KL tolerance or independent release approval |

## Reproduce or inspect an experiment

The prototype uses synthetic inputs without model downloads or CLI flags. Its
independent workspace and committed lockfile support direct runs from this
checkout. CPU tests cover the output validator; CUDA runs check a constant-fill
readback, launch/execution errors and the shape plus every finite attention
output value. These checks verify interoperability. Model accuracy and production performance
require their own measurements. Read its
[source and dependencies](prototype-candle-stack/README.md#source-and-dependencies)
before rebuilding the pinned CUDA stack.

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
