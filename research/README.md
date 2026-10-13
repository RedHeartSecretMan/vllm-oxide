# Research — v0.1.0

[Project](../README.md) · [Documentation](../docs/README.md) · [Architecture](../docs/architecture.md)

This directory contains the historical, standalone Candle/CUDA stack experiment. Its purpose is to investigate dependencies and kernel interoperability; it is not the production `LLM` API, model installation guide or release acceptance harness.

| Material | Purpose | Status |
|---|---|---|
| [Candle stack prototype](prototype-candle-stack/README.md) | Check Candle-to-cudarc FFI and the pinned paged FlashAttention entry point | Historical experiment, independent workspace; CUDA required |

The experiment has its own Cargo manifest, committed lockfile, build script,
CUDA kernel and CPU validation tests. Root workspace checks do not include it;
use the [direct run procedure](prototype-candle-stack/README.md#prepare-and-run)
for its independent workspace. Its seeded synthetic input, full finite-output
check and experiment geometry establish interoperability, without describing
production memory use or model numerical accuracy.

Use [getting started](../docs/getting-started.md) for this source's engine commands and [validation](../docs/validation/README.md) for its original numerical limitations. The [golden-generation ADR](../docs/adr/0005-golden-generation-correctness-strategy.md) records a historical design decision. Current application work should use [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0).

When recording experiments, identify the exact code/dependency revisions, toolchain, device, command and output. A successful kernel invocation establishes interoperability, not model accuracy or performance parity. Keep generated binaries, logs and model/evidence data out of git; preserve any retained result before cleaning a build directory.
