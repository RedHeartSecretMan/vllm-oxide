# Maintainer tools — RC source

[Documentation](../docs/README.md) · [Development](../docs/development.md) · [Validation](../docs/validation/README.md)

Use the RC tools below to generate and inspect validation evidence. This version predates `public_gpu_smoke`, release source-role adapters, and direct-NVML tools.

| Entrypoint | Purpose | Execution and result |
|---|---|---|
| [golden-gen package](golden-gen/README.md) | Layered numerical/operator/behavior collection, CPU validation and schema5 artifact readers | Python 3.12; CPU tests/readers, GPU model owners; source-bound raw local records |
| [Diagnostic scripts](golden-gen/scripts/README.md) | Tiny BF16 rounding probes and frozen-case layer localization | CPU/CUDA operator probe or CUDA reference capture plus CPU comparison; always nonaccepting |
| [validate-release.sh](validate-release.sh) | Earlier ADR-0012/schema-v4 staged workflow | Historical compatibility tool with fixed paths; `authoritative` and `publish` disabled |
| [Rust test harness](../crates/vllm-oxide-test/Cargo.toml) | Fixed-prefix/operator/behavior captures, benchmark producer and bundle consumer | Cargo workspace binaries; CUDA only for owning model execution; CPU for transport |
| [Candle stack prototype](../research/prototype-candle-stack/README.md) | Independent raw-pointer and paged-kernel experiment | Separate Cargo manifest, synthetic GPU tensors; not Qwen3 validation |

## Choose the correct tool

- To run Qwen3, use the public [CLI/API](../docs/getting-started.md).
- Develop the harness through its [complete CPU tests](golden-gen/README.md#development-setup-and-cpu-checks). A synthetic dry-run never establishes model accuracy.
- Inspect numerical evidence with the [layered contract](../docs/validation/layered-accuracy-contract.md); schema-v4 data cannot satisfy schema5 layered acceptance.
- Use [diagnostic scripts](golden-gen/scripts/README.md) to answer their fixed arithmetic/localization questions, preserving their nonaccepting status.
- Read [RC results](../docs/releases/v0.2.0-rc.1.en.md) before citing performance or accuracy: full RC performance was incomplete and 18 numerical cases failed.

## State, dependencies and evidence

Prepare locked development/worker environments outside the source tree. Keep generated fixtures, logs, caches, captures, release bundles and model weights local and out of git. The historical publication implementations remain available for source inspection, but this repository's current policy does not upload raw evidence and these instructions do not invoke publication.

Source-bound readers require the original clean source, model identity, inputs and complete raw dependency closure. Editing documentation does not make an older capture a new measurement; historical source hashes and failures remain unchanged. Ordinary CPU CI needs no GitHub GPU runner. WSL2 can host relevant local CUDA checks.
