# Validation — historical RC

[Documentation](../README.md) · [Development](../development.md) · [Tools](../../tools/README.md)

The RC's recorded numerical result is **48 of 66 PASS, 18 FAIL**. All 487 capture items and 153 groups were checked; seven operator profiles and 22 public behavior checks passed. Basic CPU checks and short CUDA CLI generation passed, but full performance validation and the historical Rust-minimum declaration were incomplete. See the original results, budgets and hashes in the [RC notes](../releases/v0.2.0-rc.1.en.md).

| Check | Execution | Establishes |
|---|---|---|
| Rust 1.94.0/stable checks and native dependency audit | Maintained CPU CI | Default/internal-golden compilation and tested structural behavior |
| Python unit/adapter and transport tests | Local CPU | Evidence-reader, producer, failure and archive contracts |
| CUDA CLI/example | Local GPU, including WSL2 | Build and selected public generation behavior |
| Layered accuracy | Fixed source/model/history, GPU collection plus CPU evaluation | Each declared operator, numerical and behavior check under frozen budgets |
| Performance | Local GPU with original synchronized timestamps and memory telemetry | Only the fixed measured workload and environment |

CPU tests and same-token smoke results cannot cancel a KL failure. Final v0.2.0 acceptance uses later source identities; it does not change this RC result. A documentation refresh is not a model measurement.

## Contracts in this tree

- [Layered accuracy](layered-accuracy-contract.md): L0 operator, L1 full-logit numerical and L2 public behavior checks.
- [Bounded supervision](layered-supervision-contract.md): separate measurement and guard records, ownership and interrupted-stage evidence.
- [Layered assets](layered-release-assets.md): historical schema5 packing and clean-reader design.
- [Mixed-phase pressure repair](t45-mixed-phase-pressure-repair.md): scoped earlier diagnosis and verification.
- [Frozen budget decision](../adr/0018-freeze-layered-accuracy-budgets.md) and [calibration history](../adr/README.md).

The original protocol documents retain historical source approval and publication machinery. Current repository policy keeps raw evidence local and does not instruct contributors to upload an evidence bundle. Do not re-label old failures, pending budgets, incomplete inventories or diagnostic probes as release acceptance.
