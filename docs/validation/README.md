# Validation

[Documentation](../README.md) · [Architecture](../architecture.md) · [Benchmarks](../benchmarks/README.md)

v0.2.0 completed numerical, behavior, public CUDA, and performance validation. The [release validation summary](../releases/goldens-v0.2.md) contains the results and their source commits, model revision, environment, and report hashes. Use the checks below according to the behavior you are changing.

## Choose the check for the change

| Check | Environment | What it establishes | Entry point |
|---|---|---|---|
| Ordinary CI | CPU GitHub runner or local CPU environment | Formatting, compilation, Clippy, Rust tests, Python tooling/adapters, and dependencies | [Development guide](../development.md), [CI workflow](../../.github/workflows/ci.yml) |
| Public CUDA smoke | Local CUDA GPU; optional manual self-hosted runner | Production build, real model warmup, repeated `LLM::generate`, mixed prompt types, and CLI generation | [Public GPU smoke](public-gpu-smoke.md), [manual workflow](../../.github/workflows/cuda-smoke.yml) |
| Layered numerical and behavior checks | Pinned GPU model and oracle environments | Operator invariants, same-prefix prediction distributions, reference-relative token choices, and generation behavior | [Layered accuracy contract](layered-accuracy-contract.md), [golden-gen tooling](../../tools/golden-gen/README.md) |
| Performance measurements | Fixed GPU environment | Throughput, first-token and inter-token timings, and sampled device memory for declared workloads | [Benchmarks](../benchmarks/README.md) |

CPU tests, generation smoke checks, and numerical comparisons answer different questions. Run the checks relevant to your change and report which ones executed; model accuracy requires the numerical comparisons.

## Completed v0.2.0 results

| Area | Recorded result |
|---|---|
| Accuracy and behavior | Independent `PASS`; 747/747 capture items and 231/231 execution groups, including 104 numerical, 44 behavior, and 7 operator checks |
| Public CUDA interface | Production build, real warmup, 100 repeated API calls, three mixed requests, 107 unique request IDs, and CLI output `Paris.` |
| Performance | Two fixed workloads, three measured repetitions each; all six sample/config/output and memory records validated |
| CPU CI | Rust 1.94.0/stable, Python quality and complete CPU adapters, and dependency audit passed for the integrated implementation |

The release scope is Qwen3-0.6B at revision `7e4ae267688d671ddfca3122e4528ee980cf3234`, BF16, on one GPU. Measurements used RTX 4080/SM 8.9 with WSL2, which is a supported runtime environment. Local GPU checks supply the release evidence; a hosted GitHub GPU runner is optional.

Results remain bound to their actual measurement and evaluation sources. The summary does not reclassify older failures, certify every model/hardware configuration, or claim GPU exclusivity. Full raw captures and logs are local; the source release does not provide a downloadable golden evidence package.

## Protocol and maintainer reference

| Document | Role |
|---|---|
| [Layered accuracy contract](layered-accuracy-contract.md) | Numerical and behavior methodology, written at its original design checkpoint |
| [Frozen budgets](layered-accuracy-budgets.json) and [case registry](layered-accuracy-cases.json) | Machine-readable comparison limits and case identities |
| [Baseline determinism](baseline-determinism.md) | Reference/baseline replay requirements and diagnostics |
| [Calibration policy binding](calibration-policy-binding.md) | Binding measured observations to the approved policy |
| [Resource supervision](layered-supervision-contract.md) and [policy](layered-supervision-policy.json) | Guard behavior and recorded resource limits |
| [Causal GEMM layouts](causal-gemm-layouts.md) | Token/history layout coverage for numerical execution |
| [Local evidence decision](../adr/0028-local-release-evidence.md) | Local evidence retention and source-release publication scope |
| [Release guide](../releases/README.md) | Checks and metadata for preparing a release |

## Historical investigation and transport designs

The following records explain how the final implementation and approval developed. Preserve their original status, thresholds, and failures when reading or citing them; use the completed release summary for current results.

| Record | Topic |
|---|---|
| [Accuracy diagnosis](v0.2.0-accuracy-diagnosis.md) | Earlier model-output mismatch investigation |
| [Mixed-phase pressure repair](t45-mixed-phase-pressure-repair.md) | Scheduler/cache pressure regression investigation |
| [Fresh calibration observation](fresh-calibration-observation.md) | A non-accepting measurement stage |
| [Layered release assets](layered-release-assets.md) | Earlier evidence manifest/archive transport design |
| [Performance and calibration protocol](../adr/0012-goldens-v0.2-calibration-and-performance-protocol.md) | Original workloads and release-stage design before later revisions |
| [Source-role proposal](../adr/0029-proposed-release-source-roles.md) | Historical proposal to retain original numerical and performance identities |

Read pending checkpoints and publication plans as records of the protocol at that time. The completed release summary gives the shipped source release's results.
