# Documentation

Start with the [English README](../README.md) or [中文 README](../README.zh-CN.md). v0.2.0 supports synchronous, offline Qwen3-0.6B BF16 generation on one CUDA GPU. The project is released as source; the links below separate usage, development, and the records behind that release.

## Use and upgrade

| Document | What you will find |
|---|---|
| [Getting started](getting-started.md) | Source installation, model sources, CLI, Rust API, configuration, and common errors |
| [Migrate from v0.1 to v0.2](migration-v0.2.0.md) | Breaking API changes, request identity, and stricter input validation |
| v0.2.0 release notes: [English](releases/v0.2.0.en.md) / [中文](releases/v0.2.0.md) | Supported scope, fixes, validation, and known limitations |
| [Changelog](../CHANGELOG.md) | Version-by-version changes |
| [Getting help](../SUPPORT.md) | Where to ask questions and report problems |

## Contribute

| Document | What you will find |
|---|---|
| [Contributing](../CONTRIBUTING.md) | CPU development setup, coding conventions, checks, and pull requests |
| [Security policy](../SECURITY.md) | How to report vulnerabilities |
| [Issue workflow](agents/issue-tracker.md) | Ticket structure and dependencies |
| [Triage labels](agents/triage-labels.md) | How issues move toward implementation |
| [Agent instructions](../AGENTS.md) | Repository conventions for coding agents |

## Understand the implementation

The synchronous `LLM` composition root connects request scheduling, paged cache management, model execution, and sampling. Internal engine and model types are intentionally hidden from the public API.

| Document | What you will find |
|---|---|
| [Project context and vocabulary](../CONTEXT.md) | Domain terms and module responsibilities |
| [Engine architecture](adr/0004-engine-architecture-and-dependency-dag.md) | Module ownership and dependency boundaries |
| [Public generation contract](adr/0011-public-generation-contract.md) | The six-type public API and internal diagnostics boundary |
| [Request identity](adr/0008-request-identity-and-output-order.md) | Stable request IDs and input-order outputs |
| [Bounded reference arithmetic](adr/0020-bounded-reference-arithmetic.md) | Why the release uses FP32 intermediate arithmetic |
| [All architecture decisions](adr/) | Accepted decisions and historical proposals, each with its own status |

## Validate and release

Automatic CI runs on CPU. Local GPU checks establish model accuracy, public CUDA behavior, and performance. No GPU GitHub runner is required for ordinary contributions. The release records retain their actual source and environment identities.

| Document | What you will find |
|---|---|
| [v0.2.0 validation summary](releases/goldens-v0.2.md) | Completed accuracy, behavior, CUDA smoke, performance, and CPU CI results |
| [Public GPU smoke](validation/public-gpu-smoke.md) | Production CUDA build, repeated API calls, mixed requests, and CLI checks |
| [Layered accuracy contract](validation/layered-accuracy-contract.md) | Operator, numerical, and behavior comparison rules |
| [Frozen accuracy budgets](validation/layered-accuracy-budgets.json) | Machine-readable numerical limits |
| [Validation tooling](../tools/golden-gen/README.md) | Maintainer setup and evaluation entrypoints |
| [Performance measurement protocol](adr/0012-goldens-v0.2-calibration-and-performance-protocol.md) | Fixed workloads and measurement methodology |
| [Sampler microbenchmark](benchmarks/greedy-sampling.md) | A focused kernel benchmark, separate from end-to-end throughput |
| [Maintainer release guide](releases/README.md) | Release checks, source tags, publication, and description updates |
| [Release notes template](releases/template.md) | User-facing highlights, compatibility, artifacts, and validation |
| [GitHub Releases](https://github.com/RedHeartSecretMan/vllm-oxide/releases) | Published source tags and release descriptions |

The [RC1 notes](releases/v0.2.0-rc.1.md), validation diagnoses, and older evidence transport designs remain available as historical records. Use the final [v0.2.0 release notes](releases/v0.2.0.md) and [validation summary](releases/goldens-v0.2.md) for the shipped release. Raw release evidence remains local; this source release does not publish a downloadable golden evidence package.
