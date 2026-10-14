# Documentation — v0.2.0.rc.1

[English overview](../README.md) · [中文介绍](../README.zh-CN.md)

This is the historical experimental RC source tree, refreshed to the final release's documentation format. The archive includes reviewed source maintenance; the original measurements and **18/66 numerical failures** retain their source identities. New users should use [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0).

| Goal | Read |
|---|---|
| Build, model sources, CLI, Rust API and troubleshooting | [Getting started](getting-started.md) |
| Move from the original v0.1 public surface | [Migration](migration-v0.2.0.md) |
| Navigate execution and cache ownership | [Architecture](architecture.md), [vocabulary](../CONTEXT.md) |
| Contribute and run relevant checks | [Contributing](../CONTRIBUTING.md), [development](development.md) |
| Understand what was actually verified | [Validation](validation/README.md), RC notes: [English](releases/v0.2.0-rc.1.en.md) / [中文](releases/v0.2.0-rc.1.md) |
| Understand benchmark boundaries | [Benchmarks](benchmarks/README.md) |
| Inspect the standalone CUDA experiment | [Research](../research/README.md) |
| Reproduce maintainer tooling and diagnostics | [Tools](../tools/README.md), [golden-gen](../tools/golden-gen/README.md) |
| Review decisions at their original checkpoints | [Architecture decision records](adr/README.md) |
| Find source/archive provenance and release history | [Documentation maintenance](version-doc-update.md), [releases](releases/README.md), [changelog](../CHANGELOG.md) |
| Ask for help or report a vulnerability | [Support](../SUPPORT.md), [security](../SECURITY.md), [community standards](../CODE_OF_CONDUCT.md) |

Ordinary GitHub CI in this tree is CPU-only. Its Rust 1.94.0/stable matrix runs formatting, Clippy, default/internal-golden tests and a separate native dependency audit; Python adapters and real CUDA/model validation are separate local checks. No GitHub GPU runner is required for a documentation contribution. Raw evidence stays local.

Automation conventions: [AGENTS.md](../AGENTS.md), [issue workflow](agents/issue-tracker.md), [triage labels](agents/triage-labels.md), [domain rules](agents/domain.md).
