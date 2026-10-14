# Documentation

[English project overview](../README.md) · [中文项目介绍](../README.zh-CN.md)

vllm-oxide v0.2.0 runs synchronous, offline [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) BF16 generation on one CUDA GPU. It is available as source. Start with Getting started to build the engine and run your first prompt, then use the guides below for API details, development, or validation.

## Use the engine

| Document | What it helps you do |
|---|---|
| [Getting started](getting-started.md) | Build the released source and generate with a pinned model revision |
| [CLI reference](getting-started.md#cli) | Run one prompt or stdin, select sampling options, and understand model-source syntax |
| [Rust API](getting-started.md#rust-api) | Reuse an `LLM`, submit batches, configure the engine, and handle outputs |
| [Migrate from v0.1](migration-v0.2.0.md) | Update removed internal imports, request identity, and stricter input validation |
| [Troubleshooting](getting-started.md#common-errors) | Resolve CUDA, checkpoint, context-budget, and memory errors |
| [Support](../SUPPORT.md) | Ask a question or report a reproducible problem |

## Develop and contribute

| Document | What it helps you do |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Choose a contribution and follow repository conventions |
| [Development](development.md) | Set up CPU checks, navigate the workspace, and select relevant GPU validation |
| [Architecture](architecture.md) | Follow a generation call through scheduling, execution, cache, and sampling |
| [Project vocabulary](../CONTEXT.md) | Understand domain names and ownership boundaries |
| [Security policy](../SECURITY.md) | Report a vulnerability privately |

## Validate and measure

| Document | What it helps you do |
|---|---|
| [Validation guide](validation/README.md) | Distinguish automatic CPU checks, CUDA function checks, and numerical evidence |
| [v0.2.0 validation summary](releases/goldens-v0.2.md) | Inspect the completed release checks and their actual source/environment identities |
| [Benchmarks](benchmarks/README.md) | Understand end-to-end throughput and memory measurements alongside focused microbenchmarks |
| [Validation tooling](../tools/golden-gen/README.md) | Prepare maintainer evaluation environments and run the validation tools |

Ordinary GitHub CI runs on CPU. Real CUDA API/CLI behavior, numerical accuracy, and performance are validated separately on a GPU; no GitHub GPU runner is required for ordinary contributions. Raw release evidence remains local.

## Releases and history

| Document | What you will find |
|---|---|
| [Release index and maintainer guide](releases/README.md) | All published versions, shared release format, and publication checks |
| v0.2.0: [English](releases/v0.2.0.en.md) / [中文](releases/v0.2.0.md) | Current supported scope, compatibility, validation, and installation |
| v0.2.0.rc.1: [English](releases/v0.2.0-rc.1.en.md) / [中文](releases/v0.2.0-rc.1.md) | Earlier candidate and its recorded validation limits; filenames retain the Cargo version spelling |
| v0.1.0: [English](releases/v0.1.0.en.md) / [中文](releases/v0.1.0.md) | Initial release and historical API scope |
| [Changelog](../CHANGELOG.md) | Changes grouped by version |
| [GitHub Releases](https://github.com/RedHeartSecretMan/vllm-oxide/releases) | Source tags, archives, and published descriptions |
| [Architecture decision records](adr/README.md) | Topic guide, complete record index, original statuses and dated proposals |
| [Research and prototypes](../research/README.md) | Exploratory work and community-practice references |

Use the final v0.2.0 notes and validation summary for the shipped release. Earlier calibration, diagnosis, and evidence-transport documents preserve historical decisions; a historical “pending” or “proposed” status does not describe the current product release.

## Repository automation

Contributor-facing rules live in [CONTRIBUTING.md](../CONTRIBUTING.md). For task automation, see [AGENTS.md](../AGENTS.md), [issue workflow](agents/issue-tracker.md), [triage labels](agents/triage-labels.md), and [domain documentation rules](agents/domain.md).
