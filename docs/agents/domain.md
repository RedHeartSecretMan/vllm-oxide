# Domain documentation

[Documentation index](../README.md) · [Project context](../../CONTEXT.md)

This repository uses one project context and a shared ADR directory. Start with
[CONTEXT.md](../../CONTEXT.md) when investigating module ownership, public API
contracts or domain terminology. Read the ADRs relevant to the proposed change
before deciding how to implement it.

```text
CONTEXT.md                         Project vocabulary and module responsibilities
crates/vllm-oxide/                  Inference library
crates/vllm-oxide-cli/              Offline command-line entrypoint
crates/vllm-oxide-test/             Test harness and internal validation tooling
docs/adr/                          Architecture decisions and historical proposals
docs/validation/                   Model validation contracts and investigation records
docs/releases/                     Published release notes and validation summaries
```

## Current behavior and historical decisions

The [getting started guide](../getting-started.md) describes the supported user
interface. Published [release notes](../releases/) and the release validation
summary state what was shipped and measured for a version.

ADRs explain design choices. Check each record's status, version and scope;
accepted decisions can be superseded, and proposals or investigation records
are not promises of implemented behavior. In particular, future parallelism
seams and historical evidence-transport designs do not imply supported tensor
parallelism or a downloadable validation package in v0.2.0.

If a proposed change conflicts with an accepted decision, name that ADR and
explain the reason for reopening it. Preserve the historical record and add a
new decision or explicit status update rather than silently rewriting past
results.

## Vocabulary

Use the defined terms in issue descriptions, code, tests and design proposals.
The “Avoid” notes in `CONTEXT.md` distinguish concepts that are easy to confuse,
such as request identity and a batch position, or the model registry and weight
loader.

When a concept is missing, first check whether an existing term already covers
it. If a new concept is necessary, define its responsibility and update the
project context with the corresponding design change. Keep implementation
claims grounded in the source and current release scope.
