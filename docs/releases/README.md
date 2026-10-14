# Releases — v0.1.0 source archive

[Documentation](../README.md) · [English notes](v0.1.0.en.md) · [中文说明](v0.1.0.md)

This archive contains version-specific notes for v0.1.0. The [current release history](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/docs/releases/README.md) describes later releases and the active publishing workflow.

| Version | Status | Scope |
|---|---|---|
| [v0.1.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.1.0) | Historical experiment | Original offline Qwen3 engine and schema-1 fixture assets; known numerical differences |
| [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0) | Recommended current release | Current implementation, stricter numerical acceptance and validated API/CLI |

The historical source documentation has been refreshed; [the maintenance record](../version-doc-update.md) identifies the original implementation commit and what changed. The update does not modify version numbers, feature support, original fixture assets, measurements or historical verdicts.

Release notes follow the shared structure: highlights, compatibility/migration, obtaining source, validation/performance, limitations, and artifacts/links. Report actual test scope and source provenance; CPU checks and GPU acceptance remain separate. Keep raw evidence local and never convert historical FAIL/skipped results into PASS.

## Authorized source maintenance of existing versions

The authorized 2026-10-14 maintenance includes code fixes in all three source
snapshots, including `research` and `tools`. Review the executable diff and run
regression checks for each version. Keep the original model definitions,
numerical budgets, measurement identities, and historical verdicts. Record new
checks against their actual source commits.

Before updating a tag, record its exact object locally. Publish a reviewed
maintenance descendant with an exact old-object force-with-lease check. Preserve
Release IDs, original dates, status and attached assets, and synchronize the
source identity and reviewed notes. Download each new archive and compare every
tracked file with the reviewed commit. Before removing a temporary remote work
branch, confirm its maintained release commits and documented historical sources
are ancestors of the remote default branch. Keep the three official tags and
store any backup refs locally. See the
[source maintenance record](../version-doc-update.md).
