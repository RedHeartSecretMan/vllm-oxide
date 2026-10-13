# Releases and source history

[Documentation](../README.md) · [Changelog](../../CHANGELOG.md) · [GitHub Releases](https://github.com/RedHeartSecretMan/vllm-oxide/releases)

| Version | Status and notes |
|---|---|
| [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0) | Current source release; use its own source and validation summary |
| v0.2.0.rc.1: [English](v0.2.0-rc.1.en.md) / [中文](v0.2.0-rc.1.md) | Historical experimental candidate, 18/66 numerical failures |
| [v0.1.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.1.0) | Initial source release; original wider Rust API |

This checkout belongs to RC1. [Documentation maintenance](../version-doc-update.md) updates the historical archive while retaining the original implementation, result, release date, status and attached checksums. Final release results do not apply retrospectively to earlier versions.

## Shared release format

Use [the template](template.md) for English and Chinese descriptions: highlights, compatibility/migration, source installation, actual validation/performance, limitations, artifacts and links. Explain Git-tag spelling separately from Cargo version when they differ.

| Maintainer check | Required detail |
|---|---|
| Source | Exact implementation and documentation commits, clean intended changes |
| Scope | Version-specific model, API and execution boundaries |
| Checks | Actual CPU results; relevant local GPU results for implementation changes |
| Historical maintenance | Documentation-only diff, original measurement identities and verdicts retained |
| Artifacts | Source archives, existing asset purpose/size/SHA and no invented binaries |
| Publication | Correct tag, release date/status and bilingual body; preserve original attached assets |

No model rerun or GitHub GPU runner is required for wording-only maintenance. Raw evidence stays local. Do not imply a historical verification asset was regenerated for refreshed docs, and do not use an older disabled golden publication path for source-release maintenance.
