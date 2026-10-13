# Release guide

[v0.2.0](v0.2.0.md) is the current source release. This guide is for maintainers preparing future releases or improving an existing Release description.

## Prepare the release

1. Choose the supported model, dtype, hardware and public API scope. Keep user-visible changes in [CHANGELOG.md](../../CHANGELOG.md), including migration instructions for breaking changes.
2. Align the workspace version in `Cargo.toml`, workspace packages in `Cargo.lock`, README examples, release notes and tag. Release tags use `vMAJOR.MINOR.PATCH`; prereleases use `vMAJOR.MINOR.PATCH-rc.N`. Historical tag names stay unchanged.
3. Run the applicable checks and record their actual source identity:

   | Change | Validation |
   |---|---|
   | Documentation or community metadata | Document links, workflow/template structure and examples against the actual source |
   | Rust or Python implementation | The complete CPU CI matrix in [CONTRIBUTING.md](../../CONTRIBUTING.md) |
   | Inference, model, kernels, cache or scheduling | Local CUDA API/CLI smoke, affected numerical acceptance and relevant performance measurements |
   | Formal product release | CPU CI plus source-bound numerical, public CUDA functionality and performance evidence for the supported scope |

   GPU validation can run on a supported Linux/WSL2 machine. The optional self-hosted CUDA workflow is a convenience; GitHub-hosted GPU availability is not a release requirement. Preserve the declared numerical limits and record hardware-specific performance as a baseline.
4. Start the user-facing notes from [template.md](template.md). Lead with behavior, fixes and compatibility; put source identities and detailed validation in a linked report. Describe exactly which artifacts are provided.
5. Open an integration PR. Use `breaking-change`, `enhancement`, `bug` or `documentation` labels as applicable. [`.github/release.yml`](../../.github/release.yml) groups GitHub-generated PR summaries; review the generated list before including it in the final notes.
6. Merge the approved, checked candidate. Verify the merged tree matches the tested candidate, or validate any additional changes before selecting the release commit. Preserve each original evidence source rather than renaming previous results as new measurements.

A documentation correction does not require repeating GPU measurements when the implementation and dependency identities are unchanged. It also cannot turn a previous failed or incomplete measurement into PASS.

## Publish the reviewed source

Use the exact reviewed commit and a prepared notes file. The following commands are a template; replace the values before running them:

```bash
RELEASE_VERSION=0.2.1
RELEASE_COMMIT=replace-with-full-reviewed-commit
RELEASE_NOTES=/absolute/path/to/reviewed-release-notes.md

git tag -a "v$RELEASE_VERSION" "$RELEASE_COMMIT" -m "vllm-oxide v$RELEASE_VERSION"
git push origin "refs/tags/v$RELEASE_VERSION"
gh release create "v$RELEASE_VERSION"   --repo RedHeartSecretMan/vllm-oxide   --verify-tag --target "$RELEASE_COMMIT"   --title "v$RELEASE_VERSION — vllm-oxide"   --notes-file "$RELEASE_NOTES" --latest
```

For a prerelease, use its prerelease version, add `--prerelease` and set `--latest=false`. Publishing follows the user's existing authorization; this guide does not itself authorize a new publication.

GitHub supplies source `.tar.gz` and `.zip` archives. Full golden captures, logits, raw logs and acceptance bundles remain local by default, outside git. Evidence uploads require an explicit request for those assets. A source Release must not imply that an unpublished evidence archive or a separate consumer gate completed.

## Verify and maintain

- Verify the remote tag resolves to the intended commit and the Release is non-draft, with the intended prerelease/Latest status.
- Download a source archive and confirm its version, examples and release notes match the tag. Check any explicitly published asset names and checksums.
- Link the migration guide, complete changelog, supported environment, validation summary and known limitations in the Release description.
- Close only the release work completed under the agreed scope; record deferred work separately from a passing product check.

Published tags identify immutable source snapshots. Improve wording with `gh release edit --notes-file` and update current documentation through a PR. A code fix receives a new patch release; it never silently moves an existing version tag.

The maintained [v0.2.0 English notes](v0.2.0.en.md) are also the GitHub Release
description source. To synchronize a reviewed wording change after its PR lands:

```bash
gh release edit v0.2.0 --repo RedHeartSecretMan/vllm-oxide \
  --notes-file docs/releases/v0.2.0.en.md
```

## v0.2.0 records

- [Product release notes](v0.2.0.md)
- [English release notes](v0.2.0.en.md)
- [Validation and performance summary](goldens-v0.2.md)
- [Historical RC1 notes](v0.2.0-rc.1.md)
- [GitHub Releases](https://github.com/RedHeartSecretMan/vllm-oxide/releases)
