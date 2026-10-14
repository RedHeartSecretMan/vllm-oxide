# Releases and maintainer guide

[Documentation home](../README.md) · [GitHub Releases](https://github.com/RedHeartSecretMan/vllm-oxide/releases)

## Version history

Use **v0.2.0** for the currently supported single-GPU Qwen3-0.6B/BF16 offline generation API. Earlier releases remain available for reproducibility; their historical failures and incomplete checks do not describe the current version.

| Version | Published date | Status and validation | Release notes |
|---|---|---|---|
| [`v0.2.0`](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0) | 2026-10-13 | Current release; numerical, public CUDA functionality and performance validation passed | [English](v0.2.0.en.md) · [中文](v0.2.0.md) |
| [`v0.2.0.rc.1`](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0.rc.1) | 2026-09-10 | Historical experimental prerelease; 18 numerical failures, incomplete performance/minimum-toolchain verification | [English](v0.2.0-rc.1.en.md) · [中文](v0.2.0-rc.1.md) |
| [`v0.1.0`](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.1.0) | 2026-07-26 | Historical experimental release; four of eight full-output token checks differed, with skipped regression positions | [English](v0.1.0.en.md) · [中文](v0.1.0.md) |

All three versions provide source archives. v0.1.0 also retains its original fixture archive and manifest; RC1 retains its original verification report and checksum file. v0.2.0 has no attached binary, model or raw-evidence assets. See the version's notes for exact download links and original asset digests.

The historical RC Git tag is `v0.2.0.rc.1`; its Cargo package version and documentation filenames use `0.2.0-rc.1`. Do not substitute the filename spelling into a checkout or Release URL.

## Prepare the release

1. Choose the supported model, dtype, hardware and public API scope. Keep user-visible changes in [CHANGELOG.md](../../CHANGELOG.md), including migration instructions for breaking changes.
2. Align the workspace version in `Cargo.toml`, workspace packages in `Cargo.lock`, README examples, release notes and tag. New release tags use `vMAJOR.MINOR.PATCH`; new prereleases use `vMAJOR.MINOR.PATCH-rc.N`. Historical tag names stay unchanged.
3. Run the applicable checks and record their actual source identity:

   | Change | Validation |
   |---|---|
   | Documentation or community metadata | Document links, workflow/template structure and examples against the actual source |
   | Rust or Python implementation | The complete CPU CI matrix in [CONTRIBUTING.md](../../CONTRIBUTING.md) |
   | Inference, model, kernels, cache or scheduling | Local CUDA API/CLI smoke, affected numerical acceptance and relevant performance measurements |
   | Formal product release | CPU CI plus source-bound numerical, public CUDA functionality and performance evidence for the supported scope |

   GPU validation can run on a supported Linux/WSL2 machine. The optional self-hosted CUDA workflow is a convenience; GitHub-hosted GPU availability is not a release requirement. Preserve the declared numerical limits and record hardware-specific performance as a baseline.
4. Create matching English and Chinese user-facing notes from [template.md](template.md). Use the same section order: opening/version identity, highlights, compatibility/migration, getting the source, validation/performance, known limitations, and artifacts/links. Lead with behavior and fixes; link detailed validation rather than pasting raw logs.
5. Open an integration PR. Use `breaking-change`, `enhancement`, `bug` or `documentation` labels as applicable. [`.github/release.yml`](../../.github/release.yml) groups GitHub-generated PR summaries; review the generated list before including it in the final notes.
6. Merge the approved, checked candidate. Verify the merged tree matches the tested candidate, or validate any additional changes before selecting the release commit. Preserve each original evidence source rather than renaming previous results as new measurements.

A documentation correction does not require repeating GPU measurements when the implementation and dependency identities are unchanged. It also cannot turn a previous failed or incomplete measurement into PASS.

## Publish the reviewed source

Use the exact reviewed commit and a prepared notes file. The following commands are a template; replace the values before running them:

```bash
RELEASE_VERSION=0.2.1
RELEASE_COMMIT=replace-with-full-reviewed-commit
RELEASE_NOTES=/absolute/path/to/reviewed-release-notes.md

git fetch origin main &&
git merge-base --is-ancestor "$RELEASE_COMMIT" origin/main &&
git tag -a "v$RELEASE_VERSION" "$RELEASE_COMMIT" -m "vllm-oxide v$RELEASE_VERSION" &&
git push origin "refs/tags/v$RELEASE_VERSION" &&
gh release create "v$RELEASE_VERSION" \
  --repo RedHeartSecretMan/vllm-oxide \
  --verify-tag --target "$RELEASE_COMMIT" \
  --title "v$RELEASE_VERSION — release headline" \
  --notes-file "$RELEASE_NOTES" --latest
```

For a prerelease, use its prerelease version, add `--prerelease` and set `--latest=false`. Publishing follows the user's existing authorization; this guide does not itself authorize a new publication.

The ancestry check must pass before publishing. A tag alone keeps its objects
available but does not give its commit a branch association on GitHub.

GitHub supplies source `.tar.gz` and `.zip` archives. Full golden captures, logits, raw logs and acceptance bundles remain local by default, outside git. Evidence uploads require an explicit request for those assets. A source Release must not imply that an unpublished evidence archive or a separate consumer gate completed.

## Verify and maintain

- Verify the remote tag resolves to the intended commit and the Release is non-draft, with the intended prerelease/Latest status.
- Confirm the release commit is an ancestor of the remote default branch. Check
  repository-local measurement and diagnostic commit references as well: deleting their
  last work branch can leave a warning or make an unretained commit unavailable.
- Download a source archive and confirm its version, examples and release notes match the tag. Check any explicitly published asset names and checksums.
- Link the migration guide, complete changelog, supported environment, validation summary and known limitations in the Release description.
- Close only the release work completed under the agreed scope; record deferred work separately from a passing product check.

Published tags normally identify immutable source snapshots. Improve wording with `gh release edit --notes-file` and update current documentation through a PR. A code fix receives a new patch release.

## Authorized documentation refresh of an existing version

The user explicitly requested that all three existing source archives include
the improved version-specific documentation, including their `research` and
`tools` directories. This is a documentation-maintenance exception, not a
backport of v0.2.0 behavior into older releases.

For this operation:

1. Create one documentation commit directly from each original release commit.
   Keep executable source, dependencies, lockfiles, registries, raw evidence and
   recorded numerical results unchanged. Documentation must describe that
   version's actual commands and implementation.
2. Review and validate each version's guides, local links, shell examples and
   complete source-file diff. Record the original implementation commit and
   annotated tag object locally for provenance and rollback; do not publish
   additional backup tags.
3. Use temporary named branches to review the maintenance commits. Retain the
   reviewed commits in `main` history before deleting those branches. Update existing
   release tags only under the user's explicit authority, with a force-with-lease
   check against each original remote tag object. A changed remote ref stops the
   update; it must be inspected rather than overwritten.
4. Sync the Release's source commit and notes to the documentation revision,
   naming the original implementation and measurement sources separately.
   Preserve the Release ID, original publication date, draft/prerelease/Latest
   status and attached evidence assets.
5. Download the refreshed GitHub source archive for each tag. Verify its new
   documentation and that all non-documentation runtime files are byte-identical
   to the original implementation. Verify the original implementation commits
   remain available in Git history.

Each source archive includes `docs/version-doc-update.md` to explain the
maintenance revision. It does not claim new GPU measurements or change a
historical failure into PASS. Regular releases continue to use new version tags
for behavior changes.

Each `*.en.md` file is the maintained source of that version's English GitHub
Release description. Prepare a separate GitHub-ready notes file after its
documentation PR lands:

- Omit the document's H1 when it duplicates the Release title.
- Expand relative Markdown links against `docs/releases/` in the selected tag;
  use complete GitHub URLs so they work on the Release page.
- Include the full maintained source commit, verified against the peeled tag,
  while retaining the original implementation and measurement identities.
- Review the prepared body before uploading it. Passing the repository Markdown
  directly to `--notes-file` can restore broken relative links or omit the exact
  maintained source identity.

| Tag | English source |
|---|---|
| `v0.1.0` | `docs/releases/v0.1.0.en.md` |
| `v0.2.0.rc.1` | `docs/releases/v0.2.0-rc.1.en.md` |
| `v0.2.0` | `docs/releases/v0.2.0.en.md` |

Select the intended tag and prepared file, then update the existing description:

```bash
RELEASE_TAG=v0.2.0
RELEASE_NOTES=/absolute/path/to/reviewed-github-release-notes.md
gh release edit "$RELEASE_TAG" --repo RedHeartSecretMan/vllm-oxide \
  --notes-file "$RELEASE_NOTES"
```

Before and after editing, compare the Release ID, actual peeled tag commit,
original `published_at`, draft/prerelease/Latest status and asset identities.
Compare assets by ID, not API list order. As documented in the
[Release API](https://docs.github.com/en/rest/releases/releases#update-a-release),
`target_commitish` is a tag-creation hint; an existing tag selects the source.
A tag update can also change the API's
`created_at`; keep it separate from the original publication timestamp.
Description and title improvements preserve the source and publication state;
original failed or diagnostic records keep their verdicts.

## Authorized source maintenance of existing versions

The 2026-10-14 maintenance explicitly includes code fixes in the three existing
source snapshots, including `research` and `tools`. Use version-specific
regression checks and review the executable diff. Preserve the original model
validation definitions, budgets, measurement identities and historical verdicts.
Record new checks separately; old PASS reports are not new measurements.

Before updating a tag, record its exact object locally. Publish a reviewed
maintenance descendant with an exact old-object force-with-lease check. Preserve
Release IDs, original dates, status and attached assets, and synchronize the
source identity and reviewed notes. Download each new archive and compare every
tracked file with the reviewed commit. Remove temporary remote work branches;
retain the three official tags without additional backup tags. See the
[source maintenance record](../version-doc-update.md).

Before removing any temporary remote branch, verify every maintained tag and
every linked repository-local historical source commit is reachable from `origin/main`:

```bash
git fetch origin main
for RELEASE_TAG in v0.1.0 v0.2.0.rc.1 v0.2.0; do
  git merge-base --is-ancestor "$RELEASE_TAG^{commit}" origin/main || exit 1
done
```

Confirm local tag objects match the remote tags before this check. For linked
historical sources, run the same ancestry check with each original full commit
ID. If version-specific maintenance or a diagnostic source has a separate
history, review that history before retaining it on `main`. An ancestry-only
merge with `-s ours` can preserve those sources while keeping the current file
tree unchanged. Verify the tree IDs and each required ancestor before pushing;
do not substitute `-X ours`, which can still merge old file contents. Recheck
the remote default branch and original tag objects after publication.

## Validation records

- [v0.2.0 validation and performance summary](goldens-v0.2.md)
- [Validation guide](../validation/README.md)
- [Migration to v0.2.0](../migration-v0.2.0.md)
- [Contributing](../../CONTRIBUTING.md)
