# Release notes template

Write matching English and Chinese notes, using the same section order. Prepare the GitHub Release description from the English file and include a visible link to the Chinese version. Replace placeholders with verified facts and remove template instructions before publishing.

---

# vVERSION — release headline

**Released: YYYY-MM-DD · Status: current release / prerelease / historical release · [中文](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/docs/releases/vVERSION.md)**

One or two sentences describing the user-visible result and the supported inference scope. Historical versions should point new users to the current release and describe their limitations as historical results.

| Release | Source commit | Distribution |
|---|---|---|
| `vVERSION` | Full reviewed commit, linked with a short label | Source archives and actual attached assets |

## Highlights

- The most useful behavior or feature added.
- A correctness or performance problem fixed, with its observable effect.

## Compatibility and migration

State the verified Rust/toolchain, operating-system, model/dtype and CUDA requirements. List breaking API or behavior changes and link a version-specific migration guide. Distinguish an old declared minimum from a minimum actually tested. Preserve the exact spelling of historical tags.

## Get the source and run

Provide a complete command that selects the actual published tag and a supported model. State which features enable inference and what the first run downloads. For historical releases without a verified run command, provide the source checkout and direct users to the current version's guide.

```bash
git clone --branch vVERSION --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
cd vllm-oxide
# Add the actual, checked build or run command here.
```

## Validation and performance

Summarize the actual CPU, numerical and public CUDA checks. Name the model revision, dtype and GPU environment. Give workload/configuration, sample count and units for any performance numbers, with a link to complete provenance. Hardware-specific timing is a baseline rather than a universal guarantee. Never transfer a later release's passing verdict onto an earlier source snapshot.

## Known limitations

Describe unsupported functionality and outstanding issues that affect the stated version. Keep original failed or incomplete validation results visible. Put long failure inventories and source hashes in a labeled `<details>` section when useful.

## Artifacts and links

List GitHub source archives and each actual attached asset's purpose. State whether tested binaries, model weights or evidence are supplied. Preserve original asset names, hashes and download URLs when editing old descriptions.

Link the changelog, previous-to-current comparison, language counterpart, release history and support entrypoint. Review any GitHub-generated PR/contributor summary before adding it here; focus on changes users can observe.
