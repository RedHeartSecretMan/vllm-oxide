# RC source documentation update

[Release notes](releases/v0.2.0-rc.1.en.md) · [中文发布说明](releases/v0.2.0-rc.1.md) · [Documentation](README.md)

The published tag is `v0.2.0.rc.1`; Cargo's package version and documentation filenames retain `0.2.0-rc.1`. The original RC implementation commit is **`21cd848fb1256dd908a29d4f3b08596fe1dd39fc`**.

This source maintenance update brings the historical archive's user, contributor, research and tool documentation into the final v0.2.0 presentation format. It changes Markdown and generated-state ignore rules only. Engine code, CUDA kernels, Python/Rust/shell executables, lockfiles, model inputs, frozen validation definitions and budgets are unchanged. No numerical or performance measurement is rerun or re-labeled.

The original release date, prerelease status and attached assets remain historical:

| Preserved record | Identity |
|---|---|
| Measurement source | `8f21331a9d00b36b372eed0039d97f162e2f9e25` |
| Supervision/evaluation source | `3a1cc8ebb093985fa11b659988a7b66de2b0b131` |
| Numerical result | 48/66 PASS, 18 FAIL |
| Original verification JSON SHA-256 | `4ec7cf413f3d80f5d0c07ec5da9f7f9b024f033f70c7521672cbc1c4c4c42a4a` |
| Original SHA256SUMS asset SHA-256 | `a2d39dbd1758f6013bd98c1ab1ac9b56ccf8d975775c2d7be48e448b36a0e08d` |

The original attached verification report names its original source. It is not a new verification of the documentation commit. Moving the source tag to a documentation descendant refreshes GitHub's automatic source archives while preserving those original asset bytes. Existing local clones may retain the original tag until they explicitly refresh it; use an exact commit for immutable source identity.

The release page records the final documentation commit after publication. New users should use [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0); this refreshed historical RC retains its numerical and performance limitations. Raw acceptance packages stay local.

Original implementation: [`21cd848`](https://github.com/RedHeartSecretMan/vllm-oxide/commit/21cd848fb1256dd908a29d4f3b08596fe1dd39fc).
