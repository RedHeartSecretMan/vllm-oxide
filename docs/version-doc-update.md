# Source documentation maintenance — v0.1.0

[Documentation](README.md) · [Release notes](releases/v0.1.0.en.md)

The release's original implementation source is [`52037493b38c2d01c4c07a98d2d8f22ca68f2509`](https://github.com/RedHeartSecretMan/vllm-oxide/commit/52037493b38c2d01c4c07a98d2d8f22ca68f2509). The maintained source archive adds and corrects documentation to the v0.2.0 presentation standard while describing the v0.1.0 implementation.

The refresh adds bilingual project overviews, usage/development/architecture/validation navigation, contribution/support/security guidance, detailed research and tools instructions, and version-specific release notes. Commands were checked against the old manifests and source. The CLI dependency feature, Python 3.12 requirement, BF16/FP32 distinction, batch expansion, schema-1 tolerances, comparison gaps and artifact handling are documented explicitly.

Rust, CUDA, Python, shell scripts, dependency locks and original measurement artifacts are unchanged. This maintenance is not a code backport, new GPU collection, accuracy fix, performance acceptance, or new product version. The four token-divergent groups, six regression skip cases and large original tolerance retain their historical meaning.

The original commit remains available in Git history by its full object ID. The release tag is maintained on a documentation-only descendant; its selected commit differs from the original implementation commit without changing implementation bytes. Source archives selected by the maintained tag include these documentation corrections; original attached fixture assets and their checksums remain unchanged. For current application development use [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0).

Original implementation: [`5203749`](https://github.com/RedHeartSecretMan/vllm-oxide/commit/52037493b38c2d01c4c07a98d2d8f22ca68f2509).
