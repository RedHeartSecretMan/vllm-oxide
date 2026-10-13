# v0.2.0 source documentation revision

[Release history](README.md) · [English notes](v0.2.0.en.md) · [中文发布说明](v0.2.0.md) · [Shared maintenance record](../version-doc-update.md)

This revision brings the documentation in the v0.2.0 source archive up to the
project's current presentation standard. It updates Markdown and ignore rules;
it does not change Rust/Python/CUDA code, package versions, dependency lockfiles,
machine-readable validation definitions, model files or historical measurements.

| Identity | Meaning |
|---|---|
| Original implementation commit | [`289cb65b475bb472b5c31db2615c3a5d4cbba0db`](https://github.com/RedHeartSecretMan/vllm-oxide/commit/289cb65b475bb472b5c31db2615c3a5d4cbba0db) |
| Original publication | 2026-10-13; v0.2.0 source release |
| Maintained source tag | [`v0.2.0`](https://github.com/RedHeartSecretMan/vllm-oxide/tree/v0.2.0); its maintained documentation is distinct from the original implementation commit |
| Original validation | Numerical measurement `33e0e27`, independent evaluation `ba5eb79`, performance producer `e785a21`, public CUDA smoke `3b1cdceb` |

The tag and automatically generated source archives receive a documentation
maintenance commit. The original commit remains addressable for exact historical
reproduction. Existing clones should fetch tags and compare the source revision
before assuming that a cached archive includes the revised documentation.

The update adds aligned English/Chinese project and release introductions,
usage/API and architecture guides, contribution and community guidance,
documentation/release/validation/benchmark navigation, and detailed research and
tool instructions. Commands are checked against this version's actual code.
Tools added later on `main`, such as `tools/check_docs.py`, are linked as external
maintenance resources rather than presented as bundled executables.

The [validation summary](goldens-v0.2.md) retains the original sources, limits,
counts and report hashes. Documentation validation does not represent a new GPU
measurement, turn historical failures into PASS, or publish local raw evidence.
Original Release assets, publication date and version status retain their
historical meaning.

本文更新的是 v0.2.0 源码归档中的文档和忽略规则，不改变实现、依赖、
机器可读的验收定义或原始测量。原发布实现 `289cb65` 保持可访问；维护后的
标签及源码归档包含本次文档修订。精度、功能和性能结果始终对应原始测量来源，
没有将文档检查描述为重新执行 GPU 验收。

Original tag archive: [archive/before-doc-refresh-2026-10-13/v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/tree/archive/before-doc-refresh-2026-10-13/v0.2.0).
