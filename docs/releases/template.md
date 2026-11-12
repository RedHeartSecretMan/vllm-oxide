# Release notes template

Each version ships two artefacts that share one skeleton:

1. **Full note** — `docs/releases/vVERSION.md` (中文) and `docs/releases/vVERSION.en.md` (English). This is the versioned record kept in the repository and the source of truth.
2. **GitHub Release body** — a summary of the full note: same section names, same order, no duplicated version heading, ending with links to both language files.

Only include a section that has content. Replace placeholders with verified facts and delete these instructions before publishing.

## Section order

| 中文 | English | 用途 |
|---|---|---|
| 摘要 | Summary | 版本身份、日期、状态，加一两句用户可见结果 |
| 变更 | Changes | 用户可观察的变化；必要时在下面分 新增 / 改进 / 修复 |
| 安装与升级 | Install and upgrade | 环境要求、破坏性变更、迁移指引与可复制命令 |
| 验证 | Verification | 实际跑过的检查与数字，附完整证据链接 |
| 边界 | Limitations | 不在本版范围的功能、已知失败、未完成验证 |
| 链接 | Links | 源码归档、资产、变更日志、对比、语言对照 |

Each language file uses its own language's headings. Do not mix both languages in one file.

---

# vVERSION — release headline

**Released: YYYY-MM-DD · Status: current release / prerelease / historical release**

## Summary

One or two sentences describing the user-visible result and the supported inference scope. Historical versions point new users to the current release and describe their limitations as historical results.

| Release | Source commit | Distribution |
|---|---|---|
| `vVERSION` | Full reviewed commit, linked with a short label | Source archives and actual attached assets |

## Changes

- The most useful behaviour or feature added.
- A correctness or performance problem fixed, with its observable effect.

## Install and upgrade

State the verified Rust/toolchain, operating-system, model/dtype and CUDA requirements. List breaking API or behaviour changes and link a version-specific migration guide. Distinguish an old declared minimum from a minimum actually tested. Preserve the exact spelling of historical tags.

```bash
git clone --branch vVERSION --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
cd vllm-oxide
# Add the actual, checked build or run command here.
```

State which features enable inference and what the first run downloads. For historical releases without a verified run command, provide the source checkout and direct users to the current version's guide.

## Verification

Summarise the actual CPU, numerical and public CUDA checks. Name the model revision, dtype and GPU environment. Give workload/configuration, sample count and units for any performance number, with a link to complete provenance. Hardware-specific timing is a baseline rather than a universal guarantee. Never transfer a later release's passing verdict onto an earlier source snapshot.

## Limitations

Describe unsupported functionality and outstanding issues that affect the stated version. Keep original failed or incomplete validation results visible. Put long failure inventories and source hashes in a labelled `<details>` section when useful.

## Links

List GitHub source archives and each actual attached asset's purpose. State whether tested binaries, model weights or evidence are supplied. Preserve original asset names, hashes and download URLs when editing old notes.

Link the changelog, previous-to-current comparison, language counterpart, release history and support entrypoint. Review any GitHub-generated PR/contributor summary before adding it here; focus on changes users can observe.

---

## GitHub Release body

The Release body is a summary of the full note, not a copy:

- start with `## Summary` / `## 摘要`; never repeat `# vVERSION`, the release title already carries it;
- keep only the sections that earn their place, in the same order as the table above;
- reuse the numbers and caveats from the full note verbatim instead of recomputing them;
- end with links to both language files, the changelog and any migration guide.

For this bilingual repository the body mirrors both languages:

```markdown
## English

### Summary
...
### Changes
- ...
### Verification
...

## 简体中文

### 摘要
...
### 变更
- ...
### 验证
...
```

Install commands and anything else language-neutral appear once, in the first language, and the second section references them.

## Publish

Prepare the GitHub Release description from the summary and keep the full note in the repository under the tag. Both must agree; if they diverge, the full note wins and the summary is corrected.
