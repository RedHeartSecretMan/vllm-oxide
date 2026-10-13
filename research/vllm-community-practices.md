# vLLM 社区做法与 vllm-oxide 仓库建设

核实日期：2026-10-13。目标是改善首次使用、贡献和版本发布体验，范围保持为单 GPU Qwen3 离线推理。

以下建议依据 vLLM 官方文档、仓库源码和 Release 页面核实；Context7 文档查询使用 `/vllm-project/vllm`。

## 建议落实的六项

| 优先级 | 官方做法 | 适合本项目的落实方式 |
| --- | --- | --- |
| 1 | vLLM README 把安装、Quickstart、支持模型和贡献入口集中展示。[README](https://github.com/vllm-project/vllm/blob/main/README.md) | 英文和中文 README 使用相同顺序：定位与支持边界、源码安装、固定模型的最小 CLI 示例、Rust 示例、文档和贡献入口。先让用户完成第一次生成，再链接架构和详细参数。注明本项目独立维护，与上游的关系是参考和借鉴。 |
| 2 | vLLM 文档区分用户指南、开发指南、基准测试及 API/CLI 参考；贡献页要求用户可见行为变化同步更新文档，并说明如何预览文档。[贡献文档](https://docs.vllm.ai/en/latest/contributing/#documentation) | 用一个 `docs/README.md` 整理当前 Markdown，分别指向使用、开发、验收和架构资料；先检查内部链接及示例，不必新增 MkDocs/Read the Docs 部署。贡献指南准确列出仓库现有 CPU CI 命令，另列本地 CUDA 功能、精度和性能检查。 |
| 3 | vLLM 的 bug 表单要求环境、最小复现和实际结果；PR 模板分为概要、具体主张、验证和设计细节。[Bug 表单](https://github.com/vllm-project/vllm/blob/main/.github/ISSUE_TEMPLATE/400-bug-report.yml)、[PR 模板](https://github.com/vllm-project/vllm/blob/main/.github/PULL_REQUEST_TEMPLATE.md) | 增加简短的 bug、功能建议、文档问题表单及 PR 模板。bug 收集版本/提交、模型 revision、构建 feature、Rust/CUDA/GPU、命令或最小 Rust 代码、预期和实际结果。PR 写清修复行为和实际检查；GPU 相关变化注明本地验证环境与结果。保留现有五种 triage 标签和 Conventional Commits。 |
| 4 | vLLM 将安全报告指向私密 GitHub Security Advisories，并在 README 将普通问题与安全报告分流。[安全政策](https://github.com/vllm-project/vllm/blob/main/SECURITY.md)、[README](https://github.com/vllm-project/vllm/blob/main/README.md#contact-us) | 增加简短 `SECURITY.md`：可信模型来源、敏感信息不写入公开报告、支持版本和私密报告入口。采用仓库真实可用的渠道；如尚未启用 GitHub 私密报告，先由维护者启用，不能写成已经可用。不承诺响应时限、安全团队或 CVE 流程。 |
| 5 | 核实时最新 vLLM Release 为 v0.31.0，说明按亮点、发行物、模型/引擎变化等组织，并明确列出破坏性变更、平台和安装方式。[v0.31.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) | 完善 v0.2.0 的公开说明：本版重点、相对上一版的变化、支持范围、升级/安装步骤、已知限制和验证摘要。明确当前发行物只有源码，没有预编译包或 crates.io 发布。基准数字绑定 GPU、模型、dtype 和工作负载，不写跨硬件保证或 vLLM 对等性能。将来源角色、原始证据和历史诊断放到链接的验收文档中，公开说明使用读者能直接理解的产品语言。 |
| 6 | vLLM 发布流程记录版本与 release branch/RC/final tag 的关系，并把手工端到端性能验证绑定到发布提交；流程按固定硬件比较回归。[发布流程](https://docs.vllm.ai/en/latest/contributing/release_process/) | 增加维护者发布清单与发布说明模板：版本一致、变更汇总、支持边界、CPU CI、受影响的 GPU 检查、发行物清单、tag/commit/Release 核对。发布说明以仓库文件为维护来源，再同步到 GitHub Release；保留已经发布的 v0.2.0 tag，文档补充不改写历史标签。源码发布继续让 GitHub 提供源码归档，原始验收证据按当前约定留在本地。 |

## 明确的取舍

- **保留 CPU CI 与 GPU 验收的职责分工。** vLLM 的贡献文档承认部分测试依赖 GPU，其 PR 模板允许说明 CI 资源限制及 CI 外验证。本项目普通 CI 继续验证编译、测试和工具质量；真实模型输出与性能由本地 GPU 检查完成。参考这一原则不等于照搬 vLLM 的 GPU 集群或增加 GPU runner 发布门槛。[测试说明](https://docs.vllm.ai/en/latest/contributing/#testing)、[PR 模板](https://github.com/vllm-project/vllm/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
- **先补齐可发现性和可复现性。** vLLM 有专门的 forum、Slack、多种 issue 表单及治理文档。本项目目前用 GitHub Issues 和少量模板即可，不新建论坛、Slack、委员会、区域负责人体系或复杂 RFC 程序。[README 社区入口](https://github.com/vllm-project/vllm/blob/main/README.md#contact-us)、[Issue 模板目录](https://github.com/vllm-project/vllm/tree/main/.github/ISSUE_TEMPLATE)
- **发布流程按本项目规模执行。** 可以借鉴 vLLM 的变更分类、发行物标注和固定提交验证；不照搬双周发布频率、多平台 wheel/Docker 矩阵、强制 DCO 或上游的 PR 标题标签。现有约定由本仓库贡献指南保持一致。[发布流程](https://docs.vllm.ai/en/latest/contributing/release_process/)、[贡献规范](https://docs.vllm.ai/en/latest/contributing/)
- **文档建设不扩大功能承诺。** 当前支持边界以 [v0.2.0 发布说明](../docs/releases/v0.2.0.md) 为准；仓库外观改善不表示新增服务、多 GPU、其他模型或量化能力。

这六项均为适配建议。vLLM 的源码和官方文档是参考来源，其规则不会自动成为 vllm-oxide 的贡献或发布要求。
