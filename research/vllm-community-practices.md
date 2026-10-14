# vLLM 文档与社区实践参考

核实日期：2026-10-13。此次范围覆盖 README、用户与开发文档、社区入口、仓库元数据和全部已发布版本的说明。功能边界仍以 [v0.2.0](../docs/releases/v0.2.0.md) 为准：单 GPU、Qwen3-0.6B、BF16、同步离线生成。

以下建议来自研究日的 vLLM 官方文档、仓库源码和 Release 页面，记录了最初文档建设阶段的考虑。2026-10-14 后续源码维护及检查见[维护记录](../docs/version-doc-update.md)。项目支持哪些能力，仍以自身实现和发布记录为准。

## 建议落实的六项

| 优先级 | 官方做法 | 适合本项目的落实方式 |
| --- | --- | --- |
| 1 | vLLM README 把安装、Quickstart、支持模型和贡献入口集中展示。[README](https://github.com/vllm-project/vllm/blob/main/README.md) | 英文和中文 README 使用相同顺序：定位与支持边界、源码安装、固定模型的最小 CLI 示例、Rust 示例、文档和贡献入口。先让用户完成第一次生成，再链接架构和详细参数。架构参考处直接链接 nano-vllm 与 vLLM。 |
| 2 | vLLM 文档区分用户指南、开发指南、基准测试及 API/CLI 参考；贡献页要求用户可见行为变化同步更新文档，并说明如何预览文档。[贡献文档](https://docs.vllm.ai/en/latest/contributing/#documentation) | 用一个 `docs/README.md` 整理当前 Markdown，分别指向使用、开发、验收和架构资料；先检查内部链接及示例，不必新增 MkDocs/Read the Docs 部署。贡献指南准确列出仓库现有 CPU CI 命令，另列本地 CUDA 功能、精度和性能检查。 |
| 3 | vLLM 的 bug 表单要求环境、最小复现和实际结果；PR 模板分为概要、具体主张、验证和设计细节。[Bug 表单](https://github.com/vllm-project/vllm/blob/main/.github/ISSUE_TEMPLATE/400-bug-report.yml)、[PR 模板](https://github.com/vllm-project/vllm/blob/main/.github/PULL_REQUEST_TEMPLATE.md) | 增加简短的 bug、功能建议、文档问题表单及 PR 模板。bug 收集版本/提交、模型 revision、构建 feature、Rust/CUDA/GPU、命令或最小 Rust 代码、预期和实际结果。PR 写清修复行为和实际检查；GPU 相关变化注明本地验证环境与结果。保留现有五种 triage 标签和 Conventional Commits。 |
| 4 | vLLM 将安全报告指向私密 GitHub Security Advisories，并在 README 将普通问题与安全报告分流。[安全政策](https://github.com/vllm-project/vllm/blob/main/SECURITY.md)、[README](https://github.com/vllm-project/vllm/blob/main/README.md#contact-us) | 增加简短 `SECURITY.md`：可信模型来源、敏感信息不写入公开报告、支持版本和私密报告入口。采用仓库真实可用的渠道；如尚未启用 GitHub 私密报告，先由维护者启用，不能写成已经可用。不承诺响应时限、安全团队或 CVE 流程。 |
| 5 | 核实时最新 vLLM Release 为 v0.31.0，说明按亮点、发行物、模型/引擎变化等组织，并明确列出破坏性变更、平台和安装方式。[v0.31.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) | 完善 v0.2.0 的公开说明：本版重点、相对上一版的变化、支持范围、升级/安装步骤、已知限制和验证摘要。明确当前发行物只有源码，没有预编译包或 crates.io 发布。基准数字绑定 GPU、模型、dtype 和工作负载，不写跨硬件保证或 vLLM 对等性能。将来源角色、原始证据和历史诊断放到链接的验收文档中，公开说明使用读者能直接理解的产品语言。 |
| 6 | vLLM 发布流程记录版本与 release branch/RC/final tag 的关系，并把手工端到端性能验证绑定到发布提交；流程按固定硬件比较回归。[发布流程](https://docs.vllm.ai/en/latest/contributing/release_process/) | 增加维护者发布清单与发布说明模板：版本一致、变更汇总、支持边界、CPU CI、受影响的 GPU 检查、发行物清单、tag/commit/Release 核对。发布说明以仓库文件为维护来源，再同步到 GitHub Release；历史源码文档修订按下文的本次授权范围执行，保留原受测代码与证据身份。源码发布继续让 GitHub 提供源码归档，原始验收证据按当前约定留在本地。 |

## 明确的取舍

- 普通 CI 负责本项目的编译、测试和工具质量，真实模型输出与性能由本地 GPU 检查。vLLM 的贡献文档也区分 GPU 测试，并允许在 PR 中说明 CI 资源限制及外部验证。沿用这项分工即可，本项目无需部署 vLLM 规模的 GPU 集群，也不将 GitHub GPU runner 作为发布条件。[测试说明](https://docs.vllm.ai/en/latest/contributing/#testing)、[PR 模板](https://github.com/vllm-project/vllm/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
- 求助和贡献入口应容易找到，问题也应能复现。vLLM 提供 forum、Slack、多种 issue 表单及治理文档；本项目目前使用 GitHub Issues 和少量模板即可，暂不另设论坛、Slack、委员会、区域负责人或复杂 RFC 程序。[README 社区入口](https://github.com/vllm-project/vllm/blob/main/README.md#contact-us)、[Issue 模板目录](https://github.com/vllm-project/vllm/tree/main/.github/ISSUE_TEMPLATE)
- 发布流程可采用 vLLM 的变更分类、发行物标注和固定提交验证。发布频率、wheel/Docker 平台矩阵、DCO 要求和 PR 标题标签则按本仓库贡献指南执行，无需随上游采用双周发布或扩展构建矩阵。[发布流程](https://docs.vllm.ai/en/latest/contributing/release_process/)、[贡献规范](https://docs.vllm.ai/en/latest/contributing/)
- 支持范围以 [v0.2.0 发布说明](../docs/releases/v0.2.0.md) 为准。服务、多 GPU、其他模型和量化仍需各自的实现与验证，文档调整不改变这一范围。

这些建议用于完善本仓库的文档和工作流程，具体贡献与发布要求由本项目的指南规定。

## 整个公开展示层的补充

前述六项是基础入口；本次应把同一套定位和导航落实到整个仓库。

| 展示面 | 建议立即落实 | 来源与依据 |
| --- | --- | --- |
| 项目身份 | README 双语、GitHub About/topics、Cargo 包描述和许可信息保持一致。链接 nano-vllm、vLLM 和 Candle；徽章只指向真实 CI、许可和版本，支持范围按项目已实现能力填写。 | vLLM README 用简短定位及真实的文档、安装和联系入口组织首页。[官方 README](https://github.com/vllm-project/vllm/blob/main/README.md) |
| 使用与参考 | README 保留最短首次生成命令；安装条件、CLI、Rust API、批处理、本地 checkpoint、模型 revision 和配置放进对应指南。已删除的 API 不继续出现在当前示例中。 | 官方文档区分用户指南、API 和 CLI 参考。[文档入口](https://docs.vllm.ai/en/latest/)、[CLI 指南](https://docs.vllm.ai/en/latest/cli/) |
| 架构导览 | 用实际 `LLM` → `Scheduler` → `EngineCore` 路径解释调度和 KV cache，链接源码、领域词汇及相关 ADR；未来 TP seam 标明未支持。 | vLLM 将 Architecture Overview 放在开发设计资料中。[架构概览](https://docs.vllm.ai/en/latest/design/arch_overview/) |
| 排错与求助 | 为构建、CPU 开发/CUDA 推理、模型来源、显存预算和输出差异提供短排查入口；错误报告收集最小复现及必要环境。文档或 CPU 问题允许 GPU 字段不适用。 | vLLM 有独立排错指南，其 bug 表单要求环境与复现。[排错指南](https://docs.vllm.ai/en/latest/usage/troubleshooting/)、[Bug 表单](https://github.com/vllm-project/vllm/blob/main/.github/ISSUE_TEMPLATE/400-bug-report.yml) |
| 子目录入口 | 为 ADR、validation、benchmarks 和 research 增加简短目录，说明当前入口、历史记录和诊断资料；每个入口都能返回文档索引。首次运行不需要先阅读内部验收设计。 | vLLM 把用户、开发设计、基准与社区分成独立导航；这里借鉴信息层级，无需复制网站部署。[文档导航](https://docs.vllm.ai/en/latest/) |
| 社区入口 | Support、Security、贡献、Issues/PR 模板和 README 统一指向本仓库真实可用渠道。保留 best-effort 措辞；不创建空论坛/Slack、虚构邮箱、安全团队或响应时限。 | vLLM 按技术问题、讨论、贡献和私密漏洞报告分流；本项目按规模保留 GitHub 入口。[联系入口](https://github.com/vllm-project/vllm/blob/main/README.md#contact-us)、[安全政策](https://github.com/vllm-project/vllm/blob/main/SECURITY.md) |

推荐文档入口以读者要完成的事情排序：第一次运行 → 使用和 API/CLI 参考 → 架构与开发 → 验证和性能 → 版本与迁移。具体文件名可以沿用现有路径。开发与验收资料的详尽程度不应迫使首页使用者理解来源角色、监督协议或执行 DAG。

## 当前版本与两个早期版本统一格式

vLLM 将版本重点、兼容性、变化类别与发行物集中呈现，并将重点变化链接到 PR。本项目可采用共同标题顺序，按各版真实内容填写；无需复制其全部模型或硬件分类。[官方 v0.31.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)

1. 版本标题、原发布日期、状态和原标签链接。
2. 本版重点：读者可观察到的能力或修复。
3. 支持范围与环境。
4. 获取源码与运行：使用该版本真实命令。
5. 升级与兼容性。
6. 验证摘要与已知限制。
7. 发行物、完整变更和相关版本链接。

本项目历史处理依据已经存在的 [RC1 记录](../docs/releases/v0.2.0-rc.1.md) 和 [正式版本记录](../docs/releases/v0.2.0.md)：

- `v0.1.0` 标为早期版本，能力和命令按其标签源码核对；无法确认的历史 GPU 或模型验收不写为通过，迁移入口指向 0.2.0。
- `v0.2.0.rc.1` 保留 prerelease 身份及当时的数值 **FAIL**；其 smoke 成功不升级为完整精度或性能通过。已被正式版替代的“尚未完成”项目标明是 RC 发布时状态，并链接正式版的后续结果。
- `v0.2.0` 保持当前正式版本身份和原验证来源；基准数字继续绑定硬件、模型、dtype 与工作负载。
- 三个 Release 的标题、正文、目录导航和工具说明采用同一结构，能力与命令仍按各版代码填写。原发布日期、prerelease/Latest 状态及附件保持历史事实。旧附件的适用版本和是否属于完整验收凭证写清。

2026-10-13 的文档维护从 `main` 和 Release 页面扩展到三个版本的源码归档，
包括 `research/` 与 `tools/` 的详细说明。该阶段从各自原始发布提交派生，
只修改文档。同步标签时需要记录原代码提交、文档修订提交和新标签目标，
并保留原提交的可追溯性。旧精度、性能和实验结果继续对应原受测源码，
文档修订本身不构成新一轮 GPU 验收。这次既有标签修订来自本仓库的明确授权，
vLLM 的历史标签处理方式未作为操作依据。

三个快照的工具说明应逐个核对已有源码：原型是否属于 Cargo workspace、依赖与
CUDA 特性、CLI 参数、模型来源、输入/输出、CPU/GPU 条件，以及结果是否仅为诊断。
早期版本不存在的脚本或新协议不能复制成该版本已实现的功能；源码中预期的 PASS
文案也不等于保留了真实测量记录。

维护者发布指南应对应这个共同结构，版本目录清楚列出正式版、历史版及 RC，并让 CHANGELOG、仓库版本文件与 GitHub Release 相互可达。代码补丁使用新版本；此次文档修订按上述授权保留代码与测量的历史身份。vLLM 的发布流程同样将版本标签和受测提交关联，但本项目不照搬其发布频率与构建矩阵。[发布流程](https://docs.vllm.ai/en/latest/contributing/release_process/)

## 避免误导的最终检查

- README 双语、GitHub About、Cargo 元数据、公共 API 示例与支持范围一致；所有入口真实可用，不新增未部署网站或未发布 docs.rs/crates.io 页面。
- 当前指南不继续引导读者使用已移除 API、RC 参数、过时 Rust 下限或尚未实现的模型/TP 功能。
- 历史 ADR 和诊断文件标明记录时间及适用范围；旧 `FAIL`、proposed/budget-pending 状态和旧资产发布设计保留原含义，读者可找到当前正式版结论。
- CPU CI、文档 CI 与本地 GPU 验证的职责准确区分，WSL 不构成独立发布阻塞。文档修订无需重复 GPU 验收；当前源码与原始证据的来源身份保持不变。[本项目贡献指南](../CONTRIBUTING.md)、[公开 CUDA 检查](../docs/validation/public-gpu-smoke.md)
- 验证摘要不从 CPU CI 通过推导模型精度，不将显存采样约束写成端到端延迟保证，不作 vLLM 性能对等声明。
- 发行物准确区分 GitHub 源码归档与实际附件；模型权重、原始日志和完整证据包保留本地，不暗示它们已经上传或完成独立消费者门禁。
