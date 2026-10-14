# LLM 推理精度验收：行业证据与历史方案评审

[研究索引](README.md) · [历史草案](inference-accuracy-acceptance-proposal.md)

本文记录 2026-09-06 的研究和当时规则；“当前”均指该检查点，并非最新发布状态。
之后的正式协议见 [ADR-0015](../docs/adr/0015-layered-accuracy-validation.md)，
已批准预算见 [ADR-0018](../docs/adr/0018-freeze-layered-accuracy-budgets.md)，
实际发布结果见 [v0.2.0 验收摘要](../docs/releases/goldens-v0.2.md)。外部证据保留
研究日固定的源码版本，不能直接提供本项目的数值阈值或验收结论。

研究日期：2026-09-06。本文是研究建议，不批准新门槛，不改判既有失败，不修改代码、DAG Definition 或发布结论。检索限定于官方文档和官方源码，外部库文档先经 Context7 解析与查询；未运行外部源码、GPU 实验或编译。

## 研究结论

当前方案的基本方向合理：固定模型与运行身份、固定参考实现、区分基线、比较同前缀输出、完整计数、独立重放和校准/验收隔离，都服务于可审计的推理实现验收。但它不能仅凭八个 canonical 的 L2 通过，就宣称整个引擎数值正确；二十个只有 token/top-5 的 regression 无法参与完整分布审查，exact-token 规则也会把数值近似导致的近似并列与实现错误混在同一失败类别。

讨论中的“所有 case 保存完整参考 logits、固定前缀回放、KL 为主、TV 和参考侧选词损失辅助”有明确工程价值。行业支持使用这些不同层面的证据，但没有检索到与本提案完全对应的“逐 case，KL mean/max 对 MATH 绝对上限，并相对 vLLM 加固定余量”的行业标准。它是可辩护、仍需验证的项目设计，不能从 llama.cpp 的量化排行榜、某个 `atol` 或 MLPerf 的 99% 规则直接获得批准值。“KL 主门槛是最佳行业方案”这一表述过强；本次证据支持多层验收和场景化验证，不支持统一的最佳指标或统一阈值。

KL 衡量指定输入下的输出分布偏离，不能单独证明各算子正确、自由生成任务质量合格或运行确定性。正式验收仍须遵守当前批准的规则；既有九个 regression 失败不会因本文而变为通过。

## 研究时生效规则与待讨论方案

当时规则以 [ADR-0012](../docs/adr/0012-goldens-v0.2-calibration-and-performance-protocol.md) 和 [ADR-0014](../docs/adr/0014-goldens-v0.2-empirical-tolerance-policy.md) 为准。评审上下文中的 Base P 为 `e67f040`，candidate 为 `a297024`；正式测量状态由主线程核对，本文未重新运行测量。

| 项目 | 当前生效 | 讨论中，尚未实施或批准 |
| --- | --- | --- |
| 参考 | Transformers 4.57.6、PyTorch 2.10.0、BF16、SDPA MATH 唯一参考；vLLM 0.18.1 FA2 为校准/调查；Rust 为固定 Candle FA 路径 | 保留 MATH 参考；考虑让相对 vLLM 的表现成为第二门槛 |
| 覆盖 | 28 reference + 28 baseline；8 个 canonical/batch case 有完整参考 logits，20 个 regression 只有 reference token/top-5 | 所有 case 保存完整参考 logits |
| L1 | 有完整参考 logits 时，首分歧处 candidate 内 expected/selected logit gap ≤ 0.125 可分类为 near-tie；缺完整参考 logits 的 regression 必须 exact-token | KL/TV 外，检查 Rust 所选 token 在 reference 下的损失；greedy/top-k 保留何种门槛须另定 |
| L2 | 同前缀全词表逐元素绝对误差 ≤ 1.0；包括首分歧行，排除之后不同因果历史的行 | 输入预先固定的参考 token 前缀，持续回放并比较，不因自由生成分歧而失去后续覆盖 |
| 完整性 | 无缺失、跳过、失败等；同配置第二个 fresh process 重放符合既定逐值 bitwise 规则 | 必须继续保留 |
| 状态 | 原正式测量 L1 19/28 通过、9 个 regression 失败；8 个 L2 均通过 | 尚未按新协议完成验收；此前有基于旧 M 的离线探索指标，但非新规则验收。`KL=0.01`、`TV=0.05`、余量 `0.002` 只是教学例子，均不是生效配置 |

ADR-0014 明确当前容差只适用于固定 BF16/内核范围；校准来自四个指定 candidate case，不能用 vLLM 不一致覆盖 reference 失败。它也明确，holdout 失败后不能根据该结果提高阈值或排除 case。这个约束继续成立。

## 五组一手证据及其边界

### 1. llama.cpp：KL 是直接先例，但其实现与分数有适用范围

固定版本为 `ggml-org/llama.cpp` commit `74a7c897f049c17e7080423aa2111776eff6ebbf`，于研究日解析。官方 [perplexity 说明](https://github.com/ggml-org/llama.cpp/blob/74a7c897f049c17e7080423aa2111776eff6ebbf/tools/perplexity/README.md) 用保存的基线输出比较量化模型，同时报告 KL、PPL、语料实际 token 的概率变化、top-1 一致率和分位数。它也明确：PPL 不能随意跨 tokenizer/实现比较，微调后 PPL 升高也可能伴随人评质量改善。

[`log_softmax` 与 KL 汇总源码](https://github.com/ggml-org/llama.cpp/blob/74a7c897f049c17e7080423aa2111776eff6ebbf/tools/perplexity/perplexity.cpp#L191) 以基线概率加权，方向对应 `KL(P_base || P_candidate)`；输出 mean、max 和分位数。不过此版本缓存的是压缩为 `uint16` 的 log-prob，累计 KL 时只保留 `p_log_base > -16` 的项。不能把这些报告值当作本项目无截断全词表 KL 的可直接移植阈值。

**能支持：**同一 token 历史上保存基线分布、相对比较、同时保留尾部统计与 top-1 诊断。**不能支持：**“KL 小于某个统一常数便证明实现正确”，或把量化质量取舍直接作为 BF16 移植验收标准。该工具的统计输出本身不是本项目的发布契约。

### 2. NVIDIA：张量正确性与任务精度分别测试

TensorRT-LLM 固定 commit `15580f8ddd085499a9e73d58cebc2295060d62a8`，于研究日解析。[`test_llama_allclose_to_hf`](https://github.com/NVIDIA/TensorRT-LLM/blob/15580f8ddd085499a9e73d58cebc2295060d62a8/tests/unittest/_torch/modeling/test_modeling_llama.py#L210) 在相同输入上比较 HF 与实现的 prefill/decode logits。主线程进一步核验：它构造随机权重 HF 模型并复制权重，按设备内存缩小配置，使用八个输入 token 加一步 decode；这是局部模型路径单测，不是训练好 8B 模型的发布质量门槛。同文件的缓存回退相关测试还同时比较 logits 与 greedy token。该测试的 `atol=0.4, rtol=0.4` 不能移植到其他模型/内核，也不能为本项目 `atol=1.0` 背书。

任务层面的 [`accuracy_core.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/15580f8ddd085499a9e73d58cebc2295060d62a8/tests/integration/defs/accuracy/accuracy_core.py#L42) 则通过模型、dtype、量化等配置匹配数据集 reference，并使用样本数、标准差、显著性参数计算单侧阈值，报告最小可检测效应。它覆盖 ROUGE、MMLU 等任务。这个例子支持讨论“阈值能发现多大的退化”，并没有为 KL 给出通用检验公式。

Model Optimizer 固定 commit `a7f339ed08fe663ed5cf78d5fedf9ceaeb7c7c8b`；其 [Native Fake-Quantized Evaluation](https://github.com/NVIDIA/Model-Optimizer/blob/a7f339ed08fe663ed5cf78d5fedf9ceaeb7c7c8b/examples/llm_qat/README.md#L335) 用 MMLU、WikiText 等 benchmark 快速评估量化模型。由于这是 fake-quantized 路径，它也不能替代部署后的实际 kernel 验收。

**能支持：**数值测试和任务测试并存，门槛绑定模型/配置，考虑样本数与故障检测能力。**不能支持：**任务分数良好即可豁免张量错误，或照搬 NVIDIA 某个局部 `atol`、统计参数。

### 3. MLPerf：99% 是指定任务的质量门槛

采用有明确发布日期的 MLCommons 官方文章：[MLPerf Inference v5.0，2025-04-02](https://mlcommons.org/2025/04/llm-inference-v5/)。其中 Llama 3.1 405B Instruct 使用摘要 ROUGE-L，以及检索/文档问答的 exact match；closed division 的质量门槛为 FP16 reference 对应分数的 99%。该版本的 reference 实现采用 vLLM。

**能支持：**指定模型、语料、任务指标相对于指定 reference 的质量下限，以及任务需要时使用 exact match。**不能支持：**99% token 一致、99% 算子正确、`KL ≤ 0.01`，或 MATH/vLLM/Rust 三方误差预算。任务答案 exact match 与逐 token 跨引擎 exact match 也不是同一种判定。本文引用的是 v5.0 的具体先例，不声称它代表所有 MLPerf 版本与模型。

### 4. vLLM：严格相等与 top-N 容差用于不同测试

主线程核验的版本为 vLLM v0.18.1，commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`。[`tests/models/utils.py`](https://github.com/vllm-project/vllm/blob/a26e8dc7ff2111a005144d775ecf9cebf56c45b2/tests/models/utils.py#L225) 中，`check_outputs_equal` 检查文本/token 相等；`check_logprobs_close` 的生成输出分支在首个 token 不一致时，检查双方实际所选 token 是否都在对方保存的 top-N logprob 字典中，然后停止比较该分叉后的序列。

**能支持：**对跨实现数值差异采用有限 top-N 兼容规则有实际先例，exact token 并非所有测试的唯一准则。**不能支持：**“双方 top-2 集合必须完全相同”、全词表 KL 已经过验收，或 top-N 兼容足以证明整个模型精度。`prompt_logprobs` 分支的字典 key 检查与上述生成分支必须区分。[官方模型测试说明](https://github.com/vllm-project/vllm/blob/a26e8dc7ff2111a005144d775ecf9cebf56c45b2/docs/contributing/model/tests.md) 将这部分放在 Optional Tests；不能描述为 vLLM 全项目强制 GPU 发布规范。

### 5. FlashAttention：相对误差预算有算子级先例

主线程核验的 commit 为 `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`。[`test_flash_attn.py`](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/tests/test_flash_attn.py#L640) 对同一 Q/K/V 使用高精度 reference 与低精度、重排运算的对照；[具体断言](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/tests/test_flash_attn.py#L702) 中存在“FA 对 reference 的最大误差不超过低精度对照误差的两倍”。

**能支持：**相对一个已知低精度实现的数值预算，并非没有行业先例。**不能支持：**把单个 attention、相同 Q/K/V、特定 dtype/shape 的逐元素误差规则直接换成整模型对 vLLM 的 KL mean/max 规则。换指标、换计算层级后，原来的倍数没有自动延续的依据。

## 对新方案的具体判断

以下为基于上述证据和指标定义的项目设计推论，不是外部项目的规定。

### 固定前缀与完整 logits：值得做，但须保留实际生成检查

固定参考前缀可把每一步变成明确的条件分布比较：所有引擎看到同一 token 历史，早期近似并列不再遮蔽后续行。这修补了当前“首分歧之后无法比较”的覆盖限制。保存完整参考 logits 也让 regression 能参与全词表 KL/TV 分析，top-5 无法完整恢复这两个指标。参考侧选词损失则不同：若 Rust 所选 token 出现在 reference top-5 中，所存数值足以精确计算对应的参考 logit gap；若不在其中，就只能得到界或判为证据不足。

回放必须仍经过待验收的 prefill/decode、KV cache 和调度路径；用一次无缓存整段 forward 的结果，不能证明实际增量推理状态正确。固定前缀评估也不覆盖 Rust 自己生成后到达的所有历史。因此应保留代表性自由生成、终止规则、批处理/缓存行为的验收证据，明确每种测试覆盖什么。

### KL、TV 与参考侧选词损失：用途互补

令 `p = softmax(z_ref)`、`q = softmax(z_Rust)`，在同一个固定前缀、相同词表、温度 1、未做 top-k/top-p 截断的原始 logits 上计算：

- `KL(p || q) = Σ p_i log(p_i / q_i)`：衡量参考分布在候选下增加的期望对数损失。建议计算用稳定的 log-softmax，记录计算精度；非有限值、shape/词表/身份错误先行失败。
- `TV(p,q) = ½Σ|p_i-q_i|`：给出概率质量移动的直观尺度。它与 KL 并非完全独立证据，不必为了“有两个分数”任意设置两套门槛。
- 对 greedy，令 `y_ref = argmax p`、`y_Rust = argmax q`，可报告 `log p(y_ref) − log p(y_Rust)`，也即 reference 的两 token logit gap。它说明 Rust 的选词在 reference 看来损失多大；这与现行 candidate 内 gap 是不同判据。若改为采样，不能未经重新定义就套用 greedy 的损失阈值。

KL 仍不足以独立证明精度。数学上，`z_Rust = z_ref + c` 可使 KL 为零而 logits 逐元素不同；近似并列可以在 KL 很小时翻转 argmax；有限前缀上的小 KL 不保证未覆盖输入、极小概率 token 或自由生成任务结果。正确输出也不反推出内部每个算子都正确。原始 logits 误差和关键算子测试可继续作为定位/保护证据，其角色与分布质量不同。

greedy exact-match 适合验证合同明确要求相同离散行为的路径，例如既定同引擎同配置重放，或固定答案任务。对缓存状态等价性，只有合同要求且数值执行条件相同时，才适合要求 greedy exact-match；缓存命中/未命中、单条/批量可能改变数值执行路径，不能一概要求 bitwise 相同，可以分别要求离散状态结构一致与输出符合数值容差。跨引擎近似数值对齐时，top-N 或参考侧选词损失可帮助区分近似并列，但集合一致不代表概率一致，也不能据此遗漏其余词表。

### 对 MATH 的硬上限加相对 vLLM 余量：合理提案，尚无直接标准

保留 MATH 硬上限，可避免 vLLM 在某 case 误差很大时把 Rust 也自动放行；将相对 vLLM 门槛作为额外约束，则可表达“相对同一 reference，Rust 不应明显弱于选定成熟实现”。这是合理的工程目标。

MATH 仍是固定有限精度参考实现，并非理论真值。主线程核验的 [PyTorch 2.10 数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html) 明确浮点运算不满足结合律，数学等价的批处理/逐条执行、CPU/GPU 结果不保证 bitwise 相同；MATH 对 BF16/FP16 输入默认上转 FP32 中间计算，然后回到原 dtype。更高精度的中间计算不等于精确实数计算，BF16 权重与模型其余运算的误差也不会因 attention 选 MATH 而消失。

截至本次有限检索，未找到上述**整模型、KL、逐 case mean/max、加法余量、同时有绝对上限**这一完整组合的直接行业 precedent。FlashAttention 提供的是相近思想的算子测试；MLPerf 提供的是任务分数门槛。二者都不确定本项目应采用的常数。

如果后续批准这种设计，应预先写清比较对象。`max_t D_Rust(t) ≤ max_t D_vLLM(t) + δ` 与 `max_t [D_Rust(t) − D_vLLM(t)] ≤ δ` 不等价：前者的峰值可能位于不同行，从而掩盖相同位置的退化，后者要求位置配对。case 内 token 加权 mean 与 case 等权 mean 也不同。应先选择与要防的故障一致的定义，再冻结门槛，不能看到结果后切换定义。

当前 vLLM 是非验收基线。把它变成第二个硬门槛，会改变生效合约，需要正式修订，本文不作这项变更。

## 门槛应怎样获得可信度

合理阈值不仅要覆盖经过解释的正常数值差异，还要展示它能拒绝重要故障。NVIDIA 对样本数与最小可检测效应的显式建模，提供了这种思路的任务层面实例；本项目需按自己的固定 case、指标分布和相关性设计，不能直接套其正态检验公式。

1. 先固定模型/权重/tokenizer、运行身份、prefix 列表、位置范围、指标定义、聚合方式、完整性条件，以及可接受退化的含义。列明长前缀、prefill/decode、分页边界、批处理、缓存复用和回退等既定引擎风险的覆盖情况。
2. 保留只用于校准的数据，并指定真正独立的验收集。已经打开或失败过的 holdout 不能再次称为未见验证集；后续新协议须遵守 ADR-0014 的独立性约束。
3. 设计事先约定的错误样本或隔离故障注入，例如位置偏移、attention scaling 错误、mask/KV slot 错位、权重/词表错配。验证哪些由身份/结构检查拒绝，哪些由数值门槛拒绝；不能让身份错误进入容差分布。这是待授权实施的验证建议，本次未做任何注入。
4. 同时报告正常运行被误拒的情况与已知故障漏检情况；按 case/机制审查 mean、max、尾部与具体失败位置，避免少数严重偏差被总体均值淹没。相邻 token 高度相关，不能把词表元素数量当独立样本量。
5. 在验收前冻结阈值、版本及依据。验收失败时调查实现或依既定流程建立新的独立协议；不得从失败值反算恰好通过的阈值。固定配置的重复运行验证 determinism，无法代替这些精度证据。

分层后的验收声明应准确：算子测试证明已测张量路径的数值兼容；固定前缀分布测试证明已测条件分布的接近程度；任务测试证明指定数据集与指标的质量；fresh replay 证明固定环境下的可重复性。四者不能互相替代，也都不是对所有输入的证明。

## 核验限制

本次只做官方资料和源码审阅，未测出 Rust/vLLM/MATH 的新 KL、TV、任务分数或故障检测率，因此不能判断新门槛会让多少 case 通过。外部 commit 固定的是研究日读取的版本，MLPerf 采用明确标注的 v5.0 历史先例；没有声称覆盖行业全部实现或最新所有 benchmark。

vLLM、FlashAttention 和 PyTorch 的具体证据由同一研究任务主线程核验后合入；本子任务直接核验了 llama.cpp、NVIDIA、MLCommons 来源以及 t45 ADR0012/0014，主线程还交叉核验了 NVIDIA 单测的随机权重与缩小模型范围。既有测量计数由主线程提供，本文没有重新生成 golden 或重放正式测量。未提出额外的 Transformers MATH 对 Transformers FA 后端隔离实验。
