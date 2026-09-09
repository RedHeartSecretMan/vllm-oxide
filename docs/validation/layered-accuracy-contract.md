# v0.2.0 分层精度验收协议

日期：2026-09-06。

**状态：用户已批准设计与实现，数值预算和独立验收语料须后续冻结。** 本文件随 ADR-0015 的 Definition Checkpoint 生效。它不批准任何新数值阈值，不改判旧测量；预算、case 清单或必要证据未冻结时，不得输出发布验收 PASS。

## 1. 验收目标

在固定模型、权重、tokenizer、设备和运行配置下，验证 Rust：

1. 实现正确的算子与引擎行为；
2. 在相同 token 历史上接近指定参考分布；
3. 每个 case 的平均分布偏差不明显劣于 vLLM 对照；
4. 具有完整、可复查、可重复的证据。

不把“字符串完全相同”当作跨内核数值等价的唯一含义，也不声称有限测试证明所有输入正确或模型具备某种通用任务能力。固定前缀数值验收和实际自由生成行为验收均不可缺少。

## 2. 固定运行配置

| 项目 | 规则 |
| --- | --- |
| 模型范围 | Qwen3-0.6B、单 GPU、BF16；沿用当前固定权重和 tokenizer revision |
| 主参考 | Transformers + PyTorch SDPA MATH；它是有限精度参考实现，不是理论真值 |
| 工程对照 | vLLM + FlashAttention；参与本方案的相对验收门槛 |
| 候选 | Rust 生产实现；必须绑定测量 commit/tree、二进制和内核身份 |
| 后端隔离实验 | 不增加 Transformers MATH/FA 双后端隔离实验 |
| 可复现条件 | 固定版本、dtype、精度开关、seed、执行配置及模型身份；禁止静默后端 fallback |
| 数值数据 | 三方保存完整、未做 top-k/top-p 或强制选词变换的原始 logits；按 token ID 对齐 |
| 分布定义 | 分析温度固定 T=1；与实际 greedy 解码的 temperature=0 区分 |
| 指标计算 | 稳定 log-softmax，使用 FP64 计算/累计；记录算法版本，不对 top-5 单独归一化冒充分布 |

所有数值 case 共用一个已批准配置下的指标和预算。case 分类只表示覆盖内容，不再因 canonical/regression 的存储格式不同而改变数值规则。其他模型、dtype 或设备配置需要单独声明适用范围，不能自动沿用预算。

## 3. 三层必需测试

### L0 — 算子与状态不变量验证

对 RMSNorm、RoPE、SiLU、attention、KV 写入/读取和采样使用小规模可控输入。独立参考按声明的数学与 BF16 materialization 约定计算；必要时使用 FP64 或更高精度核验少量坐标。高精度计算也是验证工具，不把整个 FP32 模型替换成 BF16 生产行为标准。

- token ID、位置、mask、head mapping、请求归属、KV slot 等离散结构必须正确。
- 算子的浮点容差按算子/dtype/shape 的预先约定验证，不能套用最终 logits 的全局容差。
- 测试必须包含能明确暴露错误的构造输入；正常配置通过不代表故障检测能力已经得到证明。

### L1 — 模型数值一致性验证

每个 case 在执行前固定 prompt token IDs 和 continuation token IDs。参考可以先生成 continuation 并冻结，但评估期间三方必须读取同一条已冻结 token 流。

对每一步：

1. 三方看到相同逻辑历史、有效长度和位置；
2. 各自运行待验证的真实 prefill/decode/KV-cache 路径；
3. 记录采样或强制选词变换之前的完整 logits，以及原始 greedy 选择；
4. 将同一个冻结 token 作为下一步输入，与各自刚才预测的 token 无关；
5. 对所有预定步骤继续计算指标，不在预测分叉后停止。

强制 token 只能控制后续输入，不能污染作为证据保存的 logits。不能每一步把整个历史重新 prefill，冒充实际 decode。vLLM 若无法在指定路径提供可信的固定前缀、原始 logits，就属于缺少必要证据，不能退回不同历史上的比较。

### L2 — 解码与引擎行为验证

继续通过 `LLM::generate` 检查请求数、输出顺序、EOS、长度上限、重复调用、混合 batch、缓存复用、抢占重算及错误传播。

- 离散行为合同必须精确满足。
- 同一实现、同一冻结执行条件的独立重放，必须满足规定的 bitwise 规则。
- batch 形状、缓存命中等改变数值路径时，不能默认所有浮点结果必然 bitwise 相同；分别检查结构正确性和同前缀数值容差。
- 保留真实自由生成输出、首分叉位置和 token 一致率，作为行为证据；不能拿不同生成历史的逐行 KL 作数值验收。
- 固定前缀的通过不覆盖自由生成后可能到达的全部历史，也不替代实际接口测试。

## 4. 数值指标及精确聚合方式

对于 case c 的预定步骤 t，令 p 是 MATH 参考分布，q_R 是 Rust 分布，q_V 是 vLLM 分布。三者必须使用同一词表和历史。

### 4.1 L1：KL 主指标

```text
D_R(c,t) = KL(p || q_R)
D_V(c,t) = KL(p || q_V)

K_mean(c) = mean_t D_R(c,t)
K_peak(c) = max_t D_R(c,t)
E_mean(c) = mean_t [D_R(c,t) - D_V(c,t)]
```

方向固定为 reference → candidate，使用自然对数，单位为 nats。每个 case 单独验收：

```text
K_mean(c) <= A_mean
K_peak(c) <= A_peak
E_mean(c) <= Delta_mean
```

这三个条件分别限制持续偏差、单步极端偏差、相对于 vLLM 的平均额外偏差。相对条件不是纯倍率，因此 vLLM 偏差接近零时不会发生除零或倍率放大。绝对条件独立存在，vLLM 偏差较大也不能提高 Rust 的绝对上限。

这里明确选用每 case 的配对平均差作为 vLLM 硬门槛。`max_t(D_R-D_V)` 记录为诊断量，本版不再增加一个未校准的逐 token 相对硬门槛。不要用两个实现各自的最大值相减，假装是在比较同一最坏位置。若将来要求逐点不劣性，应另行修订。

### 4.2 L2：参考端选择损失

```text
a = reference 的最优 token
b = Rust 原始 logits 的 greedy token
G(c,t) = z_ref[a] - z_ref[b] = log(p[a] / p[b])
G_peak(c) = max_t G(c,t)

G_peak(c) <= G_limit
```

- G=0：Rust 选中了参考端并列最优集合中的一个 token；只说明该步选择在参考下最优，不代表整个 case 通过。
- G>0：明确报告参考偏好损失，并按冻结的 G_limit 判断。
- 不要求双方 top-2 集合相同；top-k 交集与 token 一致率作为诊断。
- 这与现行 candidate 内 gap 是不同规则；不自动复用 0.125。

### 4.3 辅助报告

记录每 case 的 TV mean/max/P95、KL P95、原始 logits 最大误差、配对 KL 差分布、首分叉、token 一致率和最坏坐标。

TV 用于解释概率质量变化。本版不在 KL 之外随意增加另一套 TV 硬阈值；若业务明确提出概率质量变化上限，可在批准前增加一个有依据的 TV 预算。原始 logits 不再单独充当全部输出行为的判据，但算子测试仍保留所需的张量误差检查。

总体平均值采用 case 等权报告，同时列出逐 case 原值和步骤数。任何总体平均或多数 case 的通过，都不能抵消某个必需 case 的失败。

## 5. Case 组织

现有 28 个已观察用例全部保留为开发/回归集，并补齐完整参考和对照数据。它们不再被称为未见验收集。

新增机制覆盖至少包含：

| 机制 | 必须明确触发的条件 |
| --- | --- |
| 长度与分页 | 1-token 输入；255/256/257、511/512/513；decode 跨页；上下文长度上限含 completion 预算 |
| Prefill | 短/长输入；完整与分块 prefill；末块不足一个 chunk |
| KV/prefix cache | 冷缓存、命中、部分共享、不同后缀；必须验证未复用错误历史 |
| 调度 | 真正的混合长度 batch、重复调用、等待请求进入、受控资源压力下抢占重算 |
| 采样 | 构造明显最优、双/多 token 并列、penalty/filter 边界；验证正确 token ID 而非排名列 |
| 接口错误 | 无效参数、不可满足的长度或资源约束、错误传播、零丢失输出 |

采取覆盖矩阵选择组合，不机械运行所有维度的笛卡尔积。校准集和新的独立验收集都覆盖关键机制，但不能复用相同样本或已观察失败的 holdout。具体 token 流、用例 ID、长度/路径、数量与 expected 清单在运行前冻结。覆盖项未落实成具体 case 时，不得视为验收完成。

## 6. 阈值冻结是独立阶段

以下参数尚无足够证据可给出最终数值，默认均为空：

| 参数 | 含义 | 生效前要求 |
| --- | --- | --- |
| A_mean | 每 case 平均 KL 绝对上限 | 校准、误差预算与检测能力说明 |
| A_peak | 单步 KL 绝对上限 | 最坏情况和关键错误检测说明 |
| Delta_mean | 每 case 相对 vLLM 的平均额外 KL 预算 | 同历史成对数据及工程不劣性目标 |
| G_limit | reference 选择损失上限 | 可接受选词偏好损失的明确含义 |

这些参数不能取自教学例子，不能把现有最大 KL 向上取整后自动成为批准值。也不存在可直接套用的行业统一 KL 阈值。

校准流程：

1. 先批准本方案的验证目标、指标、聚合方式、覆盖清单与校准权限；此时只允许形成非验收观测。
2. 在不查看新独立验收结果的情况下，生成三方校准数据、复核异常。
3. 对隔离的测试替身或测试开关注入事先约定的错误：位置偏移、错误 mask、漏历史、KV slot/权重/词表错配、明显错误的 logits 分布等。不得改动正式产物或把注入样本混入正常误差分布。
4. 列明哪些错误由结构/身份门槛拒绝，哪些必须由数值门槛拒绝；同时报告合法数值差异被误拒的情况。无法区分重要故障与允许误差时，应改进测试，而非无限放宽。
5. 提交完整阈值表、适用配置、依据和限制，经独立审阅及用户批准后冻结协议版本。
6. 再运行新的独立验收集。相邻 token 高度相关，不能将 token 行数或词表元素数当作独立样本量来夸大可信度。

## 7. PASS / FAIL / INVALID

```text
PASS =
    协议与全部阈值已批准且冻结
    AND 全部必需 case/步骤/fixture 完整且身份正确
    AND 全部算子与行为硬检查通过
    AND 每个数值 case 的四个硬条件均满足
    AND 三方要求的独立重放通过
    AND 没有任何必需项未验证
```

- **FAIL**：可信、完整的执行证据违反已批准的数值或行为条件。
- **INVALID**：环境/身份不匹配、缺失数据、对照无效、无法获得同前缀、阈值未批准等；表示不能作有效验收判断，仍阻止发布。
- **PASS_WITH_WARNING** 不作为单独放行通道。可在满足全部硬条件后附带接近阈值的警告；硬失败不能由人工口头豁免变成 PASS。

NaN/Inf、错误形状、缺失或跳过等均不能作为容差内样本通过。失败或 INVALID 都不生成成功阶段标记。正式失败若需改变验收含义，必须重新修订并按独立性要求验证，不能修改原结果。

## 8. 实现与资源约束

比较 Module 使用一个纯计算 Interface，接收已验证的同前缀数据及冻结政策，返回指标和逐条件判定。采集、身份/形状校验与比较分开，避免不同 case 各自复制规则。

公开行为测试仍走 `LLM::generate`；固定前缀控制放在私有测试 Interface，不能扩大公开 API。新增比较器先使用小型合成数据完成 CPU TDD：同分布、整体 logit 平移、轻微排序变化、分布明显错误、非有限值、词表错位、前缀错位和缺行等都有明确预期。

GPU 阶段串行、每个 owner 使用独立进程；保持原模型/依赖缓存，Rust 复用原 CUDA target 并限制 jobs=1，禁止触发 FlashAttention 原生重编，RAM 可用量低于 16 GiB 停止。完整 logits 可逐 case/逐段落盘，避免为了保存更多证据同时驻留多个模型或全部语料激活。

## 9. 生效与交付

执行顺序：本设计检查点 → 纯比较器/回放 TDD 和审查 → case 注册表检查点 → 非验收校准证据 → 具体预算审批检查点 → 新独立验收 → 报告和发布流程。

正式报告绑定政策版本、模型/运行/代码身份、所有输入/产物哈希、每 case 的指标与 verdict、覆盖计数、重放结果和未验证限制。旧协议报告及其失败永久保留为历史证据；本协议不承诺新标准一定让当前实现通过。

本协议的决定及历史兼容规则见 [ADR-0015](../adr/0015-layered-accuracy-validation.md)。研究笔记和教学示例不构成规范性输入，不从其他项目复制数值常数。

## 10. 版本与层级的无歧义绑定

初始协议标识为 `layered-accuracy-v1`，层级字段使用 `operator_checks`（L0）、`numerical_checks`（L1）、`behavior_checks`（L2）。L0 不是 Transformer 的第 0 层；L1/L2 不是范数。

旧协议的 L1=token、L2=logits 和旧 per-layer/L3 调试结果保持历史含义，不能重命名成新证据。新的 manifest、policy、capture、报告及 marker 必须显式绑定协议标识和各自 schema 版本；不根据旧字段名猜测新语义。未知或缺失的新协议标识不能通过新验收。旧 `1.0/0.125` 不填入 A_mean/A_peak/Delta_mean/G_limit。历史工件如需读取，应进入明确的 legacy 路径，不可满足新的成功门禁。

层级名称与数值指标是独立字段：L1 输出中包含 KL/TV/logit 诊断，L2 输出包含参考选择损失与真实生成行为；不得仅交换旧 `l1`/`l2` JSON key 来实现迁移。

## 11. 固定前缀采集的生产者/消费者契约

设 continuation 长度为 T（T>0）。第 t 行的条件历史严格为 `prompt + continuation[:t]`，t=0..T-1；恰好 T 个预测行。每行分别记录 `predicted_token_id`（原始 greedy 选择）和 `advance_token_id=continuation[t]`（逻辑输出/推进 token），不能用强制 token 冒充模型预测。

最后一行仍可记录逻辑输出 token，但不允许据此声称执行了第 T 行模型计算。固定回放关闭 EOS 提前停止，显式记录此模式；自由生成必须关闭 forcing，并恢复其真实 EOS/长度策略。两种模式不互相冒充证据。

每行绑定 case/member、调用和运行时 request identity、step、输入历史哈希、position、真实 phase、有效长度、词表及原始 logits shape。固定 case identity 不能与运行时 request ID 混用。prefill 分块或调度额外执行步不得伪装为额外预测行；预测行必须对应真实采样边界。混合 batch 的 case registry 必须声明分组和成员，并真的以该组执行，不能将成员逐条运行后标成 batch 证据。

forcing 必须保留原始 logits 和原始预测的独立副本，不论框架先执行 processor 还是先保存输出。生产者/消费者均验证 token 流、行/成员完整性和身份；真实采集启用/禁用 forcing 的受控等价检查须证明 logits 没有被强制选词污染。没有这些证据时采集结果不能用于数值解释或验收。

## 12. 指标实现与未批准状态

稳定 log-softmax、概率、自然对数 KL、TV 和聚合在 FP64 中计算；使用补偿或同等级的稳定求和。不得 top-k 截断，不得给概率加任意 epsilon 来掩盖零或计算错误。输入 logits 必须有限且具有完整非空词表。FP64 可表示范围造成的下溢与处理方法必须写入算法版本和测试，不能静默改变数学定义。

KL 理论上非负。仅对 [-1e-12,0) 内的微小计算舍入值归零，并保留原值与归零计数；小于 -1e-12 是指标计算无效，不可按模型精度通过。这个固定算术检查不是模型容差。P95 采用排序后的 (n-1)*0.95 位置线性插值；case mean 使用该 case 的全部预定预测行，总体摘要为 case 等权。

原始 greedy 的并列选择在各适配器中明示；Rust 继续遵守最小 token ID 的既有 tie-breaking。参考选择损失使用 reference 的全词表最大值，因此并列最优集合不受 reference 实际挑选哪个 ID 影响。原始预测与 argmax/既定采样参数不符时先判行为错误，不能只据 G 值放行。

空预算用显式 pending/null 表示，不等于零，也不等于无限大。A_mean、A_peak、Delta_mean、G_limit 以及 L0 浮点 profile 均须有经批准的有限、非负数值和来源；A_mean<=A_peak。缺任何适用预算、case registry 或审批绑定，新的 authoritative 入口返回 INVALID，且不生成成功 marker。非验收观测可以输出指标和完整性结果，但必须显式 `accepting=false`，不能借整体 exit0 或字段默认值形成发布成功。

## 13. 后续冻结及继承的发布约束

本检查点仅授权实现和 CPU 验证，并允许准备后续校准方案。具体 case registry 由执行者在本覆盖约束内提出，包含 L0 profile、数值/行为 case、分组、token 流或确定性生成规则、预期行/fixture 数、分割与必需故障检测用例。Coordinator 通过后续 Definition Checkpoint 审核绑定；覆盖矩阵未落实时保持未完成。之后的 GPU 阶段仍需资源与阶段授权。

新独立验收集的定义可预先审阅，但在预算批准前不得获取/解释其数值输出用于阈值选择。旧 28 个已观察用例不得作为新 holdout。预算审批必须绑定校准 corpus、源代码、完整数据、原始指标和故障检测结果，并经独立审阅和用户批准；本次设计批准不替代预算批准。

ADR-0012 未被本文件替代的模型/权重/tokenizer/环境身份、资源保护、确定性、真实性及发布流程仍有效。旧 56-fixture 固定数、四个校准/四个 holdout 分割和旧容差只属于历史协议；新协议以已冻结 registry 的完整预期清单计数，而不是取消计数。新的 collector、schema、prompt 或数值代码必须先冻结测量提交并产生自己的数据；旧 O/P/M 工件不能因 ancestry 存在就改标为新协议。

所有成功阶段均需原子、完整输出 marker 和前驱/源身份验证；阈值未知的观测 completion 与 release PASS 必须是不同状态。发布仍需 L0/L1/L2 全通过、性能记录、证据报告、bundle/clean consumer 验证，最终 evidence-only 候选的完整复审及单独的 tag/release/asset 权限。资产仍仅为 manifest 和 archive，报告不是第三资产；CPU CI 不等于 GPU 验收。此检查点不提供发布权限。

## 14. 数值用例与执行组的适用边界

每个注册的请求 member 是独立的 Numerical case，分别使用自己的 T 个预测行计算 mean KL、peak KL、paired mean KL excess 和 peak reference choice loss。Execution group 只描述一次共同执行；组通过要求所有成员通过且组级行为检查通过。组平均值仅作摘要，不能抵消任何成员的硬失败。setup 调用、控制采集及独立重放在预期清单中单列，不计入目标 case 的 T 行。

三端同前缀比较要求相同的条件 token 历史，不要求内部调度事件相同。注册表须写明各机制要求适用于哪个 engine：Rust 的分页 KV、分块 prefill、prefix cache、等待准入和抢占重算，由 Rust 实际执行事件证明；vLLM 声明启用的对应机制由其自身证据证明。Transformers MATH 不因缺少 Rust 调度机制而伪造对应事件，也不能代替 Rust 提供机制覆盖。所有适配器仍须如实记录自己的实际 phase、输入、位置、有效长度和预测边界，不以此区分放宽共同的历史、身份、完整性与重放要求。

## 15. 注册表与待批准预算

[用例注册表](layered-accuracy-cases.json) 是完整规范输入：包含逐成员 token 流、setup 调用、执行配置、各 engine 的机制与缓存范围要求、算子输入及故障定义、公开行为调用清单、检查语义、来源身份和预期数量。它不包含模型测量值、当前运行状态或验收结果；未推送的实现指针不代替内嵌的输入和数学定义。

初始注册表的数值清单如下；行数均按一个 engine 的一个基础 variant 计。

| 分割 | 执行组 | 逐成员数值 case | 目标预测行 | setup 调用 / 预测行 |
| --- | --- | --- | --- | --- |
| development | 25 | 28 | 1152 | 0 / 0 |
| calibration | 13 | 19 | 75 | 1 / 1 |
| acceptance | 13 | 19 | 75 | 1 / 1 |

每个数值组要求 reference、baseline、candidate 三端各自的 primary、replay 和 control，合计九个基础独立采集 owner。control 关闭 forcing、忽略 EOS 并保留预定 T 行；它不是 fixed-prefix 数值行的替代品。calibration 和 acceptance 各自的 prefix-hit、pressure、waiting 组另要求一个新的 candidate `control-replay` owner，用于真实公开行为的独立重放。其精确输入与用途见注册表 `owner_inventory`。跨分割的实际预测历史（含 setup）不得重用。分割内为比较执行路径或验证 prefix sharing 而明示复用输入是允许的；逐 case 独立判定不等于统计独立，不能把这种复用夸大为独立样本量。

calibration 和 acceptance 各有五个独立自由生成行为场景，每个 primary/replay variant 包含 124 次调用、116 个成功请求输出及 10 次预期拒绝。另各保留三个 fixed-prefix 结构场景，并新增三个 `unforced_control` 公开行为场景：prefix-hit、pressure、waiting。后者明确通过 `LLM::generate` 完成所有冻结的 setup 与目标调用，整个 owner 中 forcing 都关闭；仅取 continuation 的长度作为 completion 预算，不推进冻结的 token 值。每个 control/control-replay variant 另包含四次公开调用、八个成功请求输出。

`unforced_control` 必须保存排序和文本解码完成后的真实 `RequestOutput`、实际准入身份与参数、原始 logits、实际采样 token 及执行事件；公开返回值须与该 owner 自身的采样历史一致。control 和 control-replay 都须验证缓存范围、等待准入或抢占重算等声明机制，并通过完整 logits 与公开返回值的独立一致性检查。setup、返回值、原始采集、receipt、guard 及新增重放均进入证据闭包。fixed-prefix 通过不能代替这些无 forcing 的公开调用证据，也不能通过修改模式标签来满足要求。

每个新分割共有 117 个数值用途 owner、16 个公开行为用途 owner，其中三个既有 candidate control 同时承担两种用途；因此是 130 个唯一 owner，而不是相加的 133 个。其中 120 个属于组采集（含三个新增 control-replay），10 个属于独立公开行为采集。新增 control-replay 合计 30 个目标行与一个 setup 行，使本分割所有组采集共计 715 行。`expected_counts` 中的 owner identity 与 `uses` 显式记录重叠，不允许重复计数或漏掉独立重放。

L0 包含七个 profile、primary/replay 两个实际算子采集 owner 和 16 个隔离 CPU 算子故障模型。注册表另列七个全局故障条目，其中一个条目含两个独立变换；所有必需变换均须验证，不能只按条目数声称覆盖。包含 development 的完整冻结清单为 225+130+130+2=487 个唯一 GPU owner；该数量只是完整性要求，不是立即执行所有分割或提前运行 acceptance 的授权。

专门的 EOS 场景要求真实自由生成触发提前 EOS 停止。未触发时是必需机制未覆盖，返回 INVALID 并阻断验收，不能直接诊断为模型精度或 EOS 状态机错误；实际已输出 EOS 而违反停止策略时才是行为 FAIL。不得在观察 acceptance 输出后反复更换提示挑选通过结果。校准阶段若证明输入不能触发所需机制，须按前述 Definition 与独立性规则修订。

[预算政策](layered-accuracy-budgets.json) 绑定注册表原始字节的 SHA-256 和指标算法版本。`values=null` 表示 A_mean、A_peak、Delta_mean、G_limit 均未批准；七个 `operator_budgets` 同样保持 null，校准来源、证据哈希与理由也尚未提供。后续具体预算仍须校准与故障检测证据、独立审阅及用户批准。本次用例冻结不批准任何数值、不生成 PASS、不授权 GPU 阶段或发布；实际实现仍须完成 CPU 检查和新候选审查，后续 GPU 采集另按阶段权限与资源保护执行。

## 16. 压力场景与固定调度回归

两个数值分割的 pressure 组使用 prompt 长度 512/255、completion 预算 4/5、每步 token 预算 768、context 上限 768、最多两个并发请求和三个 256-token 的 Rust 物理 KV block。Rust 必须在已经推进至少一个 completion token 后，真实重新执行此前计算过的历史区间；单有 prefill 标签或 prefix-cache 命中后只计算新 token 都不足以证明重算。

baseline 按 [ADR-0017](../adr/0017-pressure-baseline-usable-capacity.md) 使用 51 个 16-token 总 block；固定 vLLM 版本保留一个 null block，因此有 50 个可用 block，可容纳两请求完整计算历史的 33+17 个 block。baseline 仍必须证明 prefill、decode、batch，但不要求它重算；Rust 的三 block 压力和重算要求不变。49 个总 block 已能容纳初始 32+16 个 block，51 不是触发 batch 的唯一或最小值，而是完整历史容量的选择，不额外规定两端调度顺序一致或 baseline 永不抢占。两个数值 pressure 条目和对应公开行为条目的配置镜像必须一致。容量计算不代替新配置下的实际 GPU 机制与数值验证；旧配置工件保持原来源和判定。

pressure 组不再声称覆盖 chunked prefill；分块覆盖仍由 chunk-remainder、mixed、waiting 等明确要求该机制的组承担。fixed-prefix、无 forcing 的 control 及独立 control-replay 都必须以自身事件证明机制，CPU 指纹替身的成功只证明调度几何与历史路径，不代替 Qwen 的 GPU 数值验收。

[混合阶段压力修复范围](t45-mixed-phase-pressure-repair.md)中的 128-token 步预算、三个 KV block 的公开接口回归仍是必须通过的 CPU 结构门槛，须与四 block 及同容量串行对照输出一致。它不计为 GPU owner，也不因 768-token 场景成功而被删除、忽略或放宽。
