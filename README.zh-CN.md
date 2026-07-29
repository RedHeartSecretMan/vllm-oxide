# vllm-oxide

[English](README.md) | **简体中文**

[CI][ci-url]
[License: Apache-2.0][license-url]

[ci-badge]: https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml
[license-badge]: https://img.shields.io/badge/license-Apache--2.0-blue.svg
[license-url]: LICENSE
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 Rust 移植版，逐步靠近 vLLM 的 V1 架构。

- **单 GPU 离线推理**——无服务器，无异步。同步引擎中的持续批处理、前缀缓存、分页 KV 缓存和仅重计算的抢占策略。
- **架构无关的模型注册表**——目前以 Qwen3 为首要支持；添加新架构只需一个文件加一条 `mod` 声明，零改动。
- **双层正确性保证**——CI 属性测试快速捕获回归；GPU 发布门禁通过黄金夹具验证数值输出，以 transformers 为预言机。

[架构概览](#架构概览) | [快速开始](#快速开始) | [测试](#测试) | [贡献指南](#贡献指南)

## 目录

- [这是什么？](#这是什么)
- [架构概览](#架构概览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [库使用方式](#库使用方式)
- [构建特性](#构建特性)
- [测试](#测试)
- [文档](#文档)
- [贡献指南](#贡献指南)
- [最低支持的 Rust 版本 (MSRV)](#最低支持的-rust-版本-msrv)
- [安全](#安全)
- [许可证](#许可证)

---

## 这是什么？

vllm-oxide 将 LLM 推理带入 Rust 生态。它构建在 [candle](https://github.com/huggingface/candle)（CUDA 内核、安全张量操作）和 flash-attention（分页注意力内核）之上，提供一个同步、进程内的推理引擎，支持：

- 持续批处理与前缀缓存
- 分页 KV 缓存（`block_size = 256`）
- 仅重计算（recompute-only）抢占策略

v0.1 面向**单 GPU、离线推理**（无服务器、无异步）。引擎以 Qwen3 为首要支持模型，但模型注册表是架构无关的：添加新架构只需新增一个文件加一条 `mod` 声明，无需修改现有代码。

### 项目目标

- 生产级 Python 推理引擎的直接替代——先求正确，再求性能。
- 架构决策以 ADR 形式记录（`docs/adr/`），领域词汇表记录在 `CONTEXT.md` 中。

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│                   LLM (composition root)              │
│  LLM::new(source, opts) → LLM::generate(prompts, …)  │
└────────────────────┬─────────────────────────────────┘
                     │ owns
┌────────────────────▼─────────────────────────────────┐
│                    EngineCore                          │
│  Scheduler → Blocks → KVCacheManager → PagedKVCache  │
│  ↓                                                     │
│  model.forward() → compute_logits() → Sampler         │
│  ↓                                                     │
│  detokenize → RequestOutput                            │
└────────────────────────────────────────────────────────┘
```

引擎运行一个同步的 `step()` 循环：调度 token、准备张量、执行模型前向传播（hidden states）、从最后一个 token 的 hidden state 计算 logits、采样下一个 token、更新 KV 缓存，然后重复直到所有序列完成。

关键设计决策（完整词汇表见 `CONTEXT.md`）：

- **分页注意力（Paged attention）**：K/V 缓存存储在固定大小的块（`block_size = 256`）中。预填充阶段使用非分页的 `flash_attn_varlen`；解码阶段使用分页的 `flash_attn_varlen_paged_windowed`。
- **前缀缓存（Prefix caching）**：`BlockPool` 中的链式 XXH64 哈希表对跨请求的公共提示前缀进行去重（写时复制语义）。
- **TP 接缝（TP seam）**：`ParallelStyle` trait 和 `TpConfig` 枚举使未来的张量并行接入成为增量式操作。v0.1 仅使用 `TpConfig::Single`。
- **CausalLM trait**：引擎面向模型的契约 —— `forward(&mut self, input_ids, positions) -> hidden_states` + `compute_logits(hidden) -> logits`。基于 inventory 的模型注册表将 HF 架构字符串映射到产生 `Box<dyn CausalLM>` 的工厂函数。

## 环境要求

### 硬件

- **仅 CPU**（测试、开发）：任意 x86-64 或 aarch64 机器，无需 GPU。
- **推理 / 发布门禁**（启用 `--features cuda`）：
  - NVIDIA GPU，计算能力 **sm_89** 及以上（Ada Lovelace RTX 40 系列、Hopper H100/H200 或更新型号）。
  - 建议至少 8 GB GPU 内存用于 Qwen3-0.6B。
  - 安装 CUDA 驱动（已在 CUDA 12.x 和 13.2 上测试）。

### 工具链

- **Rust**：edition 2021，rust-version 1.75+（见 [workspace.package] 声明）。
- **系统**：Linux（唯一支持 NVIDIA CUDA 的平台）。v0.1 不支持 Windows 或 macOS GPU。

## 快速开始

### 构建

```bash
# 仅 CPU 构建（测试、开发迭代）
cargo build

# 启用 CUDA 后端的生产构建
cargo build --features cuda --release
```

### 运行 CLI

精简 CLI（`crates/vllm-oxide-cli`）接受模型来源和可选提示：

```bash
cargo run --release -p vllm_oxide_cli --features cuda -- \
    --model Qwen/Qwen3-0.6B \
    "The meaning of life is"
```

如果命令行未提供提示，CLI 会从标准输入读取：

```bash
echo "The meaning of life is" | \
    cargo run --release -p vllm_oxide_cli --features cuda -- \
        --model Qwen/Qwen3-0.6B
```

#### CLI 参数

| 参数                   | 说明                                                                                                                       | 默认值           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `-m`, `--model`    | 本地检查点目录_或_ HuggingFace Hub 仓库 ID（例如 `Qwen/Qwen3-0.6B`）。已存在的目录解析为本地检查点；其他值解析为 Hub。 | （必填）         |
| `prompt`（位置参数） | 提示文本。未提供时从标准输入读取。                                                                                         | stdin            |
| `--temperature`      | 采样温度。`0` = 贪心解码（确定性）。                                                                                     | `0`            |
| `--top-k`            | Top-k 采样：仅保留 logit 最高的`k` 个 token。                                                                            | `None`（禁用） |
| `--top-p`            | Top-p（核）采样：保留累积概率 >=`p` 的最小 token 集合。                                                                  | `None`（禁用） |
| `--max-tokens`       | 最大生成 token 数。                                                                                                        | `16`           |

### 运行示例

```bash
# 加载并查看 Qwen3 检查点（权重、dtype、分片）
cargo run --release --example load_qwen3 --features cuda -- hub:Qwen/Qwen3-0.6B

# 在虚拟输入上运行前向传播
cargo run --release --example forward_qwen3 --features cuda -- hub:Qwen/Qwen3-0.6B
```

示例接受 `hub:<repo>` 和 `hub:<repo>@<revision>` URL，或本地目录路径。

## 库使用方式

在 `Cargo.toml` 中将 `vllm_oxide` 添加为依赖：

```toml
[dependencies]
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", features = ["cuda"] }
```

主要 API 是 `LLM::generate`，它接受批处理提示和每条提示独立的采样参数：

```rust
use vllm_oxide::{LLM, Prompt, SamplingParams, EngineOptions, Source};

fn main() -> anyhow::Result<()> {
    // 从 HuggingFace Hub 仓库构建引擎。
    let mut llm = LLM::new(
        Source::Hub {
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: None,
        },
        EngineOptions::default(),
    )?;

    // 对一批提示执行推理。
    let outputs = llm.generate(
        &[
            Prompt::Text("The meaning of life is".into()),
            Prompt::Text("Once upon a time".into()),
        ],
        &[
            SamplingParams {
                max_tokens: 64,
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                max_tokens: 32,
                temperature: 0.0, // greedy
                ..Default::default()
            },
        ],
    )?;

    for output in outputs {
        println!("[{}] {} (finished: {})", output.seq_id, output.text, output.finished);
    }

    Ok(())
}
```

### 关键类型

| 类型               | 说明                                                                                                                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LLM`            | 组合根。通过`LLM::new(source, options)` 构建，通过 `LLM::generate(prompts, params)` 调用。                                                                                                   |
| `Prompt`         | 输入枚举：`Text(String)` 用于自然语言提示，`TokenIds(Vec<u32>)` 用于预 token 化的夹具数据。同一批次中两者均可接受。                                                                          |
| `SamplingParams` | 每条提示的配置：`temperature`、`top_k`、`top_p`、`max_tokens`、`ignore_eos`、`presence_penalty`、`frequency_penalty`、`repetition_penalty`。默认为贪心解码（temperature=0）。    |
| `RequestOutput`  | 每条请求的结果：`{ seq_id, token_ids, text, finished }`。始终同时提供解码后的文本和原始 token ID。                                                                                             |
| `EngineOptions`  | 构建时的配置：`max_num_batched_tokens`（默认 16384）、`max_num_seqs`（512）、`max_model_len`、`gpu_memory_utilization`（0.9）、`enforce_eager`（v0.1 中始终为 true）、`dtype` 覆盖。 |
| `Source`         | 权重来源：`Source::Local(PathBuf)` 用于本地目录，或 `Source::Hub { repo, revision }` 用于 HuggingFace Hub。                                                                                  |

## 构建特性

vllm-oxide 使用 `cuda` 特性门控来区分仅 CPU 开发环境和 GPU 推理环境：

```toml
[features]
default = []        # 仅 CPU — 测试和开发迭代无需 CUDA。
cuda = ["dep:candle-flash-attn", "candle-core/cuda"]  # 生产后端。
```

来自 `Cargo.toml`：默认仅 CPU 以便 `cargo test` 在 CI 上无需 GPU 即可运行。生产调用方传入 `--features cuda`。

## 测试

vllm-oxide 有两个不同的测试层级，提供不同层次的保证：

### 第一层：CI 门禁（每次推送，仅 CPU）

```bash
# 单元测试、属性测试 — 无需 GPU
cargo test
```

覆盖 `EngineOptions` 默认值、`Prompt` 变体、`SamplingParams` 验证、配置解析、`Source` 分类和 CLI 参数解析。

### 第二层：发布门禁（手动，GPU）

发布门禁验证 Rust 引擎的数值输出是否与黄金夹具匹配。需要 GPU（sm_89+）、模型权重和黄金夹具存档。

```bash
# L1 token 序列精确匹配 + L2 logits 比较
cargo run --release -p vllm_oxide_test --features cuda -- \
    --model-path /path/to/Qwen3-0.6B \
    --release-tag goldens-v0.1
```

**验证内容：**

| 层级         | 验证对象                    | 验证方式                                                                                                                                       |
| ------------ | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **L1** | 贪心解码 token 序列精确匹配 | 逐位置比较生成的 token ID 与黄金 token ID。若 top-2 logit 差距在 epsilon（2 倍校准的 atol）以内，则跳过该位置（视为 BF16 精度误差）。          |
| **L2** | 每步 logits 张量比较        | 运行`LLM::generate_logits`，使用校准的绝对容差（`atol`）比较原始采样前 logits 与黄金 logits。仅比较 token 序列匹配的步骤（相同前缀比较）。 |
| **L3** | 每层激活值（调试用）        | v0.1 中为骨架代码。                                                                                                                            |

黄金夹具由 `tools/golden-gen/`（Python）生成，运行两个预言机引擎：

- **参考预言机**：transformers（BF16，`output_logits=True`，`attn_implementation=flash_attention_2`）
- **基线预言机**：vLLM（BF16，校准可接受的数值漂移）

容差：`atol = max(|transformers - vllm|, 遍历所有标准提示) x 2.0`。

夹具作为 GitHub Release 资产（标签：`goldens-v0.1`）存储，不在 git 中。完整策略见 [ADR-0005](docs/adr/0005-golden-generation-correctness-strategy.md)。

### CI 绿色与数值验证

| | CI 门禁 | 发布门禁 |
|---|---|---|
| **时机** | 每次推送 | 手动，打标签前 |
| **环境** | 仅 CPU | GPU（sm_89+） |
| **内容** | 属性测试 | 与 transformers 预言机的黄金比较 |
| **证明** | 编译通过 + 类型正确 | 数值在容差范围内正确 |

## 文档

- **[CONTEXT.md](CONTEXT.md)** —— 领域词汇表和通用语言。代码库中使用的每个术语（`CausalLM`、`BlockPool`、`PagedKVCache`、`EngineCore`、`Prompt`、`SamplingParams` 等）都在这份文档中定义，并附有不应使用的同义词的"避免"说明。
- **[docs/adr/](docs/adr/)** —— 架构决策记录（5 篇 ADR）：参数化并行层、权重加载接缝、模型注册表 + RoPE、引擎依赖 DAG、黄金生成正确性策略。
- **Crate 源代码** —— 每个模块都带有文档注释，说明其角色和 ADR-0004 依赖关系 DAG。`lib.rs` 的文档注释是最佳起点。

## 贡献指南

欢迎贡献！在提交 Pull Request 之前，请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解分支约定、提交格式和 CI 流程。

## 最低支持的 Rust 版本 (MSRV)

当前 MSRV 为 **1.75**（在 `[workspace.package]` 中声明）。我们采用滚动策略：MSRV 可能在次版本发布时提升，但仅提升至已稳定至少 6 个月的 Rust 版本。

## 安全

如需报告安全漏洞，请使用 [GitHub Security Advisories](https://github.com/RedHeartSecretMan/vllm-oxide/security/advisories/new)。请**不要**为安全报告创建公开 issue。

## 许可证

Apache-2.0 许可证。详见 [LICENSE](LICENSE)。
