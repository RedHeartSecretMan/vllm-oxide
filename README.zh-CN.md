# vllm-oxide

[English](README.md) | **简体中文**

[![CI](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/RedHeartSecretMan/vllm-oxide)](https://github.com/RedHeartSecretMan/vllm-oxide/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

vllm-oxide 是一个 Rust 推理引擎，支持**单 GPU、同步、离线的 Qwen3 推理**，基于 [Candle](https://github.com/huggingface/candle) 和项目自身的 CUDA 内核。架构参考 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 与 [vLLM](https://github.com/vllm-project/vllm)，提供持续批处理、前缀缓存、分页 KV 缓存和抢占重算。

**此源码对应历史实验预发布 v0.2.0.rc.1。** 面向单张 CUDA GPU 上的 **Qwen3-0.6B BF16**。原数值运行记录为 **48/66 PASS、18 FAIL**，完整性能和声明的 Rust 下限尚未验证。新用户请使用正式 [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0)。详情见[发布说明](docs/releases/v0.2.0-rc.1.md)、[历史验证](docs/validation/README.md)和 [GitHub Release](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0.rc.1)。

[快速开始](docs/getting-started.md) · [文档索引](docs/README.md) · [架构](docs/architecture.md) · [版本发布](docs/releases/README.md) · [贡献指南](CONTRIBUTING.md)

## 可以实现什么

在 Rust 应用中嵌入引擎，提交一批文本或 token ID 提示，获得完整的续写文本和 token ID。跨调用复用同一个 `LLM`，可保留模型及缓存分配。CLI 使用相同的生成路径，每次处理一条提示。

| 领域 | RC1 实现范围 |
|---|---|
| 模型与执行 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、BF16、单张 CUDA GPU、同步离线生成 |
| 请求处理 | 一次生成调用内的持续批处理、分块 prefill、稳定请求 ID、按输入顺序返回结果 |
| KV 缓存 | 分页存储、共享前缀缓存、写时复制、抢占重算 |
| Token 选择 | Greedy、temperature、top-k、top-p、presence/frequency/repetition penalty、EOS 与 token 上限停止 |
| 分发形式 | 带版本标签的源码、Rust 库和 CLI；没有预编译程序或已发布的 crates.io 包 |

服务接口、异步流式输出、多 GPU/TP/NCCL、其他模型族、量化、LoRA、speculative decoding 和 CUDA Graphs 均不在本版本范围内。架构参考不代表与上游 vLLM 的 API 兼容或功能相同。

## 快速开始

GPU 推理需要 Linux 或 WSL2、记录的 **Rust 1.94.0** 工具链、计算能力 **8.9 或更高**的 NVIDIA GPU、可用的 NVIDIA 驱动，以及包含 `nvcc` 的 CUDA Toolkit。RC 记录来自 RTX 4080/WSL2；需要兼容 CUDA 驱动和完整 Toolkit。维护后的源码已固定 Rust 1.94.0；原 RC 测量与本次维护检查分别记录。

克隆发布源码，通过自带示例使用验收时的固定模型 revision：

```bash
git clone --branch v0.2.0.rc.1 --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
cd vllm-oxide
cargo run --locked --release -p vllm_oxide --example generate_qwen3 --features cuda -- \
    hub:Qwen/Qwen3-0.6B@7e4ae267688d671ddfca3122e4528ee980cf3234
```

首次运行会构建 CUDA 后端并下载模型文件。示例续写 “The meaning of life is”，后续运行复用已下载文件。也可以用本地模型目录替换 `hub:` 参数。

需要自定义提示时，可通过 CLI 加载本地检查点：

```bash
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
cargo run --locked --release -p vllm_oxide_cli --features cuda -- \
    --model "$QWEN3_MODEL_DIR" --max-tokens 2 "The capital of France is"
```

CLI 也接受 `Qwen/Qwen3-0.6B` 这样的 Hub 仓库 ID，但没有 revision 参数。CLI 选项、本地检查点和离线运行方式见[使用指南](docs/getting-started.md)。

## 在 Rust 中使用

在 `Cargo.toml` 中固定源码发布版本：

```toml
[dependencies]
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", tag = "v0.2.0.rc.1", features = ["cuda"] }
anyhow = "1"
```

```rust
use vllm_oxide::{EngineOptions, Prompt, SamplingParams, Source, LLM};

fn main() -> anyhow::Result<()> {
    let mut llm = LLM::new(
        Source::Hub {
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: Some("7e4ae267688d671ddfca3122e4528ee980cf3234".into()),
        },
        EngineOptions::default(),
    )?;
    let outputs = llm.generate(
        &[Prompt::Text("The capital of France is".into())],
        &[SamplingParams {
            max_tokens: 2,
            ..Default::default()
        }],
    )?;
    println!("{}", outputs[0].text);
    Ok(())
}
```

受支持的公开类型为 `LLM`、`EngineOptions`、`Prompt`、`SamplingParams`、`RequestOutput` 和 `Source`。`generate` 接受每条提示对应的采样策略，结果保持输入顺序，并带有稳定请求 ID。文本和 token ID 提示可以放在同一批次。详细语义见[使用指南](docs/getting-started.md#rust-api)和 [v0.2 迁移指南](docs/migration-v0.2.0.md)。

## 验证与性能

RC1 保留历史实验候选状态：487 个采集项、153 个采集组、7 项算子 profile 与 22 项公开行为检查已核验，但有 **18 项数值失败**。没有完整性能结论或生产精度保证。详见[原始结果及来源](docs/releases/v0.2.0-rc.1.md#验证与性能)。本次文档维护不改变实现、数值预算、证据身份和判定，见[维护记录](docs/version-doc-update.md)。

## 开发与验证

CPU 开发无需 CUDA：

```bash
cargo build --locked --workspace
cargo test --locked --workspace
```

CPU 构建用于开发与测试；**实际推理必须启用 `--features cuda` 并使用 GPU**。此历史源码的自动 GitHub CI 在 CPU 上检查格式、Clippy、Rust 测试与依赖；Python 工具有独立的本地 CPU 测试。当前 `main` 的后续 CI 不属于此源码快照。模型精度、CUDA API/CLI 功能和性能通过本地 GPU 单独验证。CI 通过本身不代表模型精度已经验证。

开发环境与完整检查见 [CONTRIBUTING.md](CONTRIBUTING.md)和[开发指南](docs/development.md)。[架构概览](docs/architecture.md)说明一次生成调用对应的当前实现，[验证指南](docs/validation/README.md)区分日常 CI 和真实模型检查。

## 社区与项目信息

| 需要了解 | 入口 |
|---|---|
| 使用帮助、缺陷或功能建议 | [支持渠道](SUPPORT.md)与 [GitHub Issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues) |
| 参与贡献 | [贡献指南](CONTRIBUTING.md)与[开放 PR](https://github.com/RedHeartSecretMan/vllm-oxide/pulls) |
| 社区交流规范 | [社区规范](CODE_OF_CONDUCT.md) |
| 私密安全报告 | [安全政策](SECURITY.md) |
| 版本历史与升级 | [版本发布](docs/releases/README.md)、[变更日志](CHANGELOG.md)与 [v0.2 迁移](docs/migration-v0.2.0.md) |
| 实现决策与实验 | [架构](docs/architecture.md)、[ADR](docs/adr/)与[研究记录](research/README.md) |

## 许可证

[Apache-2.0](LICENSE)。
