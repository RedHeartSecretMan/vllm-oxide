# vllm-oxide

[English](README.md) | **简体中文**

[![CI](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/RedHeartSecretMan/vllm-oxide)](https://github.com/RedHeartSecretMan/vllm-oxide/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

vllm-oxide 是一个 Rust 推理引擎，通过同步 API 在单张 GPU 上运行 Qwen3 离线推理。它基于 [Candle](https://github.com/huggingface/candle) 和自有 CUDA 内核，架构参考 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 与 [vLLM](https://github.com/vllm-project/vllm)，支持持续批处理、前缀缓存、分页 KV 缓存和抢占重算。

v0.2.0 已以源码形式发布，验证配置为单张 CUDA GPU 上的 Qwen3-0.6B BF16。详情见[发布说明](docs/releases/v0.2.0.md)、[验收摘要](docs/releases/goldens-v0.2.md)和 [GitHub Release](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0)。

[快速开始](docs/getting-started.md) · [文档索引](docs/README.md) · [架构](docs/architecture.md) · [版本发布](docs/releases/README.md) · [贡献指南](CONTRIBUTING.md)

## 可以实现什么

通过 Rust 库提交一批文本或 token ID 提示，即可取得生成的文本和 token ID。多次调用时复用同一个 `LLM`，模型会保持加载，已分配的缓存也可继续使用。CLI 使用相同的生成路径，每次处理一条提示。

| 领域 | v0.2.0 支持范围 |
|---|---|
| 模型与执行 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、BF16、单张 CUDA GPU、同步离线生成 |
| 请求处理 | 一次生成调用内的持续批处理、分块 prefill、稳定请求 ID、按输入顺序返回结果 |
| KV 缓存 | 分页存储、共享前缀缓存、写时复制、抢占重算 |
| Token 选择 | Greedy、temperature、top-k、top-p、presence/frequency/repetition penalty、EOS 与 token 上限停止 |
| 分发形式 | 带版本标签的源码、Rust 库和 CLI；没有预编译程序或已发布的 crates.io 包 |

服务接口、异步流式输出、多 GPU/TP/NCCL、其他模型族、量化、LoRA、speculative decoding 和 CUDA Graphs 均不在本版本范围内。

## 快速开始

GPU 推理需要 Linux 或 WSL2、**Rust 1.94 或更新版本**、计算能力 **8.9 或更高**的 NVIDIA GPU、可用的 NVIDIA 驱动，以及包含 `nvcc` 的 CUDA Toolkit。发布实测使用 RTX 4080、驱动 595.71 和 Toolkit 13.2.51；其他硬件与工具链组合尚未经过同等发布验证。

克隆发布源码，通过自带示例使用验收时的固定模型 revision：

```bash
git clone --branch v0.2.0 --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
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
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", tag = "v0.2.0", features = ["cuda"] }
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

## 支持范围与性能

[发布验收摘要](docs/releases/goldens-v0.2.md)记录了本版本的数值、行为和公开 CUDA 检查结果。在固定 RTX 4080/WSL2 工作负载中，decode 吞吐中位数为**单请求 29.7601 token/s**、**四请求合计 115.8207 token/s**。工作负载、测量边界和显存用量见[基准测试](docs/benchmarks/README.md)。这些结果只适用于实测的工作负载与硬件，不能据此认定与 vLLM 的性能相同。

## 开发与验证

CPU 开发无需 CUDA：

```bash
cargo build --locked --workspace
cargo test --locked --workspace
```

CPU 构建用于开发与测试。运行推理需要启用 `--features cuda` 并使用 GPU。GitHub CI 检查编译、格式、Clippy、Rust 测试、Python CPU 验证工具及依赖；模型精度、CUDA API/CLI 功能和性能另有本地 GPU 检查。

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
