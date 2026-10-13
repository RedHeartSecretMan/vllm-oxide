# vllm-oxide

[English](README.md) | **简体中文**

[![CI](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/RedHeartSecretMan/vllm-oxide/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

vllm-oxide 是一个 Rust 推理引擎，支持**单 GPU、同步、离线的 Qwen3 推理**，基于 [Candle](https://github.com/huggingface/candle) 和项目自身的 CUDA 内核。架构参考 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 与 [vLLM](https://github.com/vllm-project/vllm)，提供持续批处理、前缀缓存、分页 KV 缓存和抢占重算。本项目独立于上游 vLLM 维护。

**v0.2.0 已以源码形式发布。** 本版本验证范围为单张 CUDA GPU 上的 **Qwen3-0.6B BF16**。详情见[发布说明](docs/releases/v0.2.0.md)、[验收摘要](docs/releases/goldens-v0.2.md)和 [GitHub Release](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0)。

[快速开始](docs/getting-started.md) · [文档索引](docs/README.md) · [从 v0.1 升级](docs/migration-v0.2.0.md) · [贡献指南](CONTRIBUTING.md)

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

v0.2.0 提供进程内生成 API 和精简 CLI。服务 API、异步流式输出、多 GPU/TP/NCCL、其他模型族、量化、LoRA、speculative decoding 和 CUDA Graphs 均不在本版本范围内。

当前实现优先保证数值正确性。在固定 RTX 4080/WSL2 发布工作负载中，decode 吞吐中位数为**单请求 29.7601 token/s**、**四请求合计 115.8207 token/s**。这些是可复现的基线观测，不是跨硬件保证，也不代表与 vLLM 性能相同。默认缓存配置会预留较大比例的空闲显存。工作负载、显存测量和限制见[验收摘要](docs/releases/goldens-v0.2.md)。

## 开发与验证

CPU 开发无需 CUDA：

```bash
cargo build --locked --workspace
cargo test --locked --workspace
```

CPU 构建用于开发与测试；**实际推理必须启用 `--features cuda` 并使用 GPU**。自动 GitHub CI 检查编译、格式、Clippy、Rust 测试、Python CPU 验证工具及依赖。模型精度、CUDA API/CLI 功能和性能通过本地 GPU 单独验证。CI 通过本身不代表模型精度已经验证。

开发环境与完整检查见 [CONTRIBUTING.md](CONTRIBUTING.md)。使用问题见 [SUPPORT.md](SUPPORT.md)，缺陷和提案通过 [GitHub Issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues) 提交，安全问题遵循 [SECURITY.md](SECURITY.md)。架构决策与领域词汇见[文档索引](docs/README.md)。

## 许可证

[Apache-2.0](LICENSE)。
