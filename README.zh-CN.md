# vllm-oxide v0.1.0

[English](README.md) | **简体中文**

[发布页面](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.1.0) · [文档导航](docs/README.md) · [参与贡献](CONTRIBUTING.md) · [支持渠道](SUPPORT.md)

vllm-oxide 是一个用 Rust 编写的大语言模型推理引擎。它基于 [Candle](https://github.com/huggingface/candle) 和自有 CUDA 内核，架构参考 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 与 [vLLM](https://github.com/vllm-project/vllm)。

v0.1.0 是保留供历史复现使用的实验版本。新项目建议从 [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0) 及其[使用指南](https://github.com/RedHeartSecretMan/vllm-oxide/blob/v0.2.0/docs/getting-started.md)开始。v0.1.0 原始验证的 8 组完整输出中有 4 组存在 token 差异，部分比较被跳过；这些结果仍属于本版本的验证记录。

## 本版包含什么

- 同步的 `LLM::new` / `LLM::generate` Rust API 与轻量离线 CLI。
- 单 GPU Qwen3 模型加载、连续批处理、分页 KV cache、前缀缓存与重算抢占。
- Greedy、top-k、top-p 采样，以及 Rust API 的逐提示词采样配置。
- Transformers/vLLM 夹具生成器与历史 Rust 比较工具。

记录的模型环境为 **RTX 4080/WSL2 上的 Qwen3-0.6B/BF16**。服务、异步/流式 API、多 GPU/TP/NCCL、其他模型族、量化、LoRA、speculative decoding 和 CUDA Graphs 不在本版支持范围内。CPU 构建用于开发与测试，不提供 CPU 推理。

## 开始使用

在仓库根目录运行。推理需要 CUDA Toolkit、兼容的驱动及 SM 8.9+ NVIDIA GPU：

```bash
cargo build --locked --release -p vllm_oxide_cli --features vllm_oxide/cuda
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
target/release/vllm_oxide_cli --model "$QWEN3_MODEL_DIR" \
    --max-tokens 16 "The meaning of life is"
```

CLI 通过 `vllm_oxide/cuda` 启用库依赖的 CUDA 后端，本身没有 `cuda` feature。模型文件、参数、Rust API 与排错方式见[版本使用指南](docs/getting-started.md)。维护后的源码固定使用 Rust 1.94.0，与本次回归检查一致；原历史工具链记录仍保留。

## 文档与开发

| 目的 | 入口 |
|---|---|
| 使用历史源码 | [使用指南](docs/getting-started.md) |
| 理解引擎与 API | [架构](docs/architecture.md)、[领域术语](CONTEXT.md) |
| 参与贡献或运行 CPU 检查 | [贡献指南](CONTRIBUTING.md)、[开发指南](docs/development.md) |
| 查看夹具与比较工具 | [工具导航](tools/README.md)、[golden-gen](tools/golden-gen/README.md)、[Rust 比较工具](crates/vllm-oxide-test/README.md) |
| 查看技术栈实验 | [Research](research/README.md) |
| 查看发布范围与原始附件 | [发布说明](docs/releases/v0.1.0.md) |
| 了解本次源码文档维护 | [维护记录](docs/version-doc-update.md) |

CPU 测试检查开发构建。GPU 数值结果见[验证说明](docs/validation/README.md)，其中记录了原始失败、比较覆盖限制，以及本版尚无经过验证的性能基线。

工作区声明使用 [Apache-2.0](LICENSE) 许可证。漏洞报告请遵循[安全政策](SECURITY.md)。
