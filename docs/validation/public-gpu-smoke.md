# 公开 CUDA 接口与 Quick Start 验证

这项检查在干净源码 checkout 上构建生产 CUDA 路径，执行真实模型 warmup、
同一 `LLM` 的 100 次重复调用、文本与 token ID 混合批次，以及 README 中的 CLI 命令，
覆盖 #46 的公开接口 smoke 要求。完整分层精度、性能，以及历史双资产和干净消费者
验证各有 #45/#47 中的独立协议。输出报告 `public-gpu-smoke.json` 标记为
`numerical_acceptance=false`，仅记录功能 smoke 结果，不替代 golden 成功标记。

## 先决条件

- Linux CUDA 主机，SM89+ GPU，可用 host RAM 始终至少 16 GiB。
- Rust 1.94.0（仓库固定工具链）、CUDA 工具链和 `nvidia-smi`。
- 干净 checkout，模型本地目录对应
  `Qwen/Qwen3-0.6B@7e4ae267688d671ddfca3122e4528ee980cf3234`。
  工具重新计算 config、tokenizer 和权重 SHA-256，不凭目录名判定身份。
- 一个不存在的输出目录及足够磁盘；拒绝覆盖旧结果。

先准备依赖，再进入离线 GPU 阶段：

```bash
uv sync --project tools/golden-gen --locked --extra dev
cargo fetch --locked
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
tools/golden-gen/.venv/bin/python -m golden_gen.public_gpu_smoke \
    --repo-root "$PWD" --model-dir "$QWEN3_MODEL_DIR" \
    --output /tmp/vllm-public-gpu-smoke-new-run
```

公开 API 测试检查独立请求 ID、输出顺序、相同请求重复结果、混合批次与单独运行结果，
以及每行不同的生成长度。模型缺失、CUDA 不可用、测试未执行、空 CLI 输出、源码变化
或任一子进程失败都会使门禁失败，不会写成功记录。

每个构建、API 测试和 CLI 阶段都使用已有资源监督器，保留 stdout、stderr、guard 和
清理证据。RAM 低于下限、出现其他 CUDA 计算进程或监督失败时，阶段会停止。构建阶段各自限时
30 分钟，API 与 CLI 阶段各限时 5 分钟。失败时保留已有日志，用新目录重跑。

## GitHub Actions

`.github/workflows/cuda-smoke.yml` 只由 `workflow_dispatch` 手动触发，运行在带有
`self-hosted`, `linux`, `x64`, `cuda` 标签的当前受支持 GitHub runner 上。
`model_directory` 输入是该机器上的绝对目录。工作流调用与本地相同的门禁模块，并
上传原始日志和资源记录；没有配置 CUDA runner 时，不应把排队或未运行视为通过。

普通 CPU CI 使用 Rust 1.94.0/stable 矩阵。Python 检查使用两个锁定环境：CPU 开发环境
运行 pytest/Ruff/mypy，完整 worker 环境仅供 CPU Torch/vLLM adapter 子进程测试使用。
CPU transport reader 由 Rust job 构建并交给 Python job。pytest 的跳过或空执行结果会
被已有发布输出检查器拒绝，因此 CI 不会把缺少 worker/transport 工具误报为通过。
