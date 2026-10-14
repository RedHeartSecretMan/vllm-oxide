# v0.2.0 验收摘要

v0.2.0 的产品验证结果及其原始来源如下。完整 captures、原始 logits、运行日志和性能遥测保存在本地，schema5 manifest/archive 证据包有独立的验证流程。旧版组合证据包尚未完成完整消费者验收。

## 来源与实现一致性

| 用途 | 原始提交 |
|---|---|
| 数值测量 | `33e0e27268285368c8c28b47df5cb1df9b2df174` |
| 独立数值评估 | `ba5eb799acbf156fcab588d71aad09b6f1b15615` |
| 性能 producer | `e785a217af6031a1890c0e52de6f61d86f731713` |
| 发布实现与公开 CUDA smoke | `3b1cdceb3f105713d0c71e23cab63871db03f4c5` |

以上来源分工已获用户批准。表中原数值测量、性能 producer 与原发布实现的 `crates/vllm-oxide`、`crates/vllm-oxide-cli` Git 对象完全一致。该一致性描述原始发布；后续[源码维护](../version-doc-update.md)另有修复和回归记录，不将旧精度、smoke 或性能结果改标为维护后提交的新测量。

## 精度与行为

原独立验收结果为 `PASS`，747/747 个采集项和 231/231 个执行组完整，包含 104 项数值比较、44 项行为检查和 7 项算子检查。

| 每个 case 的指标 | 原批准上限 | 实测最大值 |
|---|---:|---:|
| 平均 KL | 0.002 | 0.0019322289489712226 |
| 峰值 KL | 0.005 | 0.004058524233118184 |
| 相对 vLLM 的配对平均额外 KL | 0.001 | 0.0002877092501205178 |
| 参考选择损失 | `ln(1.25)` ≈ 0.22314355131420976 | 0.125 |

报告使用同一冻结历史的完整 logits 比较；自由生成差异单独记录。原始独立报告 `authoritative.json` 的 SHA-256 为 `451d2e200f41356761a3b1bda5b08f7daed6037f1c82ff68b7dee98339da8d17`。

## 公开 CUDA 功能

发布实现完成生产构建、真实 warmup、100 次重复公开 API 调用、3 个混合请求及 Quick Start CLI，实际进程退出码为 0。107 个请求 ID 唯一，CLI 输出 `Paris.`。报告 `public-gpu-smoke.json` 的 SHA-256 为 `23a44126c3ac20e560bed4bfb8fea2bf092d7bde999901535fa5f4322f0faa3a`。

## 实际性能基线

原性能 producer 的 `canonical_04`、`canonical_05` 各丢弃一次 warmup，再测量三次。下表为固定 RTX 4080/WSL2 环境中的中位数；batch 吞吐为四请求总吞吐。

六次原始 telemetry 绑定相同配置：BF16、`max_model_len=4096`、每步 token 预算 16384、最大并发 512、`gpu_memory_utilization=0.50`。采样为 greedy（temperature 0）、每请求 64 个 completion token、`ignore_eos=true`，三个 penalty 均为 0，top-k/top-p 关闭。API/CLI 默认显存比例为 0.90，本基准的 0.50 及其显存结果不是默认配置。NVML 记录的是整张设备已用显存，可能包含其他 GPU 分配。

| 工作负载 | Prefill token/s | Decode token/s | TTFT ms | 设备显存用量峰值 MiB |
|---|---:|---:|---:|---:|
| 单请求 `canonical_04` | 13910.2653 | 29.7601 | 40.9622 | 10756 |
| 四请求 `canonical_05` | 1715.8632 | 115.8207 | 36.7626 | 10692 |

全部六次测量的 token/config/输出身份、同步时间、统计和显存摘要均经验证。显存采样保留 5ms 目标及严格 50ms 时序上限；共 1831 个原始样本，最大完成间隔 13.359862ms，最长 API 调用 8.274851ms，全部 primary failure 为空。没有声明 GPU 独占或跨硬件性能保证。

性能 producer 的构建输入指纹为 `7c9a6957f18919e2aa0ca3b74cd1e8a331eb505f`，保存的二进制 SHA-256 为 `44107b3a5ad5ff8c656525aed66e942174ea104078d9bb75243649f05ff93c75`。新原始 `benchmark.json` 的 SHA-256 为 `807d63db8a12e0d144d9f49c2d757f82fc1d0330652e14ed18c296aa766bbb33`；旧失败和 diagnostic 工件保持原样。

## CPU CI 与环境

发布集成 [PR #71](https://github.com/RedHeartSecretMan/vllm-oxide/pull/71) 的实现提交通过 [CI 38362611490](https://github.com/RedHeartSecretMan/vllm-oxide/actions/runs/38362611490)：Rust 1.94.0、Rust stable、Python quality and complete CPU adapters、Dependency audit 四项均为 SUCCESS。发布文档变更继续使用相同自动 CI。

实际 GPU 验证使用 Qwen3-0.6B 固定 revision `7e4ae267688d671ddfca3122e4528ee980cf3234`、BF16、RTX 4080/SM 8.9、WSL2、驱动 595.71、CUDA Toolkit 13.2.51、Rust 1.94、PyTorch 2.10.0（CUDA 12.8）、Transformers 4.57.6 和 vLLM 0.18.1。普通 GitHub CI 是 CPU-only；本地 GPU 验收与自动 CI 分别记录。
