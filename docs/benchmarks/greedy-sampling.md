# 批量 greedy 采样微基准

本次优化将整批无 penalty 的 greedy 请求合并为一次 CUDA kernel launch，每个 block
处理一行；原实现逐行 launch。`temperature == 0` 与 `top_k == 1` 可以出现在同一批中。
任何一行有 penalty 或需要随机采样时，整批继续使用原逐行路径。最低 token ID 的
并列最大值规则、每行 RNG 消耗、验证和返回前同步保持不变。

以下结果仅测量 `Sampler::forward`，不是模型吞吐、端到端生成延迟或正式发布性能证据。
本次优化没有重新判定已有的模型数值失败。

## 环境与测量范围

- 测量日期：2026-10-09；RTX 4080，驱动 595.71，CUDA 13.2.51。
- Linux 6.18.33.2-microsoft-standard-WSL2；Rust 1.94.0；release 构建。
- 基线：`0e9b766b9faf9e51ff142b7f8132bcf06c1dfd94`，包含相同的微基准和原逐行 kernel。
- 优化版本：本报告所在提交；kernel SHA-256 记录于原始样本文件。
- 固定 F32 logits，词表 151,936，默认 greedy，无 penalty，空历史；不加载模型。
- 每个 batch 预热 32 次并核对输出，然后测 7 轮，每轮连续调用 128 次。
- 每轮前后同步，CUDA adapter 每次返回前也同步。包含 Rust 参数准备、输出 tensor
  分配/释放、复用工作区和 CUDA 执行；不包含 logits 准备或结果 token 的 D2H。
- 两次独立运行，顺序为基线 → 优化 → 优化 → 基线。表中是合并 14 个轮次平均值后的
  中位数，**不是单次调用的 p50/p99**。
- GPU 时钟未锁定，桌面后台负载未隔离；结果用于确认这条路径的优化方向，不构成 SLA。

## 结果

| Batch | 基线 μs/次 | 优化后 μs/次 | 基线 / 优化后 |
| ---: | ---: | ---: | ---: |
| 1 | 133.214 | 135.889 | 0.98× |
| 8 | 687.596 | 151.303 | 4.54× |
| 32 | 2,593.242 | 134.068 | 19.34× |
| 128 | 20,457.442 | 233.919 | 87.46× |

batch=1 未显示稳定收益。批量场景的改善来自减少 launch 和让多行并行执行，不能将这些
倍率套用到完整推理：模型前向、调度、KV cache、penalty 和随机采样均不在这项结论内。

全部原始轮次和未舍入的计算结果见
[greedy-sampling-2026-10-09.json](greedy-sampling-2026-10-09.json)。

## 复现

在本报告所在的优化提交上创建两个独立 checkout，避免修改当前工作目录：

```bash
git worktree add --detach /tmp/vllm-greedy-baseline 0e9b766b9faf9e51ff142b7f8132bcf06c1dfd94
git worktree add --detach /tmp/vllm-greedy-optimized HEAD

for checkout in /tmp/vllm-greedy-baseline /tmp/vllm-greedy-optimized /tmp/vllm-greedy-optimized /tmp/vllm-greedy-baseline; do
    CARGO_TARGET_DIR=/tmp/vllm-greedy-target cargo test \
        --manifest-path "$checkout/Cargo.toml" --locked --release \
        -p vllm_oxide --features cuda \
        sampler::tests::cuda_device::benchmark_greedy_batches \
        -- --ignored --exact --nocapture --test-threads=1
done
```

微基准被明确标为手动运行，普通测试不执行耗时测量。CUDA 行为回归单独运行：

```bash
cargo test --locked --release -p vllm_oxide --features cuda \
    sampler::tests::cuda_device -- --test-threads=1
```

行为回归覆盖单行和多行、不同词表长度、F32/BF16、行偏移、最低 ID tie、混合 greedy
参数、整批 greedy 中的 penalty 回退，以及既有的混合采样、随机性和工作区复用。
