# 修复源码后的新校准观察

旧监督政策用于按原始源码和 binary 身份保留、补齐既有采集记录。它不应把修复后的
源码重标为旧源码。`golden_gen.layered_cli collect/collect-aux --fresh-observation`
提供独立的新观察入口；已有监督采集的默认行为保持不变。

这个入口只允许冻结 observation inventory 中的 calibration 数值组、calibration
公开行为和算子检查。development、acceptance 或未登记 owner 被拒绝；它不能与
`--measurement-repo` 混用，也不能传给 authoritative、assembly 或 worker 命令。

## 采集

先准备干净 checkout、锁定的完整 Python runtime、与 checkout 对应的 CUDA candidate
binary，以及固定 revision 的本地模型。环境和资源要求继续遵守分层精度协议：
16 GiB 可用 host RAM、独占 GPU owner、完整 watchdog、日志及子进程清理。

以下仅示范一个 owner，不能构成完整校准证据。`PYTHON` 指向上述 Python runtime，
`QWEN3_MODEL_DIR` 指向已经验证的固定模型目录，`RUN_DIR` 必须是尚未使用的
`/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts/` 子目录。

```bash
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/tools/golden-gen/src" PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m golden_gen.layered_cli collect \
  --repo-root "$PWD" --run-dir "$RUN_DIR" --model-dir "$QWEN3_MODEL_DIR" \
  --group calibration-length-1 --engine reference --variant primary \
  --fresh-observation
```

按 `frozen_owner_inventory(registry, authoritative=False)` 串行采集所有 owner：
数值 owner 使用 `collect`，辅助 owner 使用 `collect-aux`；candidate owner 必须另外
传入 `--candidate-binary`。每个 owner 使用新目录，失败记录不能覆盖。所有 worker
从声明的 checkout 启动，显式绑定 cwd/PYTHONPATH，并检查实际 worker 与源码身份。

完整采集后使用既有 `assemble-observation`、`observe` 和 `faults` 路径检查新证据。
缺失 owner、重放失败、来源不一致或监督失败仍会阻止完整观察记录。

## 不授予新的验收权限

新结果保留新的 commit/tree、capture、receipt 与证据哈希，且 `accepting=false`。
不能将它冒充 ADR-0018 绑定的原始校准文件，也不能作为旧的 retained-owner ledger。
原始批准证据缺失时，单凭重新采集或数值相同不能恢复其批准身份。

现有 authoritative assembly、校准批准绑定、监督 schema 2、原始 ledger 和四项数值
预算要求没有改变。若后续需要采用新校准来源，必须先完成 ADR-0018 要求的证据审阅
与批准；本入口本身不会批准预算、接受 #45 或授权发布。
