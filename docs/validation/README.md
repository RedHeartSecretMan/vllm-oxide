# Historical validation — v0.1.0

[Documentation](../README.md) · [Release notes](../releases/v0.1.0.en.md) · [Tooling](../../tools/README.md)

The original v0.1.0 record was **not fully passing**. A documentation refresh does not rerun inference, establish current GPU performance, or replace the original outcomes.

| Recorded item | Historical result |
|---|---|
| Fixture coverage | 28 expanded prompt groups × two oracles = 56 fixtures |
| Full logit coverage | Eight canonical groups |
| Regression coverage | 20 groups with top-5 data |
| Full-output token agreement | Four of eight evaluated groups passed; four differed |
| Absolute logit tolerance | `66.46875` |
| Regression exclusions | Six regression cases contain skipped positions |
| Release performance baseline | No validated baseline claimed |

The five canonical prompt specifications expand to eight groups because `canonical_05` contains four separate sub-prompts. Both oracle adapters process that batch as four requests; Transformers uses a padded batch, not one concatenated sequence.

## What the historical tools actually check

- Both oracles use BF16 model weights/computation. Stored full/top-5 logits are FP32 transport values. Transformers uses SDPA; vLLM uses eager generation and raw-logit capture.
- Schema 1 records model revision, oracle versions, fixture SHA-256 values and a single calibrated `atol`. Calibration uses twice the maximum absolute difference across paired canonical logits and shared lengths, without rtol or a teacher-forced same-prefix guarantee.
- The Rust runner loads only `canonical.jsonl`, so regression IDs and expanded batch IDs may be absent and skipped. It iterates canonical fixtures from both oracle identities rather than enforcing a single reference-only coverage set.
- L1 compares token IDs and lengths, with separately reported near-tie and regression exclusions. Its near-tie helper only recognizes the expected token when it is the generated row's top-1, despite the historical comment describing top-2; this refresh documents the implementation without changing it.
- L2 compares rows whose current generated token equals the fixture token. It does not stop after the first earlier divergence, so equal tokens later do not prove an identical prefix. Rows with different tokens are omitted; zero compared elements can still yield a passing result.
- The CLI can log generation errors and continue. Report aggregation uses `all()` over collected results, so an empty/incomplete report can pass. A zero exit code therefore does not prove full coverage.
- `--debug` adds an L3 placeholder message. Its `passed=true` field is a stub, not measured activation accuracy, and L3 does not enter the overall verdict.

These limitations explain why this historical report must not be presented as the stricter v0.2.0 acceptance contract. Read the [comparison README](../../crates/vllm-oxide-test/README.md) for commands and reporting details; keep expected coverage, actual coverage, mismatches, exclusions and errors visible when investigating old runs.

## Memory and performance

`LLM::new` uses CUDA free-memory queries and warmup to allocate its KV pool. `gpu_memory_utilization` controls allocation from available memory; it is not a sampled peak-memory bound. The generation log combines prompt and completion tokens over elapsed call time, not separate prefill/decode throughput, TTFT or TBT.

This archive contains no accepted benchmark protocol, direct NVML sampling record, synchronized stage timing or release performance threshold. Values from v0.2.0 hardware runs must not be copied into v0.1.0 as historical measurements.

Original release assets and checksums remain in the [release notes](../releases/v0.1.0.en.md). Keep raw packages local and preserve their manifest/source identity. For accepted current numerical and performance behavior, use the [v0.2.0 validation docs](https://github.com/RedHeartSecretMan/vllm-oxide/blob/v0.2.0/docs/validation/README.md).
