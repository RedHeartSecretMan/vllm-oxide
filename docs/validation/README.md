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

## Original protocol and maintained tool behavior

- Both oracles use BF16 model execution and save FP32 logits. Schema 1 records
  model revision, oracle versions, fixture hashes and one calibrated `atol`.
  Calibration uses twice the paired canonical maximum on shared output lengths,
  without a teacher-forced same-prefix guarantee or a modern KL budget.
- The original runner loaded only canonical prompt IDs, skipped missing
  regression/batch members and could continue after generation errors. Empty
  `all()` reports could pass. The maintained runner loads both corpora, expands
  batch members, preflights every declared fixture, propagates failures and
  rejects empty or incomplete declared comparisons.
- Each fixture is captured as a separate diagnostic request; expanding its ID
  does not reconstruct the oracle's original batched execution geometry.
- L1 retains legacy near-tie and regression exclusions, counted separately from
  exact matches. The maintained helper recognizes an expected runner-up only
  when the actual token is candidate argmax and the top-two gap is strictly
  below the original epsilon (default `2 × atol`). The six-case regression skip
  map is unchanged. This is not the modern reference-choice-loss criterion, and
  oracle disagreement does not itself prove nondeterminism.
- The original L2 filtered rows by current-token equality, which could compare
  different causal histories. The maintained comparison includes the first
  divergent row and excludes every subsequent row. Empty/malformed geometry,
  invalid tolerance and non-finite compared values cannot establish a PASS.
- L3 activation comparison remains unimplemented; `--debug` is now rejected
  explicitly. Mutually exclusive layer flags are rejected, and a declared
  partial run is evidence only for its selected layer.

These repairs improve the comparison tool; they do not rerun the original
fixtures or convert historical failures into a new accepted result. Schema 1
still lacks the current protocol's independent expected registry, full-source
provenance and complete L0/L1/L2 evidence. See the
[comparison guide](../../crates/vllm-oxide-test/README.md) and
[maintenance record](../version-doc-update.md).

## Memory and performance

`LLM::new` uses CUDA free-memory queries and warmup to allocate its KV pool. `gpu_memory_utilization` controls allocation from available memory; it is not a sampled peak-memory bound. The generation log combines prompt and completion tokens over elapsed call time, not separate prefill/decode throughput, TTFT or TBT.

This archive contains no accepted benchmark protocol, direct NVML sampling record, synchronized stage timing or release performance threshold. Values from v0.2.0 hardware runs must not be copied into v0.1.0 as historical measurements.

Original release assets and checksums remain in the [release notes](../releases/v0.1.0.en.md). Keep raw packages local and preserve their manifest/source identity. For accepted current numerical and performance behavior, use the [v0.2.0 validation docs](https://github.com/RedHeartSecretMan/vllm-oxide/blob/v0.2.0/docs/validation/README.md).
