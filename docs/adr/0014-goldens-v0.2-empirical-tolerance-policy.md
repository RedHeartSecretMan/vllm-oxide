# Approve the measured goldens-v0.2 tolerance policy

The complete fresh, non-accepting observation required by ADR-0012/0013 proposes `l2_atol=1.0` and `l1_near_tie_max_abs_logit_gap=0.125`. We approve exactly that smallest-covering pair for the fixed BF16 reference-versus-candidate kernel scope, with all observation data retained. This approves a numerical policy for subsequent authoritative comparison, not any fixture, Ticket #45, or release result.

## Immutable observation binding

The Definition input [goldens-v0.2-calibration-observation.json](../releases/goldens-v0.2-calibration-observation.json) contains the complete normalized observation plus the reviewed `approval` annotation. Removing that annotation reproduces the original normalized observation exactly; no measured value, identity, access-set or proposal has been replaced.

- Observation source `O`: `4c5c2cb6df5ecdffa3d7c07926f9ff3d03580d70`, tree `2867ae0b63bcf7312ff107632bb33bd540e0be7e`.
- Original normalized observation SHA-256: `a41eea0f5bdbe4c2e25649f793b545e3a5c7124b16ec5a3a9c64430fdf300723`.
- Definition observation SHA-256: `50330ab87bd541ab6192a505620a3741a60fb503482a8654f89c3e6382e20de9`, Git blob `f61e289b43ded163d00d659c08a530f8fbe6a9b8`.
- Raw-evidence SHA-256: `b993531605a8c42271c78b069b241facd660f1de28cd25d1bce7fda614f30c00`.
- Primary/replay capture-index SHA-256: `707b69e90af1f90d4c5d95f9ee9728b5d85367b8334f4668d21918d76677a995`.
- Observation binary SHA-256: `f17be08bd7903e3b10d11bcaf0af6d0f130c6aafb5765deb8f08bbc1d3ecc1c1`; build-source fingerprint `99c8c4f0750754ededdfe95978ea56deb3e9e7dd`.
- Calibrated manifest SHA-256: `cf93a232645b1784679c67eafd2c8120302ac90b542986204ad4a4257bfe962d`.
- Normalized `manifest.runtime` SHA-256: `eda7287a85f86539bd0419b78edb1963410f2a0041cc980cf240844b42aed9b6`. This is the compact runtime-object digest, not the environment-record file digest `3f534bc963af48116daf4339df992998dcf55b2ac2e046e0c1aaef53632adc96`.

The scope is `transformers-4.57.6/torch-2.10.0/sdpa-math::vs::vllm-oxide/candle-27f20fea993c81ea6d32ce44018f42b68466525e/flash-attn-varlen+paged-windowed`. All model, tokenizer, weight, environment and kernel identities remain those fixed by ADR-0012. The actual tolerance is derived only from the four candidate calibration cases, never from the vLLM baseline.

## Evidence and accepted discrepancy classes

Reference and baseline each generated 28 fixtures and passed independent fresh-process tensor-bitwise replay. Candidate capture opened only `canonical_01`, `canonical_02`, `canonical_03` and `canonical_05a`, with byte-identical raw replay files. The observation retained input thresholds `0/0`, status `non_accepting_calibration_observation` and `accepting=false`; adding policy approval does not change those historical facts.

| Case | Same-prefix rows | First divergence, zero-based | Maximum absolute error | Candidate token gap |
| --- | ---: | ---: | ---: | ---: |
| `canonical_01` | 21 | 20 | 0.375 | 0.125 |
| `canonical_02` | 50 | 49 | 0.3125 | 0.0625 |
| `canonical_03` | 51 | 50 | 0.78125 | 0 |
| `canonical_05a` | 29 | 28 | 0.4375 | 0.125 |

All 151 same-prefix rows and 22,942,336 finite elements participate. Aggregate mean absolute error is `0.044454190205318066`, RMS `0.05942146930456196`, p50 `0.03125`, p95 `0.125`, p99 `0.18359375`, p99.9 `0.265625`, and maximum `0.78125`; non-finite count is zero. L2 `0.5` cannot cover the maximum, so `1.0` is the smallest permitted L2 rung. L1 `0.0625` cannot cover both `0.125` gaps, so `0.125` is the smallest permitted L1 rung. No multiplier, percentile-only cutoff, sample exclusion or threshold above the revised ceilings is used.

The accepted L2 class is an empirical budget for finite same-prefix logit discrepancies under this fixed BF16 kernel comparison after the verified residual, RoPE and SiLU materialization repairs. ADR-0013 binds the same-input attention experiments and the layer-0 intervention evidence. They establish a concrete numerical-path mechanism within that layer, not unique attribution of every later-layer or full-model error. The approved budget is not a universal floating-point error bound; its adequacy remains subject to independent authoritative validation.

The accepted L1 class is a documented first-divergence near-tie measured from the candidate logits. It is not an exact-token match or a skipped comparison. Reference expected-token identity and the candidate expected-versus-selected gap remain explicit. Regression cases without full logits remain exact-token only, and baseline disagreement never overrides a reference failure.

## Required continuation and limits

This checkpoint `P` is Definition-only and does not absorb the unaccepted implementation at `O`. The subsequent frozen measurement commit `M` must truly descend from both `O` and `P`, preserve all executable inputs from `O`, and preserve the selected Definition bytes from `P`. Original runtime, manifests, observations and completion markers keep their original identities. Rehashing, ancestor checks, independent approval recomputation and source continuity checks remain required; an executable rename into an evidence path must also be rejected before holdout access.

Only after this checkpoint is accepted may the authoritative stage apply the policy to a separate manifest copy, bind it to `M`, and open the still-sealed `canonical_04`, `canonical_05b`, `canonical_05c` and `canonical_05d`. It must compare all eight full-logit cases and all 28 reference L1 cases, preserve first-divergence semantics, and require zero missing, unexpected, skipped, failed, duplicate, stale or unmatched cases. Authoritative candidate output must pass its independent replay before becoming evidence.

If holdout or authoritative comparison fails, this policy cannot be raised from that result, and cases cannot be excluded or reclassified to force acceptance. Code/root cause must be repaired or a new disjoint holdout and protocol must be separately approved. Performance work remains gated on authoritative success; the final report, artifact verification, tag, release and exact two-asset publication requirements remain unchanged and require their existing authority boundaries.

**Status**: approved empirical numerical policy; authoritative validation and release acceptance are not granted by this decision.
