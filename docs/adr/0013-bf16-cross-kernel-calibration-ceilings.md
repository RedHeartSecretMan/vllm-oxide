# Bound recalibration for the fixed BF16 kernel comparison

Historical calibration decision: [ADR-0015](0015-layered-accuracy-validation.md) defines the subsequent layered protocol. These ceilings and observations remain historical evidence, not budgets or approval for that new protocol.

After three BF16 materialization repairs and guarded numerical investigations, the four readable calibration cases still exceeded ADR-0012's initial ceilings. The user authorized a formal tolerance-protocol revision while retaining independent holdout validation. We retain the fixed BF16 reference and candidate kernels and revise only the admissible calibration envelope: L2 at most `1.0`, and L1 near-tie gap at most `0.125`; neither number is an approved acceptance tolerance.

## Evidence and its limits

The four-case diagnostic at source `0e09fb5fb9096f700d675f66b8c0ff81e56c4f81` included every finite same-prefix element through each first divergence. Its primary and fresh-process replay files were bit-identical. The 151 compared rows contained 22,942,336 elements with no non-finite values; mean absolute error was `0.0444541902`, p95 `0.125`, p99 `0.18359375`, and maximum `0.78125`.

| Calibration case | Compared rows | Maximum absolute error | Candidate gap at first divergence |
| --- | ---: | ---: | ---: |
| `canonical_01` | 21 | 0.375 | 0.125 |
| `canonical_02` | 50 | 0.3125 | 0.0625 |
| `canonical_03` | 51 | 0.78125 | 0 |
| `canonical_05a` | 29 | 0.4375 | 0.125 |

Those measurements already include the residual-addition, RoPE intermediate-rounding, and SiLU FP32-opmath repairs. The maximum remained a non-top logit in `canonical_03` at the first decode row; it was not excluded or relabeled to reduce the maximum. The diagnostic report SHA-256 is `8b6c7418c66024d5ba64adb7a8400d7882255c72d2e99647c67004bf04f63a58`; both capture-index files have SHA-256 `87e4fdbbaf563edf37e06ee49f984f0f8ba77d79c89df5bbae859485fecb5aea`.

Further diagnostics at source `29748bde68016c17cf805844db1b878162dc9514` reproduced each engine's original logits before interpreting intermediate values. In the first layer of the 199-token `canonical_03` prefill, raw Q/K/V were identical. A single Q-normalization element differed at token 129, but it could not explain the first-layer maximum at the earlier token 44 under causal attention. Actual visible inputs, mask, lengths and head mapping at token 44 were identical; the recorded scale values differed only in their argument precision.

A Torch-only replay first reproduced all 407,552 stored Torch attention-context elements bit-for-bit. Changing only the scale to the exact Rust value changed no context element. Running MATH with the captured Rust logical inputs and Rust scale still differed from the captured FA context at 66,319 elements, with maximum absolute error `0.0078125`; token 44 differed at 506 of 2,048 elements, with maximum `0.00390625`. Its result SHA-256 is `b0b918842fd03bf8cb1d7ea88875baa7641279d7b4985b50604e7d95b9602225`; diagnostic script SHA-256 is `0df75e2626b00e6552f62daa99a48ac4a88d4cdae1166e16bef3524eaa5525fe`.

A second replay used only the five required layer-0 weight tensors from the fixed model file and verified their GPU readback bits. The reference remainder first reproduced all 203,776 stored Torch layer-output elements. Replacing only its attention context with the Rust context then reproduced every stored Rust layer-output element, including the `0.03125` maximum difference at token 44. Its result SHA-256 is `39cfd41782d53f56db0cfe8d3798f4891e5bfbfffc0aa688c8e846b48ff4db87`; script SHA-256 is `c85a395d081415cc58ee865d489eba37fbe60536add0218a072a52dad79db63f`.

These experiments establish a concrete cross-path numerical difference and its causal effect within this prefill layer. They do **not** prove that every later-layer or full-model error is explained, that either backend is defective, or that the original ceilings are mathematically impossible to satisfy. Original GPU strides were not captured; replay equivalence establishes the observed output under the declared reconstructed layout, not original physical-layout identity. The complete-context intervention also includes the small token-129 upstream perturbation; token 44 is the same-input backend witness.

## Revised calibration boundary

The new upper endpoints are the smallest powers of two covering the complete repaired four-case diagnostic maxima: `1.0` covers `0.78125`, and `0.125` covers `0.125`. This deliberately relaxes the initial `0.25` and `0.0625` ceilings for the already-fixed kernel comparison. It adds no safety multiplier, does not use the vLLM baseline to select either endpoint, and does not establish a universal floating-point error bound. It is a one-time, explicitly authorized calibration-envelope revision made before opening the sealed candidate holdout.

The L2 ladder is exactly `0`, then `2^-12` through `2^0`; the L1 ladder is exactly `0`, then `2^-12` through `2^-3`. Selection remains the smallest covering member, using all valid same-prefix elements and every actual divergence from the four calibration cases. A fresh observation exceeding either revised ceiling blocks approval; no further increase is authorized by this decision.

At the ceiling-revision checkpoint, actual tolerances remain pending at `0/0` until the separate empirical approval. These historical diagnostics justify reconsidering the envelope but are not an authoritative calibration observation, approved policy, successful stage marker, or release payload. Ticket #45 must update the implementation and tests to this Definition, freeze a newly reviewed observation source, and execute the complete official environment/generation/replay/calibration/observation sequence in a fresh staging directory. Earlier manifests, partial stages and diagnostic captures cannot substitute for that evidence.

The separate empirical-approval Definition Checkpoint remains mandatory. It must bind the fresh normalized observation and raw-evidence hashes, state the exact mechanically proposed tolerance pair, explain the accepted numerical error classes and remaining limitations, and pass independent review. This ceiling revision is not that approval checkpoint. No full-model root-cause conclusion may be claimed from the layer-0 experiments alone.

The observation source commit `O` remains immutable, unaccepted Ticket evidence. The empirical Definition checkpoint `P` is built from the Accepted Integration Tip and changes only Definition inputs; it binds `O` by its original commit/tree and evidence identities rather than importing unaccepted implementation code. `P` therefore need not descend from `O`. The subsequent measurement commit `M` must truly descend from both `O` and `P`, preserving `O`'s executable inputs and the approved Definition from `P`. Original stage markers, manifests, runtime records and observation hashes remain unchanged; only already-allowed Definition/evidence paths may differ from `O`, and all source-fingerprint, ancestor, blob, index and recursive stage-output checks still apply. Changing an observation's commit label or using a merge to conceal changed executable bytes is invalid. If implementation inputs differ, a new measured observation and approval are required.

Only after empirical approval may the candidate open `canonical_04`, `canonical_05b`, `canonical_05c`, and `canonical_05d` and perform authoritative comparison. All eight full-logit cases and all 28 reference L1 cases remain required. Regression cases without full logits remain exact-token only. A holdout or authoritative failure cannot be repaired by increasing a threshold, excluding or reclassifying a case, or consulting the baseline as a correctness override. Code/root cause must be repaired, or a new disjoint holdout and protocol must be separately approved.

## Scope preserved

ADR-0012's model/tokenizer/weights identities, versions, BF16 dtype, MATH reference, fixed FA candidate and baseline paths, deterministic fresh-process replays, complete fixture accounting, same-prefix comparison, zero missing/skipped/failed release cases, performance prerequisites, evidence/source binding, resource guards, and separately authorized publication remain unchanged. The public API, Tickets #29 through #47, native dependency edges, and non-numerical acceptance obligations do not change.

The trade-off is explicit: this revision permits larger empirically justified absolute differences within the existing kernel scope. It does not replace the production attention path to seek bitwise parity, relax a threshold after viewing holdout results, or accept the failed historical calibration. Independent holdout and the fail-closed release workflow remain the tests of the revised policy's adequacy.

**Supersedes**: only ADR-0012's initial numerical ladder upper endpoints and calibration ceilings.

**Status**: accepted calibration-ceiling revision. This decision alone approves neither an actual tolerance nor release acceptance; [ADR-0014](0014-goldens-v0.2-empirical-tolerance-policy.md) records the later empirical policy decision.
