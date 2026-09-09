# Freeze the calibrated layered accuracy budgets

The user approved the explicit operator and model budget table after reviewing its calibration consequences and the meaning of all four model conditions. We freeze `layered-budget-v1` in the [Tolerance policy](../validation/layered-accuracy-budgets.json), under the unchanged `layered-accuracy-v1` protocol and metric algorithm. These are engineering error preferences supported by the bound observations and fault experiments, not universal tolerances, mathematical error bounds or downstream quality guarantees.

**Status:** accepted numerical-budget decision. This decision supersedes only the statements that these exact budgets are still pending in ADR-0015, ADR-0017 and the [layered contract](../validation/layered-accuracy-contract.md)'s opening status, section 6 and section 15. Their initial approval history remains intact. The contract's pending/unknown-policy fail-closed rules remain applicable whenever a required value, source, approval binding or artifact is missing; its other requirements are unchanged.

## Values and applicable scope

Every registered numerical case uses all four existing inclusive (`<=`) hard conditions. A mean over other cases cannot cancel a failing case, and baseline error cannot raise an absolute ceiling.

| Policy key | Approved upper bound | Meaning |
| --- | ---: | --- |
| `a_mean` | 0.002 nats | Mean reference-weighted log-score loss over that case's fixed histories |
| `a_peak` | 0.005 nats | Maximum single-step reference-weighted log-score loss |
| `delta_mean` | 0.001 nats | Mean paired extra reference cross-entropy relative to vLLM, per case |
| `g_limit` | 0.22314355131420976 | Binary64 representation of `ln(1.25)`; the chosen token has at least 80% of the best token's probability in the reference distribution |

The 2/5/1 millinat allowances and 80% reference-choice ratio express accepted engineering trade-offs. A positive paired allowance is not strict non-inferiority to vLLM. Exponentiating a KL budget describes a geometric reference-weighted log-score ratio, not measured hard-label corpus perplexity or task accuracy. No TV gate, candidate-only gap, top-2 identity condition or pointwise paired-KL gate is added.

| Operator profile | Approved `max_abs_error` |
| --- | ---: |
| `rms-residual` | 0.0 |
| `rope-position` | 0.0078125 |
| `silu-opmath` | 0.0 |
| `attention-prefill` | 0.0078125 |
| `attention-paged` | 0.0078125 |
| `kv-slots` | 0.0 |
| `sampler-ids` | 0.0 |

Operator limits apply only to each frozen profile's `input_rule`, `input_definition`, dtype, input/output shapes and pinned runtime. RMS and SiLU are conservative fixed materialization/rounding probes, not zero-error promises for arbitrary inputs. RoPE and attention use the absolute engineering scale `1/128`; it is neither one ULP at every coordinate nor a BF16 operator error theorem. KV and sampler identities, positions, masks, ownership, raw selection and all required replay rules remain exact independently of floating limits.

Model scope remains the registry's Qwen3-0.6B weights/tokenizer and single-GPU BF16 configuration: Transformers SDPA MATH reference, vLLM FlashAttention baseline, and the Rust candidate. The approved calibration's complete runtime profile, model/config/tokenizer/weight identities, binary and kernel receipts remain required by the existing provenance checks. A different model, device/runtime/kernel scope or changed numerical implementation does not automatically inherit this approval. All current corpus inputs, counts and split independence rules remain unchanged, including the 51-total-block baseline and three-block Rust pressure obligations.

## Evidence and alternatives

The calibration was collected from commit `6b7ae7d9ec5e4736a504aa1a58aa36005986bfd3`, tree `f0f931e75eefa722bad41fb9491b6cb32ce38c4d`. Budget selection occurred after the calibration results were known; it was not blind and retains calibration-selection bias. The approval considers 19 cases / 75 correlated prediction steps and seven controlled operator profiles, not 75 independent samples or independent semantic-quality labels.

Offline evaluation under this table rejects none of those normal observations and rejects all 16 registered operator fault models. Seven global fault definitions cover eight actual transformations: negative token identity, wrong raw argmax, incorrect vocabulary/weights, wrong prefix history, missing rows, partial-prefix KV misuse, and the numerical distribution swap. Identity/structure/selection faults remain independently rejected; the distribution swap violates all four numerical/choice conditions. Normal/fault separation supports these finite probes only, not detection of every small or unregistered defect.

For the fixed RMS/RoPE/SiLU profiles, observed normal maximum errors were zero and the smallest registered fault errors were 0.00390625 / 0.865234375 / 0.00390625. Both attention profiles had normal maximum error 0.00390625, versus minimum registered fault errors 0.1796875 (prefill) and 0.2421875 (paged). KV/sampler faults violate exact structure. The zero limits intentionally reject any nonzero rounding discrepancy on the designated fixed probes; a later discrepancy requires investigation, not automatic tolerance widening.

The alternative of strict paired non-inferiority (`delta_mean=0`) rejects eight calibration cases. Separately tightening `a_mean` to 0.001, `a_peak` to 0.002, or `g_limit` to `ln(1.2)` rejects 13, six, or one case respectively. These sensitivity results expose the trade-offs; they are not proof that the rejected observations are semantically acceptable or that a particular threshold is uniquely correct. Zero observed normal rejection is not a population false-rejection-rate guarantee. The proposal's predicate boundary calculations are not additional production comparator tests.

The following SHA-256 identities bind the original evidence and independently reviewed proposal. Artifact names identify their evidence roles, not extra release assets or paths to execute. The existing release evidence closure must preserve the original bytes and sources; hashes do not replace the underlying data.

| Evidence role / original artifact | SHA-256 |
| --- | --- |
| Registry / `layered-accuracy-cases.json` | `88f89084fe7b0c14a877c4c72f275314ad2a8e593d6fbc3ac63a194b91fb98f4` |
| Calibration execution manifest / `manifest.json` | `73b1700490d29ab04072529f740318ac7034608357b99a3fd2721d423587ab91` |
| Calibration observation / `observe.json` | `d423774300f50f4602dbbfeb23a65a4ef8d8bbd645ca27764e87072af0101faa` |
| Original observation marker / `observation.complete.json` | `71cedc463ae26ab7e3013a3217748183727d291ed6b515f7cdfcdfdec1a45d30` |
| Original fault models / `fault-evidence.json` | `1bccf67c82ff8da33df3ccc3110942366ea2278c46d651d49c8eadb81e83cec8` |
| Original complete evidence inventory / `summary.json` | `9ebea57589a60215c0dae4b0ba49c5d6f679edb4dfa14666e4abff46c4492d41` |
| Selected budget proposal / `proposal.json` | `8eee41ee8895f706a4cdfa259fe9e1768bba3fbf1e7f5246de5df5d89ab9ee0e` |
| Budget rationale and limitations / `proposal-report.md` | `3269430f7b563db5d20ee1b83e5cc4c93131d2784c0272d37acf5bf3f4ae2314` |
| Offline numerical/fault evaluation / `evaluation.json` | `2fccc2ce0223c10d2f46d744cbbc1a77cb6d7d2a1839a0e127d08528c07e9517` |
| Independent mathematical and evidence review / `independent-reviews.md` | `3ec6f6c155fee88a7d7e482f44cccd5f3ec684371030ecd7a3f6c14f9f0d8143` |

The policy's `calibration_evidence_sha256` names the original observation, not the proposal evaluation or an altered report. `fault_evidence_sha256` names the original source-bound fault payloads. The independent proposal reviews are not substitutes for a candidate-bound code review or the fresh Definition Checkpoint review.

## Consequences and unchanged gates

The original observation and proposal retain their original `accepting=false`, pending labels and sources. They must not be rewritten into PASS, assigned the approval commit as their measurement source, or used as independent acceptance data. The existing ancestry, unchanged-execution-source, runtime, closure, replay and fault-verification checks still govern reuse of approved calibration; ancestry alone does not prove equivalence.

Acceptance must use the frozen policy and untouched independent split under the existing stage authorization and resource protections. A subsequent FAIL or INVALID is preserved and investigated; it does not authorize changing thresholds after seeing holdout results. No new GPU run, native FlashAttention compilation, benchmark or publication is authorized by budget approval alone. Budget freezing accepts no Ticket, and does not replace the remaining L0/L1/L2, performance, final review, bundle/clean-consumer or separately authorized release gates.
