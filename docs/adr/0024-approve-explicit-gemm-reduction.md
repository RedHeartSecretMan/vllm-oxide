---
status: accepted
---

# Approve explicit FP32 GEMM reduction and fresh release evidence

The user explicitly approved the reviewed [ADR-0023 proposal](0023-proposed-explicit-gemm-reduction.md) on 2026-10-09. The Transformers reference and Rust candidate use FP32 intermediate reductions for BF16 GEMMs; the reference explicitly disables reduced-precision BF16 reduction and TF32. BF16 model storage and materialization, the pinned Qwen3 model/tokenizer/runtime, the registry and every ADR-0018 model/operator budget value remain unchanged. This approves the new precision scope and calibration bindings, and permits independent acceptance to proceed; it does not assert acceptance or authorize publication by itself.

The waiting-case investigation isolated shape-dependent truncation in the large reference MLP GEMM. Exact binary-rational probes, same-input operator interventions and three failed candidate-only alternatives support making the precision contract explicit. This evidence concerns the tested inputs and arithmetic scope, not a universal error bound or task-quality guarantee. The original waiting failure remains a failure under its original reference.

## Bound evidence

Measurement source is `880e22c8823d30c0a52775b26a193f175824e80c`, tree `fb39df4d40239700bbf76ccfa76d341f924382b5`. Reference kernel identity is `transformers-4.57.6/torch-2.10.0/fp32-gemm-reduction/sdpa-math`; candidate identity is `vllm-oxide/candle-27f20fea993c81ea6d32ce44018f42b68466525e/fp32-gemm-reduction+causal-math-v2`.

The complete calibration contains 132 fresh owners, 19 numerical cases / 75 correlated prediction steps, 39 engine groups, 11 behavior checks and seven operator profiles. Every numerical case satisfies all four unchanged bounds; the across-case maxima are mean KL 0.0011968224723076534, peak KL 0.0021945758736470573, paired mean 0.0007057754292802969, and choice loss 0.0625. All 16 operator faults and the eight transformations described by seven global fault definitions are detected. Both independent review axes verified the identities and evidence; the Spec review reproduced the original observation through the production evaluator.

| Original artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `c232a03f0d2ff5c8b743ccd112956dab56d0118b26eed0f634a4d31074e0728e` |
| `observe.json` | `40863458ef4cb693ce7a39fe476bdab3b68af8708ca967079ad00f4abb509ab8` |
| `layered-markers/observation.complete.json` | `cd76879f1ac0e226b0ea9852ee422982fdb2bcba4caf12571789b58d206a8e98` |
| `fault-evidence.json` | `f514943614e30b63406ca211a9bafdd680bd309f533d85cdf23e4d2a6481f9a2` |
| `proposed-scope-evaluation.json` | `08c8b52a8c6999cc64fb5245922ed0a6945c258efc350c042b488667ce29be03` |
| `proposed-scope-evaluator.py` | `698f86723690b6f6f2185ffbe14154c0fa1281bdcbd199a9b395d0a96d0ca75f` |
| `independent-scope-reviews.md` | `6ea524b0bd295ffc09c67495c095af810f728683da862233a222b3359854c6f7` |
| fixed-prefix binary | `996a1b6b6709bd7bde2f11c7c458558b6833f70a267a8f5651f0dc4e881a9656` |
| benchmark binary | `3eb06a8ab3a3d3b5d43e33102c08331309a26a0b76f9bdfe9213e544238262f3` |

The original observation remains `INVALID / budgets_pending / accepting=false`, and the pre-approval derived evaluation remains `approved_scope=false`. Approval binds those original bytes without rewriting them. Calibration and development results were observed before this decision and retain their selection bias. They cannot serve as independent acceptance.

## Fresh supervision and remaining gates

Select ADR-0021's `bounded-telemetry-fresh-v1` policy and execution manifest schema 3. The old retained ledger is unavailable, so retain zero owners and collect the complete unchanged 487-owner authoritative inventory anew, including all registered splits and exact replay/control obligations. Every owner requires original worker metadata and a validated schema-2 lifecycle/resource guard. The 16 GiB RAM floor, serial GPU ownership, bounded deadlines and cleanup requirements remain unchanged.

The policy binds the actual immutable measurement checkout, capture binary and benchmark binary. The evaluator/supervisor is a separately frozen descendant; every existing protected measurement path, plus the measured CLI/benchmark sources and prompts, must remain Git-object-identical. Separate complete CPU gates attest the two roles. This decision activates the reviewed fresh-evidence transport and performance identity routing; it does not broaden their source-equivalence checks or reuse old owners.

Independent authoritative accuracy/behavior acceptance, performance evidence, the complete asset closure, clean-consumer validation, final review and separately authorized publication remain required. A later FAIL or INVALID is preserved and investigated; no threshold, input, precision or skip rule may be changed merely to obtain PASS.
