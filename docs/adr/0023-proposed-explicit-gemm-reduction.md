---
status: proposed
---

# Propose an explicit FP32 GEMM reduction reference

This experimental branch proposes full FP32 intermediate reductions for BF16 GEMMs in both the Transformers reference and the Rust candidate. Weights, stored activations, logits materialization, token histories, corpus splits and all four numerical budget values remain unchanged. The reference still uses PyTorch SDPA MATH. This is a proposed precision-scope revision, not approval under ADR-0018 and not release acceptance.

The waiting calibration exposed a shape-dependent reference operation. The first two admitted requests execute MLP GEMMs with two rows, while the reference's complete padded batch has 771 rows. Their first-layer embedding, normalization, Q/K/V, attention context, output projection and post-attention normalization agree bitwise. The first gate/up projections differ in 1744 and 1717 BF16 coordinates. Disabling PyTorch's reduced-precision BF16 reduction makes the large projection agree exactly with the original small projection; independent FP64 accumulation is also much closer to those results. [PyTorch documents this optional intermediate truncation and its default enablement](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms).

The original candidate's short waiting case exceeds the unchanged paired-mean limit: 0.001048677 versus 0.001. Global FP32 candidate reduction repairs that case but fails ten original-reference development peaks; phase/one-query alternatives introduce other regressions. These failed experiments remain failed under their original oracle. They motivate testing a consistent arithmetic reference, rather than declaring an existing result accepted.

The proposed reference explicitly disables both reduced-precision BF16 GEMM reduction and TF32, and receipts retain the observed flag values. The candidate configures its fresh, owned cuBLAS handle before any model work. Distinct kernel identities prevent old captures from being presented as evidence for this scope. The original production reference, historical artifacts, registry, budgets and approval bindings are unchanged outside this experiment.

Before adoption, collect and independently check new development/calibration/replay/control/fault evidence, present the results and exact source/binary identities for scope approval, and only then run the unchanged independent acceptance split and remaining release gates. No tensor threshold or reference distribution is selected case by case.
