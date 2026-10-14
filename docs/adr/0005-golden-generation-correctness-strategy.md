# Golden generation correctness strategy

Historical v0.1.0 design record. The intended comparison strategy below is preserved; implementation and coverage gaps are explained in [historical validation](../validation/README.md). The documentation refresh does not change this decision or its measured evidence.

The original oracle triangle (transformers / nano-vllm / vLLM V1) flagged all
8 canonical prompts as SUSPECT. Oracle-to-oracle L2 divergence exceeded 0.1
in every case, and calibrating from that disagreement produced a tolerance of
4623.99. The resulting fixtures could not serve as ground truth: the process
accepted the same noise from which it had calculated the tolerance. The revised
strategy assigns expected values to one reference oracle and measures tolerance
against a known-good BF16 engine.

**Decision**: transformers BF16 (`output_logits=True`, `attn_implementation=sdpa`) is the reference. vLLM BF16 is the baseline for acceptable BF16 numerical drift. vllm-oxide must be no further from the reference than vLLM is, with a 2× relaxation factor. nano-vllm is dropped (dependency conflict with vLLM's torch 2.10). vLLM V1 is collapsed to "vLLM" (V1 is now the default engine in vllm>=0.10). Both oracles run in a single host venv.

**Tolerance**:
- atol only (no rtol). `atol = max(|transformers - vllm| per-element, across all canonical prompts) × 2.0`
- Calibrated from the 5 canonical prompts' full logits in a separate `golden-gen calibrate` step.

**Verification layers**:
- L1 canonical retains t23's near-tie rule: skip positions where the top-2 logit gap < ε (ε = atol × 2.0).
- L1 regression: skip positions where vLLM also disagrees with transformers (no full logits available, so no ε-based gap detection).
- L2 uses `compare_l2_same_prefix` to skip steps after the first token divergence and avoid comparisons affected by that divergence. It replaces the old `compare_l2`.

**CLI**: two-step. `golden-gen generate` produces fixtures from both oracles with tolerance fields left pending. `golden-gen calibrate` loads the canonical fixtures, computes atol, and fills the manifest.

**Status**: accepted

**Considered Options**:
- Oracle triangle (rejected): it calibrated tolerance from its own disagreement, creating a circular check. All prompts were SUSPECT.
- Transformers FP32 as ground truth (rejected): Qwen3-0.6B weights are BF16; running the model in FP32 would change the computational path, not just the output dtype.
- Single oracle without baseline (rejected): it supplies no measured basis for choosing atol.
- Retaining nano-vllm (rejected): torch 2.9 requirement conflicts with vLLM 0.26's torch 2.10. Also, nano-vllm forbids temperature=0, requiring workarounds.

**Consequences**:
- Manifest schema: `cross_validation` removed, `tolerance` updated (atol only, added `observed_max_abs_diff`, `calibration_factor`, `method`).
- `oracle_versions` reduced to `transformers` and `vllm`.
- Calibration requires both oracle outputs. If either is corrupt, the resulting tolerance and all L2 comparisons are invalid.
- L3 per-layer activations comparison remains a skeleton (debug-only, not in CI).
