# Golden generation correctness strategy

The original oracle triangle (transformers / nano-vllm / vLLM V1) produced golden
fixtures where all 8 canonical prompts were flagged SUSPECT — oracle-vs-oracle L2
divergence exceeded 0.1 across the board, tolerance was calibrated at 4623.99 from
the circular "observed noise → tolerance → accept noise" loop, and the golden
fixtures were unusable as ground truth. We redesigned the strategy around two
principles: (1) a single reference oracle eliminates the need for
cross-validation, and (2) tolerance is measured from a known-good BF16 engine
rather than guessed from cycle noise.

**Decision**: transformers BF16 (`output_logits=True`, `attn_implementation=sdpa`) is the reference. vLLM BF16 is the baseline for acceptable BF16 numerical drift. vllm-oxide must be no further from the reference than vLLM is, with a 2× relaxation factor. nano-vllm is dropped (dependency conflict with vLLM's torch 2.10). vLLM V1 is collapsed to "vLLM" (V1 is now the default engine in vllm>=0.10). Both oracles run in a single host venv.

**Tolerance**:
- atol only (no rtol). `atol = max(|transformers - vllm| per-element, across all canonical prompts) × 2.0`
- Calibrated from the 5 canonical prompts' full logits in a separate `golden-gen calibrate` step.

**Verification layers**:
- L1 canonical: near-tie skip — positions where top-2 logit gap < ε (ε = atol × 2.0) are skipped. Stays as-is from t23.
- L1 regression: skip positions where vLLM also disagrees with transformers (no full logits available, so no ε-based gap detection).
- L2: `compare_l2_same_prefix` only — skips steps after the first token divergence, avoiding chain-divergence false positives. Replaces the old `compare_l2`.

**CLI**: two-step. `golden-gen generate` produces fixtures from both oracles with tolerance fields left pending. `golden-gen calibrate` loads the canonical fixtures, computes atol, and fills the manifest.

**Status**: accepted

**Considered Options**:
- Oracle triangle (rejected): cross-validation was circular — tolerance calibrated from noise it produced. All prompts were SUSPECT.
- Transformers FP32 as ground truth (rejected): Qwen3-0.6B weights are BF16; running the model in FP32 would change the computational path, not just the output dtype.
- Single oracle without baseline (rejected): no objective way to set tolerance — any atol would be an arbitrary guess.
- Retaining nano-vllm (rejected): torch 2.9 requirement conflicts with vLLM 0.26's torch 2.10. Also, nano-vllm forbids temperature=0, requiring workarounds.

**Consequences**:
- Manifest schema: `cross_validation` removed, `tolerance` updated (atol only, added `observed_max_abs_diff`, `calibration_factor`, `method`).
- `oracle_versions` reduced to `transformers` and `vllm`.
- Both oracle outputs are co-dependent for calibration — if one oracle's output is corrupt, tolerance is wrong and all L2 comparisons are invalid.
- L3 per-layer activations comparison remains a skeleton (debug-only, not in CI).
