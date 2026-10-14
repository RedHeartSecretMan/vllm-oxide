# Golden generation correctness strategy

The original oracle triangle (transformers / nano-vllm / vLLM V1) flagged all
8 canonical prompts as SUSPECT. Oracle-to-oracle L2 divergence exceeded 0.1
in every case, and calibrating from that disagreement produced a tolerance of
4623.99. The resulting fixtures could not serve as ground truth: the process
accepted the same noise from which it had calculated the tolerance. The revised
strategy assigns expected values to one reference oracle and measures tolerance
against a known-good BF16 engine.

**Decision**: transformers BF16 (`output_logits=True`, `attn_implementation=sdpa`) is the authoritative reference. vLLM BF16 supplies calibration and investigation evidence but cannot override a reference failure. nano-vllm is dropped (dependency conflict with vLLM's torch 2.10). vLLM V1 is collapsed to "vLLM" (V1 is now the default engine in vllm>=0.10). Both oracles run in a single host venv.

**Tolerance**:
- Acceptance thresholds are explicit, versioned, and justified per dtype and
  kernel path from observed same-prefix error distributions.
- The legacy global `max(|transformers - vllm|) × 2.0` rule is calibration
  evidence, not an automatic v0.2.0 acceptance threshold.

**Verification layers**:
- L1 accepts the reference token or an explicit near-tie classification derived
  from the relevant same-prefix candidate logits and versioned policy.
- A baseline-oracle disagreement never creates an L1 skip map or pass.
- L2 compares only a matching generated prefix and stops at the first token
  divergence; later logits are excluded, while the divergence stays visible.
- Release acceptance requires zero missing, unexpected, skipped, or failed
  fixtures.

**CLI**: two-step. `golden-gen generate` produces fixtures from both oracles with tolerance fields left pending. `golden-gen calibrate` loads the canonical fixtures, computes atol, and fills the manifest.

**Status**: accepted; v0.2.0 policy revised by ADR-0006

**Considered Options**:
- Oracle triangle (rejected): it calibrated tolerance from its own disagreement, creating a circular check. All prompts were SUSPECT.
- Transformers FP32 as ground truth (rejected): Qwen3-0.6B weights are BF16; running the model in FP32 would change the computational path, not just the output dtype.
- Treating the baseline as a second pass/fail oracle (rejected): calibration
  evidence cannot override the authoritative reference.
- Retaining nano-vllm (rejected): torch 2.9 requirement conflicts with vLLM 0.26's torch 2.10. Also, nano-vllm forbids temperature=0, requiring workarounds.

**Consequences**:
- The manifest records the observed distributions, selected tolerance policy,
  dtype/kernel scope, and rationale as versioned evidence.
- `oracle_versions` reduced to `transformers` and `vllm`.
- Both oracle outputs are required for calibration, but only the reference
  supplies expected correctness values.
- L3 per-layer activations comparison remains a skeleton (debug-only, not in CI).
