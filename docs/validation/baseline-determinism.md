# vLLM baseline admission determinism

Each release capture remains a fresh, guarded process. Within that owner,
`VllmOracle` sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` before importing vLLM and
rejects a previously cached enabled setting. This keeps EngineCore in the
owner's process, so offline `LLM.generate` queues its entire input batch before
stepping. The pinned vLLM 0.18.1 source supports this mode; it is also the
[documented offline reproducibility setting](https://docs.vllm.ai/en/stable/usage/reproducibility/).

The 479bf82 calibration exposed why deterministic PyTorch arithmetic alone was
insufficient. In the baseline waiting group, primary and replay initially ran
only the short request; control initially ran short and long together. The
short request's identical first history then differed in 145508 logits, with
maximum absolute difference 0.25. The production equivalence gate rejected the
whole observation. Those original captures and their INVALID result remain.

A one-variable diagnostic disabling only EngineCore multiprocessing made all
three owners use the same scheduling geometry and restored bitwise replay and
shared-history collection equivalence. This does not establish release
acceptance or waive subsequent numerical conditions.

Production receipts now carry the actual EngineCore client class, disabled
multiprocessing flag, owner PID and GPU-worker PID. The evaluator requires an
`InprocClient`, exact false flag and matching owner/worker identities, together
with the existing before-CUDA determinism, FlashAttention-v2 and lifecycle
proofs. Missing or inconsistent scheduling evidence is invalid. Baseline
identity adds `inproc-scheduler`; old receipts cannot silently acquire it.

BF16 arithmetic, eager mode, attention backend, model, inputs and every approved
budget value are unchanged. No batch-invariant kernel replacement or tolerance
relaxation is enabled. New source-bound calibration and release evidence remain
mandatory.
