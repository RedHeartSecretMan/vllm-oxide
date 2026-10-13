# Preserve bounded causal GEMM layouts

The independent confirmation run at measurement c0e8b3d, supervised by 5a0c904,
stopped at 32 of 617 owners. The mixed batch's medium member exceeded the unchanged
absolute budgets: mean KL 0.002199434040322365 and peak KL 0.006178046492977655.
The original captures, guards and failure remain intact. Exact candidate replay
reproduced all three members bit for bit; token agreement did not waive the error.

Two independent operator effects were isolated using the same captured operands:

- BF16 down projection changed its FP32 reduction with matrix row count. Inputs
  through attention, output projection, post-normalization, gate/up and SiLU were
  identical. CUDA 12.8 and 13 reproduced the shape-dependent result; a zero
  cuBLAS workspace selected the reference-compatible matrix reduction in the
  tested shapes. Some original small-matrix values were closer to exact rational
  dot products, so compatibility must not be described as improved exact accuracy.
- Small causal query suffixes selected a different FP32 QK reduction. Swapping
  QK scores swapped the BF16 attention result; varying PV geometry alone did not.
  Padding QK to 32 physical query rows reproduced the full-prefill scores. The
  scores are cropped and made contiguous before the unchanged causal mask,
  softmax and PV operations. The one-query decode path is preserved.

ProjectionLayout keeps the existing bounded row floor. Multi-request BF16 CUDA
matrix tiles above two physical rows use zero workspace for that GEMM only.
Single-request and one/two-row GEMMs retain their existing behavior. A narrow
handle-control module restores the same stream's default workspace on success,
errors and unwinding, and preserves both operation and restoration errors if
both fail. Each public LLM owns a fresh device/handle and generation is exclusive.
The helper installs no user buffer and never retries inference with different math.

The QK padding is bounded by 31 extra rows. Both full and split-head workspace
planning account for those physical score rows and temporary query storage;
logical query ranges, cache writes, causal positions and scheduled token budgets
remain unchanged. These are compatibility choices for the recorded release
kernel/hardware scope, not a proof of shape-independent cuBLAS arithmetic on
arbitrary hardware or inputs.

The combined prototype 91e7764 reduced the original medium-member mean KL to
0.0014010984944923528 and peak KL to 0.00255655425556236. Its two-request/two-step
minimal reproduction also passed, and the full packed-prefill counterfactual
became bitwise equal to the unchanged reference. These are diagnostic comparisons
against retained observations, not fresh independent acceptance.

Current candidate kernel identity advances to call-layout-v2. Reference arithmetic,
model identity and all model/operator budgets remain unchanged. Actual guarded
CUDA regressions use the small operands documented in
[the test-data record](../../crates/vllm-oxide/src/llm/test_data/README.md).
Fresh calibration, source-bound CPU/GPU evidence, an unobserved independent
checkpoint, performance and release-asset verification remain mandatory.

The cuBLAS handle contract is documented in the official
[workspace and stream API](https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace):
zero workspace disables the default pool, and setting the stream unconditionally
restores it. Zero workspace can reduce performance or fail for unsupported
routines; errors propagate and must be covered by the actual release gates.
