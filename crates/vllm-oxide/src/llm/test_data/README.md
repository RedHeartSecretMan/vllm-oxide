# Small-batch projection rounding regression

`projection-input.bf16` and `projection-weight.bf16` each contain 3072
little-endian BF16 operands (6144 bytes). They isolate one dot product from
Qwen3-0.6B revision `7e4ae267688d671ddfca3122e4528ee980cf3234`:
layer 0 SwiGLU activation for token 33603 and down-projection weight row 883.
The weight column is repeated across 1024 outputs; all added input rows are zero.
These are operator operands, not generated model-output golden fixtures.

On the measured RTX 4080, with FP32 intermediate reductions, row counts 1/2
produce -0.31640625; 3/8/771 produce -0.314453125. The exact dot is approximately
-0.31543077422247734: the smaller matrix result is mathematically closer.
The test checks compatibility with the approved reference's matrix geometry,
not a claim that padding is universally more accurate. It also checks the
unchanged single-request path and output slicing/bias through real CUDA GEMM.
The regression is an explicit guarded GPU test; CPU CI cannot substitute for it.

SHA-256:

- input: `443c0a7508d8191ec3b9f128c5a470abbc6d77abc9ef016ee6e13b44f13d71ca`
- weight: `330f54bfe7546482886f839168dfd143f946e98af408991a28c682fad61dda23`


# Batched workspace and causal suffix regressions

The workspace operands isolate the same model revision's layer-0 down projection
in the observed confirmation mixed-batch failure: one 3072-value SwiGLU input
and weight row 50 (6144 bytes each, little-endian BF16). The CUDA test embeds
those operands in the original 1024-output geometry. With FP32 accumulation,
258 rows yield 0xb954 and514 rows yield 0xb955 on the validated GPU. Zero workspace
reproduces 0xb955 for the multi-request matrix path. The exact dot is approximately
-0.00020244751249265391; the original 258-row value is mathematically closer.
This is reference-layout compatibility, not a higher-accuracy theorem. The test
also verifies single-request/two-row preservation and default-workspace recovery
after a successful operation and a real dimension error.

The query-suffix operands contain three 128-wide queries for query head 4, and 257
128-wide keys/values for KV head 2. They reconstruct the first divergent
layer-0 paged-attention operation at positions 254–256 of the same model.
The actual Q/K/V bits are identical in the chunked and full-prefill executions.
The original three-query output at query 0/head 4/component 52 is 0xbcd5; the recorded
full-prefill output is 0xbcd4. The regression exercises real paged gathering,
GQA head mapping, QK, causal softmax and PV, and expects 0xbcd4. Zero query rows
are confined to QK and discarded before masking; they do not create tokens.
These small operator inputs are not publishable model-output golden fixtures.

Additional SHA-256:

- query-suffix-k.bf16: 9ab2dce6f73b6ca1d0206dcfb08fdc3570ec2c000ccea6de5545726fac674b4c
- query-suffix-q.bf16: c10fcd260fdf203b506cffad3ef47f99526ad5bb558d2e8f8ab1e5c04ca654f9
- query-suffix-v.bf16: 79cccbf59776d8b232dc5dcbc23880a8a25b516d1dcf01da338454597dad7b8e
- workspace-input.bf16: a0d122dd97dae8fb4e93c1a85cb4316f63592e10876ead798a8751e493467519
- workspace-weight.bf16: 6a75ad0b988ff952288f604ddefbddc10feeae31684e5ee356b26566a7ccb323
