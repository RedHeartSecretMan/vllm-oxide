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
