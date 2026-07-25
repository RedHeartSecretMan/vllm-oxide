# ⚠️ THROWAWAY PROTOTYPE — `prototype-candle-stack`

Part of vllm-oxide **T10 (#11)**. **Not** the vllm-oxide crate layout (T9 is undecided).
Code here is throwaway from day one — it exists only to validate two risky
assumptions in [T3 — Stack A (candle hybrid)](https://github.com/RedHeartSecretMan/vllm-oxide/issues/5):

1. **cudarc re-export seam** — unwrap a `candle::Tensor` (CUDA) to raw
   `cudarc::driver::CudaSlice` + `CUstream` and call a hand-written
   `extern "C"` CUDA kernel through it (the mistral.rs pattern).
2. **`candle-flash-attn` paged kernel API** — the Rust-side entry point
   added by [candle PR #3655](https://github.com/huggingface/candle/pull/3655).

## Run

```bash
cd research/prototype-candle-stack
cargo run --release
```

Requires: CUDA toolkit (nvcc on PATH), an NVIDIA GPU (sm80+; tested on RTX 4080 / sm_89).

## What this is NOT

- Not the vllm-oxide engine.
- Not production code — no error handling beyond runnable, no tests, no abstractions.
- Will be deleted (or archived on a `research/*` branch) once T10 resolves.
