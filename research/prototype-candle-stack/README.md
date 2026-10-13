# Candle/CUDA interoperability prototype

[Research index](../README.md) · [Engine architecture](../../docs/architecture.md)

This historical, standalone experiment answered two early implementation questions tracked in [T10 / #11](https://github.com/RedHeartSecretMan/vllm-oxide/issues/11) and [Stack A / #5](https://github.com/RedHeartSecretMan/vllm-oxide/issues/5): can Candle-owned CUDA storage be passed safely to a handwritten CUDA launch, and can the selected paged attention entrypoint run? It is not a member of the engine workspace and publishes no crate.

## Source and dependency boundaries

| File | Responsibility |
|---|---|
| [Cargo.toml](Cargo.toml) | Pins Candle core and FlashAttention to commit `27f20fea993c81ea6d32ce44018f42b68466525e`; enables CUDA and declares `cudaforge` |
| [build.rs](build.rs) | Builds `kernels/*.cu` as `libfillconst.a`, links `cudart`, and lets `cudaforge` detect compute capability |
| [fill_const.cu](kernels/fill_const.cu) | C-linkage launch wrapper that fills F32 storage on the supplied CUDA stream |
| [main.rs](src/main.rs) | Allocates tensors, calls both phases and checks their observed outputs |
| [.gitignore](.gitignore) | Excludes its standalone target, build log and generated dependency lock |

It takes no CLI arguments, model path or prompt. Its dependency lockfile is intentionally untracked: the Candle Git revision is pinned, while other declared dependencies are resolved independently. Record the generated lockfile if reproducing the experiment; do not claim it uses the main workspace's release lock.

## Requirements and run

Use a CUDA-capable Linux or WSL2 environment, NVIDIA driver, full CUDA Toolkit with `nvcc`, a compatible host compiler/linker and Rust/Cargo. The historical README recorded RTX 4080/SM 8.9, with SM 8.0+ as the prototype's intended scope; this is not an engine hardware validation promise. There is no CPU test mode because both Candle dependencies and the build script always require CUDA.

The historical manifest is nested under the main workspace but is not listed as a workspace member or exclusion. Running `cargo run` directly in this directory therefore produces “current package believes it is in a workspace when it is not.” Preserve the original manifest and run an isolated copy instead, from the repository root:

```bash
PROTO_RUN_DIR=$(mktemp -d /tmp/vllm-oxide-rc-prototype.XXXXXX)
cp -a research/prototype-candle-stack/. "$PROTO_RUN_DIR/"
cargo run --release --manifest-path "$PROTO_RUN_DIR/Cargo.toml"
```

The copy uses exactly the original Rust/CUDA and dependency declarations. Its artifacts and independently resolved lockfile stay under the temporary project. Keep that directory while inspecting results; the command does not modify workspace membership or historical source.

The first run resolves this separate manifest and compiles CUDA code. No model weights, golden corpus or downloaded engine cache is needed. It prints human-readable stdout and creates ordinary build artifacts; it does not produce a release marker or a JSON result. Save stdout/stderr and the process exit status externally if a reviewer needs a reproduction.

## Phase A: raw pointer and stream

The program allocates 1,024 F32 zeros on GPU 0, obtains `CudaStorage`, applies the layout offset to its `CudaSlice`, and obtains a device pointer guard and `CUstream`. The guard remains live while `fill_const_f32_launch` writes value 42 on that stream.

Readback checks sum **43,008**, minimum **42**, maximum **42**; sum tolerance is `1e-3` and min/max tolerance `1e-6`. The `[proto][A] PASS` message means this one buffer and launch succeeded. A mismatch exits 1; a Candle/build error also fails the process. This is a concrete interoperability probe, not a general unsafe-FFI safety proof.

## Phase B: paged attention API

The pinned [Candle PR #3655](https://github.com/huggingface/candle/pull/3655) entrypoint is `flash_attn_varlen_paged_windowed`.

| Input | Fixed experiment dimensions |
|---|---|
| Queries | FP16 `[16, 4, 64]` |
| Keys and values | FP16 `[4, 32, 4, 64]` each |
| Query/key prefix sums | U32 `[0, 16]` / `[0, 64]` |
| Block table | U32 `[1, 4]`, physical indices 0 through 3 |
| Attention | MHA, scale `1/sqrt(64)`, causal right window 0 |
| Expected output | `[16, 4, 64]` |

Cumulative sequence lengths are prefix sums, not vLLM's per-sequence context lengths. The program checks the output shape and samples values for NaNs. It uses random tensors without recording a seed, checks only sampled NaNs and has no reference attention comparison. Its `[proto][B] PASS` therefore means the selected API ran and met these limited checks; it does not establish deterministic output, full finite coverage or Qwen3 precision.

## Relating the experiment to the engine

This prototype uses 32-token pages and FP16 synthetic MHA. The RC engine uses 256-token pages and a Qwen3 BF16 execution profile. Reproduce the supported engine through [getting started](../../docs/getting-started.md), and assess numerical behavior through [the RC validation guide](../../docs/validation/README.md). There are no unit tests or production abstractions in this experiment. No current GPU execution is claimed by this documentation maintenance.

CUDA build failures usually indicate an unavailable Toolkit/host compiler or incompatible GPU target; `expected CUDA storage` indicates no CUDA device was selected. Inspect the actual build error before changing the pinned stack. Keep experiment changes separate from engine fixes.
