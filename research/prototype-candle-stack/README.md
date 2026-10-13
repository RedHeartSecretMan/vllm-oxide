# Candle/CUDA interoperability prototype

[Research index](../README.md) · [Current architecture](../../docs/architecture.md) · [Development guide](../../docs/development.md)

This retained [T10 / issue #11](https://github.com/RedHeartSecretMan/vllm-oxide/issues/11)
experiment exercises two early stack decisions from
[Stack A / issue #5](https://github.com/RedHeartSecretMan/vllm-oxide/issues/5): calling a project CUDA kernel
through Candle's tensor storage, and invoking Candle's paged FlashAttention
entrypoint introduced by [Candle PR #3655](https://github.com/huggingface/candle/pull/3655).
It uses synthetic tensors only: **no Qwen3 checkpoint, tokenizer,
prompt, or generation API is loaded**. The current engine is in
[`crates/vllm-oxide`](../../crates/vllm-oxide/).

## What the executable checks

| Phase | Inputs and operation | Success condition in the source |
|---|---|---|
| A: Candle → cudarc → CUDA | Allocate 1,024 F32 zeros on device 0; obtain the storage slice, apply the tensor's start offset, retain the device-pointer guard, and pass the pointer and CUDA stream to `fill_const_f32_launch` | Read back a sum of 43,008, minimum 42 and maximum 42; absolute error below `1e-3` for the sum and `1e-6` for each extremum |
| B: paged attention | Random F16 Q `[16, 4, 64]`; K/V `[4, 32, 4, 64]`; one request with 16 query and 64 key tokens; U32 cumulative lengths and a physical block table | Output shape `[16, 4, 64]` and no NaN among the sampled elements |

Phase B calls `flash_attn_varlen_paged_windowed` with scale `1/sqrt(64)`, an
unbounded left window and right window 0. Its page size is 32 tokens. The
`cu_seqlens_q` and `cu_seqlens_k` arguments are **cumulative lengths**, rather
than the per-request `context_lens` used by other paged-attention APIs. The
experiment checks that this concrete call executes; it does not compare the
attention output with a numerical oracle.

## Source and dependency map

| File or dependency | Role |
|---|---|
| [`src/main.rs`](src/main.rs) | Both phases, tensor/stream extraction, readback checks and terminal PASS/FAIL messages |
| [`kernels/fill_const.cu`](kernels/fill_const.cu) | `extern "C"` launcher and a 256-thread fill kernel |
| [`build.rs`](build.rs) | Compile `kernels/*.cu` into `libfillconst.a`; link the static kernel library and CUDA runtime `cudart` |
| [`Cargo.toml`](Cargo.toml) | Private package `prototype-candle-stack` 0.0.0, Rust edition 2021; no publish target |
| [Candle Core and FlashAttention](https://github.com/huggingface/candle/tree/27f20fea993c81ea6d32ce44018f42b68466525e) | Both pinned to git revision `27f20fea993c81ea6d32ce44018f42b68466525e`; Candle Core enables `cuda` directly |
| `cudaforge` | Manifest requirement `0.1.2`; detects the compute target and invokes `nvcc` |
| `anyhow`, `half` | Manifest requirements `1` and `2.7`, respectively |

The CUDA build uses C++17, `-O3`, `--use_fast_math`, and position-independent
code. There is no `--features cuda` switch for this prototype: its dependency
already enables CUDA. The retained directory has no committed `Cargo.lock` or
package-specific Rust minimum; the Candle revisions are fixed, while compatible
registry dependencies are resolved on the first standalone build. The current
repository pins Rust 1.94.0; the command below selects it explicitly after copying
the experiment outside the workspace.

## Prepare and run

Use a Linux CUDA environment, including WSL2, with an NVIDIA GPU, a compatible
driver, CUDA Toolkit (`nvcc` on `PATH`), the CUDA runtime, and a host C++ compiler.
The original experiment targeted SM 8.0+ and documented RTX 4080 / SM 8.9.
That historical target is not a replacement for the shipped engine's current
hardware requirements. Compilation also builds the pinned Candle FlashAttention
dependency, so allow time, host memory and disk for a CUDA dependency build.
No model download or Python environment is needed.

The prototype is not a member of the root Cargo workspace. Running
`cargo run` directly in this directory currently fails with Cargo's
"current package believes it's in a workspace when it's not" error. To run the
retained experiment without changing the production workspace, copy its retained
sources to a fresh directory outside the repository. From the
repository root:

```bash
PROTOTYPE_ROOT="$(mktemp -d /tmp/vllm-oxide-candle-prototype.XXXXXX)"
cp -R research/prototype-candle-stack "$PROTOTYPE_ROOT/"
cd "$PROTOTYPE_ROOT/prototype-candle-stack"
cargo +1.94.0 run --release
```

The first build needs access to the pinned Candle git source and registry
dependencies unless they are already cached. Cargo writes the standalone
`Cargo.lock`, compiled objects and executable under this temporary copy. Keep
them with any experiment logs when reproducing an old result. The program has
**no CLI arguments** and always selects device 0; changing shapes, seed behavior,
or device requires a source change and a separate experiment record.

## Read the result

A successful run prints `[proto][A] PASS` followed by `[proto][B] PASS`. A
failed explicit value/shape/NaN check exits 1; setup, allocation and backend
errors propagate as nonzero execution failures. Phase A may pass before phase B
fails, so retain the complete output and exit status.

These messages describe the checks performed by that invocation. The original
README reported an RTX 4080 experiment, but this directory contains no retained
run receipt or raw log establishing a newly reproduced PASS. Documentation
maintenance does not rerun the GPU experiment or infer a current result from the
printed strings in the source.

This executable does not establish BF16 model accuracy, decoding behavior,
throughput, KV-cache reuse or scheduler correctness. Phase B samples NaNs rather
than checking every element and has no reference comparison or seeded replay.
There is no dedicated test suite or ordinary CI job for this package. The
production engine uses the same pinned Candle Core revision, but its scheduling,
attention implementation, kernels and acceptance checks live in the main
workspace; this prototype's paged-FlashAttention call is not a specification of
the released backend.

For current engine checks, use the [validation guide](../../docs/validation/README.md),
[public GPU smoke](../../docs/validation/public-gpu-smoke.md), and
[v0.2.0 results](../../docs/releases/goldens-v0.2.md).
