# Candle/CUDA interoperability prototype

[Research index](../README.md) · [Engine architecture](../../docs/architecture.md) · [Development guide](../../docs/development.md)

This retained [T10 / issue #11](https://github.com/RedHeartSecretMan/vllm-oxide/issues/11)
experiment checks two early [Stack A / issue #5](https://github.com/RedHeartSecretMan/vllm-oxide/issues/5)
decisions: passing Candle tensor storage to a project CUDA kernel, and calling
the paged FlashAttention entrypoint from [Candle PR #3655](https://github.com/huggingface/candle/pull/3655).
It uses synthetic tensors. It loads no checkpoint, tokenizer, prompt or
generation API; the engine lives in [`crates/vllm-oxide`](../../crates/vllm-oxide/).

## What the executable checks

| Phase | Inputs and operation | Success condition |
|---|---|---|
| A: Candle → cudarc → CUDA | Allocate 1,024 F32 zeros on device 0; retain storage and pointer guards, launch a 256-thread fill kernel on the tensor's stream, synchronize and read back | Sum 43,008, minimum 42, maximum 42; errors below `1e-3` for sum and `1e-6` for each extremum |
| B: paged attention | Seed 11; F16 Q `[16, 4, 64]`, K/V `[4, 32, 4, 64]`; one request with 16 query and 64 key tokens | Shape `[16, 4, 64]`, exactly 4,096 output values, all finite |

Phase B uses 32-token pages, four Q/KV heads, scale `1/sqrt(64)`, an unbounded
left window and causal right window 0. U32 `cu_seqlens_q` `[0, 16]` and
`cu_seqlens_k` `[0, 64]` are cumulative lengths, rather than per-sequence context
lengths. The physical block table has shape `[1, 4]` and indices 0 through 3.
The output has no independent numerical oracle. A fixed input seed helps replay
within the same stack; it does not promise bitwise identity across GPUs or CUDA
versions. The experiment's 32-token pages are separate from the engine's geometry.

## Source and dependencies

| File or dependency | Responsibility |
|---|---|
| [`src/main.rs`](src/main.rs) | Tensor/stream extraction, both phases, errors and PASS messages |
| [`src/validation.rs`](src/validation.rs) | CPU-only output shape, length and complete finite-value checks |
| [`kernels/fill_const.cu`](kernels/fill_const.cu) | C-linkage launcher returning immediate CUDA launch status |
| [`build.rs`](build.rs) | Compile `kernels/*.cu` into `libfillconst.a` and link CUDA runtime `cudart` |
| [`Cargo.toml`](Cargo.toml) | Private package 0.0.0 with its own workspace; Rust 1.94 minimum |
| [`Cargo.lock`](Cargo.lock) | Independent dependency lock, committed for `--locked` reproduction |
| [Candle Core and FlashAttention](https://github.com/huggingface/candle/tree/27f20fea993c81ea6d32ce44018f42b68466525e) | Both pin git revision `27f20fea993c81ea6d32ce44018f42b68466525e` |
| `cudaforge` | Manifest requirement `0.1.2`; compute-target detection and `nvcc` invocation |
| `anyhow`, `half` | Manifest requirements `1` and `2.7` |

The CUDA build uses C++17, `-O3`, `--use_fast_math` and position-independent code.
There is no prototype `--features cuda` switch: its Candle dependencies already
enable CUDA. Its local `[workspace]` makes this package directly runnable while
keeping CUDA dependencies out of the root workspace's CPU-only tests. Its own
lockfile is independent of the root engine lockfile and retains the same pinned
Candle revision.

## Prepare and run

Use Linux or WSL2 with an NVIDIA GPU, a compatible driver, full CUDA Toolkit
(`nvcc` on `PATH`), CUDA runtime and host C++ compiler. The original experiment
recorded RTX 4080 / SM 8.9 and intended SM 8.0+; this is not an engine hardware
validation promise. Allow host memory, disk and time to build Candle's CUDA
and FlashAttention dependencies. No model download or Python environment is needed.

From the repository root:

```bash
cargo +1.94.0 metadata --locked --no-deps \
  --manifest-path research/prototype-candle-stack/Cargo.toml
cargo +1.94.0 run --release --locked \
  --manifest-path research/prototype-candle-stack/Cargo.toml
```

The first build needs the pinned Git source and registry dependencies unless
cached. Cargo writes objects and the executable into the prototype's own
`target/` directory; `.gitignore` excludes that directory and `build.log`.
The program takes no CLI arguments and always selects CUDA device 0. Changing
shapes or device selection is a separate experiment source change.

Phase A's launcher reports immediate CUDA status and Rust explicitly synchronizes
before releasing its pointer/storage guards, catching asynchronous execution
errors before readback. It returns a nonzero process status on any failure.
Phase B reads all output values back in one transfer, rejecting both NaN and Inf.
A successful complete run prints `[proto][A] PASS` followed by `[proto][B] PASS`.
Preserve full stdout/stderr, exit status, environment and source revision; Phase A
can succeed before Phase B fails.

## CPU regression checks and limits

The readback validator can be tested without CUDA, dependency compilation or a
GPU. This runs the same helper used by Phase B, including a NaN at an index the
old sampling stride skipped, positive/negative Inf, wrong shapes and incomplete
or empty readbacks:

```bash
PROTO_TEST_BIN="$(mktemp /tmp/vllm-oxide-prototype-validation.XXXXXX)"
rustc +1.94.0 --edition 2021 --test \
  research/prototype-candle-stack/src/validation.rs -o "$PROTO_TEST_BIN"
"$PROTO_TEST_BIN"
rm "$PROTO_TEST_BIN"
```

Building or running the full Cargo package still requires CUDA. Ordinary engine
CI does not execute it. This prototype does not establish BF16 model accuracy,
decoding behavior, throughput, KV-cache reuse or scheduler correctness. Its
paged-FlashAttention call is not a specification of the released backend.
The historical experiment's claimed result does not become a new GPU PASS when
source or documentation is maintained. Use the [validation guide](../../docs/validation/README.md)
for engine checks and the [release notes](../../docs/releases/) for version results.
