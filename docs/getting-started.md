# Getting started

[Documentation index](README.md) · [中文项目介绍](../README.zh-CN.md)

This guide uses the **historical v0.2.0.rc.1 source prerelease** and its fixed **Qwen3-0.6B BF16** checkpoint. It covers offline generation through the Rust API and CLI. There is no inference server or streaming API in this release.

[Requirements](#requirements) · [Install](#install-the-released-source) · [Model sources](#load-a-model) · [CLI](#cli) · [Rust API](#rust-api) · [Troubleshooting](#common-errors)

| Your goal | Choose |
|---|---|
| Try the exact model revision without writing code | The [pinned model example](#use-a-pinned-hub-revision) |
| Generate once from your own prompt | The [CLI](#cli) with a local checkpoint or Hub repo ID |
| Batch requests, reuse a model, or adjust engine settings | The [Rust API](#rust-api) |
| Contribute without a GPU | The [development guide](development.md) |

## Requirements

| Component | Requirement |
|---|---|
| OS | Linux, including WSL2 |
| Rust | Use the recorded Rust 1.94.0 toolchain with Rustfmt and Clippy; the maintained manifest and toolchain use Rust 1.94 |
| GPU | One NVIDIA CUDA GPU with compute capability 8.9 or newer |
| Build tools | CUDA Toolkit with `nvcc`, plus a compatible host C/C++ compiler and linker |
| Runtime | A working NVIDIA CUDA driver and enough free GPU memory for weights, activations, and the KV cache |

The RC was checked on RTX 4080/WSL2. Historical numerical results include 18 failures; full performance and the declared Rust minimum were not validated. Do not use the final v0.2.0 measurements as RC results. See the [RC notes](releases/v0.2.0-rc.1.en.md).

CPU development and tests do not require a GPU or CUDA Toolkit. A default CPU build does **not** provide CPU inference.

## Install the released source

```bash
git clone --branch v0.2.0.rc.1 --depth 1 https://github.com/RedHeartSecretMan/vllm-oxide.git
cd vllm-oxide
cargo build --locked --release -p vllm_oxide_cli --features cuda
```

The CLI executable is `target/release/vllm_oxide_cli`. `--locked` uses the repository's dependency lockfile. This release distributes source; there is no prebuilt executable or crates.io package for this RC.

## Load a model

### Use a pinned Hub revision

The provided example accepts a model source as its only argument and generates from a fixed prompt:

```bash
cargo run --locked --release -p vllm_oxide --example generate_qwen3 --features cuda -- \
    hub:Qwen/Qwen3-0.6B@7e4ae267688d671ddfca3122e4528ee980cf3234
```

The first run downloads the checkpoint and caches its files. The example continues “The meaning of life is” with greedy decoding and at most 16 completion tokens, then prints the continuation. It understands `hub:<repo>`, `hub:<repo>@<revision>`, and local directory paths. Supplying a commit revision keeps model selection reproducible; omitting it follows the Hub's default branch. The fixed [checkpoint files on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B/tree/7e4ae267688d671ddfca3122e4528ee980cf3234) are the same model revision used for RC measurements.

### Use a local checkpoint

A complete local checkpoint contains `config.json`, `tokenizer.json`, and its safetensors weights. Sharded checkpoints also need their index and every referenced shard. For the RC-recorded checkpoint, use the revision shown above.

```bash
export QWEN3_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
cargo run --locked --release -p vllm_oxide_cli --features cuda -- \
    --model "$QWEN3_MODEL_DIR" --max-tokens 2 "The capital of France is"
```

The original RC includes only short CUDA CLI smoke checks; a successful generation does not establish numerical accuracy. A local directory is used directly, without Hub downloads. The loader resolves configuration, tokenizer, weights, dtype, and special tokens together for the lifetime of an `LLM`.

For a cached Hub source, `HF_HUB_OFFLINE=1` disables model network fetching. All files for the requested revision must already exist in the cache; missing files produce an error. Build the application and populate its model files before disconnecting. `Source::Local` is the simplest way to select a specific complete checkpoint for fully offline generation.

| Entry point | Local checkpoint | Hub source | Pinned Hub revision |
|---|---|---|---|
| Provided example | Directory argument | `hub:Qwen/Qwen3-0.6B` | `hub:Qwen/Qwen3-0.6B@<revision>` |
| CLI | `--model <existing-directory>` | `--model Qwen/Qwen3-0.6B` | Use a local checkpoint from that revision |
| Rust API | `Source::Local(path)` | `Source::Hub { repo, revision: None }` | `Source::Hub { repo, revision: Some(commit) }` |

## CLI

```bash
target/release/vllm_oxide_cli --help
target/release/vllm_oxide_cli --model "$QWEN3_MODEL_DIR" \
    --temperature 0 --max-tokens 16 "The meaning of life is"
```

Without a positional prompt, the CLI reads all of stdin:

```bash
printf '%s' 'The meaning of life is' | \
    target/release/vllm_oxide_cli --model "$QWEN3_MODEL_DIR" --max-tokens 16
```

| Option | Meaning | Default |
|---|---|---|
| `-m`, `--model` | Existing local directory or Hub repo ID | Required |
| Positional prompt | Plain text to continue | Read stdin |
| `--temperature` | `0` selects greedy generation | `0` |
| `--top-k` | Keep the highest-logit `k` tokens | Disabled |
| `--top-p` | Nucleus probability threshold | Disabled |
| `--max-tokens` | Maximum number of completion tokens | `16` |

The CLI treats an existing directory as local and every other model argument as a Hub repo ID. It has **no revision flag**, and it does not parse the example's `hub:` or `@revision` syntax. To pin a revision, use the example, the Rust `Source::Hub` API, or an existing local checkpoint from that revision.

CLI option parsing rejects NaN or negative temperature, zero `top-k` or
`max-tokens`, and non-finite or out-of-range `top-p` before loading a model.
Positive-infinity temperature remains accepted, matching the Rust sampling
contract.

Prompts are raw completion text. The CLI does not apply a chat template; format chat conversations before calling the engine if your application needs them.

The CLI emits completion text to stdout and propagates construction/generation errors as a failed process. It submits one request, uses default engine settings, and has no switches for batch input, token-ID prompts, penalties, `ignore_eos`, cache budgets, dtype, GPU selection, or random seed. Use the Rust API for the supported controls it exposes.

## Rust API

Add the Git dependency to your application:

```toml
[dependencies]
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", tag = "v0.2.0.rc.1", features = ["cuda"] }
anyhow = "1"
```

Construct one `LLM`, then reuse it for generation calls:

```rust
use vllm_oxide::{EngineOptions, Prompt, SamplingParams, Source, LLM};

fn main() -> anyhow::Result<()> {
    let mut llm = LLM::new(
        Source::Hub {
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: Some("7e4ae267688d671ddfca3122e4528ee980cf3234".into()),
        },
        EngineOptions::default(),
    )?;
    let prompts = [
        Prompt::Text("The capital of France is".into()),
        Prompt::Text("Once upon a time".into()),
    ];
    let params = [
        SamplingParams {
            max_tokens: 2,
            ..Default::default()
        },
        SamplingParams {
            max_tokens: 32,
            temperature: 0.7,
            ..Default::default()
        },
    ];
    for (prompt, output) in prompts.iter().zip(llm.generate(&prompts, &params)?) {
        println!("{prompt:?} -> [{}] {}", output.request_id, output.text);
    }
    Ok(())
}
```

`LLM::generate(&mut self, prompts: &[Prompt], params: &[SamplingParams])` returns `anyhow::Result<Vec<RequestOutput>>`. Provide exactly one `SamplingParams` value per prompt. Each result contains `request_id`, completion `token_ids`, completion `text`, and `finished`; it does not include the prompt in the completion. Results preserve input order, and request IDs stay independent of that call's batch positions. `Prompt::Text` and `Prompt::TokenIds` can share a batch.

### Public types and output fields

| Export | Purpose |
|---|---|
| `LLM` | Load and own one inference engine; call `new` and `generate` |
| `Source` | Select a complete local checkpoint or a Hub repository/revision |
| `EngineOptions` | Configure context, scheduling, cache allocation, and checkpoint dtype at construction |
| `Prompt` | Supply raw `Text(String)` or pre-tokenized `TokenIds(Vec<u32>)` |
| `SamplingParams` | Set each request's token-selection and completion-stopping policy |
| `RequestOutput` | Read the completed request's identity, tokens, and text |

| `RequestOutput` field | Type | Meaning |
|---|---|---|
| `request_id` | `usize` | Stable identity within the owning `LLM`; do not use it as a batch-array index |
| `token_ids` | `Vec<u32>` | Generated completion tokens |
| `text` | `String` | Decoded completion with special tokens skipped |
| `finished` | `bool` | Completion state; ordinary successful offline generation returns finished requests |

An empty prompt batch with an empty parameter slice returns an empty result. A non-empty batch validates sampling values and every prompt's context budget before admission. Token-ID prompts must use the selected checkpoint's vocabulary; the engine does not apply a chat template to either input variant.

Model loading and CUDA warmup happen in `LLM::new`. Construction and generation can fail, so propagate or handle their errors. Calls are synchronous and use `&mut self`; the API does not expose per-token streaming, cancellation, or concurrent request submission. Internal model, scheduler, cache, and sampler types are not public extension points. Generate local API documentation with `cargo doc --locked -p vllm_oxide --no-deps`. See [architecture](architecture.md) for the execution path beneath the public API.

### Engine configuration

| `EngineOptions` field | Default | Meaning |
|---|---:|---|
| `max_num_batched_tokens` | `16384` | Maximum tokens processed across prefill and decode in one scheduler step |
| `max_num_seqs` | `512` | Maximum concurrently running sequences |
| `max_model_len` | `4096` | Combined prompt and completion context budget |
| `gpu_memory_utilization` | `0.9` | Fraction of remaining free GPU memory assigned to the KV cache |
| `enforce_eager` | `true` | Must stay true; CUDA Graphs are unsupported |
| `dtype` | `None` | Read dtype from checkpoint metadata; RC measurements use BF16 |

`max_num_batched_tokens` and `max_num_seqs` must be positive;
`max_model_len` must be at least 2 and within the checkpoint context limit.
`gpu_memory_utilization` must be finite and in `(0, 1]`. A dtype override
accepts `candle_core::DType::BF16` or `F16`; the release validation scope
remains BF16. Invalid engine options fail before model resolution.

Do not infer low memory use from the 0.6B parameter count: default cache sizing deliberately uses much of the remaining GPU memory. To leave more free memory for other applications, reduce `gpu_memory_utilization` in the Rust API. Reducing concurrency or prefill token budgets can also reduce workload pressure; no minimum-memory configuration is certified by this release. The CLI uses `EngineOptions::default()` and does not expose these settings.

### Sampling and input constraints

| `SamplingParams` field | Default | Accepted values and effect |
|---|---|---|
| `temperature` | `0.0` | Non-negative, not NaN; `0` selects greedy. Positive infinity is an accepted uniform pre-filter corner case. |
| `top_k` | `None` | `None` or `Some(k)` with `k >= 1`; values at least vocabulary size do nothing |
| `top_p` | `None` | `None` or a finite value in `(0, 1]` |
| `max_tokens` | `16` | At least `1`; counts completion tokens only |
| `ignore_eos` | `false` | Ignore EOS stopping when true; still honor `max_tokens` |
| `presence_penalty` | `0.0` | Finite in `[-2, 2]`; subtract once from the logit of each previously seen token |
| `frequency_penalty` | `0.0` | Finite in `[-2, 2]`; subtract in proportion to the previous token count |
| `repetition_penalty` | `0.0` | Finite and non-negative; `0` is the supported no-op convention |

Penalties apply before token selection. `temperature == 0` or `top_k == Some(1)` selects the greedy path; later top-k/top-p filtering is then inert. This project's sampling representation is not interchangeable with Python vLLM's.

Prompts must tokenize to at least one token. Their token count plus `max_tokens` must fit `max_model_len`; that limit must also fit the checkpoint's context length. Invalid sampling parameters fail before request admission.

## Common errors

| Symptom | What to check |
|---|---|
| “requires one CUDA GPU” | Build with `--features cuda`, verify the driver exposes a GPU to Linux/WSL2 |
| GPU compute capability below the minimum | Use a GPU with compute capability 8.9 or newer |
| `nvcc` or CUDA compilation failure | Install a full CUDA Toolkit and ensure the compiler and CUDA libraries are available to the build |
| Missing checkpoint files | Use a complete local snapshot, or populate the requested Hub revision before enabling offline mode |
| Dtype metadata error | Ensure checkpoint `config.json` declares its dtype; v0.2 no longer silently guesses |
| Context budget exceeded | Reduce prompt/completion length or choose an appropriate `max_model_len` within the model limit |
| Insufficient device memory | Free other allocations and adjust Rust engine budgets; the CLI uses the default cache configuration |

For an issue report, include the source tag/commit, model revision, Rust and CUDA versions, GPU/driver, exact command, and full error. Ask through the [support channels](../SUPPORT.md). The [development guide](development.md) describes local checks; [RC release notes](releases/v0.2.0-rc.1.en.md), [validation](validation/README.md), and [benchmarks](benchmarks/README.md) explain the shipped scope and measurements.

## Upgrade to the final release

For maintained usage and completed release validation, use [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0). This guide documents RC functionality only; it does not incorporate later numerical or telemetry repairs. The [source maintenance record](version-doc-update.md) explains the refreshed historical archive.
