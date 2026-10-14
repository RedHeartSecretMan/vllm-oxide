# Migrate from v0.1 to v0.2

[Documentation index](README.md) · [Getting started](getting-started.md) · [RC release notes](releases/v0.2.0-rc.1.en.md)

The v0.2 API contraction is already present in RC1. This historical checkout is an experimental prerelease; prefer the final [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0) for new applications. Applications that already construct `LLM` and call `LLM::generate` keep those entrypoints, but must update output identity and respect stricter validation. Applications that imported internal engine or model types must move to the supported generation API.

| Upgrade step | Action |
|---|---|
| Dependency | Pin `v0.2.0`, use Rust 1.94 or newer, and enable `cuda` for inference |
| Imports | Retain only the six public generation types |
| Outputs | Replace `seq_id` with `request_id`; associate results by vector position |
| Inputs | Validate completion lengths, sampling policies, and checkpoint dtype/context metadata |
| Verification | Exercise repeated calls and your application's real prompts on a GPU |

## Pin the release

```toml
[dependencies]
vllm_oxide = { git = "https://github.com/RedHeartSecretMan/vllm-oxide.git", tag = "v0.2.0", features = ["cuda"] }
anyhow = "1"
```

Use Rust 1.94 or newer and build inference applications with the `cuda` feature. CPU-only builds remain available for development and tests, but model inference requires one CUDA GPU. The validated release scope is Qwen3-0.6B BF16; v0.2 does not add tensor parallelism or other model families.

## Use the six supported types

The crate root now exposes exactly:

```rust
use vllm_oxide::{EngineOptions, Prompt, RequestOutput, SamplingParams, Source, LLM};
```

`LLM::new(source, options)` and `LLM::generate(prompts, params)` are the supported operations. Both continue to return `anyhow::Result`. `EngineOptions::dtype` still uses the external `candle_core::DType`; it is not re-exported from `vllm_oxide`.

| v0.1 use | v0.2 migration |
|---|---|
| `EngineCore`, `Scheduler`, `Sequence`, `SequenceGroup`, schedule types | Construct `LLM` and call `generate`; request lifecycle and scheduling are internal |
| `PagedKVCache`, `KvCacheManager`, attention metadata and builders | Let `LLM` manage cache allocation and attention preparation |
| `load_weights`, loader helpers, `CausalLM`, `ModelEntry`, `BuiltModel`, `build_model` | Select the checkpoint through `Source`; model construction is internal |
| `Sampler` | Configure per-request `SamplingParams` |
| Root utility and config helpers | Keep application utilities in your application; use `Source` and `EngineOptions` for model construction |
| `LLM::generate_logits` | No supported public replacement; normal application generation uses `generate` |

There are no compatibility aliases for the removed exports. The `internal-golden` feature is unsupported maintainer validation tooling, not an application raw-logits API. See [ADR-0011](adr/0011-public-generation-contract.md) for the boundary.

## Replace `seq_id` with `request_id`

v0.1 exposed `RequestOutput.seq_id`, which mixed scheduler identity with call-local batch position. v0.2 replaces it with `request_id`:

```rust
// v0.1
println!("{}: {}", output.seq_id, output.text);

// v0.2
println!("{}: {}", output.request_id, output.text);
```

Treat the ID as request identity, not as a prompt-array index. Associate output with the input by vector position:

```rust
let outputs = llm.generate(&prompts, &params)?;
for (prompt, output) in prompts.iter().zip(outputs.iter()) {
    println!("{prompt:?}: {}", output.text);
}
```

The output vector preserves input order and contains one result per accepted prompt. Reusing the same `LLM` across calls no longer relies on internal sequence IDs starting from zero. `RequestOutput` contains `request_id`, `token_ids`, `text`, and `finished`; there is no `seq_id` field.

## Handle validation errors

Supply one complete `SamplingParams` value per prompt. v0.2 validates sampling parameters before admitting requests, so invalid values return an error instead of entering the generation loop.

| Input | v0.2 requirement |
|---|---|
| `max_tokens` | At least `1` |
| `temperature` | Non-negative and not NaN |
| `top_k` | `None` or `Some(k)` with `k >= 1` |
| `top_p` | `None` or finite in `(0, 1]` |
| Presence/frequency penalties | Finite and in `[-2, 2]` |
| Repetition penalty | Finite and non-negative; `0` remains a no-op |
| Prompt | At least one token after tokenization |
| Context budget | Prompt tokens plus `max_tokens` fit `max_model_len` |
| `enforce_eager` | `true`; CUDA Graphs are unsupported |

Default sampling is still greedy. `ignore_eos` ignores EOS stopping but never removes the completion-token limit. See the [sampling guide](getting-started.md#sampling-and-input-constraints) for the complete supported combinations and corner cases.

Each `LLM` resolves one model source at construction, keeping configuration, tokenizer, weights, dtype, and special tokens together. Supply checkpoint dtype metadata or an explicit override; malformed or missing metadata no longer silently falls back to BF16. Use a complete checkpoint directory whose tokenizer, configuration, and weights come from the same revision.

## Check behavior after upgrading

Use the [pinned model example](getting-started.md#use-a-pinned-hub-revision), then exercise your application's prompts, sampling policies, and repeated calls on a GPU. CPU CI checks code and tooling, while CUDA function and numerical checks are separate. The [validation guide](validation/README.md) and [RC result](releases/v0.2.0-rc.1.en.md) describe the RC implementation and its incomplete validation; they do not promise that every hardware/model configuration or application prompt produces identical output to Python vLLM.

Keep your own sampling and model-source settings explicit during the upgrade. The CLI still handles one raw-text prompt per invocation, has no revision flag, and uses default engine options; see its [current limits](getting-started.md#cli). The historical source tag includes explicitly requested documentation and code maintenance; applications pinned to an exact implementation commit retain that source identity. See [the maintenance record](version-doc-update.md).
