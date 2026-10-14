# Changelog

## Source maintenance — 2026-10-14

The three existing release snapshots received version-specific source
maintenance. It covers sampling numeric boundaries, model
configuration validation, fixture output confinement, unaligned tensor reads,
early EOS handling, and the research prototype's workspace and safety checks.
The follow-up review additionally covers input/token bounds, Qwen3 configuration
semantics, cached/indexed checkpoint discovery, CLI preflight, degenerate sampling
and fixture publication. Version-specific comparator and CPU workflow fixes are
detailed in the maintenance record. Original measurements, budgets, and
historical verdicts remain tied to the source that produced them. See [source maintenance](docs/version-doc-update.md) for scope and
validation records.


All notable changes to vllm-oxide will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Documentation maintenance

- Align English and Chinese project introductions, usage, architecture,
  migration, contribution and community guidance with the released v0.2.0 API.
- Add documentation, validation, benchmark, release and ADR navigation, plus
  matching English and Chinese historical release notes.
- Explain research experiments and validation tools with their actual inputs,
  outputs, prerequisites, commands and evidence limits in this source snapshot.
- The initial documentation revision preserved the original implementation,
  dependency lockfiles, GPU measurements, historical failures, and asset
  identities. Later source fixes are listed above; see
  [source documentation revision](docs/releases/version-doc-update.md).

## [0.2.0] - 2026-10-13

Publication status and evidence requirements follow the [release notes](docs/releases/v0.2.0.md).

- Rebuilt the `goldens-v0.2` workflow around pinned model/tokenizer/runtime and
  kernel identities, independent operator checks, same-prefix three-engine
  comparisons, bit-identical replay, and calibrated per-case budgets. Fixed
  benchmark telemetry and content-bound stage markers preserve original
  evidence identities. The v0.2.0 product release retains raw evidence locally.

### Changed

- Rust 1.94 is the minimum supported toolchain; production requires `--features cuda`.
- Qwen3 uses independent projections, reference-ordered CUDA RMS statistics and
  bounded FP32 causal attention over BF16/F16 weights and paged KV storage.
  Batch padding and workspace planning preserve correctness without per-layer
  host mask/index uploads.
- Greedy batch sampling selects all rows in one CUDA launch, and step histories
  share immutable token buffers.
- Release validation records complete CPU checks, public GPU API/CLI smoke,
  and fresh source-bound calibration with strict per-owner resource evidence.
  These checks do not replace the numerical, performance or publication gates.

- **Breaking (v0.2.0):** contracted the default `vllm_oxide` crate root to
  exactly `LLM`, `EngineOptions`, `Prompt`, `SamplingParams`, `RequestOutput`,
  and `Source`. The supported operations on `LLM` are now only `LLM::new` and
  `LLM::generate`; callers should migrate all inference through that
  composition-root interface.
- The GPU golden harness now obtains raw logits through the default-off,
  diagnostic-only `internal-golden` capture protocol while still constructing
  and invoking the engine exclusively through the supported generation
  interface. Capture is inert without explicit process configuration and
  publishes one caller-private, self-validated artifact with atomic
  no-replace semantics.

### Removed

- Removed crate-root access to scheduler and sequence types, block/KV-cache
  types, attention metadata and context, model loader and resolved identity,
  registry/model factories, `CausalLM`, `Sampler`, and internal utility
  helpers. No compatibility aliases remain.
- Removed the public `LLM::generate_logits` diagnostic method and the temporary
  public-field/direct-forward attention compatibility bridge.

`LLM::new` and `LLM::generate` continue to return `anyhow::Result`, and
`EngineOptions::dtype` continues to use `Option<candle_core::DType>`; these are
transitive external signature types, not new crate-root re-exports.

## [0.2.0-rc.1] - 2026-09-10

### Added

- Experimental source prerelease for trying the contracted public generation
  API and CUDA Quick Start before complete numerical/performance acceptance.
- A source-bound verification report and checksum file. Historical numerical
  failures and incomplete checks remain documented in the [RC1 notes](docs/releases/v0.2.0-rc.1.en.md).

The original Git tag is `v0.2.0.rc.1`; the hyphenated heading follows the Cargo
package version. This candidate was superseded by v0.2.0.

## [0.1.0] - 2026-07-26

### Added

- Historical experimental baseline; the original validation limits and two
  fixture assets are described in the [v0.1.0 notes](docs/releases/v0.1.0.en.md).
- Core inference engine: scheduler, block pool, KV cache manager, paged attention
- Qwen3 model support via architecture-agnostic model registry (CausalLM trait)
- Continuous batching with recompute-only preemption
- Prefix caching via chained XXH64 hash table (CoW block semantics)
- Sampling: greedy, top-k, top-p (nucleus), repetition/presence/frequency penalties
- CLI (`vllm-oxide-cli`): model source + prompt, stdin support, sampling flags
- Library API: `LLM::new` + `LLM::generate` with per-prompt `SamplingParams`
- Golden comparison harness (`vllm-oxide-test`): L1 token-sequence match, L2 logits comparison
- Python golden fixture generator (`tools/golden-gen/`): transformers + vLLM oracle triangle
- Weight loader: HuggingFace sharded safetensors, local directory support
- TP seam: `ParallelStyle` trait + `TpConfig` enum (v0.2 contract, Single-only in v0.1)

### Documentation

- CONTEXT.md: domain vocabulary and ubiquitous language
- ADR-0005: golden generation correctness strategy

[Unreleased]: https://github.com/RedHeartSecretMan/vllm-oxide/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/RedHeartSecretMan/vllm-oxide/compare/v0.1.0...v0.2.0
[0.2.0-rc.1]: https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0.rc.1
[0.1.0]: https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.1.0
