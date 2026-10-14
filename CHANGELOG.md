# Changelog

## Source maintenance — 2026-10-14

Version-specific maintenance is applied to the three existing release source
snapshots at the user's request. It covers sampling numeric boundaries, model
configuration validation, fixture output confinement, unaligned tensor reads,
early EOS handling, and the research prototype's workspace and safety checks.
The follow-up review additionally covers input/token bounds, Qwen3 configuration
semantics, cached/indexed checkpoint discovery, CLI preflight, degenerate sampling
and fixture publication. Version-specific comparator and CPU workflow fixes are
detailed in the maintenance record. These changes do not relabel original model
acceptance or performance measurements.
Original measurements, budgets and historical verdicts retain their source
identities. See [source maintenance](docs/version-doc-update.md) for scope and
validation records.


[Release history](docs/releases/README.md) · [GitHub Releases](https://github.com/RedHeartSecretMan/vllm-oxide/releases)

Historical implementation entries are retained below. The source tree belongs to experimental RC1; **48/66 numerical checks passed and 18 failed**, while full performance and the declared Rust minimum remained unverified. The [RC release notes](docs/releases/v0.2.0-rc.1.en.md) retain original evidence and asset identities.

## Documentation maintenance after v0.2.0 publication

The initial documentation refresh aligned the bilingual overview, user/developer guides, research/prototype detail, tools/diagnostic instructions and release presentation without changing implementation. Subsequent source fixes are recorded separately above; locks, model inputs, budgets, measurements and original assets retain their original identities. See [the update record](docs/version-doc-update.md). The final [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0) uses its own later implementation and acceptance.

## v0.2.0.rc.1 — 2026-09-10

- Rebuilt the `goldens-v0.2` workflow around pinned model/tokenizer/runtime and
  kernel identities, bit-identical oracle/candidate replay, a sealed four-case
  tolerance holdout, fixed private benchmark telemetry, content-bound stage
  markers, and a separately authorized exact-two-asset publication stage.

### Changed

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

## v0.1.0 — 2026-07-26

### Added

- Core inference engine: scheduler, block pool, KV cache manager, paged attention
- Qwen3 model support via architecture-agnostic model registry (CausalLM trait)
- Continuous batching with recompute-only preemption
- Prefix caching via chained XXH64 hash table (CoW block semantics)
- Sampling: greedy, top-k, top-p (nucleus), repetition/presence/frequency penalties
- CLI (`vllm-oxide-cli`): model source + prompt, stdin support, sampling flags
- Library API: `LLM::new` + `LLM::generate` with per-prompt `SamplingParams`
- Golden comparison harness (`vllm-oxide-test`): L1 token-sequence match, L2 logits comparison
- Python golden fixture generator (`tools/golden-gen/`): transformers + vLLM oracle triangle
- CI pipeline: fmt, clippy (12 lints), test, cargo-deny audit
- Cargo-deny: advisory, license, ban, and source audits (deny.toml)
- Weight loader: HuggingFace sharded safetensors, local directory support
- TP seam: `ParallelStyle` trait + `TpConfig` enum (v0.2 contract, Single-only in v0.1)

### Documentation

- README with architecture overview, quick start, library usage, testing tiers
- CONTEXT.md: domain vocabulary and ubiquitous language
- ADR-0001 through ADR-0004: parametric parallel layers, weight loader seam, model registry + RoPE, engine dependency DAG
- ADR-0005: golden generation correctness strategy
