# Changelog

All notable changes to vllm-oxide will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-07-29

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

[Unreleased]: https://github.com/RedHeartSecretMan/vllm-oxide/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.1.0
