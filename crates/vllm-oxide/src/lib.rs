//! `vllm_oxide` — Rust port of nano-vllm trending toward vLLM V1.
//!
//! v0.1 scope: single-GPU, Qwen3-only, offline `LLM::generate` Rust API,
//! paged KV cache (`block_size = 256`), continuous batching, prefix caching,
//! BF16/FP16, recompute-only preemption. TP=1 hardcoded with the
//! `ParallelStyle` seam preserved for v0.2 NCCL wiring.
//!
//! This file is the **only** module that issues top-level `pub use`
//! (ADR-0004 R4). Internal modules default to `pub(crate)` or stricter;
//! downstream callers should never reach below the re-exports curated here.

pub(crate) mod utils;

// Public API surface: per ADR-0004 R4, `lib.rs` is the ONLY module that issues
// top-level `pub use`. Internal modules stay at `pub(crate)` or stricter.
pub use utils::{kv_cache_layout_shape, round_up};
pub use sampler::{Sampler, SamplingParams};

// Module stubs — working code lands in downstream tickets (T2 engine,
// T3 layers/loader, T4 attention, T5 model). Dependency DAG per ADR-0004:
// layers / attention / loader / sampler are leaves; models depends on
// layers + attention + loader; engine does not depend on models; llm is
// the only composition root.
pub(crate) mod attention;
pub(crate) mod config;
pub(crate) mod engine;
pub(crate) mod layers;
pub(crate) mod llm;
pub(crate) mod loader;
pub(crate) mod models;
pub(crate) mod sampler;
