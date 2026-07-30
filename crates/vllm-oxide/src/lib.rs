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
pub use sampler::{Sampler, SamplingParams};
pub use utils::{kv_cache_layout_shape, round_up};

pub use attention::{
    build_decode_metadata, build_prefill_metadata, AttentionContext, AttnMetadata, PagedKVCache,
};

// T2/#20 — engine data model: Sequence (carries its own request_id), BlockPool,
// KVCacheManager. The former 1:1 SequenceGroup wrapper has been absorbed.
// T2/#21 — Scheduler + EngineCore: the live control loop.
// The scheduler imports only KvCacheManager (and Sequence) — never BlockPool
// or PagedKVCache directly (ADR-0004 seam contract).
pub use engine::{
    BlockPoolError, EngineCore, KvCacheManager, RequestOutput, ScheduleMode, ScheduleOutput,
    Scheduler, Sequence, SequenceStatus,
};

// T15 — weight loader + config (ADR-0002). Loader is model-agnostic: returns
// a candle `ShardedVarBuilder`; all fusion (q/k/v, gate/up) lives in
// `Linear::<P>::from_vb` (T3, lands later).
pub use config::{
    default_dtype, default_dtype_from_config_json, is_hf_hub_offline, is_offline_value, HFConfig,
    Source, HF_HUB_OFFLINE_ENV,
};
pub use loader::{load_weights, load_weights_vb};

pub use causal_lm::CausalLM;
pub use models::registry::{build as build_model, BuiltModel, ModelEntry};

pub use llm::{EngineOptions, Prompt, LLM};

// Module stubs — working code lands in downstream tickets (T2 engine,
// T3 layers/loader, T4 attention, T5 model). Dependency DAG per ADR-0004:
// layers / attention / loader / sampler are leaves; models depends on
// layers + attention + loader; engine does not depend on models; llm is
// the only composition root.
pub(crate) mod attention;
pub(crate) mod causal_lm;
pub(crate) mod config;
pub(crate) mod engine;
pub(crate) mod layers;
pub(crate) mod llm;
pub(crate) mod loader;
pub(crate) mod models;
pub(crate) mod sampler;
