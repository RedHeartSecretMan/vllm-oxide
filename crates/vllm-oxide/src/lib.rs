//! Synchronous, offline Qwen3 generation on one CUDA GPU.
//!
//! The validated v0.2.0 workload is Qwen3-0.6B/BF16. Inference requires the
//! `cuda` feature, Linux or WSL2, and an NVIDIA GPU with compute capability 8.9
//! or newer. CPU-only builds support development and tests.
//!
//! The supported v0.2.0 interface is the composition root [`LLM`], its
//! constructor [`LLM::new`], and generation through [`LLM::generate`]. The
//! crate root exposes only the six types needed to use that interface:
//! [`LLM`], [`EngineOptions`], [`Prompt`], [`SamplingParams`],
//! [`RequestOutput`], and [`Source`].
//! `LLM::new` and `LLM::generate` retain `anyhow::Result` signatures, and
//! `EngineOptions::dtype` retains `Option<candle_core::DType>`; those are
//! transitive external types, not additional root exports.
//!
//! ```no_run
//! use vllm_oxide::{
//!     EngineOptions, LLM, Prompt, RequestOutput, SamplingParams, Source,
//! };
//!
//! let source = Source::Hub {
//!     repo: "Qwen/Qwen3-0.6B".to_string(),
//!     revision: Some("7e4ae267688d671ddfca3122e4528ee980cf3234".to_string()),
//! };
//! let mut llm = LLM::new(source, EngineOptions::default())?;
//! let outputs: Vec<RequestOutput> = llm
//!     .generate(
//!         &[Prompt::Text("The meaning of life is".to_string())],
//!         &[SamplingParams::default()],
//!     )?;
//! assert_eq!(outputs.len(), 1);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! See the [usage guide](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/docs/getting-started.md)
//! for batching, model sources and configuration, and the
//! [migration guide](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/docs/migration-v0.2.0.md)
//! for upgrading applications from v0.1.
//!
//! Engine, scheduler, cache, attention, loader, registry, model, sampler, and
//! utility mechanics are intentionally internal. These compile-fail examples
//! guard representative paths as an external crate would see them.
//!
//! <details>
//! <summary>Internal API boundary checks</summary>
//!
//! ```compile_fail
//! use vllm_oxide::Scheduler;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::PagedKVCache;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::AttentionContext;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::Sequence;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::ResolvedModel;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::load_weights;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::Sampler;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::CausalLM;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::ModelEntry;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::build_model;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::build_prefill_metadata;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::kv_cache_layout_shape;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::engine::Scheduler;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::golden_capture;
//! ```
//!
//! ```compile_fail
//! use vllm_oxide::{LLM, Prompt};
//!
//! fn raw_logits_are_not_a_supported_method(llm: &mut LLM, prompt: &Prompt) {
//!     let _ = llm.generate_logits(prompt, 1);
//! }
//! ```
//!
//! </details>
//!
//! This file is the **only** module that issues top-level `pub use`
//! (ADR-0004 R4). Internal modules default to `pub(crate)` or stricter;
//! downstream callers should never reach below the re-exports curated here.

pub(crate) mod utils;

#[cfg(feature = "internal-golden")]
mod golden_capture;

#[cfg(any(feature = "internal-golden", test))]
mod benchmark_telemetry;

// Public generation contract (ADR-0011). Keep this list exact: internal
// modules and their implementation types are reachable only within the crate.
pub use config::Source;
pub use engine::scheduler::RequestOutput;
pub use llm::{EngineOptions, Prompt, LLM};
pub use sampler::SamplingParams;

// Internal module DAG per ADR-0004:
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
