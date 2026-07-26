//! `engine/` — V1 three-layer data model (ADR-0004).
//!
//! `Scheduler` / `BlockPool` / `KVCacheManager` / `EngineCore` / `Sequence`.
//! The potential `engine ↔ attention` cycle is broken by the rule: engine
//! holds `Arc<Mutex<PagedKVCache>>` and builds `AttnMetadata` from its own
//! scheduler state; `attention/` never imports `engine/`.
//!
//! # Current status
//!
//! - **T2/#20 (landed)**: `Sequence`, `SequenceGroup`, `SequenceStatus`,
//!   `Block`, `BlockPool`, `KVCacheManager` — the full data-model leaf below
//!   the scheduler.
//! - **T2/#21 (stub)**: `EngineCore`, `Scheduler`, `RequestOutput` — the
//!   control loop that wires data model → attention → model → sampler.
//!
//! `EngineCore` itself collapses V1/nano-vllm `ModelRunner` — `step()`
//! performs scheduler → tensor prep → `model.forward()` → sampler → KV
//! update in one method. No `model_runner` sub-module at v0.1. Split trigger
//! (ADR-0004 R5): `step()` exceeds ~300 LOC or CUDA graph capture lands.

#![allow(dead_code)]

pub mod block_pool;
pub mod kv_cache_manager;
pub mod scheduler;
pub mod sequence;

// Re-export the seam-facing types so lib.rs can lift them to crate-level `pub use`.
// `Block` and `BlockPool` stay crate-internal (ADR-0004: scheduler never names them).
pub use block_pool::BlockPoolError;
pub use kv_cache_manager::KvCacheManager;
pub use sequence::{Sequence, SequenceGroup, SequenceStatus};
