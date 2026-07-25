//! `engine/` — V1 three-layer data model (ADR-0004).
//!
//! `Scheduler` / `BlockPool` / `KVCacheManager` / `EngineCore` / `Sequence`.
//! The potential `engine ↔ attention` cycle is broken by the rule: engine
//! holds `Arc<Mutex<PagedKVCache>>` and builds `AttnMetadata` from its own
//! scheduler state; `attention/` never imports `engine/`.
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
