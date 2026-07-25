//! `attention/` — T4 paged-attention contract.
//!
//! Model code calls `flash_attn_varlen` (prefill) / `flash_attn_varlen_paged_windowed`
//! (decode) directly — NO `AttentionBackend` trait for v0.1 (YAGNI). The
//! `engine ↔ attention` cycle is broken by `attention/` never importing
//! `engine/`; EngineCore holds `Arc<Mutex<PagedKVCache>>` and builds
//! `AttnMetadata` from scheduler state.

#![allow(dead_code)]

pub mod flash_attn;
pub mod metadata;
