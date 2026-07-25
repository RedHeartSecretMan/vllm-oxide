//! Stub — `AttnMetadata` (T10 finding honoured).
//!
//! Carries `cu_seqlens_q` / `cu_seqlens_k` (flash-attn cumulative convention)
//! — NOT `context_lens` (vLLM scheduler convention). The conversion lives in
//! a `build_paged_metadata` helper inside the attention backend, NOT in
//! `BlockPool`/`KVCacheManager`. Prefill fields: `cu_seqlens_q/k`,
//! `max_seqlen_q/k`, `slot_mapping`. Decode fields: `context_lens`,
//! `block_table`, `slot_mapping`. Lands in T4.

#![allow(dead_code)]
