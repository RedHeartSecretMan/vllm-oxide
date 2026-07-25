//! Stub — `Sequence` / `SequenceGroup` (V1 data model; ADR-0004 M2).
//!
//! `SequenceStatus { Waiting, Running, Finished }`; per-sequence `block_table`,
//! `num_tokens`, `num_cached_tokens`, `num_scheduled_tokens`, `is_prefill`,
//! `last_token`. v0.1 uses 1:1 group:sequence — n>1 sampling deferred.
//! Lands in T2.

#![allow(dead_code)]
