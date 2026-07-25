//! Stub — Scheduler-facing seam over `BlockPool` + `PagedKVCache` (V1 parity).
//!
//! The ONLY Scheduler-facing seam over `BlockPool` + physical `PagedKVCache`.
//! Scheduler never imports `BlockPool` or `attention::PagedKVCache` directly.
//! Owns the mapping from logical block tables to physical block ids and to
//! paged-cache slot indices. Lands in T2.

#![allow(dead_code)]
