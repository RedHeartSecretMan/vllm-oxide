//! Stub — model-agnostic weight loader (ADR-0002).
//!
//! `load_weights(Source, DType, &Device) -> ShardedVarBuilder`. Lazy mmap
//! via `MmapedSafetensors::multi`. No fusion logic — fusion lives in
//! `Linear::<P>::from_vb`. Lands in T3.

#![allow(dead_code)]
