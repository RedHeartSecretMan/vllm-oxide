//! Stub — `Linear<P: ParallelStyle>` + `LinearSpec` (ADR-0001 / ADR-0002).
//!
//! One generic struct over a type-level style tag. v0.1 style tags:
//! `QkvMerged`, `GateUpMerged`, `Row`. Plain `Column` deferred (YAGNI).
//! Constructed via `Linear::<P>::from_vb(vb, spec, dev)`. QKV/gate-up
//! fusion happens here, NOT in the loader. Lands in T3.

#![allow(dead_code)]
