//! Stub — `ParallelStyle` trait + `QkvMerged` / `GateUpMerged` / `Row` + `TpConfig`.
//!
//! TP seam: `slice_for_rank(full, rank, world, shard_id) -> Result<Cow<Tensor>>`
//! (v0.1 identity `Cow::Borrowed`, zero-cost; v0.2 per-style w/ GQA replica
//! math for `QkvMerged`). `TpConfig::Single` hardcoded; `TpConfig::Sharded`
//! is the named-but-unbuildable v0.2 contract. Lands in T3.

#![allow(dead_code)]
