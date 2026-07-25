//! Stub — `inventory::collect!(ModelEntry)` registry + `build()` query.
//!
//! Pure query function — no model-specific knowledge. Keyed off
//! `config.json["architectures"][0]`. Adding a model is one
//! `inventory::submit!` in the model file, no edit here. Lands in T3.

#![allow(dead_code)]
