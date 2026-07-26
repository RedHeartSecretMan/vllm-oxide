//! `models/` — model implementations (ADR-0003).
//!
//! Each architecture is one module + one `inventory::submit!` self-registration.
//! Adding an architecture is purely additive: new file + `mod xxx;` line here,
//! zero edits to `registry.rs` or any existing model file.

#![allow(dead_code)]

pub mod qwen3;
pub mod registry;

use candle_core::{Device, Result, Tensor};

pub trait CausalLM: Send + Sync {
    fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor>;
    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor>;
    fn vocab_size(&self) -> usize;
    fn device(&self) -> &Device;
}
