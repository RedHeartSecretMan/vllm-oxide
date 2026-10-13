//! Neutral engine-facing model contract — the `CausalLM` trait.
//!
//! Lives outside `models/` (and outside `engine/`) so that `engine/` and
//! `models/` can both depend on it without either depending on the other
//! (ADR-0004 dependency rule). The composition root (`llm.rs`) wires a
//! `Box<dyn CausalLM>` from the registry into `EngineCore`.
//!
//! v0.2 adds new task traits (`SequenceClassifier`, `Embedder`) as sibling
//! traits in this module — not by overloading `CausalLM`.

use candle_core::{Result, Tensor};

pub trait CausalLM: Send + Sync {
    /// Begin a fully validated generation call, before any request is admitted.
    /// Its logical size remains fixed as requests wait or finish.
    fn begin_generation(&mut self, _request_count: usize) {}
    /// Idempotently release call-local state after completion or an execution failure.
    fn end_generation(&mut self) {}
    fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor>;
    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor>;
    fn vocab_size(&self) -> usize;
}
