//! Stub — `Qwen3ForCausalLM` (5-layer nesting 1:1 with nano-vllm `qwen3.py`).
//!
//! `ForCausalLM` / `Model` / `DecoderLayer` / `Attention` / `MLP`. QKV and
//! gate/up projections fused into `Linear<QkvMerged>` / `Linear<GateUpMerged>`.
//! `q_norm`/`k_norm` present iff `attention_bias = false`. Tied vs non-tied
//! embeddings is a construction-time branch driven by `tie_word_embeddings`.
//! Lands in T3.

#![allow(dead_code)]
