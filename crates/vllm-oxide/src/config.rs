//! Stub — `Source` enum + HF dtype resolution + `HF_HUB_OFFLINE` shim (ADR-0002).
//!
//! Default dtype reads `config.json`'s `torch_dtype` (fixes nano-vllm's
//! `hf_config.dtype` latent bug). User-overridable via the `dtype` parameter.
//! Lands in T3.

#![allow(dead_code)]
