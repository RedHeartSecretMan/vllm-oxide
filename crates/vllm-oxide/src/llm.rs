//! Stub — `LLM::generate` composition root (ADR-0004).
//!
//! The composition root: the only module that simultaneously imports
//! `engine`, `models::registry`, `loader`, and `sampler`. Port of nano-vllm
//! `llm.py` / `llm_engine.py`. Lands in T2.

#![allow(dead_code)]
