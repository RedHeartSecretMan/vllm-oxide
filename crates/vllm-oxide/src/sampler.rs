//! Stub — `SamplingParams` + `Sampler` (ADR-0004 micro-decision M1).
//!
//! SamplingParams lives here (not in `engine/sequence.rs`) per nano-vllm
//! parity (`sampling_params.py` is top-level). `Sampler` upcasts logits to
//! FP32, applies penalties, scales by temperature (greedy path short-circuits
//! to argmax), then top-k mask, top-p nucleus, sample. Lands in T5.

#![allow(dead_code)]
