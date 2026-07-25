//! Stub — token-level scheduling + recompute-only preemption (V1 parity).
//!
//! One `schedule()` step is either pure-prefill or pure-decode (nano-vllm
//! parity). Chunked prefill allowed only for the first sequence in a step.
//! Lands in T2.

#![allow(dead_code)]
