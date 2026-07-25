//! Stub — RMSNorm with FP32 upcast (parity with nano-vllm `layernorm.py:21-25`).
//!
//! Upcast to FP32 before square-and-mean, cast back. Takes
//! `(hidden, Option<residual>) -> (normed, residual)` (V1 add+norm pattern).
//! Lands in T3.

#![allow(dead_code)]
