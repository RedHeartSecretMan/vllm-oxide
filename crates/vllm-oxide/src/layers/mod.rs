//! `layers/` — model-agnostic layers (ADR-0001 / ADR-0002 / ADR-0003).
//!
//! Leaves of the dependency DAG — no internal deps.

#![allow(dead_code)]

pub mod activation;
pub mod linear;
pub mod parallel;
pub mod rmsnorm;
pub mod rope;
