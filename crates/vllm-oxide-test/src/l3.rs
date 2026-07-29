//! L3 — per-layer activations comparison (debug-only, NOT in CI).
//!
//! This module provides a skeleton for comparing per-layer hidden states
//! and attention outputs between the Rust engine and the golden oracle.
//! It is only compiled and usable when the `--debug` flag is passed.
//!
//! In v0.1 this is a stub — the engine does not yet expose per-layer
//! intermediate tensors. When model introspection lands (v0.2), this
//! module will be wired up.

use std::path::Path;

use anyhow::Result;

use crate::types::Manifest;

/// Result of an L3 per-layer activations comparison.
#[derive(Debug, Default)]
pub struct L3Result {
    pub prompt_id: String,
    pub passed: bool,
    pub message: String,
}

/// L3 comparison is not yet implemented.
///
/// When implemented, this will:
/// 1. Run the Rust model while capturing per-layer hidden states.
/// 2. Load the per-layer golden activations from disk.
/// 3. Compare layer-by-layer to localise where divergence occurs.
pub fn compare_l3(_manifest: &Manifest, _fixture_dir: &Path, _prompt_id: &str) -> Result<L3Result> {
    Ok(L3Result {
        prompt_id: _prompt_id.to_string(),
        passed: true,
        message: "L3 comparison not yet implemented (v0.2)".to_string(),
    })
}
