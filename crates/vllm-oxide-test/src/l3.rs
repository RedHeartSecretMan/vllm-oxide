//! Legacy L3 entrypoint. No per-layer numerical comparison is implemented here.
//! Requesting this layer returns an error; the placeholder never counts as a pass.

use crate::types::Manifest;
use anyhow::Result;
use std::path::Path;

/// Reserved result type retained for the existing API.
#[derive(Debug, Default)]
pub struct L3Result {
    pub prompt_id: String,
    pub passed: bool,
    pub message: String,
}

/// Reject the unavailable legacy per-layer comparator explicitly.
pub fn compare_l3(_manifest: &Manifest, _fixture_dir: &Path, _prompt_id: &str) -> Result<L3Result> {
    anyhow::bail!("L3 per-layer comparison is not implemented; no comparison was executed")
}
