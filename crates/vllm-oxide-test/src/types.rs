//! Manifest and fixture data types matching `tools/golden-gen/src/golden_gen/schema.py`.
//!
//! These types are the Rust-side parse targets for `manifest.json` and the
//! `.safetensors` fixture files produced by the Python golden generator.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Top-level manifest describing a set of golden fixtures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub generated_at: String,
    pub model: ModelInfo,
    pub oracle_versions: OracleVersions,
    pub generation: GenerationConfig,
    pub tolerance: ToleranceCalibration,
    pub fixtures: Vec<FixtureMetadata>,
    /// Mapping from prompt_id to a list of token positions where vLLM (the
    /// reference BF16 engine) disagrees with transformers. These positions are
    /// skipped during L1 regression comparison.
    #[serde(default)]
    pub regression_skip_map: HashMap<String, Vec<usize>>,
}

/// Provenance of the model used to generate goldens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub revision: String,
    pub arch: String,
    pub dtype: String,
    pub vocab_size: usize,
}

/// Versions of the oracle engines used during generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleVersions {
    pub transformers: String,
    pub vllm: String,
}

/// Parameters used during golden generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub canonical_max_tokens: u32,
    pub regression_max_tokens: u32,
    pub temperature: f64,
    pub attn_implementation: String,
}

/// Calibrated tolerances from oracle cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceCalibration {
    pub atol: f64,
    pub observed_max_abs_diff: f64,
    pub calibration_factor: f64,
    pub method: String,
}

/// Metadata for a single fixture file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureMetadata {
    pub prompt_id: String,
    pub category: PromptCategory,
    pub oracle: OracleName,
    pub num_tokens: u32,
    pub logits_dtype: LogitsDtype,
    pub logits_shape: (usize, usize),
    pub sha256: String,
    pub filename: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PromptCategory {
    Canonical,
    Regression,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OracleName {
    Transformers,
    Vllm,
    Fake,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogitsDtype {
    Float32,
}

/// Loaded fixture data from a `.safetensors` file.
#[derive(Debug, Clone)]
pub struct FixtureData {
    pub prompt_id: String,
    pub category: PromptCategory,
    pub oracle: OracleName,
    pub num_tokens: u32,
    /// Generated token IDs, shape `[n]`.
    pub token_ids: Vec<i64>,
    /// Number of prompt tokens (scalar).
    pub n_prompt_tokens: i64,
    /// Full logits tensor, shape `[n, vocab_size]` (canonical only, None for regression).
    pub logits: Option<Vec<f32>>,
    pub logits_shape: (usize, usize),
    /// Top-5 indices, shape `[n, 5]` (regression only, None for canonical).
    pub top5_indices: Option<Vec<i64>>,
    /// Top-5 logits, shape `[n, 5]` (regression only, None for canonical).
    pub top5_logits: Option<Vec<f32>>,
}

impl FixtureData {
    pub fn model_vocab_size(&self) -> usize {
        if self.logits_shape.1 > 0 {
            self.logits_shape.1
        } else {
            0
        }
    }
}
