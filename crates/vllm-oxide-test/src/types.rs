//! Manifest and fixture data types matching `tools/golden-gen/src/golden_gen/schema.py`.
//!
//! These types are the Rust-side parse targets for `manifest.json` and the
//! `.safetensors` fixture files produced by the Python golden generator.

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
    #[serde(default)]
    pub cross_validation: Vec<KnownDeviation>,
    pub fixtures: Vec<FixtureMetadata>,
    #[serde(default)]
    pub suspect_prompt_ids: Vec<String>,
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
    pub nanovllm: String,
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
    pub rtol: f64,
    pub observed_max_l2: f64,
    pub calibration_factor: f64,
    pub method: String,
}

/// A known disagreement between two oracles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownDeviation {
    pub pair: (String, String),
    pub prompt_id: String,
    pub max_l2: f64,
    pub argmax_mismatches: u32,
    pub note: String,
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
    Nanovllm,
    #[serde(rename = "vllm_v1")]
    VllmV1,
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
