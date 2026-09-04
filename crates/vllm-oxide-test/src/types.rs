//! Manifest and fixture data types matching `tools/golden-gen/src/golden_gen/schema.py`.
//!
//! These types are the Rust-side parse targets for `manifest.json` and the
//! `.safetensors` fixture files produced by the Python golden generator.

use serde::{Deserialize, Serialize};

/// Top-level manifest describing a set of golden fixtures.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub product_version: String,
    pub golden_version: String,
    pub archive: ArchiveInfo,
    pub generated_at: String,
    pub model: ModelInfo,
    pub oracle_versions: OracleVersions,
    pub generation: GenerationConfig,
    /// Versioned acceptance inputs consumed by the reference comparator.
    pub tolerance_policy: TolerancePolicy,
    /// Baseline-oracle calibration observations; never a correctness oracle.
    pub baseline_calibration: BaselineCalibration,
    pub expected_fixtures: Vec<ExpectedFixture>,
    pub fixtures: Vec<FixtureMetadata>,
    /// Baseline fixture identifiers successfully consumed by calibration.
    #[serde(default)]
    pub calibrated_fixtures: Vec<String>,
}

/// Identity of the sole compressed fixture archive release asset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArchiveInfo {
    pub filename: String,
    pub sha256: String,
}

/// Contract for one oracle artifact required by the release manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedFixture {
    pub fixture_id: String,
    pub prompt_id: String,
    pub family: FixtureFamily,
    pub model_revision: String,
    pub dtype: String,
    pub oracle: OracleName,
    pub oracle_role: OracleRole,
    pub required_comparison: RequiredComparison,
    pub filename: String,
}

/// Provenance of the model used to generate goldens.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInfo {
    pub id: String,
    pub revision: String,
    pub arch: String,
    pub dtype: String,
    pub vocab_size: usize,
}

/// Versions of the oracle engines used during generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleVersions {
    pub transformers: String,
    pub vllm: String,
}

/// Parameters used during golden generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    pub canonical_max_tokens: u32,
    pub regression_max_tokens: u32,
    pub temperature: f64,
    pub attn_implementation: String,
}

/// Baseline-oracle calibration observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineCalibration {
    pub candidate_atol: f64,
    pub observed_max_abs_diff: f64,
    pub calibration_factor: f64,
    pub method: String,
}

/// Explicit, versioned mathematical policy for reference-oracle comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TolerancePolicy {
    pub version: String,
    pub dtype: String,
    pub kernel: String,
    pub l1_near_tie_max_abs_logit_gap: f64,
    pub l2_atol: f64,
    pub rationale: String,
    pub evidence: Vec<String>,
}

/// Metadata for a single fixture file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum PromptCategory {
    Canonical,
    Regression,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum FixtureFamily {
    Canonical,
    Batch,
    Regression,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum OracleName {
    Transformers,
    Vllm,
    Fake,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum OracleRole {
    Reference,
    Baseline,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RequiredComparison {
    L1,
    L1L2,
    Calibration,
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
    pub fn model_vocab_size(&self, logits: &[f32]) -> usize {
        if self.logits_shape.1 > 0 {
            self.logits_shape.1
        } else if self.logits_shape.0 > 0 && !logits.is_empty() {
            logits.len() / self.logits_shape.0
        } else {
            0
        }
    }
}
