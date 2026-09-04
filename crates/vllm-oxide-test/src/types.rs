//! Manifest and fixture data types matching `tools/golden-gen/src/golden_gen/schema.py`.
//!
//! These types are the Rust-side parse targets for `manifest.json` and the
//! `.safetensors` fixture files produced by the Python golden generator.

use serde::{Deserialize, Serialize};

pub const MANIFEST_SCHEMA_VERSION: u32 = 4;
pub const PRODUCT_VERSION: &str = "v0.2.0";
pub const GOLDEN_VERSION: &str = "goldens-v0.2";
pub const MANIFEST_FILENAME: &str = "manifest.json";
pub const ARCHIVE_FILENAME: &str = "goldens-v0.2.tar.gz";
pub const MODEL_ID: &str = "Qwen/Qwen3-0.6B";
pub const MODEL_REVISION: &str = "7e4ae267688d671ddfca3122e4528ee980cf3234";
pub const MODEL_CONFIG_SHA256: &str =
    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd";
pub const TOKENIZER_SHA256: &str =
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";
pub const MODEL_WEIGHTS_SHA256: &str =
    "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b";
pub const REFERENCE_KERNEL_PATH: &str = "transformers-4.57.6/torch-2.10.0/sdpa-math";
pub const BASELINE_KERNEL_PATH: &str = "vllm-0.18.1/flash-attn-v2/eager";
pub const CANDIDATE_KERNEL_PATH: &str =
    "vllm-oxide/candle-27f20fea993c81ea6d32ce44018f42b68466525e/flash-attn-varlen+paged-windowed";

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
    pub runtime: RuntimeInfo,
    pub kernel_paths: KernelPaths,
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
    pub tokenizer_revision: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub weights_sha256: String,
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

/// Software and live-host identity for one release evidence run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeInfo {
    pub evidence_mode: String,
    pub registry_install_mode: String,
    pub pythonhashseed: String,
    pub cublas_workspace_config: String,
    pub python_version: String,
    pub torch_version: String,
    pub torch_cuda_version: String,
    pub transformers_version: String,
    pub vllm_version: String,
    pub xgrammar_version: String,
    pub triton_version: String,
    pub cuda_toolkit_version: String,
    pub rustc_version: String,
    pub nvidia_driver_version: String,
    pub gpu_name: String,
    pub compute_capability: String,
    pub os_kernel: String,
    pub generator_commit: String,
    pub uv_lock_sha256: String,
    pub wheels: Vec<WheelIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WheelIdentity {
    pub name: String,
    pub version: String,
    pub filename: String,
    pub sha256: String,
}

/// Exact reference, baseline, and candidate kernel identities.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelPaths {
    pub reference: String,
    pub baseline: String,
    pub candidate: String,
}

impl KernelPaths {
    pub fn comparison_scope(&self) -> String {
        format!("{}::vs::{}", self.reference, self.candidate)
    }
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
