//! `vllm_oxide_test` — golden comparison crate (release gate, NOT CI).
//!
//! Validates the Rust inference engine against golden fixtures produced by
//! the Python golden generator (`tools/golden-gen/`). Three comparison
//! levels:
//!
//! - **L1**: reference-token match or explicit same-prefix near-tie classification
//! - **L2**: same-prefix logits comparison with a versioned absolute tolerance
//! - **L3**: per-layer activations (debug-only, skeleton in v0.1)
//!
//! This is a **release gate** (manual, GPU). CI green (CPU property tests)
//! does NOT imply numerical validation — see README.

pub mod download;
pub mod driver;
pub mod l1;
pub mod l2;
pub mod l3;
pub mod lifecycle;
pub mod manifest;
pub mod prompts;
pub mod report;
pub mod types;

pub use download::{download_release, load_from_dir};
pub use driver::{run_comparison, DriverOptions};
pub use l1::{compare_l1, compare_l1_tokens_only, L1Result};
pub use l2::{compare_l2, L2Result};
pub use l3::{compare_l3, L3Result};
pub use lifecycle::LifecycleTracker;
pub use manifest::{load_fixture, parse_manifest, parse_manifest_bytes};
pub use report::{print_report, ComparisonReport, LifecycleTotals};
pub use types::{
    ArchiveInfo, BaselineCalibration, ExpectedFixture, FixtureData, FixtureFamily, FixtureMetadata,
    Manifest, OracleRole, RequiredComparison, TolerancePolicy,
};
