//! `vllm_oxide_test` — golden comparison crate (release gate, NOT CI).
//!
//! Validates the Rust inference engine against golden fixtures produced by
//! the Python golden generator (`tools/golden-gen/`). Three comparison
//! levels:
//!
//! - **L1**: greedy token-sequence exact match with near-tie skipping
//! - **L2**: per-step logits tensor comparison with calibrated atol+rtol
//! - **L3**: per-layer activations (debug-only, skeleton in v0.1)
//!
//! This is a **release gate** (manual, GPU). CI green (CPU property tests)
//! does NOT imply numerical validation — see README.

pub mod download;
pub mod l1;
pub mod l2;
pub mod l3;
pub mod manifest;
pub mod prompts;
pub mod report;
pub mod types;

pub use download::{download_release, load_from_dir};
pub use l1::{compare_l1, compare_l1_regression, L1Result};
pub use l2::{compare_l2, L2Result};
pub use l3::{compare_l3, L3Result};
pub use manifest::{load_fixture, parse_manifest};
pub use report::{print_report, ComparisonReport};
pub use types::{FixtureData, FixtureMetadata, Manifest, ToleranceCalibration};
