//! Fail-closed fixture lifecycle accounting.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use crate::manifest::{load_fixture, sha256_hex};
use crate::prompts::PromptEntry;
use crate::report::LifecycleTotals;
use crate::types::{
    ExpectedFixture, FixtureData, FixtureFamily, FixtureMetadata, Manifest, OracleRole,
    PromptCategory,
};

/// One validated reference-oracle case ready for GPU comparison.
#[derive(Debug, Clone)]
pub struct ReferenceCase {
    pub expected: ExpectedFixture,
    pub metadata: FixtureMetadata,
    pub fixture: FixtureData,
    pub prompt: PromptEntry,
}

/// CPU preflight result consumed by the GPU comparison driver.
#[derive(Debug, Clone)]
pub struct LifecyclePreflight {
    pub tracker: LifecycleTracker,
    pub reference_cases: Vec<ReferenceCase>,
    pub errors: Vec<String>,
}

/// Match manifest expectations to concrete prompt cases before GPU work starts.
pub fn preflight(
    manifest: &Manifest,
    fixture_dir: &Path,
    prompts: &HashMap<String, PromptEntry>,
) -> LifecyclePreflight {
    let mut tracker = LifecycleTracker::new(
        manifest
            .expected_fixtures
            .iter()
            .map(|expected| expected.fixture_id.clone()),
    );
    let mut errors = Vec::new();
    let expected_ids: HashSet<_> = manifest
        .expected_fixtures
        .iter()
        .map(|expected| expected.fixture_id.as_str())
        .collect();
    let expected_filenames: HashSet<_> = manifest
        .expected_fixtures
        .iter()
        .map(|expected| expected.filename.as_str())
        .collect();
    for expected in &manifest.expected_fixtures {
        if let Some(prompt) = prompts.get(&expected.prompt_id) {
            tracker.record_discovered(&expected.fixture_id);
            if prompt.family != expected.family {
                tracker.record_failed(&expected.fixture_id);
                errors.push(format!(
                    "fixture {} family {:?} does not match discovered {:?}",
                    expected.fixture_id, expected.family, prompt.family
                ));
            }
        }
    }
    for prompt in prompts.values() {
        for oracle in ["transformers", "vllm"] {
            let fixture_id = format!("{}.{}", prompt.id, oracle);
            if !expected_ids.contains(fixture_id.as_str()) {
                tracker.record_discovered(&fixture_id);
            }
        }
    }

    match std::fs::read_dir(fixture_dir) {
        Ok(entries) => {
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        tracker.record_all_failed();
                        errors.push(format!("reading fixture directory entry: {error}"));
                        continue;
                    }
                };
                let path = entry.path();
                if path.extension() != Some(std::ffi::OsStr::new("safetensors")) {
                    continue;
                }
                let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
                    tracker.record_all_failed();
                    errors.push(format!(
                        "fixture path is not valid UTF-8: {}",
                        path.to_string_lossy()
                    ));
                    continue;
                };
                if !expected_filenames.contains(filename) {
                    tracker.record_generated(filename);
                }
            }
        }
        Err(error) => {
            for expected in &manifest.expected_fixtures {
                tracker.record_failed(&expected.fixture_id);
            }
            errors.push(format!(
                "reading fixture directory {}: {error}",
                fixture_dir.display()
            ));
        }
    }

    let mut reference_cases = Vec::new();
    for metadata in &manifest.fixtures {
        let Some(expected) = manifest.expected_fixtures.iter().find(|expected| {
            expected.prompt_id == metadata.prompt_id
                && expected.oracle == metadata.oracle
                && expected.filename == metadata.filename
        }) else {
            continue;
        };
        let fixture_id = &expected.fixture_id;
        let path = fixture_dir.join(&metadata.filename);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                errors.push(format!("reading fixture {}: {error}", path.display()));
                continue;
            }
        };
        tracker.record_generated(fixture_id);
        if sha256_hex(&bytes) != metadata.sha256 {
            tracker.record_failed(fixture_id);
            errors.push(format!("fixture {fixture_id} checksum mismatch"));
            continue;
        }
        let fixture = match load_fixture(&path, metadata) {
            Ok(fixture) => fixture,
            Err(error) => {
                tracker.record_failed(fixture_id);
                errors.push(format!("fixture {fixture_id}: {error:#}"));
                continue;
            }
        };
        if let Err(error) = validate_fixture_values(&fixture, manifest.model.vocab_size) {
            tracker.record_failed(fixture_id);
            errors.push(format!("fixture {fixture_id}: {error}"));
            continue;
        }
        let category_matches = match expected.family {
            FixtureFamily::Canonical | FixtureFamily::Batch => {
                metadata.category == PromptCategory::Canonical
                    && metadata.logits_shape.1 == manifest.model.vocab_size
            }
            FixtureFamily::Regression => {
                metadata.category == PromptCategory::Regression && metadata.logits_shape == (0, 0)
            }
        };
        if !category_matches {
            tracker.record_failed(fixture_id);
            errors.push(format!("fixture {fixture_id} has unsupported family shape"));
            continue;
        }
        if expected.oracle_role == OracleRole::Reference {
            if let Some(prompt) = prompts.get(&expected.prompt_id) {
                if prompt.family == expected.family {
                    reference_cases.push(ReferenceCase {
                        expected: expected.clone(),
                        metadata: metadata.clone(),
                        fixture,
                        prompt: prompt.clone(),
                    });
                }
            }
        }
    }

    for fixture_id in &manifest.calibrated_fixtures {
        let Some(expected) = manifest
            .expected_fixtures
            .iter()
            .find(|expected| expected.fixture_id == *fixture_id)
        else {
            tracker.record_compared(fixture_id);
            continue;
        };
        if expected.oracle_role != OracleRole::Baseline
            || !tracker.was_generated(fixture_id)
            || tracker.was_failed(fixture_id)
        {
            tracker.record_failed(fixture_id);
            errors.push(format!("invalid calibration evidence for {fixture_id}"));
        } else {
            tracker.record_compared(fixture_id);
        }
    }

    LifecyclePreflight {
        tracker,
        reference_cases,
        errors,
    }
}

fn validate_fixture_values(fixture: &FixtureData, vocab_size: usize) -> Result<(), &'static str> {
    if fixture.n_prompt_tokens <= 0 {
        return Err("n_prompt_tokens must be positive");
    }
    if fixture
        .token_ids
        .iter()
        .any(|token_id| !token_is_in_vocab(*token_id, vocab_size))
    {
        return Err("token id outside model vocabulary");
    }
    if fixture.top5_indices.as_ref().is_some_and(|indices| {
        indices
            .iter()
            .any(|token_id| !token_is_in_vocab(*token_id, vocab_size))
    }) {
        return Err("top-5 token id outside model vocabulary");
    }
    Ok(())
}

fn token_is_in_vocab(token_id: i64, vocab_size: usize) -> bool {
    usize::try_from(token_id).is_ok_and(|token_id| token_id < vocab_size)
}

/// Tracks each expected fixture through discovery, generation, and comparison.
#[derive(Debug, Clone)]
pub struct LifecycleTracker {
    expected: HashSet<String>,
    discovered: HashSet<String>,
    generated: HashSet<String>,
    compared: HashSet<String>,
    skipped: HashSet<String>,
    failed: HashSet<String>,
    unexpected: HashSet<String>,
}

impl LifecycleTracker {
    pub fn new(expected_ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            expected: expected_ids.into_iter().collect(),
            discovered: HashSet::new(),
            generated: HashSet::new(),
            compared: HashSet::new(),
            skipped: HashSet::new(),
            failed: HashSet::new(),
            unexpected: HashSet::new(),
        }
    }

    pub fn record_discovered(&mut self, fixture_id: &str) {
        Self::record_stage(
            &self.expected,
            &mut self.discovered,
            &mut self.unexpected,
            fixture_id,
        );
    }

    pub fn record_generated(&mut self, fixture_id: &str) {
        Self::record_stage(
            &self.expected,
            &mut self.generated,
            &mut self.unexpected,
            fixture_id,
        );
    }

    pub fn record_compared(&mut self, fixture_id: &str) {
        Self::record_stage(
            &self.expected,
            &mut self.compared,
            &mut self.unexpected,
            fixture_id,
        );
    }

    pub fn record_skipped(&mut self, fixture_id: &str) {
        Self::record_stage(
            &self.expected,
            &mut self.skipped,
            &mut self.unexpected,
            fixture_id,
        );
    }

    pub fn record_failed(&mut self, fixture_id: &str) {
        Self::record_stage(
            &self.expected,
            &mut self.failed,
            &mut self.unexpected,
            fixture_id,
        );
    }

    pub fn totals(&self) -> LifecycleTotals {
        let missing = self
            .expected
            .iter()
            .filter(|fixture_id| {
                !self.discovered.contains(*fixture_id) || !self.generated.contains(*fixture_id)
            })
            .count();
        LifecycleTotals {
            expected: self.expected.len(),
            discovered: self.discovered.len(),
            generated: self.generated.len(),
            compared: self.compared.len(),
            missing,
            unexpected: self.unexpected.len(),
            skipped: self.skipped.len(),
            failed: self.failed.len(),
        }
    }

    pub fn was_generated(&self, fixture_id: &str) -> bool {
        self.generated.contains(fixture_id)
    }

    pub fn was_failed(&self, fixture_id: &str) -> bool {
        self.failed.contains(fixture_id)
    }

    pub fn was_skipped(&self, fixture_id: &str) -> bool {
        self.skipped.contains(fixture_id)
    }

    fn record_all_failed(&mut self) {
        self.failed.extend(self.expected.iter().cloned());
    }

    fn record_stage(
        expected: &HashSet<String>,
        stage: &mut HashSet<String>,
        unexpected: &mut HashSet<String>,
        fixture_id: &str,
    ) {
        if expected.contains(fixture_id) {
            stage.insert(fixture_id.to_string());
        } else {
            unexpected.insert(fixture_id.to_string());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::preflight;
    use crate::driver::{run_prepared_comparisons, CaseComparison, DriverOptions};
    use crate::l1::L1Result;
    use crate::l2::L2Result;
    use crate::manifest::sha256_hex;
    use crate::prompts::PromptEntry;
    use crate::types::{
        ExpectedFixture, FixtureFamily, FixtureMetadata, LogitsDtype, Manifest, OracleName,
        OracleRole, PromptCategory, RequiredComparison,
    };

    fn test_manifest() -> Manifest {
        serde_json::from_value(json!({
            "schema_version": 3,
            "generated_at": "2026-09-04T00:00:00Z",
            "model": {
                "id": "model", "revision": "rev", "arch": "arch",
                "dtype": "bfloat16", "vocab_size": 3
            },
            "oracle_versions": {"transformers": "5", "vllm": "0.26"},
            "generation": {
                "canonical_max_tokens": 1, "regression_max_tokens": 1,
                "temperature": 0.0, "attn_implementation": "sdpa"
            },
            "tolerance": {
                "atol": 0.01, "observed_max_abs_diff": 0.005,
                "calibration_factor": 2.0, "method": "test"
            },
            "comparison_policy": {
                "version": "same-prefix-v1",
                "l1_near_tie_max_abs_logit_gap": 0.02,
                "l2_atol": 0.01
            },
            "expected_fixtures": [
                {
                    "fixture_id": "canonical_01.transformers",
                    "prompt_id": "canonical_01", "family": "canonical",
                    "model_revision": "rev", "dtype": "bfloat16",
                    "oracle": "transformers", "oracle_role": "reference",
                    "required_comparison": "l1_l2",
                    "filename": "canonical_01.transformers.safetensors"
                },
                {
                    "fixture_id": "canonical_01.vllm",
                    "prompt_id": "canonical_01", "family": "canonical",
                    "model_revision": "rev", "dtype": "bfloat16",
                    "oracle": "vllm", "oracle_role": "baseline",
                    "required_comparison": "calibration",
                    "filename": "canonical_01.vllm.safetensors"
                }
            ],
            "fixtures": [],
            "calibrated_fixtures": []
        }))
        .unwrap()
    }

    fn test_prompts() -> HashMap<String, PromptEntry> {
        HashMap::from([(
            "canonical_01".to_string(),
            PromptEntry {
                id: "canonical_01".to_string(),
                family: FixtureFamily::Canonical,
                prompt: "hello".to_string(),
            },
        )])
    }

    fn write_canonical_fixture(path: &std::path::Path) -> String {
        write_canonical_fixture_with_token(path, 1)
    }

    fn write_canonical_fixture_with_token(path: &std::path::Path, token_id: i64) -> String {
        let token_bytes = token_id.to_ne_bytes();
        let prompt_bytes = 1_i64.to_ne_bytes();
        let logits_bytes: Vec<u8> = [0.0_f32, 1.0, 0.5]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect();
        let tensors = vec![
            (
                "token_ids".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::I64,
                    vec![1],
                    &token_bytes,
                )
                .unwrap(),
            ),
            (
                "n_prompt_tokens".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::I64,
                    vec![],
                    &prompt_bytes,
                )
                .unwrap(),
            ),
            (
                "logits".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 3],
                    &logits_bytes,
                )
                .unwrap(),
            ),
        ];
        safetensors::tensor::serialize_to_file(tensors, &None, path).unwrap();
        sha256_hex(&std::fs::read(path).unwrap())
    }

    fn canonical_metadata(
        prompt_id: &str,
        oracle: OracleName,
        filename: &str,
        sha256: String,
    ) -> FixtureMetadata {
        FixtureMetadata {
            prompt_id: prompt_id.to_string(),
            category: PromptCategory::Canonical,
            oracle,
            num_tokens: 1,
            logits_dtype: LogitsDtype::Float32,
            logits_shape: (1, 3),
            sha256,
            filename: filename.to_string(),
        }
    }

    fn write_regression_fixture(path: &std::path::Path) -> String {
        let token_bytes = 1_i64.to_ne_bytes();
        let prompt_bytes = 1_i64.to_ne_bytes();
        let indices_bytes: Vec<u8> = [0_i64, 1, 2, 0, 1]
            .into_iter()
            .flat_map(i64::to_ne_bytes)
            .collect();
        let logits_bytes: Vec<u8> = [1.0_f32, 0.8, 0.6, 0.4, 0.2]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect();
        let tensors = vec![
            (
                "token_ids".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::I64,
                    vec![1],
                    &token_bytes,
                )
                .unwrap(),
            ),
            (
                "n_prompt_tokens".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::I64,
                    vec![],
                    &prompt_bytes,
                )
                .unwrap(),
            ),
            (
                "top5_indices".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::I64,
                    vec![1, 5],
                    &indices_bytes,
                )
                .unwrap(),
            ),
            (
                "top5_logits".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 5],
                    &logits_bytes,
                )
                .unwrap(),
            ),
        ];
        safetensors::tensor::serialize_to_file(tensors, &None, path).unwrap();
        sha256_hex(&std::fs::read(path).unwrap())
    }

    fn regression_metadata(oracle: OracleName, filename: &str, sha256: String) -> FixtureMetadata {
        FixtureMetadata {
            prompt_id: "regression_01".to_string(),
            category: PromptCategory::Regression,
            oracle,
            num_tokens: 1,
            logits_dtype: LogitsDtype::Float32,
            logits_shape: (0, 0),
            sha256,
            filename: filename.to_string(),
        }
    }

    fn add_expected_pair(manifest: &mut Manifest, prompt_id: &str, family: FixtureFamily) {
        for (oracle, oracle_name, role, comparison) in [
            (
                OracleName::Transformers,
                "transformers",
                OracleRole::Reference,
                if family == FixtureFamily::Regression {
                    RequiredComparison::L1
                } else {
                    RequiredComparison::L1L2
                },
            ),
            (
                OracleName::Vllm,
                "vllm",
                OracleRole::Baseline,
                RequiredComparison::Calibration,
            ),
        ] {
            let fixture_id = format!("{prompt_id}.{oracle_name}");
            manifest.expected_fixtures.push(ExpectedFixture {
                fixture_id: fixture_id.clone(),
                prompt_id: prompt_id.to_string(),
                family: family.clone(),
                model_revision: "rev".to_string(),
                dtype: "bfloat16".to_string(),
                oracle,
                oracle_role: role,
                required_comparison: comparison,
                filename: format!("{fixture_id}.safetensors"),
            });
        }
    }

    #[test]
    fn preflight_reports_discovered_but_missing_generated_fixtures() {
        let manifest = test_manifest();
        let prompts = test_prompts();
        let fixtures = tempfile::tempdir().unwrap();

        let result = preflight(&manifest, fixtures.path(), &prompts);
        let totals = result.tracker.totals();

        assert_eq!(totals.expected, 2);
        assert_eq!(totals.discovered, 2);
        assert_eq!(totals.generated, 0);
        assert_eq!(totals.missing, 2);
        assert!(!totals.release_passed());
    }

    #[test]
    fn successful_lifecycle_dispatches_all_families_through_fake_adapter() {
        let fixtures = tempfile::tempdir().unwrap();
        let mut manifest = test_manifest();
        add_expected_pair(&mut manifest, "batch_01a", FixtureFamily::Batch);
        add_expected_pair(&mut manifest, "regression_01", FixtureFamily::Regression);
        let mut prompts = test_prompts();
        prompts.insert(
            "batch_01a".to_string(),
            PromptEntry {
                id: "batch_01a".to_string(),
                family: FixtureFamily::Batch,
                prompt: "batch".to_string(),
            },
        );
        prompts.insert(
            "regression_01".to_string(),
            PromptEntry {
                id: "regression_01".to_string(),
                family: FixtureFamily::Regression,
                prompt: "regression".to_string(),
            },
        );

        for (prompt_id, family) in [
            ("canonical_01", FixtureFamily::Canonical),
            ("batch_01a", FixtureFamily::Batch),
            ("regression_01", FixtureFamily::Regression),
        ] {
            for (oracle, oracle_name) in [
                (OracleName::Transformers, "transformers"),
                (OracleName::Vllm, "vllm"),
            ] {
                let filename = format!("{prompt_id}.{oracle_name}.safetensors");
                let path = fixtures.path().join(&filename);
                let sha256 = if family == FixtureFamily::Regression {
                    write_regression_fixture(&path)
                } else {
                    write_canonical_fixture(&path)
                };
                let is_baseline = oracle == OracleName::Vllm;
                let metadata = if family == FixtureFamily::Regression {
                    regression_metadata(oracle, &filename, sha256)
                } else {
                    canonical_metadata(prompt_id, oracle, &filename, sha256)
                };
                manifest.fixtures.push(metadata);
                if is_baseline {
                    manifest
                        .calibrated_fixtures
                        .push(format!("{prompt_id}.vllm"));
                }
            }
        }

        let prepared = preflight(&manifest, fixtures.path(), &prompts);
        assert!(prepared.errors.is_empty(), "{:?}", prepared.errors);
        assert_eq!(prepared.reference_cases.len(), 3);
        let options = DriverOptions {
            l1_only: false,
            l2_only: false,
            debug: false,
        };
        let report = run_prepared_comparisons(prepared, &options, |case| {
            let l1 = L1Result {
                prompt_id: case.expected.prompt_id.clone(),
                passed: true,
                total_positions: 1,
                compared_positions: 1,
                exact_matches: 1,
                near_ties: 0,
                mismatches: 0,
                first_divergence: None,
                excluded_positions: 0,
                policy_version: "same-prefix-v1".to_string(),
                near_tie_max_abs_logit_gap: 0.02,
                details: Vec::new(),
            };
            let l2 =
                (case.expected.required_comparison == RequiredComparison::L1L2).then(|| L2Result {
                    prompt_id: case.expected.prompt_id.clone(),
                    passed: true,
                    compared_steps: 1,
                    first_divergence: None,
                    excluded_steps: 0,
                    total_elements: 3,
                    max_abs_diff: 0.0,
                    elements_exceeding_tol: 0,
                });
            Ok(CaseComparison {
                l1: Some(l1),
                l2,
                l3: None,
            })
        });

        assert_eq!(report.lifecycle.expected, 6);
        assert_eq!(report.lifecycle.discovered, 6);
        assert_eq!(report.lifecycle.generated, 6);
        assert_eq!(report.lifecycle.compared, 6);
        assert_eq!(report.lifecycle.skipped, 0);
        assert_eq!(report.lifecycle.failed, 0);
        assert_eq!(report.l1_results.len(), 3);
        assert_eq!(report.l2_results.len(), 2);
        assert!(report.overall_passed());
    }

    #[test]
    fn preflight_reports_checksum_failure_and_unexpected_asset() {
        let fixtures = tempfile::tempdir().unwrap();
        let reference_filename = "canonical_01.transformers.safetensors";
        let baseline_filename = "canonical_01.vllm.safetensors";
        write_canonical_fixture(&fixtures.path().join(reference_filename));
        let baseline_sha = write_canonical_fixture(&fixtures.path().join(baseline_filename));
        write_canonical_fixture(&fixtures.path().join("unexpected.safetensors"));
        let mut manifest = test_manifest();
        manifest.fixtures = vec![
            canonical_metadata(
                "canonical_01",
                OracleName::Transformers,
                reference_filename,
                "wrong-checksum".to_string(),
            ),
            canonical_metadata(
                "canonical_01",
                OracleName::Vllm,
                baseline_filename,
                baseline_sha,
            ),
        ];
        manifest.calibrated_fixtures = vec!["canonical_01.vllm".to_string()];

        let result = preflight(&manifest, fixtures.path(), &test_prompts());
        let totals = result.tracker.totals();

        assert_eq!(totals.generated, 2);
        assert_eq!(totals.unexpected, 1);
        assert_eq!(totals.failed, 1);
        assert!(!totals.release_passed());
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("checksum mismatch")));
    }

    #[test]
    fn preflight_rejects_fixture_values_outside_model_contract() {
        let fixtures = tempfile::tempdir().unwrap();
        let reference_filename = "canonical_01.transformers.safetensors";
        let baseline_filename = "canonical_01.vllm.safetensors";
        let reference_sha =
            write_canonical_fixture_with_token(&fixtures.path().join(reference_filename), -1);
        let baseline_sha = write_canonical_fixture(&fixtures.path().join(baseline_filename));
        let mut manifest = test_manifest();
        manifest.fixtures = vec![
            canonical_metadata(
                "canonical_01",
                OracleName::Transformers,
                reference_filename,
                reference_sha,
            ),
            canonical_metadata(
                "canonical_01",
                OracleName::Vllm,
                baseline_filename,
                baseline_sha,
            ),
        ];
        manifest.calibrated_fixtures = vec!["canonical_01.vllm".to_string()];

        let result = preflight(&manifest, fixtures.path(), &test_prompts());
        let totals = result.tracker.totals();

        assert_eq!(totals.generated, 2);
        assert_eq!(totals.failed, 1);
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("token id outside model vocabulary")));
    }

    #[cfg(unix)]
    #[test]
    fn preflight_fails_closed_for_non_utf8_fixture_filename() {
        use std::os::unix::ffi::OsStringExt;

        let fixtures = tempfile::tempdir().unwrap();
        let mut filename = vec![0xff];
        filename.extend_from_slice(b".safetensors");
        std::fs::write(
            fixtures.path().join(std::ffi::OsString::from_vec(filename)),
            b"bad",
        )
        .unwrap();

        let result = preflight(&test_manifest(), fixtures.path(), &test_prompts());
        let totals = result.tracker.totals();

        assert_eq!(totals.failed, 2);
        assert!(!totals.release_passed());
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("not valid UTF-8")));
    }
}
