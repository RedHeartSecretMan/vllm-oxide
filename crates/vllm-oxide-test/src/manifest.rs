use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;

use crate::types::{
    FixtureData, FixtureFamily, FixtureMetadata, Manifest, OracleName, OracleRole, PromptCategory,
    RequiredComparison, ARCHIVE_FILENAME, GOLDEN_VERSION, MANIFEST_SCHEMA_VERSION, PRODUCT_VERSION,
};

/// Parse a `manifest.json` file.
pub fn parse_manifest(path: &Path) -> Result<Manifest> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading manifest from {}", path.display()))?;
    parse_manifest_bytes(content.as_bytes(), &path.display().to_string())
}

/// Parse and validate manifest bytes received from a non-filesystem source.
pub fn parse_manifest_bytes(content: &[u8], source: &str) -> Result<Manifest> {
    let manifest: Manifest = serde_json::from_slice(content)
        .map_err(|error| anyhow::anyhow!("parsing manifest from {source}: {error}"))?;
    validate_manifest_contract(&manifest)?;
    Ok(manifest)
}

fn validate_manifest_contract(manifest: &Manifest) -> Result<()> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        anyhow::bail!(
            "unsupported manifest schema_version {}; expected {MANIFEST_SCHEMA_VERSION}",
            manifest.schema_version,
        );
    }
    if manifest.product_version != PRODUCT_VERSION {
        anyhow::bail!("unsupported product version: {}", manifest.product_version);
    }
    if manifest.golden_version != GOLDEN_VERSION {
        anyhow::bail!("unsupported golden version: {}", manifest.golden_version);
    }
    if manifest.archive.filename != ARCHIVE_FILENAME {
        anyhow::bail!(
            "unsupported golden archive filename: {}",
            manifest.archive.filename
        );
    }
    if !is_lowercase_sha256(&manifest.archive.sha256) {
        anyhow::bail!("archive sha256 must be 64 lowercase hexadecimal characters");
    }
    if manifest.model.dtype != "bfloat16" || manifest.generation.attn_implementation != "sdpa" {
        anyhow::bail!("golden manifest requires the Transformers BF16 SDPA reference oracle");
    }
    let policy = &manifest.tolerance_policy;
    if policy.version != "same-prefix-v1" {
        anyhow::bail!("unsupported tolerance policy version: {}", policy.version);
    }
    if !policy.l1_near_tie_max_abs_logit_gap.is_finite()
        || policy.l1_near_tie_max_abs_logit_gap < 0.0
        || !policy.l2_atol.is_finite()
        || policy.l2_atol < 0.0
    {
        anyhow::bail!("tolerance policy thresholds must be finite and non-negative");
    }
    if policy.dtype != manifest.model.dtype
        || policy.kernel != manifest.generation.attn_implementation
    {
        anyhow::bail!("tolerance policy scope does not match model dtype and kernel");
    }
    if policy.rationale.trim().is_empty()
        || policy.evidence.is_empty()
        || policy.evidence.iter().any(|item| item.trim().is_empty())
    {
        anyhow::bail!("tolerance policy requires non-empty rationale and evidence");
    }
    let calibration = &manifest.baseline_calibration;
    if !calibration.candidate_atol.is_finite()
        || calibration.candidate_atol < 0.0
        || !calibration.observed_max_abs_diff.is_finite()
        || calibration.observed_max_abs_diff < 0.0
        || !calibration.calibration_factor.is_finite()
        || calibration.calibration_factor <= 0.0
        || calibration.method.trim().is_empty()
    {
        anyhow::bail!("baseline calibration observations are invalid");
    }
    if manifest.expected_fixtures.is_empty() {
        anyhow::bail!("manifest expected_fixtures is empty");
    }
    if manifest.model.revision.is_empty()
        || manifest.model.dtype.is_empty()
        || manifest.model.vocab_size == 0
    {
        anyhow::bail!("manifest model identity is incomplete");
    }
    let mut expected_ids = HashSet::new();
    let mut portable_filenames = HashSet::new();
    let mut roles_by_prompt: HashMap<&str, HashSet<OracleRole>> = HashMap::new();
    let mut families_by_prompt: HashMap<&str, HashSet<FixtureFamily>> = HashMap::new();
    for expected in &manifest.expected_fixtures {
        if expected.prompt_id.is_empty()
            || !expected
                .prompt_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            anyhow::bail!("unsupported fixture identifier: {}", expected.prompt_id);
        }
        if !expected_ids.insert(expected.fixture_id.as_str()) {
            anyhow::bail!(
                "duplicate expected fixture identifier: {}",
                expected.fixture_id
            );
        }
        let oracle_label = match expected.oracle {
            OracleName::Transformers => "transformers",
            OracleName::Vllm => "vllm",
            OracleName::Fake => "fake",
        };
        let canonical_id = format!("{}.{}", expected.prompt_id, oracle_label);
        if expected.fixture_id != canonical_id
            || expected.filename != format!("{canonical_id}.safetensors")
        {
            anyhow::bail!(
                "non-canonical fixture identifier or filename: {}",
                expected.fixture_id
            );
        }
        if !portable_filenames.insert(expected.filename.to_ascii_lowercase()) {
            anyhow::bail!("case-colliding fixture filename: {}", expected.filename);
        }
        if expected.model_revision != manifest.model.revision
            || expected.dtype != manifest.model.dtype
            || expected.model_revision.is_empty()
            || expected.dtype.is_empty()
        {
            anyhow::bail!(
                "fixture {} model revision or dtype does not match manifest",
                expected.fixture_id
            );
        }
        let reference_comparison = match expected.family {
            FixtureFamily::Canonical | FixtureFamily::Batch => RequiredComparison::L1L2,
            FixtureFamily::Regression => RequiredComparison::L1,
        };
        let valid_reference = expected.oracle == OracleName::Transformers
            && expected.oracle_role == OracleRole::Reference
            && expected.required_comparison == reference_comparison;
        let valid_baseline = expected.oracle == OracleName::Vllm
            && expected.oracle_role == OracleRole::Baseline
            && expected.required_comparison == RequiredComparison::Calibration;
        if !valid_reference && !valid_baseline {
            anyhow::bail!("invalid oracle role contract: {}", expected.fixture_id);
        }
        roles_by_prompt
            .entry(&expected.prompt_id)
            .or_default()
            .insert(expected.oracle_role.clone());
        families_by_prompt
            .entry(&expected.prompt_id)
            .or_default()
            .insert(expected.family.clone());
    }
    let required_roles = HashSet::from([OracleRole::Reference, OracleRole::Baseline]);
    for (prompt_id, roles) in roles_by_prompt {
        if roles != required_roles {
            anyhow::bail!("fixture {prompt_id} must declare reference and baseline oracle roles");
        }
        if families_by_prompt[prompt_id].len() != 1 {
            anyhow::bail!("fixture {prompt_id} declares inconsistent families");
        }
    }
    let expected_by_id: HashMap<_, _> = manifest
        .expected_fixtures
        .iter()
        .map(|expected| (expected.fixture_id.as_str(), expected))
        .collect();
    let generated_ids: Vec<_> = manifest
        .fixtures
        .iter()
        .map(|fixture| (&fixture.prompt_id, &fixture.oracle))
        .collect();
    if generated_ids.iter().collect::<HashSet<_>>().len() != generated_ids.len() {
        anyhow::bail!("duplicate generated fixture identifier");
    }
    let mut calibrated_ids = HashSet::new();
    for calibrated in &manifest.calibrated_fixtures {
        if !calibrated_ids.insert(calibrated.as_str()) {
            anyhow::bail!("duplicate calibrated fixture identifier: {calibrated}");
        }
        let Some(expected) = expected_by_id.get(calibrated.as_str()) else {
            anyhow::bail!("unmatched calibrated fixture: {calibrated}");
        };
        let was_generated = manifest.fixtures.iter().any(|fixture| {
            fixture.prompt_id == expected.prompt_id && fixture.oracle == expected.oracle
        });
        if expected.oracle_role != OracleRole::Baseline || !was_generated {
            anyhow::bail!("invalid baseline calibration evidence: {calibrated}");
        }
    }
    let expected_keys: HashSet<_> = manifest
        .expected_fixtures
        .iter()
        .map(|expected| {
            (
                expected.prompt_id.as_str(),
                &expected.oracle,
                expected.filename.as_str(),
            )
        })
        .collect();
    for fixture in &manifest.fixtures {
        if !is_lowercase_sha256(&fixture.sha256) {
            anyhow::bail!(
                "fixture {} sha256 must be 64 lowercase hexadecimal characters",
                fixture.filename
            );
        }
        let key = (
            fixture.prompt_id.as_str(),
            &fixture.oracle,
            fixture.filename.as_str(),
        );
        if !expected_keys.contains(&key) {
            anyhow::bail!(
                "unmatched generated fixture: {} ({})",
                fixture.prompt_id,
                fixture.filename
            );
        }
        let expected_id = format!(
            "{}.{}",
            fixture.prompt_id,
            match fixture.oracle {
                OracleName::Transformers => "transformers",
                OracleName::Vllm => "vllm",
                OracleName::Fake => "fake",
            }
        );
        let expected = expected_by_id[expected_id.as_str()];
        let category_matches = match expected.family {
            FixtureFamily::Canonical | FixtureFamily::Batch => {
                fixture.category == PromptCategory::Canonical
            }
            FixtureFamily::Regression => fixture.category == PromptCategory::Regression,
        };
        if !category_matches {
            anyhow::bail!(
                "generated fixture {} family does not match expectation",
                fixture.prompt_id
            );
        }
    }
    Ok(())
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

/// Load a single `.safetensors` fixture file into a `FixtureData`.
pub fn load_fixture(path: &Path, meta: &FixtureMetadata) -> Result<FixtureData> {
    let file_bytes =
        std::fs::read(path).with_context(|| format!("reading fixture from {}", path.display()))?;

    let tensors = SafeTensors::deserialize(&file_bytes)
        .with_context(|| format!("deserializing safetensors from {}", path.display()))?;

    if meta.num_tokens == 0 {
        anyhow::bail!("unsupported empty fixture shape for {}", meta.filename);
    }
    validate_tensor_shape(
        &tensors,
        "token_ids",
        safetensors::Dtype::I64,
        &[meta.num_tokens as usize],
    )?;
    validate_tensor_shape(&tensors, "n_prompt_tokens", safetensors::Dtype::I64, &[])?;

    let actual_names: HashSet<_> = tensors.names().into_iter().map(String::as_str).collect();
    match meta.category {
        crate::types::PromptCategory::Canonical => {
            let expected_names = HashSet::from(["token_ids", "n_prompt_tokens", "logits"]);
            if actual_names != expected_names {
                anyhow::bail!("unsupported canonical tensor set for {}", meta.filename);
            }
            if meta.logits_shape.0 != meta.num_tokens as usize || meta.logits_shape.1 == 0 {
                anyhow::bail!("unsupported logits shape for {}", meta.filename);
            }
            validate_tensor_shape(
                &tensors,
                "logits",
                safetensors::Dtype::F32,
                &[meta.logits_shape.0, meta.logits_shape.1],
            )?;
        }
        crate::types::PromptCategory::Regression => {
            let expected_names = HashSet::from([
                "token_ids",
                "n_prompt_tokens",
                "top5_indices",
                "top5_logits",
            ]);
            if actual_names != expected_names || meta.logits_shape != (0, 0) {
                anyhow::bail!("unsupported regression fixture shape for {}", meta.filename);
            }
            validate_tensor_shape(
                &tensors,
                "top5_indices",
                safetensors::Dtype::I64,
                &[meta.num_tokens as usize, 5],
            )?;
            validate_tensor_shape(
                &tensors,
                "top5_logits",
                safetensors::Dtype::F32,
                &[meta.num_tokens as usize, 5],
            )?;
        }
    }

    let token_ids: Vec<i64> = read_named_tensor_as_i64(&tensors, "token_ids")?;
    let n_prompt_tokens: i64 = read_named_scalar_as_i64(&tensors, "n_prompt_tokens")?;

    let (logits, top5_indices, top5_logits) = match meta.category {
        crate::types::PromptCategory::Canonical => {
            let logits_vec: Vec<f32> = read_named_tensor_as_f32(&tensors, "logits")?;
            (Some(logits_vec), None, None)
        }
        crate::types::PromptCategory::Regression => {
            let indices: Vec<i64> = read_named_tensor_as_i64(&tensors, "top5_indices")?;
            let logits_top5: Vec<f32> = read_named_tensor_as_f32(&tensors, "top5_logits")?;
            (None, Some(indices), Some(logits_top5))
        }
    };

    Ok(FixtureData {
        prompt_id: meta.prompt_id.clone(),
        category: meta.category.clone(),
        oracle: meta.oracle.clone(),
        num_tokens: meta.num_tokens,
        token_ids,
        n_prompt_tokens,
        logits,
        logits_shape: meta.logits_shape,
        top5_indices,
        top5_logits,
    })
}

fn validate_tensor_shape(
    tensors: &SafeTensors<'_>,
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[usize],
) -> Result<()> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor '{name}' not found in safetensors"))?;
    if view.dtype() != dtype || view.shape() != shape {
        anyhow::bail!(
            "unsupported {name} shape or dtype: got {:?} {:?}, expected {:?} {:?}",
            view.dtype(),
            view.shape(),
            dtype,
            shape
        );
    }
    Ok(())
}

/// Compute SHA-256 hex digest of file bytes.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn read_named_tensor_as_i64(tensors: &SafeTensors, name: &str) -> Result<Vec<i64>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor '{}' not found in safetensors", name))?;
    match view.dtype() {
        safetensors::Dtype::I64 => {
            let len = view.data().len() / 8;
            let data: &[i64] = bytemuck::cast_slice(view.data());
            Ok(data[..len].to_vec())
        }
        other => anyhow::bail!("tensor '{}' has dtype {:?}, expected I64", name, other),
    }
}

fn read_named_scalar_as_i64(tensors: &SafeTensors, name: &str) -> Result<i64> {
    let data = read_named_tensor_as_i64(tensors, name)?;
    data.first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("tensor '{}' is empty (expected scalar)", name))
}

fn read_named_tensor_as_f32(tensors: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor '{}' not found in safetensors", name))?;
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let len = view.data().len() / 4;
            let data: &[f32] = bytemuck::cast_slice(view.data());
            Ok(data[..len].to_vec())
        }
        other => anyhow::bail!("tensor '{}' has dtype {:?}, expected F32", name, other),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use serde_json::{json, Value};

    use super::{load_fixture, parse_manifest};
    use crate::types::{FixtureMetadata, LogitsDtype, OracleName, PromptCategory};

    fn valid_manifest_json() -> Value {
        json!({
            "schema_version": 4,
            "product_version": "v0.2.0",
            "golden_version": "goldens-v0.2",
            "archive": {
                "filename": "goldens-v0.2.tar.gz",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "generated_at": "2026-09-04T00:00:00Z",
            "model": {
                "id": "Qwen/Qwen3-0.6B",
                "revision": "7e4ae267688d671ddfca3122e4528ee980cf3234",
                "arch": "Qwen3ForCausalLM",
                "dtype": "bfloat16",
                "vocab_size": 151_936
            },
            "oracle_versions": {"transformers": "5.0", "vllm": "0.26"},
            "generation": {
                "canonical_max_tokens": 64,
                "regression_max_tokens": 32,
                "temperature": 0.0,
                "attn_implementation": "sdpa"
            },
            "baseline_calibration": {
                "candidate_atol": 0.01,
                "observed_max_abs_diff": 0.005,
                "calibration_factor": 2.0,
                "method": "test"
            },
            "tolerance_policy": {
                "version": "same-prefix-v1",
                "dtype": "bfloat16",
                "kernel": "sdpa",
                "l1_near_tie_max_abs_logit_gap": 0.02,
                "l2_atol": 0.01,
                "rationale": "Reviewed synthetic policy",
                "evidence": ["synthetic:manifest"]
            },
            "expected_fixtures": [
                {
                    "fixture_id": "canonical_01.transformers",
                    "prompt_id": "canonical_01",
                    "family": "canonical",
                    "model_revision": "7e4ae267688d671ddfca3122e4528ee980cf3234",
                    "dtype": "bfloat16",
                    "oracle": "transformers",
                    "oracle_role": "reference",
                    "required_comparison": "l1_l2",
                    "filename": "canonical_01.transformers.safetensors"
                },
                {
                    "fixture_id": "canonical_01.vllm",
                    "prompt_id": "canonical_01",
                    "family": "canonical",
                    "model_revision": "7e4ae267688d671ddfca3122e4528ee980cf3234",
                    "dtype": "bfloat16",
                    "oracle": "vllm",
                    "oracle_role": "baseline",
                    "required_comparison": "calibration",
                    "filename": "canonical_01.vllm.safetensors"
                }
            ],
            "fixtures": [],
            "calibrated_fixtures": []
        })
    }

    #[test]
    fn shared_python_manifest_v4_fixture_is_compatible() {
        let bytes = include_bytes!("../../../tools/golden-gen/tests/fixtures/manifest-v4.json");

        let manifest =
            super::parse_manifest_bytes(bytes, "shared Python manifest fixture").unwrap();

        assert_eq!(manifest.schema_version, 4);
        assert_eq!(manifest.archive.sha256, "a".repeat(64));
    }

    #[test]
    fn schema_v4_asset_contract_is_accepted() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["schema_version"] = json!(4);
        manifest["product_version"] = json!("v0.2.0");
        manifest["golden_version"] = json!("goldens-v0.2");
        manifest["archive"] = json!({
            "filename": "goldens-v0.2.tar.gz",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        });
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let parsed = parse_manifest(&path).unwrap();

        assert_eq!(parsed.schema_version, 4);
        assert_eq!(parsed.product_version, "v0.2.0");
        assert_eq!(parsed.golden_version, "goldens-v0.2");
        assert_eq!(parsed.archive.filename, "goldens-v0.2.tar.gz");
    }

    #[test]
    fn unsupported_product_golden_and_archive_contracts_are_rejected() {
        let cases = [
            (
                "product_version",
                json!("v0.3.0"),
                "unsupported product version",
            ),
            (
                "golden_version",
                json!("goldens-v0.3"),
                "unsupported golden version",
            ),
            (
                "archive.filename",
                json!("fixtures.tar.gz"),
                "unsupported golden archive filename",
            ),
            (
                "archive.sha256",
                json!("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                "64 lowercase hexadecimal",
            ),
        ];
        for (field, value, expected_error) in cases {
            let mut manifest = valid_manifest_json();
            if let Some(archive_field) = field.strip_prefix("archive.") {
                manifest["archive"][archive_field] = value;
            } else {
                manifest[field] = value;
            }

            let error = super::parse_manifest_bytes(
                &serde_json::to_vec(&manifest).unwrap(),
                "version rejection test",
            )
            .unwrap_err()
            .to_string();

            assert!(error.contains(expected_error), "{field}: {error}");
        }
    }

    #[test]
    fn noncanonical_fixture_checksum_is_rejected() {
        let mut manifest = valid_manifest_json();
        manifest["fixtures"] = json!([{
            "prompt_id": "canonical_01",
            "category": "canonical",
            "oracle": "transformers",
            "num_tokens": 1,
            "logits_dtype": "float32",
            "logits_shape": [1, 151_936],
            "sha256": "ABC123",
            "filename": "canonical_01.transformers.safetensors"
        }]);

        let error = super::parse_manifest_bytes(
            &serde_json::to_vec(&manifest).unwrap(),
            "fixture checksum test",
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("64 lowercase hexadecimal"), "{error}");
    }

    #[test]
    fn case_colliding_fixture_names_are_rejected() {
        let mut manifest = valid_manifest_json();
        let duplicates: Vec<_> = manifest["expected_fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .map(|fixture| {
                let mut duplicate = fixture.clone();
                for field in ["fixture_id", "prompt_id", "filename"] {
                    duplicate[field] = json!(duplicate[field]
                        .as_str()
                        .unwrap()
                        .replace("canonical_01", "CANONICAL_01"));
                }
                duplicate
            })
            .collect();
        manifest["expected_fixtures"]
            .as_array_mut()
            .unwrap()
            .extend(duplicates);

        let error = super::parse_manifest_bytes(
            &serde_json::to_vec(&manifest).unwrap(),
            "case collision test",
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("case-colliding"), "{error}");
    }

    #[test]
    fn manifest_exposes_explicit_versioned_tolerance_policy() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let manifest = valid_manifest_json();
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let parsed = parse_manifest(&path).unwrap();

        assert_eq!(parsed.tolerance_policy.version, "same-prefix-v1");
        assert_eq!(parsed.tolerance_policy.l1_near_tie_max_abs_logit_gap, 0.02);
        assert_eq!(parsed.tolerance_policy.l2_atol, 0.01);
        assert_eq!(parsed.tolerance_policy.evidence, ["synthetic:manifest"]);
    }

    #[test]
    fn unsupported_manifest_schema_version_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["schema_version"] = json!(999);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(
            error.contains("unsupported manifest schema_version"),
            "{error}"
        );
    }

    #[test]
    fn unsupported_tolerance_policy_version_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["tolerance_policy"]["version"] = json!("latest");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(
            error.contains("unsupported tolerance policy version"),
            "{error}"
        );
    }

    #[test]
    fn tolerance_policy_without_provenance_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["tolerance_policy"]["evidence"] = json!([]);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("rationale and evidence"), "{error}");
    }

    #[test]
    fn tolerance_policy_scope_must_match_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["tolerance_policy"]["kernel"] = json!("flash_attention_2");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("scope does not match"), "{error}");
    }

    #[test]
    fn self_consistent_non_bf16_reference_contract_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["model"]["dtype"] = json!("float32");
        manifest["tolerance_policy"]["dtype"] = json!("float32");
        for expected in manifest["expected_fixtures"].as_array_mut().unwrap() {
            expected["dtype"] = json!("float32");
        }
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("BF16 SDPA reference oracle"), "{error}");
    }

    #[test]
    fn self_consistent_non_sdpa_reference_contract_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["generation"]["attn_implementation"] = json!("eager");
        manifest["tolerance_policy"]["kernel"] = json!("eager");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("BF16 SDPA reference oracle"), "{error}");
    }

    #[test]
    fn legacy_regression_skip_map_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["regression_skip_map"] = json!({"canonical_01": [0]});
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("unknown field"), "{error}");
    }

    #[test]
    fn empty_expected_fixture_set_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["expected_fixtures"] = json!([]);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("expected_fixtures is empty"), "{error}");
    }

    #[test]
    fn duplicate_expected_fixture_identifier_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        let duplicate = manifest["expected_fixtures"][0].clone();
        manifest["expected_fixtures"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(
            error.contains("duplicate expected fixture identifier"),
            "{error}"
        );
    }

    #[test]
    fn unknown_manifest_field_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["looks_complete"] = json!(true);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("unknown field"), "{error}");
    }

    #[test]
    fn unmatched_generated_fixture_identifier_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["fixtures"] = json!([{
            "prompt_id": "canonical_99",
            "category": "canonical",
            "oracle": "transformers",
            "num_tokens": 1,
            "logits_dtype": "float32",
            "logits_shape": [1, 151_936],
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "filename": "canonical_99.transformers.safetensors"
        }]);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("unmatched generated fixture"), "{error}");
    }

    #[test]
    fn unmatched_calibrated_fixture_identifier_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["calibrated_fixtures"] = json!(["canonical_99.vllm"]);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("unmatched calibrated fixture"), "{error}");
    }

    #[test]
    fn invalid_oracle_role_contract_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let mut manifest = valid_manifest_json();
        manifest["expected_fixtures"][0]["oracle_role"] = json!("baseline");
        manifest["expected_fixtures"][0]["required_comparison"] = json!("calibration");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = parse_manifest(&path).unwrap_err().to_string();

        assert!(error.contains("invalid oracle role contract"), "{error}");
    }

    #[test]
    fn unsupported_canonical_fixture_shape_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("canonical_01.transformers.safetensors");
        let token_bytes = 1_i64.to_ne_bytes();
        let prompt_bytes = 1_i64.to_ne_bytes();
        let logits_bytes: Vec<u8> = [0.0_f32, 1.0]
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
                    vec![1, 2],
                    &logits_bytes,
                )
                .unwrap(),
            ),
        ];
        safetensors::tensor::serialize_to_file(tensors, &None, &path).unwrap();
        let metadata = FixtureMetadata {
            prompt_id: "canonical_01".to_string(),
            category: PromptCategory::Canonical,
            oracle: OracleName::Transformers,
            num_tokens: 1,
            logits_dtype: LogitsDtype::Float32,
            logits_shape: (1, 3),
            sha256: "unused".to_string(),
            filename: "canonical_01.transformers.safetensors".to_string(),
        };

        let error = load_fixture(&path, &metadata).unwrap_err().to_string();

        assert!(error.contains("unsupported logits shape"), "{error}");
    }
}
