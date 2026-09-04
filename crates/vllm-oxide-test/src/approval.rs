//! Independent gate that must pass before authoritative holdout comparison.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::types::Manifest;

const CALIBRATION_IDS: [&str; 4] = [
    "canonical_01",
    "canonical_02",
    "canonical_03",
    "canonical_05a",
];
const HOLDOUT_IDS: [&str; 4] = [
    "canonical_04",
    "canonical_05b",
    "canonical_05c",
    "canonical_05d",
];

pub fn validate_authoritative_approval(manifest: &Manifest, path: &Path) -> Result<()> {
    if path.file_name().and_then(|name| name.to_str())
        != Some("goldens-v0.2-calibration-observation.json")
    {
        bail!("authoritative comparison requires the tracked calibration observation path");
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading approved observation {}", path.display()))?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let required_evidence = format!("definition-observation-sha256:{sha256}");
    if !manifest
        .tolerance_policy
        .evidence
        .contains(&required_evidence)
    {
        bail!("manifest policy does not bind the approved Definition observation SHA-256");
    }
    let observation: Value =
        serde_json::from_slice(&bytes).context("parsing approved calibration observation")?;
    let opened = observation["opened_fixture_ids"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("approved observation opened IDs are missing"))?;
    let holdout = observation["sealed_holdout_ids"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("approved observation holdout IDs are missing"))?;
    let expected_opened = CALIBRATION_IDS.map(Value::from).to_vec();
    let expected_holdout = HOLDOUT_IDS.map(Value::from).to_vec();
    if observation["schema_version"] != 1
        || observation["status"] != "non_accepting_calibration_observation"
        || observation["accepting"] != false
        || observation["input_l1_threshold"] != 0.0
        || observation["input_l2_threshold"] != 0.0
        || opened.as_slice() != expected_opened.as_slice()
        || holdout.as_slice() != expected_holdout.as_slice()
    {
        bail!("approved observation does not preserve the sealed holdout protocol");
    }
    let proposed_l1 = observation["proposal"]["l1_near_tie_max_abs_logit_gap"]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("approved observation L1 proposal is missing"))?;
    let proposed_l2 = observation["proposal"]["l2_atol"]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("approved observation L2 proposal is missing"))?;
    if proposed_l1 != manifest.tolerance_policy.l1_near_tie_max_abs_logit_gap
        || proposed_l2 != manifest.tolerance_policy.l2_atol
        || proposed_l1 > 0.0625
        || proposed_l2 > 0.25
    {
        bail!("manifest thresholds differ from the mechanically approved proposal");
    }
    if observation["identity"]["kernel_scope"] != manifest.kernel_paths.comparison_scope() {
        bail!("approved observation kernel scope differs from the manifest");
    }
    let runtime_bytes = serde_json::to_vec(&manifest.runtime)?;
    let runtime_sha256 = format!("{:x}", Sha256::digest(runtime_bytes));
    if observation["identity"]["runtime_sha256"] != runtime_sha256 {
        bail!("approved observation runtime identity differs from the manifest");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use serde_json::json;
    use sha2::Digest;

    use super::validate_authoritative_approval;

    #[test]
    fn approval_binds_definition_sha_proposal_runtime_kernel_and_sealed_holdout() {
        let bytes = include_bytes!("../../../tools/golden-gen/tests/fixtures/manifest-v4.json");
        let mut manifest = crate::manifest::parse_manifest_bytes(bytes, "shared fixture").unwrap();
        let runtime_sha = format!(
            "{:x}",
            sha2::Sha256::digest(serde_json::to_vec(&manifest.runtime).unwrap())
        );
        let observation = json!({
            "schema_version": 1,
            "status": "non_accepting_calibration_observation",
            "accepting": false,
            "input_l1_threshold": 0.0,
            "input_l2_threshold": 0.0,
            "opened_fixture_ids": ["canonical_01", "canonical_02", "canonical_03", "canonical_05a"],
            "sealed_holdout_ids": ["canonical_04", "canonical_05b", "canonical_05c", "canonical_05d"],
            "identity": {
                "runtime_sha256": runtime_sha,
                "kernel_scope": manifest.kernel_paths.comparison_scope()
            },
            "proposal": {
                "l1_near_tie_max_abs_logit_gap": 0.02,
                "l2_atol": 0.01
            }
        });
        let temp = tempfile::tempdir().unwrap();
        let path = temp
            .path()
            .join("goldens-v0.2-calibration-observation.json");
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        std::fs::write(&path, &observation_bytes).unwrap();
        manifest.tolerance_policy.evidence = vec![format!(
            "definition-observation-sha256:{:x}",
            sha2::Sha256::digest(observation_bytes)
        )];

        validate_authoritative_approval(&manifest, &path).unwrap();

        let mut tampered = observation;
        tampered["sealed_holdout_ids"] = json!([]);
        std::fs::write(&path, serde_json::to_vec(&tampered).unwrap()).unwrap();
        assert!(validate_authoritative_approval(&manifest, &path).is_err());
    }
}
