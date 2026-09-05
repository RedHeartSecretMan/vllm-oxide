//! Independent gate that must pass before authoritative holdout comparison.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::measurement::{validate_measurement_identity, MeasurementIdentity};
use crate::observation::{CALIBRATION_IDS, HOLDOUT_IDS};
use crate::types::Manifest;

const L1_LADDER: [f64; 10] = [
    0.0,
    0.000_244_140_625,
    0.000_488_281_25,
    0.000_976_562_5,
    0.001_953_125,
    0.003_906_25,
    0.007_812_5,
    0.015_625,
    0.031_25,
    0.062_5,
];
const L2_LADDER: [f64; 12] = [
    0.0,
    0.000_244_140_625,
    0.000_488_281_25,
    0.000_976_562_5,
    0.001_953_125,
    0.003_906_25,
    0.007_812_5,
    0.015_625,
    0.031_25,
    0.062_5,
    0.125,
    0.25,
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApprovedObservation {
    schema_version: u32,
    status: String,
    accepting: bool,
    input_l1_threshold: f64,
    input_l2_threshold: f64,
    opened_fixture_ids: Vec<String>,
    sealed_holdout_ids: Vec<String>,
    identity: ObservationIdentity,
    cases: Vec<CaseObservation>,
    aggregate: AggregateObservation,
    proposal: ThresholdProposal,
    approval: ObservationApproval,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationIdentity {
    measurement_commit: String,
    measurement_tree: String,
    manifest_sha256: String,
    candidate_binary_sha256: String,
    runtime_sha256: String,
    kernel_scope: String,
    raw_evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseObservation {
    prompt_id: String,
    compared_rows: usize,
    compared_elements: usize,
    first_divergence: Option<usize>,
    excluded_rows: usize,
    candidate_token_gap: Option<f64>,
    mean_abs_error: f64,
    rms_abs_error: f64,
    p50_abs_error: f64,
    p95_abs_error: f64,
    p99_abs_error: f64,
    p999_abs_error: f64,
    maximum_abs_error: f64,
    non_finite_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AggregateObservation {
    compared_rows: usize,
    compared_elements: usize,
    divergence_count: usize,
    mean_abs_error: f64,
    rms_abs_error: f64,
    p50_abs_error: f64,
    p95_abs_error: f64,
    p99_abs_error: f64,
    p999_abs_error: f64,
    maximum_abs_error: f64,
    non_finite_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdProposal {
    l1_near_tie_max_abs_logit_gap: f64,
    l2_atol: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationApproval {
    status: String,
    accepted_error_classes: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DefinitionIndex {
    schema_version: u32,
    inputs: Vec<DefinitionInput>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DefinitionInput {
    path: String,
    blob_oid: String,
}

pub fn validate_authoritative_approval(
    manifest: &Manifest,
    path: &Path,
    repo_root: &Path,
    expected_measurement: &MeasurementIdentity,
) -> Result<()> {
    let measurement = validate_measurement_identity(
        repo_root,
        &expected_measurement.commit,
        &expected_measurement.tree,
    )?;
    let expected_path = repo_root.join("docs/releases/goldens-v0.2-calibration-observation.json");
    if path
        .canonicalize()
        .context("canonicalizing approved observation")?
        != expected_path
            .canonicalize()
            .context("canonicalizing tracked approved observation")?
    {
        bail!("authoritative comparison requires the tracked calibration observation path");
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading approved observation {}", path.display()))?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let blob = git_text(repo_root, &["hash-object", path.to_string_lossy().as_ref()])?;
    let tracked_blob = git_text(
        repo_root,
        &[
            "rev-parse",
            "HEAD:docs/releases/goldens-v0.2-calibration-observation.json",
        ],
    )?;
    if blob != tracked_blob {
        bail!("approved observation bytes differ from the tracked HEAD blob");
    }
    let index: DefinitionIndex = serde_json::from_slice(&std::fs::read(
        repo_root.join(".dag/definition-index.json"),
    )?)?;
    let selected = index
        .inputs
        .iter()
        .filter(|input| input.path == "docs/releases/goldens-v0.2-calibration-observation.json")
        .collect::<Vec<_>>();
    if index.schema_version != 1 || selected.len() != 1 || selected[0].blob_oid != tracked_blob {
        bail!("approved observation is not selected by the bound Definition index");
    }
    let observation: ApprovedObservation =
        serde_json::from_slice(&bytes).context("parsing approved calibration observation")?;
    validate_observation_contract(&observation)?;
    let observation_measurement = validate_measurement_identity(
        repo_root,
        &observation.identity.measurement_commit,
        &observation.identity.measurement_tree,
    )?;

    if observation.identity.kernel_scope != manifest.kernel_paths.comparison_scope() {
        bail!("approved observation kernel scope differs from the manifest");
    }
    let runtime_bytes = serde_json::to_vec(&manifest.runtime)?;
    let runtime_sha256 = format!("{:x}", Sha256::digest(runtime_bytes));
    if observation.identity.runtime_sha256 != runtime_sha256 {
        bail!("approved observation runtime identity differs from the manifest");
    }
    if observation.proposal.l1_near_tie_max_abs_logit_gap
        != manifest.tolerance_policy.l1_near_tie_max_abs_logit_gap
        || observation.proposal.l2_atol != manifest.tolerance_policy.l2_atol
    {
        bail!("manifest thresholds differ from the mechanically approved proposal");
    }
    let rationale = observation.approval.accepted_error_classes.join("; ");
    if manifest.tolerance_policy.rationale != rationale {
        bail!("manifest rationale differs from the reviewed accepted error classes");
    }

    let required_evidence = [
        format!("definition-observation-sha256:{sha256}"),
        format!("definition-observation-blob:{tracked_blob}"),
        format!(
            "observation-measurement-commit:{}",
            observation_measurement.commit
        ),
        format!(
            "observation-measurement-tree:{}",
            observation_measurement.tree
        ),
        format!("measurement-commit:{}", measurement.commit),
        format!("measurement-tree:{}", measurement.tree),
        format!(
            "candidate-binary-sha256:{}",
            observation.identity.candidate_binary_sha256
        ),
        format!("runtime-sha256:{}", observation.identity.runtime_sha256),
        format!(
            "raw-evidence-sha256:{}",
            observation.identity.raw_evidence_sha256
        ),
    ];
    let actual = manifest
        .tolerance_policy
        .evidence
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    if actual.len() != manifest.tolerance_policy.evidence.len()
        || actual != required_evidence.into_iter().collect()
    {
        bail!("manifest policy evidence is incomplete, duplicated, or unexpected");
    }
    Ok(())
}

fn validate_observation_contract(observation: &ApprovedObservation) -> Result<()> {
    let expected_opened = CALIBRATION_IDS.map(str::to_owned);
    let expected_holdout = HOLDOUT_IDS.map(str::to_owned);
    if observation.schema_version != 1
        || observation.status != "non_accepting_calibration_observation"
        || observation.accepting
        || observation.input_l1_threshold != 0.0
        || observation.input_l2_threshold != 0.0
        || observation.opened_fixture_ids != expected_opened
        || observation.sealed_holdout_ids != expected_holdout
    {
        bail!("approved observation does not preserve the sealed holdout protocol");
    }
    for value in [
        &observation.identity.manifest_sha256,
        &observation.identity.candidate_binary_sha256,
        &observation.identity.runtime_sha256,
        &observation.identity.raw_evidence_sha256,
    ] {
        if !is_lowercase_hex(value, 64) {
            bail!("approved observation contains a malformed SHA-256 identity");
        }
    }
    if observation.cases.len() != CALIBRATION_IDS.len()
        || observation
            .cases
            .iter()
            .map(|case| case.prompt_id.as_str())
            .collect::<Vec<_>>()
            != CALIBRATION_IDS
    {
        bail!("approved observation does not contain the exact calibration cases");
    }
    let mut total_rows = 0_usize;
    let mut total_elements = 0_usize;
    let mut divergence_count = 0_usize;
    let mut weighted_mean = 0.0;
    let mut weighted_square = 0.0;
    let mut maximum = 0.0_f64;
    let mut maximum_gap = 0.0_f64;
    for case in &observation.cases {
        let ordered = [
            case.p50_abs_error,
            case.p95_abs_error,
            case.p99_abs_error,
            case.p999_abs_error,
            case.maximum_abs_error,
        ];
        if case.compared_rows == 0
            || case.compared_elements == 0
            || case.non_finite_count != 0
            || !ordered.into_iter().all(f64::is_finite)
            || !case.mean_abs_error.is_finite()
            || !case.rms_abs_error.is_finite()
            || ordered.windows(2).any(|pair| pair[0] > pair[1])
            || ordered.iter().any(|value| *value < 0.0)
            || !(0.0 <= case.mean_abs_error
                && case.mean_abs_error <= case.rms_abs_error
                && case.rms_abs_error <= case.maximum_abs_error)
            || case.compared_elements % case.compared_rows != 0
        {
            bail!("approved observation case statistics are invalid");
        }
        match (case.first_divergence, case.candidate_token_gap) {
            (Some(index), Some(gap))
                if index.checked_add(1) == Some(case.compared_rows)
                    && gap.is_finite()
                    && gap >= 0.0 =>
            {
                divergence_count += 1;
                maximum_gap = maximum_gap.max(gap);
            }
            (None, None) if case.excluded_rows == 0 => {}
            _ => bail!("approved observation divergence metadata is invalid"),
        }
        total_rows = total_rows
            .checked_add(case.compared_rows)
            .ok_or_else(|| anyhow::anyhow!("observation row count overflow"))?;
        total_elements = total_elements
            .checked_add(case.compared_elements)
            .ok_or_else(|| anyhow::anyhow!("observation element count overflow"))?;
        let count = f64::from(
            u32::try_from(case.compared_elements)
                .context("calibration case exceeds supported sample count")?,
        );
        weighted_mean += case.mean_abs_error * count;
        weighted_square += case.rms_abs_error.powi(2) * count;
        maximum = maximum.max(case.maximum_abs_error);
    }
    let aggregate = &observation.aggregate;
    let count = f64::from(
        u32::try_from(total_elements)
            .context("calibration aggregate exceeds supported sample count")?,
    );
    let aggregate_ordered = [
        aggregate.p50_abs_error,
        aggregate.p95_abs_error,
        aggregate.p99_abs_error,
        aggregate.p999_abs_error,
        aggregate.maximum_abs_error,
    ];
    if aggregate.compared_rows != total_rows
        || aggregate.compared_elements != total_elements
        || aggregate.divergence_count != divergence_count
        || aggregate.non_finite_count != 0
        || !close(aggregate.mean_abs_error, weighted_mean / count)
        || !close(aggregate.rms_abs_error, (weighted_square / count).sqrt())
        || aggregate.maximum_abs_error != maximum
        || !aggregate_ordered.into_iter().all(f64::is_finite)
        || aggregate_ordered.windows(2).any(|pair| pair[0] > pair[1])
    {
        bail!("approved observation aggregate does not cover every case sample");
    }
    if observation.proposal.l1_near_tie_max_abs_logit_gap
        != smallest_covering(maximum_gap, &L1_LADDER)?
        || observation.proposal.l2_atol != smallest_covering(maximum, &L2_LADDER)?
    {
        bail!("approved observation proposal is not the smallest covering ladder pair");
    }
    let mut normalized = observation.approval.accepted_error_classes.clone();
    normalized.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    normalized.dedup();
    if observation.approval.status != "approved"
        || normalized.is_empty()
        || normalized != observation.approval.accepted_error_classes
        || normalized
            .iter()
            .any(|value| value.is_empty() || value.trim() != value)
    {
        bail!("approved observation lacks normalized reviewed error classes");
    }
    Ok(())
}

fn smallest_covering(value: f64, ladder: &[f64]) -> Result<f64> {
    ladder
        .iter()
        .copied()
        .find(|threshold| value <= *threshold)
        .ok_or_else(|| anyhow::anyhow!("observation exceeds the approved tolerance ceiling"))
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-15_f64.max(right.abs() * 1e-12)
}

fn is_lowercase_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn git_text(repo_root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .with_context(|| format!("running git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8(output.stdout)
        .context("Git output is not UTF-8")?
        .trim()
        .to_owned())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use serde_json::json;
    use sha2::Digest;

    use super::validate_authoritative_approval;
    use crate::measurement::MeasurementIdentity;

    fn git(repo: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

    #[test]
    fn approval_binds_complete_definition_measurements_and_shared_holdout() {
        let bytes = include_bytes!("../../../tools/golden-gen/tests/fixtures/manifest-v4.json");
        let mut manifest = crate::manifest::parse_manifest_bytes(bytes, "shared fixture").unwrap();
        let runtime_sha = format!(
            "{:x}",
            sha2::Sha256::digest(serde_json::to_vec(&manifest.runtime).unwrap())
        );
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        git(repo, &["init", "-q"]);
        git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-qm",
                "measurement",
            ],
        );
        let observation_commit = git(repo, &["rev-parse", "HEAD"]);
        let observation_tree = git(repo, &["rev-parse", "HEAD^{tree}"]);
        let case = json!({
            "prompt_id": "placeholder",
            "compared_rows": 1,
            "compared_elements": 1,
            "first_divergence": null,
            "excluded_rows": 0,
            "candidate_token_gap": null,
            "mean_abs_error": 0.0,
            "rms_abs_error": 0.0,
            "p50_abs_error": 0.0,
            "p95_abs_error": 0.0,
            "p99_abs_error": 0.0,
            "p999_abs_error": 0.0,
            "maximum_abs_error": 0.0,
            "non_finite_count": 0
        });
        let cases = crate::observation::CALIBRATION_IDS
            .iter()
            .map(|id| {
                let mut value = case.clone();
                value["prompt_id"] = json!(id);
                value
            })
            .collect::<Vec<_>>();
        let observation = json!({
            "schema_version": 1,
            "status": "non_accepting_calibration_observation",
            "accepting": false,
            "input_l1_threshold": 0.0,
            "input_l2_threshold": 0.0,
            "opened_fixture_ids": crate::observation::CALIBRATION_IDS,
            "sealed_holdout_ids": crate::observation::HOLDOUT_IDS,
            "identity": {
                "measurement_commit": observation_commit.clone(),
                "measurement_tree": observation_tree.clone(),
                "manifest_sha256": "3".repeat(64),
                "candidate_binary_sha256": "4".repeat(64),
                "runtime_sha256": runtime_sha.clone(),
                "kernel_scope": manifest.kernel_paths.comparison_scope(),
                "raw_evidence_sha256": "6".repeat(64)
            },
            "cases": cases,
            "aggregate": {
                "compared_rows": 4,
                "compared_elements": 4,
                "divergence_count": 0,
                "mean_abs_error": 0.0,
                "rms_abs_error": 0.0,
                "p50_abs_error": 0.0,
                "p95_abs_error": 0.0,
                "p99_abs_error": 0.0,
                "p999_abs_error": 0.0,
                "maximum_abs_error": 0.0,
                "non_finite_count": 0
            },
            "proposal": {
                "l1_near_tie_max_abs_logit_gap": 0.0,
                "l2_atol": 0.0
            },
            "approval": {
                "status": "approved",
                "accepted_error_classes": ["Exact observation; no numerical error accepted."]
            }
        });
        let path = repo.join("docs/releases/goldens-v0.2-calibration-observation.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let observation_bytes = serde_json::to_vec(&observation).unwrap();
        std::fs::write(&path, &observation_bytes).unwrap();
        let blob = git(repo, &["hash-object", path.to_string_lossy().as_ref()]);
        std::fs::create_dir(repo.join(".dag")).unwrap();
        std::fs::write(repo.join(".dag/definition-index.json"), serde_json::to_vec(&json!({
            "schema_version": 1,
            "inputs": [{"path": "docs/releases/goldens-v0.2-calibration-observation.json", "blob_oid": blob}]
        })).unwrap()).unwrap();
        git(repo, &["add", ".dag/definition-index.json"]);
        git(
            repo,
            &[
                "add",
                "docs/releases/goldens-v0.2-calibration-observation.json",
            ],
        );
        git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "Definition observation",
            ],
        );
        let measurement = MeasurementIdentity {
            commit: git(repo, &["rev-parse", "HEAD"]),
            tree: git(repo, &["rev-parse", "HEAD^{tree}"]),
        };
        let blob = git(repo, &["hash-object", path.to_string_lossy().as_ref()]);
        let observation_sha = format!("{:x}", sha2::Sha256::digest(&observation_bytes));
        manifest.tolerance_policy.l1_near_tie_max_abs_logit_gap = 0.0;
        manifest.tolerance_policy.l2_atol = 0.0;
        manifest.tolerance_policy.rationale =
            "Exact observation; no numerical error accepted.".to_owned();
        manifest.tolerance_policy.evidence = vec![
            format!("definition-observation-sha256:{observation_sha}"),
            format!("definition-observation-blob:{blob}"),
            format!("observation-measurement-commit:{observation_commit}"),
            format!("observation-measurement-tree:{observation_tree}"),
            format!("measurement-commit:{}", measurement.commit),
            format!("measurement-tree:{}", measurement.tree),
            format!("candidate-binary-sha256:{}", "4".repeat(64)),
            format!("runtime-sha256:{}", runtime_sha),
            format!("raw-evidence-sha256:{}", "6".repeat(64)),
        ];

        validate_authoritative_approval(&manifest, &path, repo, &measurement).unwrap();

        manifest.tolerance_policy.evidence.pop();
        assert!(validate_authoritative_approval(&manifest, &path, repo, &measurement).is_err());
    }
}
