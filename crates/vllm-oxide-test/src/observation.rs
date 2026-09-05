//! Raw candidate capture for the non-accepting ADR-0012 calibration subset.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use vllm_oxide::{Prompt, Source, LLM};

use crate::benchmark::fixed_engine_options;
use crate::capture::generate_with_preserved_capture;
use crate::manifest::{parse_manifest, parse_observation_manifest};
use crate::measurement::{validate_measurement_identity, validate_running_binary};
use crate::prompts::PromptEntry;
use crate::types::OracleName;

pub const CALIBRATION_IDS: [&str; 4] = [
    "canonical_01",
    "canonical_02",
    "canonical_03",
    "canonical_05a",
];
pub const HOLDOUT_IDS: [&str; 4] = [
    "canonical_04",
    "canonical_05b",
    "canonical_05c",
    "canonical_05d",
];

fn capture_ids(diagnostic: bool) -> &'static [&'static str] {
    if diagnostic {
        &["canonical_03"]
    } else {
        &CALIBRATION_IDS
    }
}

fn diagnostic_case(request: &str) -> Result<(&'static str, usize)> {
    let value: serde_json::Value = serde_json::from_str(request)?;
    match value["prompt_id"].as_str() {
        Some("canonical_03") => Ok(("canonical_03", 2)),
        Some("regression_11") => Ok(("regression_11", 12)),
        _ => bail!("unsupported fixed layer diagnostic case"),
    }
}

#[derive(Debug, Serialize)]
pub struct CandidateCaptureEntry {
    pub prompt_id: String,
    pub filename: String,
    pub sha256: String,
    pub tensor_shape: (usize, usize),
    pub token_count: usize,
}

#[derive(Debug, Serialize)]
pub struct CandidateCaptureIndex {
    pub schema_version: u32,
    pub measurement_commit: String,
    pub measurement_tree: String,
    pub manifest_sha256: String,
    pub candidate_binary_sha256: String,
    pub runtime_sha256: String,
    pub kernel_scope: String,
    pub opened_fixture_ids: Vec<String>,
    pub sealed_holdout_ids: Vec<String>,
    pub captures: Vec<CandidateCaptureEntry>,
}

pub fn run_candidate_capture(
    model_path: &Path,
    manifest_path: &Path,
    prompts: &HashMap<String, PromptEntry>,
    output_dir: &Path,
    repo_root: &Path,
    measurement_commit: &str,
    measurement_tree: &str,
) -> Result<CandidateCaptureIndex> {
    if std::env::var("PYTHONHASHSEED").as_deref() != Ok("0")
        || std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":4096:8")
    {
        bail!("candidate observation requires the deterministic process environment");
    }
    let measurement =
        validate_measurement_identity(repo_root, measurement_commit, measurement_tree)?;
    validate_running_binary(repo_root)?;
    crate::measurement::validate_release_model(model_path)?;
    let trace_root = std::env::var_os("VLLM_OXIDE_INTERNAL_LAYER_TRACE_DIR");
    let diagnostic = trace_root.is_some();
    let mut selected_case = None;
    if let Some(root) = trace_root {
        let root = std::path::PathBuf::from(root).canonicalize()?;
        if output_dir != root.join("rust") || !root.join("request.json").is_file() {
            bail!("layer diagnostic must use its request root/rust destination");
        }
        selected_case = Some(diagnostic_case(&std::fs::read_to_string(
            root.join("request.json"),
        )?)?);
    }
    if output_dir.exists() || output_dir.is_symlink() {
        bail!("candidate capture output must be fresh and non-existing");
    }
    std::fs::create_dir(output_dir)
        .with_context(|| format!("creating candidate capture output {}", output_dir.display()))?;
    std::fs::set_permissions(output_dir, std::fs::Permissions::from_mode(0o700))?;
    let manifest_bytes = std::fs::read(manifest_path)?;
    let manifest = if selected_case == Some(("regression_11", 12)) {
        // The fixed post-failure diagnostic reads the original approved manifest.
        // This only captures raw rows; it cannot produce an accepting comparison.
        parse_manifest(manifest_path)?
    } else {
        parse_observation_manifest(manifest_path)?
    };
    let runtime_bytes = serde_json::to_vec(&manifest.runtime)?;
    let binary_bytes = std::fs::read(std::env::current_exe()?)?;
    let mut captures = Vec::new();
    let ids = selected_case.map_or_else(|| capture_ids(false).to_vec(), |(id, _)| vec![id]);
    for &prompt_id in &ids {
        let prompt = prompts.get(prompt_id).ok_or_else(|| {
            anyhow::anyhow!("candidate calibration prompt is missing: {prompt_id}")
        })?;
        let reference = manifest
            .fixtures
            .iter()
            .find(|fixture| {
                fixture.prompt_id == prompt_id && fixture.oracle == OracleName::Transformers
            })
            .ok_or_else(|| anyhow::anyhow!("reference metadata is missing: {prompt_id}"))?;
        let mut llm = LLM::new(
            Source::Local(model_path.to_path_buf()),
            fixed_engine_options(),
        )?;
        let filename = format!("{prompt_id}.candidate.jsonl");
        let destination = output_dir.join(&filename);
        let max_tokens = selected_case.map_or(reference.num_tokens as usize, |(_, count)| count);
        let (captured, sha256) = generate_with_preserved_capture(
            &mut llm,
            Prompt::Text(prompt.prompt.clone()),
            max_tokens,
            prompt_id,
            &destination,
        )?;
        if captured.tokens_by_input.len() != 1 || captured.tokens_by_input[0].len() != max_tokens {
            bail!("candidate capture token count does not match reference metadata");
        }
        if selected_case == Some(("canonical_03", 2)) && captured.tokens_by_input[0][0] != 151_667 {
            bail!("layer diagnostic first token differs from the shared prefix");
        }
        captures.push(CandidateCaptureEntry {
            prompt_id: prompt_id.to_string(),
            filename,
            sha256,
            tensor_shape: captured.tensor_shape,
            token_count: captured.tokens_by_input[0].len(),
        });
        drop(llm);
    }
    let index = CandidateCaptureIndex {
        schema_version: 1,
        measurement_commit: measurement.commit,
        measurement_tree: measurement.tree,
        manifest_sha256: format!("{:x}", Sha256::digest(&manifest_bytes)),
        candidate_binary_sha256: format!("{:x}", Sha256::digest(&binary_bytes)),
        runtime_sha256: format!("{:x}", Sha256::digest(&runtime_bytes)),
        kernel_scope: manifest.kernel_paths.comparison_scope(),
        opened_fixture_ids: ids.iter().map(|value| (*value).to_string()).collect(),
        sealed_holdout_ids: HOLDOUT_IDS
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        captures,
    };
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_dir.join("capture-index.json"))?;
    if diagnostic {
        let mut value = serde_json::to_value(&index)?;
        value["diagnostic_only"] = true.into();
        value["accepting"] = false.into();
        serde_json::to_writer_pretty(&mut output, &value)?;
    } else {
        serde_json::to_writer_pretty(&mut output, &index)?;
    }
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::{CALIBRATION_IDS, HOLDOUT_IDS};

    #[test]
    fn diagnostic_request_selects_only_the_fixed_case_and_row_boundary() -> anyhow::Result<()> {
        assert_eq!(
            super::diagnostic_case(r#"{"prompt_id":"regression_11"}"#)?,
            ("regression_11", 12)
        );
        assert_eq!(
            super::diagnostic_case(r#"{"prompt_id":"canonical_03"}"#)?,
            ("canonical_03", 2)
        );
        assert!(super::diagnostic_case(r#"{"prompt_id":"regression_12"}"#).is_err());
        Ok(())
    }

    #[test]
    fn candidate_capture_ids_are_disjoint_and_leave_the_holdout_sealed() {
        assert_eq!(super::capture_ids(false), CALIBRATION_IDS);
        assert_eq!(super::capture_ids(true), ["canonical_03"]);
        assert_eq!(
            CALIBRATION_IDS,
            [
                "canonical_01",
                "canonical_02",
                "canonical_03",
                "canonical_05a"
            ]
        );
        assert_eq!(
            HOLDOUT_IDS,
            [
                "canonical_04",
                "canonical_05b",
                "canonical_05c",
                "canonical_05d"
            ]
        );
        assert!(CALIBRATION_IDS.iter().all(|id| !HOLDOUT_IDS.contains(id)));
    }
}
