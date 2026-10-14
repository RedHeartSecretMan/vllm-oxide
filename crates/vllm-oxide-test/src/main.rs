//! CLI entrypoint for the golden comparison runner.
//!
//! ```text
//! vllm-oxide-test --model-path /path/to/Qwen3-0.6B \
//!     --manifest /path/to/goldens/manifest.json
//!
//! vllm-oxide-test --model-path /path/to/Qwen3-0.6B \
//!     --release-tag goldens-v0.1 --cache-dir /tmp/goldens
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;

use vllm_oxide::{EngineOptions, Prompt, Source, LLM};
use vllm_oxide_test::{
    self, compare_l1, compare_l1_regression, compare_l2, download, manifest, print_report, prompts,
    ComparisonReport,
};

/// Validate the vllm-oxide Rust engine against golden fixtures.
///
/// This is a release gate — run it manually on a GPU before tagging a release.
/// CI green (CPU property tests) does NOT imply numerical correctness.
#[derive(Parser, Debug)]
#[command(name = "vllm-oxide-test", version, about)]
struct Cli {
    /// Path to the model directory (containing config.json, tokenizer.json, weights).
    #[arg(long)]
    model_path: PathBuf,

    /// Path to a local manifest.json + fixture directory.
    #[arg(long, group = "source")]
    manifest: Option<PathBuf>,

    /// GitHub release tag to download goldens from.
    #[arg(long, group = "source")]
    release_tag: Option<String>,

    /// GitHub owner/repo (default: RedHeartSecretMan/vllm-oxide).
    #[arg(long, default_value = "RedHeartSecretMan/vllm-oxide")]
    repo: String,

    /// Local cache directory for downloaded goldens.
    #[arg(long, default_value = "/tmp/vllm-oxide-goldens")]
    cache_dir: PathBuf,

    /// Override the near-tie epsilon for L1 (default: 2× manifest atol).
    #[arg(long)]
    epsilon: Option<f64>,

    /// Request L3 per-layer comparison (currently unavailable; returns an error).
    #[arg(long)]
    debug: bool,

    /// Output results as JSON instead of human-readable.
    #[arg(long)]
    json: bool,

    /// Only run L1 comparison (skip L2).
    #[arg(long, conflicts_with = "l2_only")]
    l1_only: bool,

    /// Only run L2 comparison (skip L1).
    #[arg(long)]
    l2_only: bool,

    /// Path to the golden-gen prompts directory (canonical.jsonl).
    #[arg(long, default_value = "tools/golden-gen/prompts")]
    prompts_dir: PathBuf,
}

struct PreparedFixture {
    metadata: vllm_oxide_test::FixtureMetadata,
    fixture: vllm_oxide_test::FixtureData,
    prompt: prompts::PromptEntry,
}

fn prepare_fixtures(
    manifest: &vllm_oxide_test::Manifest,
    fixture_dir: &Path,
    prompts: &HashMap<String, prompts::PromptEntry>,
    l2_only: bool,
) -> Result<Vec<PreparedFixture>> {
    use vllm_oxide_test::types::{OracleName, PromptCategory};
    if manifest.schema_version != 1
        || manifest.fixtures.is_empty()
        || manifest.model.vocab_size == 0
    {
        anyhow::bail!("schema1 comparison requires a non-empty fixture inventory and vocabulary");
    }
    if !manifest.tolerance.atol.is_finite() || manifest.tolerance.atol < 0.0 {
        anyhow::bail!("comparison requires a finite non-negative atol");
    }
    let mut seen = HashSet::new();
    let mut prepared = Vec::new();
    for meta in &manifest.fixtures {
        if meta.filename.len() == ".safetensors".len()
            || !meta.filename.ends_with(".safetensors")
            || !meta
                .filename
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || matches!(v, b'.' | b'_' | b'-'))
            || !seen.insert(meta.filename.clone())
        {
            anyhow::bail!("unsafe or duplicate fixture filename: {}", meta.filename);
        }
        if meta.oracle == OracleName::Fake {
            anyhow::bail!("fake fixtures cannot serve as numerical acceptance evidence");
        }
        if l2_only && meta.category == PromptCategory::Regression {
            anyhow::bail!(
                "L2-only comparison does not support regression fixtures without full logits"
            );
        }
        let prompt = prompts.get(&meta.prompt_id).with_context(|| {
            format!(
                "required prompt {} is missing from the corpus",
                meta.prompt_id
            )
        })?;
        let expected_category = match meta.category {
            PromptCategory::Canonical => "canonical",
            PromptCategory::Regression => "regression",
        };
        if prompt.category != expected_category || prompt.sub_prompts.is_some() {
            anyhow::bail!(
                "unsupported or inconsistent prompt family for {}",
                meta.prompt_id
            );
        }
        let path = fixture_dir.join(&meta.filename);
        let bytes =
            std::fs::read(&path).with_context(|| format!("reading fixture {}", path.display()))?;
        if manifest::sha256_hex(&bytes) != meta.sha256 {
            anyhow::bail!("fixture checksum mismatch for {}", meta.filename);
        }
        let fixture = manifest::load_fixture(&path, meta)?;
        if fixture.n_prompt_tokens <= 0
            || fixture
                .token_ids
                .iter()
                .any(|id| usize::try_from(*id).map_or(true, |id| id >= manifest.model.vocab_size))
            || fixture.top5_indices.as_ref().is_some_and(|ids| {
                ids.iter().any(|id| {
                    usize::try_from(*id).map_or(true, |id| id >= manifest.model.vocab_size)
                })
            })
            || fixture
                .logits
                .as_ref()
                .is_some_and(|values| values.iter().any(|v| !v.is_finite()))
            || fixture
                .top5_logits
                .as_ref()
                .is_some_and(|values| values.iter().any(|v| !v.is_finite()))
            || (meta.category == PromptCategory::Canonical
                && meta.logits_shape.1 != manifest.model.vocab_size)
        {
            anyhow::bail!(
                "fixture values violate the model contract for {}",
                meta.filename
            );
        }
        prepared.push(PreparedFixture {
            metadata: meta.clone(),
            fixture,
            prompt: prompt.clone(),
        });
    }
    Ok(prepared)
}

fn generated_tokens(
    logits: &candle_core::Tensor,
    expected_rows: usize,
    expected_vocab: usize,
) -> Result<Vec<u32>> {
    if logits.dims() != [expected_rows, expected_vocab] || expected_rows == 0 || expected_vocab == 0
    {
        anyhow::bail!(
            "generated logits do not match the requested non-empty [steps, vocabulary] shape"
        );
    }
    let values = logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if expected_rows.checked_mul(expected_vocab) != Some(values.len())
        || values.iter().any(|v| !v.is_finite())
    {
        anyhow::bail!("generated logits have inconsistent geometry or non-finite values");
    }
    values
        .chunks_exact(expected_vocab)
        .map(|row| {
            let mut best = 0;
            for index in 1..row.len() {
                if row[index] > row[best] {
                    best = index;
                }
            }
            u32::try_from(best).context("logit column exceeds u32 token ID range")
        })
        .collect()
}

fn run(cli: Cli) -> Result<()> {
    if cli.debug {
        anyhow::bail!(
            "L3 comparison is not implemented; --debug cannot provide acceptance evidence"
        );
    }
    let (golden_manifest, fixture_dir) = if let Some(ref path) = cli.manifest {
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        (manifest::parse_manifest(path)?, directory.to_path_buf())
    } else if let Some(ref tag) = cli.release_tag {
        let parts: Vec<_> = cli.repo.split('/').collect();
        if parts.len() != 2 || parts.iter().any(|part| part.is_empty()) {
            anyhow::bail!("invalid repo format: expected owner/repo");
        }
        download::download_release(parts[0], parts[1], tag, &cli.cache_dir)?
    } else {
        anyhow::bail!("either --manifest or --release-tag must be provided");
    };
    let tolerance = &golden_manifest.tolerance;
    let epsilon = cli.epsilon.unwrap_or(tolerance.atol * 2.0);
    if !epsilon.is_finite() || epsilon < 0.0 {
        anyhow::bail!("near-tie epsilon must be finite and non-negative");
    }
    let prompts = prompts::load_all_prompts(&cli.prompts_dir)?;
    // Resolve and verify every requested case before the first model initialization.
    let prepared = prepare_fixtures(&golden_manifest, &fixture_dir, &prompts, cli.l2_only)?;
    let expected_l1 = if cli.l2_only { 0 } else { prepared.len() };
    let expected_l2 = if cli.l1_only {
        0
    } else {
        prepared
            .iter()
            .filter(|case| {
                case.metadata.category == vllm_oxide_test::types::PromptCategory::Canonical
            })
            .count()
    };
    let mut report = ComparisonReport {
        manifest_path: cli
            .manifest
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| cli.release_tag.clone().unwrap_or_default()),
        model_path: cli.model_path.display().to_string(),
        ..Default::default()
    };
    for case in prepared {
        let meta = case.metadata;
        tracing::info!("[{}] loading engine", meta.prompt_id);
        let mut llm = LLM::new(
            Source::Local(cli.model_path.clone()),
            EngineOptions::default(),
        )
        .with_context(|| format!("initializing engine for {}", meta.prompt_id))?;
        let logits = llm
            .generate_logits(&Prompt::Text(case.prompt.prompt), meta.num_tokens as usize)
            .with_context(|| format!("generating logits for {}", meta.prompt_id))?;
        let tokens = generated_tokens(
            &logits,
            meta.num_tokens as usize,
            golden_manifest.model.vocab_size,
        )?;
        match meta.category {
            vllm_oxide_test::types::PromptCategory::Canonical => {
                if !cli.l2_only {
                    report.l1_results.push(compare_l1(
                        &case.fixture,
                        &tokens,
                        Some(&logits),
                        tolerance,
                        Some(epsilon),
                    )?);
                }
                if !cli.l1_only {
                    report
                        .l2_results
                        .push(compare_l2(&case.fixture, &logits, &tokens, tolerance)?);
                }
            }
            vllm_oxide_test::types::PromptCategory::Regression => {
                report.l1_results.push(compare_l1_regression(
                    &case.fixture,
                    &tokens,
                    &golden_manifest.regression_skip_map,
                )?);
            }
        }
    }
    if report.l1_results.len() != expected_l1 || report.l2_results.len() != expected_l2 {
        anyhow::bail!("not every requested fixture reached its required comparison layers");
    }
    if cli.json {
        println!(
            "{}",
            vllm_oxide_test::report::json_report(&report, tolerance)
        );
    } else {
        print_report(&report, tolerance);
    }
    if !report.overall_passed() {
        anyhow::bail!("numerical comparison failed");
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    run(Cli::parse())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn corpus(root: &Path) {
        std::fs::write(
            root.join("canonical.jsonl"),
            "{\"id\":\"canonical_01\",\"category\":\"canonical\",\"prompt\":\"hello\"}\n",
        )
        .unwrap();
        std::fs::write(root.join("regression.jsonl"), "").unwrap();
    }

    fn fixture_manifest(root: &Path) -> vllm_oxide_test::Manifest {
        let mut header = serde_json::to_vec(&json!({
            "token_ids": {"dtype":"I64", "shape":[1], "data_offsets":[0,8]},
            "n_prompt_tokens": {"dtype":"I64", "shape":[], "data_offsets":[8,16]},
            "logits": {"dtype":"F32", "shape":[1,2], "data_offsets":[16,24]}
        }))
        .unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(1_i64.to_le_bytes());
        bytes.extend(1_i64.to_le_bytes());
        bytes.extend(0_f32.to_le_bytes());
        bytes.extend(1_f32.to_le_bytes());
        let filename = "canonical_01.transformers.safetensors";
        std::fs::write(root.join(filename), &bytes).unwrap();
        serde_json::from_value(json!({
            "schema_version":1, "generated_at":"test",
            "model":{"id":"model", "revision":"revision", "arch":"Qwen3ForCausalLM", "dtype":"bfloat16", "vocab_size":2},
            "oracle_versions":{"transformers":"test", "vllm":"test"},
            "generation":{"canonical_max_tokens":1, "regression_max_tokens":1, "temperature":0.0, "attn_implementation":"sdpa"},
            "tolerance":{"atol":0.0, "observed_max_abs_diff":0.0, "calibration_factor":2.0, "method":"test"},
            "fixtures":[{"prompt_id":"canonical_01", "category":"canonical", "oracle":"transformers", "num_tokens":1,
                "logits_dtype":"float32", "logits_shape":[1,2], "sha256":manifest::sha256_hex(&bytes), "filename":filename}]
        })).unwrap()
    }

    fn cli(root: &Path) -> Cli {
        Cli::try_parse_from([
            "vllm-oxide-test".to_string(),
            "--manifest".into(),
            root.join("manifest.json").display().to_string(),
            "--prompts-dir".into(),
            root.display().to_string(),
            "--model-path".into(),
            root.join("missing-model").display().to_string(),
        ])
        .unwrap()
    }

    #[test]
    fn missing_corrupt_or_undeclared_cases_fail_before_model_initialization() {
        let root = tempfile::tempdir().unwrap();
        corpus(root.path());
        let mut manifest = fixture_manifest(root.path());
        let prompts = prompts::load_all_prompts(root.path()).unwrap();
        assert_eq!(
            prepare_fixtures(&manifest, root.path(), &prompts, false)
                .unwrap()
                .len(),
            1
        );
        manifest.fixtures[0].prompt_id = "missing".into();
        assert!(prepare_fixtures(&manifest, root.path(), &prompts, false).is_err());
        manifest.fixtures[0].prompt_id = "canonical_01".into();
        let path = root.path().join(&manifest.fixtures[0].filename);
        std::fs::remove_file(&path).unwrap();
        assert!(prepare_fixtures(&manifest, root.path(), &prompts, false).is_err());
        std::fs::write(&path, b"bad").unwrap();
        assert!(prepare_fixtures(&manifest, root.path(), &prompts, false).is_err());
        manifest.fixtures[0].sha256 = manifest::sha256_hex(b"bad");
        assert!(prepare_fixtures(&manifest, root.path(), &prompts, false).is_err());
        manifest.fixtures.clear();
        assert!(prepare_fixtures(&manifest, root.path(), &prompts, false).is_err());
    }

    #[test]
    fn unsupported_l3_and_regression_l2_are_explicit_errors() {
        let root = tempfile::tempdir().unwrap();
        let mut args = cli(root.path());
        args.debug = true;
        assert!(run(args)
            .unwrap_err()
            .to_string()
            .contains("L3 comparison is not implemented"));
        let mut manifest = fixture_manifest(root.path());
        assert!(vllm_oxide_test::compare_l3(&manifest, root.path(), "canonical_01").is_err());
        manifest.fixtures[0].category = vllm_oxide_test::types::PromptCategory::Regression;
        assert!(prepare_fixtures(&manifest, root.path(), &HashMap::new(), true).is_err());
    }

    #[test]
    fn malformed_generated_logits_return_errors_instead_of_panicking_or_selecting_zero() {
        let device = &candle_core::Device::Cpu;
        assert!(generated_tokens(
            &candle_core::Tensor::new(&[1.0_f32, 2.0], device).unwrap(),
            1,
            2
        )
        .is_err());
        assert!(generated_tokens(
            &candle_core::Tensor::new(&[[f32::NAN, 2.0]], device).unwrap(),
            1,
            2
        )
        .is_err());
        assert_eq!(
            generated_tokens(
                &candle_core::Tensor::new(&[[2.0_f32, 2.0]], device).unwrap(),
                1,
                2
            )
            .unwrap(),
            [0]
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn actual_cli_path_propagates_model_initialization_failure() {
        let root = tempfile::tempdir().unwrap();
        corpus(root.path());
        let manifest = fixture_manifest(root.path());
        std::fs::write(
            root.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let error = run(cli(root.path())).unwrap_err();
        assert!(error.to_string().contains("initializing engine"));
    }
}
