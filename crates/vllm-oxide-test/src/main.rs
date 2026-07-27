//! CLI entrypoint for the golden comparison runner.
//!
//! ```text
//! vllm-oxide-test --model-path /path/to/Qwen3-0.6B \
//!     --manifest /path/to/goldens/manifest.json
//!
//! vllm-oxide-test --model-path /path/to/Qwen3-0.6B \
//!     --release-tag goldens-v0.1 --cache-dir /tmp/goldens
//! ```

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use vllm_oxide_test::{
    self, ComparisonReport,
    compare_l2, compare_l3,
    download, manifest, print_report,
};
use vllm_oxide::{
    EngineOptions, LLM, Prompt, SamplingParams, Source,
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

    /// Enable L3 per-layer activations comparison (debug-only, skeleton).
    #[arg(long)]
    debug: bool,

    /// Output results as JSON instead of human-readable.
    #[arg(long)]
    json: bool,

    /// Only run L1 comparison (skip L2).
    #[arg(long)]
    l1_only: bool,

    /// Only run L2 comparison (skip L1).
    #[arg(long)]
    l2_only: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    // 1. Load or download golden fixtures.
    let (golden_manifest, fixture_dir) = if let Some(ref manifest_path) = cli.manifest {
        let dir = manifest_path
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let m = manifest::parse_manifest(manifest_path)?;
        (m, dir)
    } else if let Some(ref tag) = cli.release_tag {
        let parts: Vec<&str> = cli.repo.split('/').collect();
        if parts.len() != 2 {
            anyhow::bail!("invalid repo format '{}': expected 'owner/repo'", cli.repo);
        }
        download::download_release(parts[0], parts[1], tag, &cli.cache_dir)?
    } else {
        anyhow::bail!("either --manifest or --release-tag must be provided");
    };

    let tolerance = &golden_manifest.tolerance;
    let model_path = cli.model_path.clone();

    let mut report = ComparisonReport {
        manifest_path: cli
            .manifest
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| cli.release_tag.unwrap_or_default()),
        model_path: model_path.display().to_string(),
        known_deviations: golden_manifest.cross_validation.clone(),
        ..Default::default()
    };

    // 2. Run comparisons for each canonical fixture.
    // L1 and L2 use separate LLM instances because generate() and generate_logits()
    // each consume the engine state. For a release gate that runs once per release,
    // loading the model twice per fixture is acceptable.
    for meta in &golden_manifest.fixtures {
        if meta.category != vllm_oxide_test::types::PromptCategory::Canonical {
            continue;
        }

        let fixture = manifest::load_fixture(&fixture_dir.join(&meta.filename), meta)?;
        let prompt_text = format!("vllm_oxide_test fixture {}", meta.prompt_id);
        let prompt = Prompt::Text(prompt_text);
        let max_tokens = meta.num_tokens as usize;

        // ── L1: Token exact match via LLM::generate ─────────
        if !cli.l2_only {
            tracing::info!("[{}/L1] loading engine for token comparison", meta.prompt_id);
            let mut llm = LLM::new(
                Source::Local(model_path.clone()),
                EngineOptions::default(),
            )?;

            let output = llm.generate(
                &[prompt.clone()],
                &[SamplingParams {
                    temperature: 0.0,
                    max_tokens,
                    ignore_eos: true,
                    ..SamplingParams::default()
                }],
            )?;

            let generated_tokens = &output[0].token_ids;
            // L1 without logits for near-tie (we don't have per-step logits from generate alone).
            let l1_result = vllm_oxide_test::l1::compare_l1_tokens_only(&fixture, generated_tokens);
            report.l1_results.push(l1_result);
        }

        // ── L2: Logits tensor comparison via generate_logits ─
        if !cli.l1_only {
            tracing::info!("[{}/L2] loading engine for logits comparison", meta.prompt_id);
            let mut llm = LLM::new(
                Source::Local(model_path.clone()),
                EngineOptions::default(),
            )?;

            match llm.generate_logits(&prompt, max_tokens) {
                Ok(logits) => {
                    let l2_result = compare_l2(&fixture, &logits, tolerance)?;
                    report.l2_results.push(l2_result);
                }
                Err(e) => {
                    tracing::warn!("L2 skipped for {}: {e}", meta.prompt_id);
                    report.skipped_l2.push(meta.prompt_id.clone());
                }
            }
        }

        // ── L3: Per-layer activations (debug-only) ──────────
        if cli.debug {
            let l3_result = compare_l3(&golden_manifest, &fixture_dir, &meta.prompt_id)?;
            report.l3_results.push(l3_result);
        }
    }

    // 3. Print report.
    if cli.json {
        println!("{}", vllm_oxide_test::report::json_report(&report, tolerance));
    } else {
        print_report(&report, tolerance);
    }

    if !report.overall_passed() {
        std::process::exit(1);
    }

    Ok(())
}
