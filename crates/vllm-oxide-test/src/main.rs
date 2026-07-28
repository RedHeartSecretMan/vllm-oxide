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
    compare_l1, compare_l1_regression, compare_l2, compare_l3,
    download, manifest, print_report, prompts,
};
use vllm_oxide::{
    EngineOptions, LLM, Prompt, Source,
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

    /// Path to the golden-gen prompts directory (canonical.jsonl).
    #[arg(long, default_value = "tools/golden-gen/prompts")]
    prompts_dir: PathBuf,
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

    let canonical_prompts = prompts::load_canonical_prompts(&cli.prompts_dir)?;

    let mut report = ComparisonReport {
        manifest_path: cli
            .manifest
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| cli.release_tag.unwrap_or_default()),
        model_path: model_path.display().to_string(),
        ..Default::default()
    };

    // 2. Run comparisons for each canonical fixture.
    // Run generate_logits once per fixture — it captures both per-step logits
    // (for L2) and tokens via argmax (for L1 with near-tie).
    for meta in &golden_manifest.fixtures {
        if meta.category != vllm_oxide_test::types::PromptCategory::Canonical {
            continue;
        }

        let prompt_entry = match canonical_prompts.get(&meta.prompt_id) {
            Some(e) => e,
            None => {
                tracing::warn!(
                    "prompt_id '{}' not found in canonical.jsonl — skipping",
                    meta.prompt_id
                );
                continue;
            }
        };

        // Batch prompts (canonical_05) have sub_prompts — skip in v0.1.
        if prompt_entry.sub_prompts.is_some() {
            tracing::info!(
                "[{}] skipping batch prompt (not supported in v0.1 generate_logits)",
                meta.prompt_id
            );
            continue;
        }

        let fixture = manifest::load_fixture(&fixture_dir.join(&meta.filename), meta)?;
        let prompt = Prompt::Text(prompt_entry.prompt.clone());
        let max_tokens = meta.num_tokens as usize;

        tracing::info!("[{}/L1+L2] loading engine", meta.prompt_id);
        let mut llm = LLM::new(
            Source::Local(model_path.clone()),
            EngineOptions::default(),
        )?;

        let logits = match llm.generate_logits(&prompt, max_tokens) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("generate_logits failed for {}: {e}", meta.prompt_id);
                continue;
            }
        };

        // Extract greedy tokens from logits via argmax.
        let logits_f32 = match logits.to_dtype(candle_core::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("logits to_dtype failed: {e}");
                continue;
            }
        };
        let logits_vals = logits_f32.flatten_all()?.to_vec1::<f32>()?;
        let vocab_size = if meta.logits_shape.1 > 0 {
            meta.logits_shape.1
        } else {
            logits_vals.len() / logits.dims()[0]
        };
        let n_steps = logits.dims()[0];

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let start = step * vocab_size;
            let end = start + vocab_size;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0u32;
            for (j, &val) in logits_vals[start..end].iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = j as u32;
                }
            }
            generated_tokens.push(max_idx);
        }

        if !cli.l2_only {
            let l1_result = compare_l1(
                &fixture,
                &generated_tokens,
                Some(&logits),
                tolerance,
                cli.epsilon,
            )?;
            report.l1_results.push(l1_result);
        }

        if !cli.l1_only {
            // ADR-0005: L2 uses same-prefix comparison (skips divergent steps)
            let l2_result = compare_l2(&fixture, &logits, &generated_tokens, tolerance)?;
            report.l2_results.push(l2_result);
        }

        if cli.debug {
            let l3_result = compare_l3(&golden_manifest, &fixture_dir, &meta.prompt_id)?;
            report.l3_results.push(l3_result);
        }
    }

    // 2b. Run L1 comparison for regression fixtures (no logits available).
    for meta in &golden_manifest.fixtures {
        if meta.category != vllm_oxide_test::types::PromptCategory::Regression {
            continue;
        }
        if cli.l2_only {
            continue;
        }

        let prompt_entry = match canonical_prompts.get(&meta.prompt_id) {
            Some(e) => e,
            None => {
                tracing::warn!(
                    "prompt_id '{}' not found in canonical.jsonl — skipping",
                    meta.prompt_id
                );
                continue;
            }
        };

        if prompt_entry.sub_prompts.is_some() {
            continue;
        }

        let fixture = manifest::load_fixture(&fixture_dir.join(&meta.filename), meta)?;
        let prompt = Prompt::Text(prompt_entry.prompt.clone());
        let max_tokens = meta.num_tokens as usize;

        tracing::info!("[{}/L1] loading engine for regression", meta.prompt_id);
        let mut llm = LLM::new(
            Source::Local(model_path.clone()),
            EngineOptions::default(),
        )?;

        let logits = match llm.generate_logits(&prompt, max_tokens) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("generate_logits failed for {}: {e}", meta.prompt_id);
                continue;
            }
        };

        let logits_f32 = match logits.to_dtype(candle_core::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("logits to_dtype failed: {e}");
                continue;
            }
        };
        let logits_vals = logits_f32.flatten_all()?.to_vec1::<f32>()?;
        let vocab_size = if meta.logits_shape.1 > 0 {
            meta.logits_shape.1
        } else {
            logits_vals.len() / logits.dims()[0]
        };
        let n_steps = logits.dims()[0];

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let start = step * vocab_size;
            let end = start + vocab_size;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0u32;
            for (j, &val) in logits_vals[start..end].iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = j as u32;
                }
            }
            generated_tokens.push(max_idx);
        }

        let l1_result = compare_l1_regression(
            &fixture,
            &generated_tokens,
            &golden_manifest.regression_skip_map,
        )?;
        report.l1_results.push(l1_result);
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
