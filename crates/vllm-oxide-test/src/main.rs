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

use vllm_oxide_test::{download, manifest, print_report, prompts, DriverOptions};

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
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
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

    let canonical_prompts = prompts::load_canonical_prompts(&cli.prompts_dir)?;

    // 2. Run all comparisons via the driver.
    let opts = DriverOptions {
        l1_only: cli.l1_only,
        l2_only: cli.l2_only,
        debug: cli.debug,
        epsilon: cli.epsilon,
    };
    let mut report =
        vllm_oxide_test::run_comparison(&golden_manifest, &fixture_dir, &cli.model_path, &canonical_prompts, &opts)?;

    report.manifest_path = cli
        .manifest
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| cli.release_tag.clone().unwrap_or_default());
    report.model_path = cli.model_path.display().to_string();

    // 3. Print report.
    let tolerance = &golden_manifest.tolerance;
    if cli.json {
        println!(
            "{}",
            vllm_oxide_test::report::json_report(&report, tolerance)
        );
    } else {
        print_report(&report, tolerance);
    }

    if !report.overall_passed() {
        std::process::exit(1);
    }

    Ok(())
}
