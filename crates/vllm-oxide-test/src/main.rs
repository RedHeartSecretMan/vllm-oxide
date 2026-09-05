//! CLI entrypoint for the golden comparison runner.
//!
//! ```text
//! vllm-oxide-test --model-path /path/to/Qwen3-0.6B \
//!     --manifest /path/to/goldens/manifest.json
//!
//! vllm-oxide-test --model-path /path/to/Qwen3-0.6B \
//!     --release-tag goldens-v0.2 --cache-dir /tmp/goldens
//! ```

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use vllm_oxide_test::{download, manifest, print_report, prompts, DriverOptions};

/// Validate the vllm-oxide Rust engine against golden fixtures.
///
/// This is a release gate — run it manually on a GPU before tagging a release.
/// CI green (CPU property tests) does NOT imply numerical correctness.
#[derive(Parser, Debug)]
#[command(name = "vllm-oxide-test", version, about)]
struct Cli {
    /// Release gate mode. Holdout access exists only in authoritative mode.
    #[arg(long, value_enum)]
    mode: GateMode,

    /// Tracked, Definition-approved calibration observation.
    #[arg(long, required_if_eq("mode", "authoritative"))]
    approved_observation: Option<PathBuf>,

    /// Path to the model directory (containing config.json, tokenizer.json, weights).
    #[arg(long)]
    model_path: PathBuf,

    /// Reviewed repository whose executable bytes produced this measurement.
    #[arg(long)]
    repo_root: PathBuf,

    /// Frozen measurement commit containing all executable and workflow bytes.
    #[arg(long)]
    measurement_commit: String,

    /// Tree object belonging to the frozen measurement commit.
    #[arg(long)]
    measurement_tree: String,

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

    /// Fresh private directory that retains candidate captures for replay audit.
    #[arg(long)]
    capture_dir: PathBuf,
}

#[derive(Clone, Debug, ValueEnum)]
enum GateMode {
    Authoritative,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let measurement = vllm_oxide_test::measurement::validate_measurement_identity(
        &cli.repo_root,
        &cli.measurement_commit,
        &cli.measurement_tree,
    )?;
    vllm_oxide_test::measurement::validate_running_binary(&cli.repo_root)?;
    vllm_oxide_test::measurement::validate_release_model(&cli.model_path)?;

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

    let all_prompts = prompts::load_all_prompts(&cli.prompts_dir)?;
    let approval = cli
        .approved_observation
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("authoritative mode requires --approved-observation"))?;
    vllm_oxide_test::approval::validate_authoritative_approval(
        &golden_manifest,
        approval,
        &cli.repo_root,
        &measurement,
    )?;
    if cli.capture_dir.exists() || cli.capture_dir.is_symlink() {
        anyhow::bail!("authoritative capture directory must be fresh and non-existing");
    }
    std::fs::create_dir(&cli.capture_dir)?;
    std::fs::set_permissions(&cli.capture_dir, std::fs::Permissions::from_mode(0o700))?;

    // 2. Run all comparisons via the driver.
    let opts = DriverOptions {
        l1_only: cli.l1_only,
        l2_only: cli.l2_only,
        debug: cli.debug,
        capture_dir: Some(cli.capture_dir.clone()),
    };
    let mut report = vllm_oxide_test::run_comparison(
        &golden_manifest,
        &fixture_dir,
        &cli.model_path,
        &all_prompts,
        &opts,
    )?;

    report.manifest_path = cli
        .manifest
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| cli.release_tag.clone().unwrap_or_default());
    report.model_path = cli.model_path.display().to_string();

    // 3. Print report.
    if cli.json {
        println!(
            "{}",
            vllm_oxide_test::report::json_report(
                &report,
                &golden_manifest.tolerance_policy,
                &golden_manifest.baseline_calibration,
                &golden_manifest.calibrated_fixtures,
            )
        );
    } else {
        print_report(
            &report,
            &golden_manifest.tolerance_policy,
            &golden_manifest.baseline_calibration,
            &golden_manifest.calibrated_fixtures,
        );
    }

    if !report.overall_passed() {
        std::process::exit(1);
    }

    Ok(())
}
