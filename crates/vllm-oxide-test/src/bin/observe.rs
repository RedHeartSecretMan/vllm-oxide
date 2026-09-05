use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "vllm-oxide-observe")]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    prompts_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    repo_root: PathBuf,
    #[arg(long)]
    measurement_commit: String,
    #[arg(long)]
    measurement_tree: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let prompts = vllm_oxide_test::prompts::load_all_prompts(&cli.prompts_dir)?;
    vllm_oxide_test::observation::run_candidate_capture(
        &cli.model_path,
        &cli.manifest,
        &prompts,
        &cli.output_dir,
        &cli.repo_root,
        &cli.measurement_commit,
        &cli.measurement_tree,
    )?;
    Ok(())
}
