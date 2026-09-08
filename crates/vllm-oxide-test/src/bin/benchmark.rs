use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "vllm-oxide-benchmark")]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    prompts_dir: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    repo_root: PathBuf,
    #[arg(long)]
    measurement_commit: String,
    #[arg(long)]
    measurement_tree: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.prompts_dir.canonicalize()?
        != cli
            .repo_root
            .join("tools/golden-gen/prompts")
            .canonicalize()?
    {
        bail!("performance prompts must come from the reviewed source checkout");
    }
    let prompts = vllm_oxide_test::prompts::load_all_prompts(&cli.prompts_dir)?;
    vllm_oxide_test::benchmark::run_release_benchmark(
        &cli.model_path,
        &prompts,
        &cli.output,
        &cli.repo_root,
        &cli.measurement_commit,
        &cli.measurement_tree,
    )?;
    Ok(())
}
