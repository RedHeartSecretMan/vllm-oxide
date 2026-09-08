//! Exercise the production downloader from a fresh local two-asset source.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::Parser;
use vllm_oxide_test::download::{download_release_with, ReleaseSource};

#[derive(Parser)]
struct Cli {
    /// Verify schema5 transport only; mathematical acceptance is checked separately.
    #[arg(long)]
    layered_transport: bool,
    #[arg(long)]
    bundle_dir: PathBuf,
    #[arg(long)]
    cache_dir: PathBuf,
}

struct LocalBundle(PathBuf);

impl ReleaseSource for LocalBundle {
    fn asset_names(&self, _: &str, _: &str, _: &str) -> Result<Vec<String>> {
        std::fs::read_dir(&self.0)?
            .map(|entry| {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    bail!("bundle contains a non-regular asset");
                }
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("non-UTF-8 asset name"))
            })
            .collect()
    }

    fn fetch_asset(&self, _: &str, _: &str, _: &str, name: &str, destination: &Path) -> Result<()> {
        std::fs::copy(self.0.join(name), destination)?;
        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.cache_dir.exists() || cli.cache_dir.is_symlink() {
        bail!("clean-consumer verification requires a non-existing cache directory");
    }
    if cli.layered_transport {
        let installed =
            vllm_oxide_test::release_transport::install_transport(&cli.bundle_dir, &cli.cache_dir)?;
        println!(
            "{}",
            serde_json::json!({"protocol":"layered-accuracy-v1",
            "schema_version":5,"transport_verified":true,"accepting":false,"installed":installed})
        );
        return Ok(());
    }
    let (manifest, _) = download_release_with(
        &LocalBundle(cli.bundle_dir),
        "RedHeartSecretMan",
        "vllm-oxide",
        "goldens-v0.2",
        &cli.cache_dir,
    )?;
    println!(
        "{}",
        serde_json::json!({"verified": true, "fixtures": manifest.fixtures.len(),
        "archive_sha256": manifest.archive.sha256})
    );
    Ok(())
}
