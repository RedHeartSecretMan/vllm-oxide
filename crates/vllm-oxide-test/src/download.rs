//! Download golden fixtures from a GitHub Release asset.
//!
//! Golden fixtures are stored as GitHub Release assets (not in git mainline).
//! This module fetches `manifest.json` and all `.safetensors` files listed in
//! it, verifies their SHA-256 digests, and writes them to a local cache
//! directory.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::manifest;
use crate::types::Manifest;

/// Download a golden fixture release by tag and verify all assets.
///
/// Returns the parsed manifest and the directory containing the downloaded
/// fixtures.
pub fn download_release(
    owner: &str,
    repo: &str,
    tag: &str,
    cache_dir: &Path,
) -> Result<(Manifest, PathBuf)> {
    let agent = ureq::Agent::new_with_defaults();

    // 1. Fetch the release metadata to get asset URLs.
    let release_url = format!(
        "https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    );
    tracing::info!("fetching release metadata from {release_url}");

    let mut release_resp = agent
        .get(&release_url)
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "vllm-oxide-golden-test/0.1")
        .call()
        .with_context(|| format!("fetching release {tag}"))?;

    let release_json: serde_json::Value = release_resp
        .body_mut()
        .read_json()
        .context("parsing release JSON")?;

    let assets = release_json["assets"]
        .as_array()
        .context("release has no assets array")?;

    // 2. Find manifest.json and parse it.
    let manifest_url = find_asset_url(assets, "manifest.json")?;
    tracing::info!("downloading manifest.json");

    let manifest_bytes = download_bytes(&agent, &manifest_url)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .context("parsing manifest.json")?;

    // 3. Create cache dir and write manifest.
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;
    let manifest_path = cache_dir.join("manifest.json");
    std::fs::write(&manifest_path, &manifest_bytes)?;

    // 4. Download and verify each fixture.
    for fixture in &manifest.fixtures {
        let expected_sha = &fixture.sha256;
        let fixture_url = find_asset_url(assets, &fixture.filename)?;
        let dest_path = cache_dir.join(&fixture.filename);

        if dest_path.exists() {
            // Check if cached file matches expected hash.
            let existing = std::fs::read(&dest_path)?;
            let actual = manifest::sha256_hex(&existing);
            if &actual == expected_sha {
                tracing::info!("{} already cached (hash verified)", fixture.filename);
                continue;
            }
            tracing::warn!(
                "{} cached but hash mismatch, re-downloading",
                fixture.filename
            );
        }

        tracing::info!("downloading {} ({} bytes)", fixture.filename, fixture.num_tokens);
        let bytes = download_bytes(&agent, &fixture_url)?;

        // Verify SHA-256.
        let actual = manifest::sha256_hex(&bytes);
        if &actual != expected_sha {
            anyhow::bail!(
                "SHA-256 mismatch for {}: expected {expected_sha}, got {actual}",
                fixture.filename,
            );
        }

        std::fs::write(&dest_path, &bytes)
            .with_context(|| format!("writing {}", dest_path.display()))?;
    }

    tracing::info!(
        "downloaded {} fixtures to {}",
        manifest.fixtures.len(),
        cache_dir.display()
    );

    Ok((manifest, cache_dir.to_path_buf()))
}

/// Load golden fixtures from a local directory (no download).
pub fn load_from_dir(dir: &Path) -> Result<(Manifest, PathBuf)> {
    let manifest_path = dir.join("manifest.json");
    let manifest = manifest::parse_manifest(&manifest_path)?;
    verify_fixture_hashes(dir, &manifest.fixtures)?;
    Ok((manifest, dir.to_path_buf()))
}

/// Verify all fixture files in a directory match their expected SHA-256.
fn verify_fixture_hashes(
    dir: &Path,
    fixtures: &[crate::types::FixtureMetadata],
) -> Result<()> {
    for fixture in fixtures {
        let path = dir.join(&fixture.filename);
        if !path.exists() {
            anyhow::bail!("fixture file not found: {}", path.display());
        }
        let bytes = std::fs::read(&path)?;
        let actual = manifest::sha256_hex(&bytes);
        if &actual != &fixture.sha256 {
            anyhow::bail!(
                "SHA-256 mismatch for {} in {}: expected {}, got {}",
                fixture.filename,
                dir.display(),
                fixture.sha256,
                actual,
            );
        }
    }
    Ok(())
}

/// Find an asset URL by filename in a GitHub release assets array.
fn find_asset_url(assets: &[serde_json::Value], filename: &str) -> Result<String> {
    for asset in assets {
        if asset["name"].as_str() == Some(filename) {
            return asset["url"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| anyhow::anyhow!("asset '{}' has no url field", filename));
        }
    }
    anyhow::bail!("asset '{}' not found in release", filename)
}

/// Download raw bytes from a URL using ureq.
///
/// Uses `Accept: application/octet-stream` for GitHub asset downloads.
/// Streams the response into a Vec<u8> while computing SHA-256 on the fly
/// (though the caller also verifies).
fn download_bytes(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>> {
    let mut response = agent
        .get(url)
        .header("Accept", "application/octet-stream")
        .header("User-Agent", "vllm-oxide-golden-test/0.1")
        .call()
        .with_context(|| format!("downloading {url}"))?;

    let status = response.status();
    if status != 200 {
        anyhow::bail!("HTTP {status} downloading {url}");
    }

    let mut buf = Vec::new();
    response
        .body_mut()
        .as_reader()
        .read_to_end(&mut buf)
        .with_context(|| format!("reading response body from {url}"))?;

    Ok(buf)
}
