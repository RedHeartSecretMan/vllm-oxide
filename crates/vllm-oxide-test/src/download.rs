//! Fetch, verify, and atomically install a golden asset bundle.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use flate2::bufread::GzDecoder;
use sha2::{Digest, Sha256};

use crate::manifest;
use crate::types::{FixtureMetadata, Manifest, OracleRole, PromptCategory};

const MANIFEST_FILENAME: &str = "manifest.json";
const ARCHIVE_FILENAME: &str = "goldens-v0.2.tar.gz";
const GOLDEN_VERSION: &str = "goldens-v0.2";
const SAFETENSORS_HEADER_LIMIT: u64 = 1024 * 1024;
static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

/// Adapter at the remote-release seam.
///
/// Production uses the GitHub adapter; tests provide a local fake. Implementations
/// must write the requested asset to `destination` and return every release asset
/// name from `asset_names`, including names outside this contract.
pub trait ReleaseSource {
    fn asset_names(&self, owner: &str, repo: &str, tag: &str) -> Result<Vec<String>>;

    fn fetch_asset(
        &self,
        owner: &str,
        repo: &str,
        tag: &str,
        name: &str,
        destination: &Path,
    ) -> Result<()>;
}

struct GitHubReleaseSource {
    agent: ureq::Agent,
}

impl GitHubReleaseSource {
    fn new() -> Self {
        Self {
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    fn release_assets(&self, owner: &str, repo: &str, tag: &str) -> Result<Vec<RemoteAsset>> {
        let release_url =
            format!("https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}");
        tracing::info!("fetching release metadata from {release_url}");
        let mut response = self
            .agent
            .get(&release_url)
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "vllm-oxide-golden-test/0.2.0")
            .call()
            .with_context(|| format!("fetching release {tag}"))?;
        let release_json: serde_json::Value = response
            .body_mut()
            .read_json()
            .context("parsing release JSON")?;
        let assets = release_json["assets"]
            .as_array()
            .context("release has no assets array")?;
        assets
            .iter()
            .map(|asset| {
                let name = asset["name"]
                    .as_str()
                    .context("release asset has no name")?;
                let url = asset["url"].as_str().context("release asset has no url")?;
                Ok(RemoteAsset {
                    name: name.to_string(),
                    url: url.to_string(),
                })
            })
            .collect()
    }
}

struct RemoteAsset {
    name: String,
    url: String,
}

impl ReleaseSource for GitHubReleaseSource {
    fn asset_names(&self, owner: &str, repo: &str, tag: &str) -> Result<Vec<String>> {
        Ok(self
            .release_assets(owner, repo, tag)?
            .into_iter()
            .map(|asset| asset.name)
            .collect())
    }

    fn fetch_asset(
        &self,
        owner: &str,
        repo: &str,
        tag: &str,
        name: &str,
        destination: &Path,
    ) -> Result<()> {
        let matching: Vec<_> = self
            .release_assets(owner, repo, tag)?
            .into_iter()
            .filter(|asset| asset.name == name)
            .collect();
        let [asset] = matching.as_slice() else {
            anyhow::bail!("release must contain exactly one asset named {name}");
        };
        let mut response = self
            .agent
            .get(&asset.url)
            .header("Accept", "application/octet-stream")
            .header("User-Agent", "vllm-oxide-golden-test/0.2.0")
            .call()
            .with_context(|| format!("downloading release asset {name}"))?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .with_context(|| format!("creating {}", destination.display()))?;
        std::io::copy(&mut response.body_mut().as_reader(), &mut output)
            .with_context(|| format!("writing {}", destination.display()))?;
        output.sync_all()?;
        Ok(())
    }
}

/// Download a golden release from GitHub and atomically install its verified bundle.
pub fn download_release(
    owner: &str,
    repo: &str,
    tag: &str,
    cache_dir: &Path,
) -> Result<(Manifest, PathBuf)> {
    download_release_with(&GitHubReleaseSource::new(), owner, repo, tag, cache_dir)
}

/// Fetch through an injected release adapter and install only after complete validation.
pub fn download_release_with<S: ReleaseSource>(
    source: &S,
    owner: &str,
    repo: &str,
    tag: &str,
    cache_root: &Path,
) -> Result<(Manifest, PathBuf)> {
    validate_asset_names(&source.asset_names(owner, repo, tag)?)?;

    let version_root = cache_root.join(GOLDEN_VERSION);
    std::fs::create_dir_all(&version_root)
        .with_context(|| format!("creating golden cache root {}", version_root.display()))?;
    let mut staging = StagingDir::create(&version_root)?;
    let manifest_path = staging.path().join(MANIFEST_FILENAME);
    source
        .fetch_asset(owner, repo, tag, MANIFEST_FILENAME, &manifest_path)
        .context("fetching standalone golden manifest")?;
    validate_regular_file(&manifest_path)?;
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest = manifest::parse_manifest_bytes(&manifest_bytes, "release manifest")?;
    validate_release_manifest(&manifest, tag)?;

    let archive_path = staging.path().join(ARCHIVE_FILENAME);
    source
        .fetch_asset(owner, repo, tag, ARCHIVE_FILENAME, &archive_path)
        .context("fetching golden fixture archive")?;
    validate_asset_names(&source.asset_names(owner, repo, tag)?)
        .context("release asset set changed while downloading")?;
    validate_regular_file(&archive_path)?;
    let actual_archive_sha256 = sha256_file(&archive_path)?;
    if actual_archive_sha256 != manifest.archive.sha256 {
        anyhow::bail!(
            "archive checksum mismatch: expected {}, got {actual_archive_sha256}",
            manifest.archive.sha256
        );
    }

    extract_verified_archive(&archive_path, staging.path(), &manifest)?;
    std::fs::remove_file(&archive_path)
        .with_context(|| format!("removing staged archive {}", archive_path.display()))?;
    verify_install(staging.path(), &manifest_bytes, &manifest)?;

    let final_dir = version_root.join(&manifest.archive.sha256);
    if path_exists_no_follow(&final_dir)? {
        verify_install(&final_dir, &manifest_bytes, &manifest)
            .context("existing content-addressed golden install is invalid")?;
        return Ok((manifest, final_dir));
    }

    match std::fs::rename(staging.path(), &final_dir) {
        Ok(()) => staging.publish(),
        Err(error) if path_exists_no_follow(&final_dir)? => {
            verify_install(&final_dir, &manifest_bytes, &manifest)
                .context("concurrent golden install winner is invalid")?;
            tracing::debug!("concurrent golden install won rename race: {error}");
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "publishing golden install {} -> {}",
                    staging.path().display(),
                    final_dir.display()
                )
            });
        }
    }
    Ok((manifest, final_dir))
}

/// Load golden fixtures from a local directory (no download).
pub fn load_from_dir(dir: &Path) -> Result<(Manifest, PathBuf)> {
    let manifest_path = dir.join(MANIFEST_FILENAME);
    let manifest = manifest::parse_manifest(&manifest_path)?;
    verify_fixture_hashes(dir, &manifest.fixtures)?;
    Ok((manifest, dir.to_path_buf()))
}

fn validate_asset_names(names: &[String]) -> Result<()> {
    let expected = BTreeSet::from([MANIFEST_FILENAME, ARCHIVE_FILENAME]);
    let actual: BTreeSet<_> = names.iter().map(String::as_str).collect();
    if names.len() != expected.len() || actual != expected {
        anyhow::bail!(
            "golden release assets must be exactly {MANIFEST_FILENAME} and {ARCHIVE_FILENAME}"
        );
    }
    Ok(())
}

fn validate_release_manifest(manifest: &Manifest, tag: &str) -> Result<()> {
    if tag != manifest.golden_version {
        anyhow::bail!(
            "release tag {tag} does not match manifest golden_version {}",
            manifest.golden_version
        );
    }
    let expected: HashSet<_> = manifest
        .expected_fixtures
        .iter()
        .map(|fixture| fixture.fixture_id.as_str())
        .collect();
    let generated: HashSet<_> = manifest.fixtures.iter().map(fixture_id).collect();
    if expected.len() != manifest.expected_fixtures.len()
        || generated.len() != manifest.fixtures.len()
        || expected != generated
    {
        anyhow::bail!("release manifest must contain every and only expected fixture");
    }
    let expected_calibration: HashSet<_> = manifest
        .expected_fixtures
        .iter()
        .filter(|fixture| fixture.oracle_role == OracleRole::Baseline)
        .map(|fixture| fixture.fixture_id.as_str())
        .collect();
    let calibrated: HashSet<_> = manifest
        .calibrated_fixtures
        .iter()
        .map(String::as_str)
        .collect();
    if expected_calibration != calibrated {
        anyhow::bail!("release manifest requires complete baseline calibration");
    }
    Ok(())
}

fn fixture_id(fixture: &FixtureMetadata) -> &str {
    fixture.filename.strip_suffix(".safetensors").unwrap_or("")
}

fn extract_verified_archive(
    archive_path: &Path,
    staging: &Path,
    manifest: &Manifest,
) -> Result<()> {
    let file = File::open(archive_path)
        .with_context(|| format!("opening archive {}", archive_path.display()))?;
    let decoder = GzDecoder::new(BufReader::new(file));
    let gzip_header = decoder
        .header()
        .context("archive has a malformed gzip header")?;
    if gzip_header.mtime() != 0 || gzip_header.filename().is_some() {
        anyhow::bail!("archive gzip header is not normalized");
    }

    let declared: HashMap<_, _> = manifest
        .fixtures
        .iter()
        .map(|fixture| (fixture.filename.as_str(), fixture))
        .collect();
    if declared.len() != manifest.fixtures.len() || declared.is_empty() {
        anyhow::bail!("archive fixture declaration is empty or contains duplicates");
    }
    let mut seen = HashSet::new();
    let mut previous_name: Option<String> = None;
    let mut archive = tar::Archive::new(decoder);
    {
        let entries = archive.entries().context("reading USTAR archive entries")?;
        for entry in entries.raw(true) {
            let mut entry = entry.context("reading USTAR archive entry")?;
            let header = entry.header();
            if header.as_ustar().is_none() || !header.entry_type().is_file() {
                anyhow::bail!("archive contains a non-regular or non-USTAR entry");
            }
            let raw_name = header.path_bytes();
            if !raw_name.is_ascii() {
                anyhow::bail!("archive entry path is not ASCII");
            }
            let name = std::str::from_utf8(&raw_name)
                .context("archive entry path is not UTF-8")?
                .to_string();
            validate_archive_name(&name)?;
            if !seen.insert(name.clone()) {
                anyhow::bail!("archive contains duplicate fixture entry: {name}");
            }
            if previous_name
                .as_deref()
                .is_some_and(|previous| previous >= name.as_str())
            {
                anyhow::bail!("archive fixture entries are not in canonical ASCII order");
            }
            previous_name = Some(name.clone());
            let metadata = declared
                .get(name.as_str())
                .with_context(|| format!("archive contains undeclared fixture: {name}"))?;
            validate_tar_metadata(header, &name)?;
            let entry_size = header
                .size()
                .with_context(|| format!("reading size for {name}"))?;
            let size_limit = fixture_size_limit(metadata, manifest)?;
            if entry_size > size_limit {
                anyhow::bail!(
                    "archive fixture {name} exceeds extraction limit: {entry_size} > {size_limit}"
                );
            }
            let expected_sha256 = metadata.sha256.clone();
            let destination = staging.join(&name);
            let actual_sha256 = copy_entry(&mut entry, &destination, entry_size)?;
            if actual_sha256 != expected_sha256 {
                anyhow::bail!(
                    "fixture checksum mismatch for {name}: expected {}, got {actual_sha256}",
                    expected_sha256
                );
            }
        }
    }
    let mut decoder = archive.into_inner();
    let mut trailing_tar_bytes = 0_u64;
    let mut trailing = [0_u8; 8192];
    loop {
        let count = decoder
            .read(&mut trailing)
            .context("finishing gzip archive stream")?;
        if count == 0 {
            break;
        }
        if trailing[..count].iter().any(|byte| *byte != 0) {
            anyhow::bail!("archive contains data after the USTAR end marker");
        }
        trailing_tar_bytes = trailing_tar_bytes
            .checked_add(u64::try_from(count).context("trailing archive size overflow")?)
            .context("trailing archive size overflow")?;
        if trailing_tar_bytes > SAFETENSORS_HEADER_LIMIT {
            anyhow::bail!("archive contains excessive trailing padding");
        }
    }
    let mut compressed_reader = decoder.into_inner();
    let mut trailing_compressed = [0_u8; 1];
    if compressed_reader
        .read(&mut trailing_compressed)
        .context("checking for data after the gzip member")?
        != 0
    {
        anyhow::bail!("archive contains trailing data or multiple gzip members");
    }

    let missing: Vec<_> = declared
        .keys()
        .filter(|name| !seen.contains(**name))
        .copied()
        .collect();
    if !missing.is_empty() {
        anyhow::bail!("archive is missing declared fixtures: {missing:?}");
    }
    Ok(())
}

fn validate_archive_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.starts_with('/')
    {
        anyhow::bail!("unsafe archive entry path: {name:?}");
    }
    Ok(())
}

fn validate_tar_metadata(header: &tar::Header, name: &str) -> Result<()> {
    let empty_link = header
        .link_name_bytes()
        .map_or(true, |link_name| link_name.is_empty());
    if header.mode()? != 0o644
        || header.uid()? != 0
        || header.gid()? != 0
        || header.mtime()? != 0
        || !header.username_bytes().map_or(true, <[u8]>::is_empty)
        || !header.groupname_bytes().map_or(true, <[u8]>::is_empty)
        || !empty_link
    {
        anyhow::bail!("archive entry metadata is not normalized: {name}");
    }
    Ok(())
}

fn fixture_size_limit(metadata: &FixtureMetadata, manifest: &Manifest) -> Result<u64> {
    let tokens = u64::from(metadata.num_tokens);
    let vocab_size = u64::try_from(manifest.model.vocab_size)
        .context("model vocabulary does not fit archive size accounting")?;
    let payload = match metadata.category {
        PromptCategory::Canonical => tokens
            .checked_mul(vocab_size)
            .and_then(|size| size.checked_mul(4))
            .and_then(|size| size.checked_add(tokens.checked_mul(8)?))
            .and_then(|size| size.checked_add(8)),
        PromptCategory::Regression => tokens
            .checked_mul(8 + 5 * 8 + 5 * 4)
            .and_then(|size| size.checked_add(8)),
    }
    .context("fixture extraction size limit overflow")?;
    payload
        .checked_add(SAFETENSORS_HEADER_LIMIT)
        .context("fixture extraction size limit overflow")
}

fn copy_entry(entry: &mut dyn Read, destination: &Path, expected_size: u64) -> Result<String> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .with_context(|| format!("creating extracted fixture {}", destination.display()))?;
    let mut hasher = Sha256::new();
    let mut written = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = entry
            .read(&mut buffer)
            .with_context(|| format!("extracting {}", destination.display()))?;
        if count == 0 {
            break;
        }
        written = written
            .checked_add(u64::try_from(count).context("extracted fixture size overflow")?)
            .context("extracted fixture size overflow")?;
        if written > expected_size {
            anyhow::bail!("archive entry exceeded its declared size");
        }
        output.write_all(&buffer[..count])?;
        hasher.update(&buffer[..count]);
    }
    if written != expected_size {
        anyhow::bail!(
            "archive entry was truncated: expected {expected_size} bytes, extracted {written}"
        );
    }
    output.sync_all()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_install(dir: &Path, manifest_bytes: &[u8], manifest: &Manifest) -> Result<()> {
    let root_metadata = std::fs::symlink_metadata(dir)
        .with_context(|| format!("reading install metadata for {}", dir.display()))?;
    if !root_metadata.file_type().is_dir() || root_metadata.file_type().is_symlink() {
        anyhow::bail!("golden install is not a real directory: {}", dir.display());
    }
    let mut expected: BTreeSet<_> = manifest
        .fixtures
        .iter()
        .map(|fixture| fixture.filename.as_str())
        .collect();
    expected.insert(MANIFEST_FILENAME);
    let mut actual = BTreeSet::new();
    for item in std::fs::read_dir(dir)? {
        let item = item?;
        let name = item
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("installed fixture name is not UTF-8"))?;
        if !item.file_type()?.is_file() {
            anyhow::bail!("installed golden entry is not a regular file: {name}");
        }
        actual.insert(name);
    }
    let expected_owned: BTreeSet<_> = expected.into_iter().map(str::to_string).collect();
    if actual != expected_owned {
        anyhow::bail!("installed golden file set does not match manifest");
    }
    if std::fs::read(dir.join(MANIFEST_FILENAME))? != manifest_bytes {
        anyhow::bail!("installed standalone manifest bytes do not match the release asset");
    }
    verify_fixture_hashes(dir, &manifest.fixtures)
}

fn verify_fixture_hashes(dir: &Path, fixtures: &[FixtureMetadata]) -> Result<()> {
    for fixture in fixtures {
        let path = dir.join(&fixture.filename);
        validate_regular_file(&path)?;
        let actual = sha256_file(&path)?;
        if actual != fixture.sha256 {
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

fn validate_regular_file(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading file metadata for {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        anyhow::bail!("expected a regular file: {}", path.display());
    }
    Ok(())
}

fn path_exists_no_follow(path: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("checking {}", path.display())),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

struct StagingDir {
    path: PathBuf,
    published: bool,
}

impl StagingDir {
    fn create(parent: &Path) -> Result<Self> {
        let process_id = std::process::id();
        for _ in 0..1024 {
            let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(".staging-{process_id}-{nonce}"));
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        published: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("creating staging directory {}", path.display()));
                }
            }
        }
        anyhow::bail!("could not allocate a unique golden staging directory")
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn publish(&mut self) {
        self.published = true;
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if !self.published {
            if let Err(error) = std::fs::remove_dir_all(&self.path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        "failed to remove unpublished golden staging directory {}: {error}",
                        self.path.display()
                    );
                }
            }
        }
    }
}
