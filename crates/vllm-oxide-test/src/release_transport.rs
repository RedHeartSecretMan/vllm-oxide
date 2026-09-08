//! Schema5 transport only. Semantic release acceptance is a separate operation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use flate2::bufread::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MANIFEST_LIMIT: u64 = 16 * 1024 * 1024;
const FILE_LIMIT: u64 = 8 * 1024 * 1024 * 1024 - 1;
const TOTAL_LIMIT: u64 = 1024 * 1024 * 1024 * 1024;
const UPLOAD_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const ARCHIVE: &str = "goldens-v0.2.tar.gz";
static NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub commit: String,
    pub tree: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub logical_path: String,
    pub filename: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveIdentity {
    pub filename: String,
    pub sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Entrypoints {
    pub authoritative_manifest: String,
    pub authoritative_marker: String,
    pub performance: String,
    pub cpu_gates: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseManifest {
    pub schema_version: u32,
    pub protocol: String,
    pub product_version: String,
    pub golden_version: String,
    pub source: Source,
    pub registry_sha256: String,
    pub policy_sha256: String,
    pub definition_index_blob: String,
    pub entrypoints: Entrypoints,
    #[serde(deserialize_with = "unique_counts")]
    pub counts: BTreeMap<String, u64>,
    pub artifacts: Vec<Artifact>,
    pub archive: ArchiveIdentity,
}

fn unique_counts<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, u64>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, u64>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique count names")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> std::result::Result<Self::Value, M::Error> {
            let mut values = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, u64>()? {
                if values.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate count name"));
                }
            }
            Ok(values)
        }
    }
    deserializer.deserialize_map(Unique)
}

fn no_symlinks(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        match std::fs::symlink_metadata(ancestor) {
            Ok(m) if m.file_type().is_symlink() => bail!("symlink in artifact or cache path"),
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn hex(value: &str, size: usize) -> bool {
    value.len() == size
        && value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}

fn safe_path(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && value.is_ascii()
        && !value.contains(['\\', '\0'])
        && value
            .split('/')
            .all(|p| !matches!(p, "" | "." | ".." | ".git"))
}

pub fn parse_manifest(bytes: &[u8]) -> Result<ReleaseManifest> {
    if u64::try_from(bytes.len())? > MANIFEST_LIMIT {
        bail!("release manifest exceeds size limit");
    }
    let value: ReleaseManifest = serde_json::from_slice(bytes)?;
    if value.schema_version != 5
        || value.protocol != "layered-accuracy-v1"
        || value.product_version != "v0.2.0"
        || value.golden_version != "goldens-v0.2"
        || value.archive.filename != ARCHIVE
        || !hex(&value.archive.sha256, 64)
        || !hex(&value.source.commit, 40)
        || !hex(&value.source.tree, 40)
        || !hex(&value.registry_sha256, 64)
        || !hex(&value.policy_sha256, 64)
        || !hex(&value.definition_index_blob, 40)
        || value.artifacts.is_empty()
        || value.artifacts.len() > 50_000
    {
        bail!("invalid schema5 release identity or bounds");
    }
    let mut names = BTreeSet::new();
    let mut previous: Option<&str> = None;
    let mut total = 0_u64;
    for (index, artifact) in value.artifacts.iter().enumerate() {
        if !safe_path(&artifact.logical_path)
            || previous.is_some_and(|p| p >= artifact.logical_path.as_str())
            || artifact.filename != format!("artifact-{index:06}.bin")
            || !hex(&artifact.sha256, 64)
            || artifact.size_bytes > FILE_LIMIT
        {
            bail!("invalid artifact inventory");
        }
        let mut prefix = String::new();
        for part in artifact.logical_path.split('/') {
            if names.contains(&prefix) {
                bail!("file/directory path conflict");
            }
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(part);
        }
        names.insert(artifact.logical_path.clone());
        previous = Some(&artifact.logical_path);
        total = total
            .checked_add(artifact.size_bytes)
            .context("size overflow")?;
    }
    if total > TOTAL_LIMIT {
        bail!("extracted size limit exceeded");
    }
    for path in [
        &value.entrypoints.authoritative_manifest,
        &value.entrypoints.authoritative_marker,
        &value.entrypoints.performance,
        &value.entrypoints.cpu_gates,
    ] {
        if !names.contains(path) {
            bail!("missing release entrypoint");
        }
    }
    Ok(value)
}

fn regular(path: &Path) -> Result<()> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        bail!("not a regular artifact");
    }
    Ok(())
}

fn digest(path: &Path) -> Result<String> {
    regular(path)?;
    let mut stream = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65_536];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn header(artifact: &Artifact) -> [u8; 512] {
    let mut bytes = [0_u8; 512];
    bytes[..artifact.filename.len()].copy_from_slice(artifact.filename.as_bytes());
    for (start, len, number) in [
        (100, 8, 0o644),
        (108, 8, 0),
        (116, 8, 0),
        (124, 12, artifact.size_bytes),
        (136, 12, 0),
    ] {
        let octal = format!("{number:0width$o}\0", width = len - 1);
        bytes[start..start + len].copy_from_slice(octal.as_bytes());
    }
    bytes[148..156].fill(b' ');
    bytes[156] = b'0';
    bytes[257..263].copy_from_slice(b"ustar\0");
    bytes[263..265].copy_from_slice(b"00");
    let checksum: u64 = bytes.iter().map(|v| u64::from(*v)).sum();
    bytes[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    bytes
}

fn verify_tree(root: &Path, bytes: &[u8], manifest: &ReleaseManifest) -> Result<()> {
    no_symlinks(root)?;
    regular(&root.join("manifest.json"))?;
    if std::fs::read(root.join("manifest.json"))? != bytes {
        bail!("installed manifest differs");
    }
    let expected = std::iter::once("manifest.json".to_string())
        .chain(
            manifest
                .artifacts
                .iter()
                .map(|a| format!("evidence/{}", a.logical_path)),
        )
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    let mut directories = BTreeSet::new();
    for artifact in &manifest.artifacts {
        let path = PathBuf::from("evidence").join(&artifact.logical_path);
        for p in path.ancestors().skip(1) {
            if !p.as_os_str().is_empty() {
                directories.insert(p.to_path_buf());
            }
        }
    }
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if !directories.contains(entry.path().strip_prefix(root)?) {
                    bail!("unexpected installed directory");
                }
                pending.push(entry.path());
            } else if entry.file_type()?.is_file() {
                actual.insert(
                    entry
                        .path()
                        .strip_prefix(root)?
                        .to_str()
                        .context("non-UTF8 path")?
                        .to_string(),
                );
            } else {
                bail!("symlink or special installed entry");
            }
        }
    }
    if actual != expected {
        bail!("immutable inventory mismatch");
    }
    for artifact in &manifest.artifacts {
        let path = root.join("evidence").join(&artifact.logical_path);
        if path.metadata()?.len() != artifact.size_bytes || digest(&path)? != artifact.sha256 {
            bail!("installed size/checksum mismatch");
        }
    }
    Ok(())
}

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn install_transport(bundle: &Path, cache: &Path) -> Result<PathBuf> {
    no_symlinks(bundle)?;
    no_symlinks(cache)?;
    let names = std::fs::read_dir(bundle)?
        .map(|e| Ok(e?.file_name()))
        .collect::<Result<BTreeSet<_>>>()?;
    if names
        != [
            std::ffi::OsString::from("manifest.json"),
            std::ffi::OsString::from(ARCHIVE),
        ]
        .into()
    {
        bail!("release must contain exactly two assets");
    }
    let path = bundle.join("manifest.json");
    regular(&path)?;
    if path.metadata()?.len() > MANIFEST_LIMIT {
        bail!("manifest size limit");
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MANIFEST_LIMIT + 1)
        .read_to_end(&mut bytes)?;
    let manifest = parse_manifest(&bytes)?;
    let archive = bundle.join(ARCHIVE);
    regular(&archive)?;
    if archive.metadata()?.len() >= UPLOAD_LIMIT || digest(&archive)? != manifest.archive.sha256 {
        bail!("archive size/checksum mismatch");
    }
    let parent = cache.join("goldens-v0.2");
    std::fs::create_dir_all(&parent)?;
    if cache.symlink_metadata()?.file_type().is_symlink()
        || parent.symlink_metadata()?.file_type().is_symlink()
    {
        bail!("symlink cache root");
    }
    let final_path = parent.join(&manifest.archive.sha256);
    if let Ok(metadata) = std::fs::symlink_metadata(&final_path) {
        if !metadata.is_dir() {
            bail!("invalid immutable install");
        }
        verify_tree(&final_path, &bytes, &manifest)?;
        return Ok(final_path);
    }
    let disk = rustix::fs::statvfs(&parent)?;
    let needed = manifest.artifacts.iter().map(|a| a.size_bytes).sum::<u64>() + MANIFEST_LIMIT;
    if disk.f_bavail.saturating_mul(disk.f_frsize) < needed {
        bail!("capacity decision required");
    }
    let mut stage = None;
    for _ in 0..1024 {
        let path = parent.join(format!(
            ".schema5-{}-{}",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => {
                stage = Some(Staging(path));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
    }
    let stage = stage.context("cannot allocate staging directory")?;
    let mut compressed = BufReader::new(File::open(archive)?);
    let mut gzip_header = [0_u8; 10];
    compressed.read_exact(&mut gzip_header)?;
    if gzip_header != [31, 139, 8, 0, 0, 0, 0, 0, 2, 255] {
        bail!("noncanonical gzip header");
    }
    use std::io::Seek;
    compressed.rewind()?;
    let mut reader = GzDecoder::new(compressed);
    let mut buffer = [0_u8; 65_536];
    for artifact in &manifest.artifacts {
        let mut block = [0_u8; 512];
        reader.read_exact(&mut block)?;
        if block != header(artifact) {
            bail!("noncanonical or unexpected USTAR header");
        }
        let target = stage.0.join("evidence").join(&artifact.logical_path);
        std::fs::create_dir_all(target.parent().context("missing parent")?)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(target)?;
        let mut remaining = artifact.size_bytes;
        let mut hash = Sha256::new();
        while remaining > 0 {
            let count = reader.read(&mut buffer[..usize::try_from(remaining.min(65_536))?])?;
            if count == 0 {
                bail!("truncated archive payload");
            }
            output.write_all(&buffer[..count])?;
            hash.update(&buffer[..count]);
            remaining -= u64::try_from(count)?;
        }
        output.sync_all()?;
        if format!("{:x}", hash.finalize()) != artifact.sha256 {
            bail!("extracted checksum mismatch");
        }
        let padding = usize::try_from((512 - artifact.size_bytes % 512) % 512)?;
        reader.read_exact(&mut buffer[..padding])?;
        if buffer[..padding].iter().any(|v| *v != 0) {
            bail!("nonzero payload padding");
        }
    }
    let mut tail = Vec::new();
    reader
        .by_ref()
        .take(1024 * 1024 + 1)
        .read_to_end(&mut tail)?;
    if tail.len() < 1024
        || tail.len() > 1024 * 1024
        || tail.len() % 512 != 0
        || tail.iter().any(|v| *v != 0)
    {
        bail!("invalid USTAR end markers or trailing content");
    }
    if reader.into_inner().read(&mut buffer[..1])? != 0 {
        bail!("trailing compressed content");
    }
    std::fs::write(stage.0.join("manifest.json"), &bytes)?;
    verify_tree(&stage.0, &bytes, &manifest)?;
    match rustix::fs::renameat_with(
        rustix::fs::CWD,
        &stage.0,
        rustix::fs::CWD,
        &final_path,
        rustix::fs::RenameFlags::NOREPLACE,
    ) {
        Ok(()) => {}
        Err(rustix::io::Errno::EXIST) => verify_tree(&final_path, &bytes, &manifest)?,
        Err(e) => return Err(e.into()),
    }
    Ok(final_path)
}
