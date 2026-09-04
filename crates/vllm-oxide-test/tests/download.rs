#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use flate2::{Compression, GzBuilder};
use serde_json::json;
use tar::{Builder, EntryType, Header};
use tempfile::TempDir;
use vllm_oxide_test::download::{download_release_with, ReleaseSource};
use vllm_oxide_test::manifest::sha256_hex;

struct FakeReleaseSource {
    assets: BTreeMap<String, Vec<u8>>,
    listed_names: Vec<String>,
    fail_asset: Option<String>,
}

struct ChangingReleaseSource {
    inner: FakeReleaseSource,
    listings: AtomicUsize,
}

impl ReleaseSource for ChangingReleaseSource {
    fn asset_names(&self, owner: &str, repo: &str, tag: &str) -> Result<Vec<String>> {
        let mut names = self.inner.asset_names(owner, repo, tag)?;
        if self.listings.fetch_add(1, Ordering::SeqCst) > 0 {
            names.push("late-extra.safetensors".to_string());
        }
        Ok(names)
    }

    fn fetch_asset(
        &self,
        owner: &str,
        repo: &str,
        tag: &str,
        name: &str,
        destination: &Path,
    ) -> Result<()> {
        self.inner.fetch_asset(owner, repo, tag, name, destination)
    }
}

impl ReleaseSource for FakeReleaseSource {
    fn asset_names(&self, _owner: &str, _repo: &str, _tag: &str) -> Result<Vec<String>> {
        Ok(self.listed_names.clone())
    }

    fn fetch_asset(
        &self,
        _owner: &str,
        _repo: &str,
        _tag: &str,
        name: &str,
        destination: &Path,
    ) -> Result<()> {
        let bytes = self
            .assets
            .get(name)
            .with_context(|| format!("missing fake asset {name}"))?;
        let mut output = std::fs::File::create(destination)?;
        if self.fail_asset.as_deref() == Some(name) {
            output.write_all(b"partial")?;
            anyhow::bail!("injected partial fetch failure for {name}");
        }
        output.write_all(bytes)?;
        Ok(())
    }
}

fn archive_bytes(fixtures: &[(&str, &[u8])]) -> Vec<u8> {
    archive_bytes_with_options(fixtures, EntryType::Regular, 0o644, false, false)
}

fn archive_bytes_with_options(
    fixtures: &[(&str, &[u8])],
    entry_type: EntryType,
    mode: u32,
    old_header: bool,
    named_gzip: bool,
) -> Vec<u8> {
    let mut gzip = GzBuilder::new().mtime(0);
    if named_gzip {
        gzip = gzip.filename("source.tar");
    }
    let encoder = gzip.write(Vec::new(), Compression::default());
    let mut archive = Builder::new(encoder);
    for (name, payload) in fixtures {
        let mut header = if old_header {
            Header::new_old()
        } else {
            Header::new_ustar()
        };
        header.set_path(name).unwrap();
        header.set_size(payload.len() as u64);
        header.set_mode(mode);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        header.set_entry_type(entry_type);
        if !old_header {
            header.set_username("").unwrap();
            header.set_groupname("").unwrap();
        }
        header.set_cksum();
        archive.append(&header, *payload).unwrap();
    }
    let encoder = archive.into_inner().unwrap();
    encoder.finish().unwrap()
}

fn archive_bytes_with_raw_name(raw_name: &[u8]) -> Vec<u8> {
    let encoder = GzBuilder::new()
        .mtime(0)
        .write(Vec::new(), Compression::default());
    let mut archive = Builder::new(encoder);
    let mut header = Header::new_ustar();
    header.set_path("placeholder").unwrap();
    header.set_size(1);
    header.set_mode(0o644);
    header.set_uid(0);
    header.set_gid(0);
    header.set_mtime(0);
    header.set_entry_type(EntryType::Regular);
    header.set_username("").unwrap();
    header.set_groupname("").unwrap();
    let raw_header = header.as_ustar_mut().unwrap();
    raw_header.name.fill(0);
    raw_header.name[..raw_name.len()].copy_from_slice(raw_name);
    header.set_cksum();
    archive.append(&header, [0_u8].as_slice()).unwrap();
    let encoder = archive.into_inner().unwrap();
    encoder.finish().unwrap()
}

fn release_source() -> (FakeReleaseSource, String) {
    let fixtures: [(&str, &[u8]); 2] = [
        ("canonical_01.transformers.safetensors", b"reference"),
        ("canonical_01.vllm.safetensors", b"baseline"),
    ];
    let archive = archive_bytes(&fixtures);
    let archive_sha256 = sha256_hex(&archive);
    let manifest = json!({
        "schema_version": 4,
        "product_version": "v0.2.0",
        "golden_version": "goldens-v0.2",
        "archive": {"filename": "goldens-v0.2.tar.gz", "sha256": archive_sha256},
        "generated_at": "2026-09-05T00:00:00Z",
        "model": {
            "id": "Qwen/Qwen3-0.6B",
            "revision": "7e4ae267688d671ddfca3122e4528ee980cf3234",
            "arch": "Qwen3ForCausalLM",
            "dtype": "bfloat16",
            "vocab_size": 151_936
        },
        "oracle_versions": {"transformers": "5.0", "vllm": "0.26"},
        "generation": {
            "canonical_max_tokens": 64,
            "regression_max_tokens": 32,
            "temperature": 0.0,
            "attn_implementation": "sdpa"
        },
        "tolerance_policy": {
            "version": "same-prefix-v1",
            "dtype": "bfloat16",
            "kernel": "sdpa",
            "l1_near_tie_max_abs_logit_gap": 0.02,
            "l2_atol": 0.01,
            "rationale": "reviewed test policy",
            "evidence": ["test:download"]
        },
        "baseline_calibration": {
            "candidate_atol": 0.01,
            "observed_max_abs_diff": 0.005,
            "calibration_factor": 2.0,
            "method": "test"
        },
        "expected_fixtures": [
            {
                "fixture_id": "canonical_01.transformers",
                "prompt_id": "canonical_01",
                "family": "canonical",
                "model_revision": "7e4ae267688d671ddfca3122e4528ee980cf3234",
                "dtype": "bfloat16",
                "oracle": "transformers",
                "oracle_role": "reference",
                "required_comparison": "l1_l2",
                "filename": "canonical_01.transformers.safetensors"
            },
            {
                "fixture_id": "canonical_01.vllm",
                "prompt_id": "canonical_01",
                "family": "canonical",
                "model_revision": "7e4ae267688d671ddfca3122e4528ee980cf3234",
                "dtype": "bfloat16",
                "oracle": "vllm",
                "oracle_role": "baseline",
                "required_comparison": "calibration",
                "filename": "canonical_01.vllm.safetensors"
            }
        ],
        "fixtures": fixtures.iter().map(|(name, payload)| {
            let (prompt_id, oracle, _) = name.split_once('.').and_then(|(prompt, rest)| {
                rest.split_once('.').map(|(oracle, suffix)| (prompt, oracle, suffix))
            }).unwrap();
            json!({
                "prompt_id": prompt_id,
                "category": "canonical",
                "oracle": oracle,
                "num_tokens": 1,
                "logits_dtype": "float32",
                "logits_shape": [1, 151_936],
                "sha256": sha256_hex(payload),
                "filename": name
            })
        }).collect::<Vec<_>>(),
        "calibrated_fixtures": ["canonical_01.vllm"]
    });
    let assets = BTreeMap::from([
        (
            "manifest.json".to_string(),
            serde_json::to_vec(&manifest).unwrap(),
        ),
        ("goldens-v0.2.tar.gz".to_string(), archive),
    ]);
    let listed_names = assets.keys().cloned().collect();
    (
        FakeReleaseSource {
            assets,
            listed_names,
            fail_asset: None,
        },
        archive_sha256,
    )
}

fn replace_archive(source: &mut FakeReleaseSource, archive: Vec<u8>) {
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&source.assets["manifest.json"]).unwrap();
    manifest["archive"]["sha256"] = json!(sha256_hex(&archive));
    source.assets.insert(
        "manifest.json".to_string(),
        serde_json::to_vec(&manifest).unwrap(),
    );
    source
        .assets
        .insert("goldens-v0.2.tar.gz".to_string(), archive);
}

fn assert_archive_rejected(archive: Vec<u8>, expected: &str) {
    let cache = TempDir::new().unwrap();
    let (mut source, _) = release_source();
    replace_archive(&mut source, archive);

    let error = install(&source, cache.path(), "goldens-v0.2")
        .unwrap_err()
        .to_string();

    assert!(error.contains(expected), "{error}");
    assert_no_unpublished_state(cache.path());
}

fn install(source: &FakeReleaseSource, cache: &Path, tag: &str) -> Result<()> {
    download_release_with(source, "RedHeartSecretMan", "vllm-oxide", tag, cache).map(|_| ())
}

fn assert_no_unpublished_state(cache: &Path) {
    let version_root = cache.join("goldens-v0.2");
    if !version_root.exists() {
        return;
    }
    assert_eq!(std::fs::read_dir(version_root).unwrap().count(), 0);
}

fn assert_no_staging_directories(cache: &Path) {
    let version_root = cache.join("goldens-v0.2");
    if !version_root.exists() {
        return;
    }
    for item in std::fs::read_dir(version_root).unwrap() {
        let name = item.unwrap().file_name().into_string().unwrap();
        assert!(
            !name.starts_with(".staging-"),
            "left staging directory {name}"
        );
    }
}

#[test]
fn downloader_installs_verified_bundle_at_content_addressed_path() {
    let cache = TempDir::new().unwrap();
    let (source, archive_sha256) = release_source();

    let (manifest, installed) = download_release_with(
        &source,
        "RedHeartSecretMan",
        "vllm-oxide",
        "goldens-v0.2",
        cache.path(),
    )
    .unwrap();

    assert_eq!(manifest.archive.sha256, archive_sha256);
    assert_eq!(
        installed,
        cache
            .path()
            .join("goldens-v0.2")
            .join(&manifest.archive.sha256)
    );
    assert_eq!(
        std::fs::read(installed.join("canonical_01.transformers.safetensors")).unwrap(),
        b"reference"
    );
    assert_eq!(
        std::fs::read(installed.join("canonical_01.vllm.safetensors")).unwrap(),
        b"baseline"
    );
    assert!(installed.join("manifest.json").is_file());
    assert!(!installed.join("goldens-v0.2.tar.gz").exists());
}

#[test]
fn downloader_rejects_missing_extra_and_duplicate_release_assets() {
    for listed_names in [
        vec!["manifest.json".to_string()],
        vec![
            "manifest.json".to_string(),
            "goldens-v0.2.tar.gz".to_string(),
            "fixture.safetensors".to_string(),
        ],
        vec![
            "manifest.json".to_string(),
            "manifest.json".to_string(),
            "goldens-v0.2.tar.gz".to_string(),
        ],
    ] {
        let cache = TempDir::new().unwrap();
        let (mut source, _) = release_source();
        source.listed_names = listed_names;

        let error = install(&source, cache.path(), "goldens-v0.2")
            .unwrap_err()
            .to_string();

        assert!(error.contains("exactly manifest.json"), "{error}");
        assert_no_unpublished_state(cache.path());
    }
}

#[test]
fn downloader_rejects_an_asset_set_that_changes_during_download() {
    let cache = TempDir::new().unwrap();
    let (source, _) = release_source();
    let source = ChangingReleaseSource {
        inner: source,
        listings: AtomicUsize::new(0),
    };

    let error = download_release_with(
        &source,
        "RedHeartSecretMan",
        "vllm-oxide",
        "goldens-v0.2",
        cache.path(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("release asset set changed"), "{error}");
    assert_no_unpublished_state(cache.path());
}

#[test]
fn downloader_rejects_malformed_manifest_and_version_upgrade() {
    for manifest_bytes in [b"not json".to_vec(), {
        let (source, _) = release_source();
        let mut value: serde_json::Value =
            serde_json::from_slice(&source.assets["manifest.json"]).unwrap();
        value["schema_version"] = json!(5);
        serde_json::to_vec(&value).unwrap()
    }] {
        let cache = TempDir::new().unwrap();
        let (mut source, _) = release_source();
        source
            .assets
            .insert("manifest.json".to_string(), manifest_bytes);

        install(&source, cache.path(), "goldens-v0.2").unwrap_err();

        assert_no_unpublished_state(cache.path());
    }
}

#[test]
fn downloader_rejects_tag_and_archive_checksum_mismatches_without_partial_install() {
    let cache = TempDir::new().unwrap();
    let (source, _) = release_source();
    let error = install(&source, cache.path(), "goldens-v9")
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not match"), "{error}");
    assert_no_unpublished_state(cache.path());

    let cache = TempDir::new().unwrap();
    let (mut source, _) = release_source();
    source
        .assets
        .get_mut("goldens-v0.2.tar.gz")
        .unwrap()
        .push(0);
    let error = install(&source, cache.path(), "goldens-v0.2")
        .unwrap_err()
        .to_string();
    assert!(error.contains("archive checksum mismatch"), "{error}");
    assert_no_unpublished_state(cache.path());
}

#[test]
fn downloader_cleans_a_partial_fetch() {
    let cache = TempDir::new().unwrap();
    let (mut source, _) = release_source();
    source.fail_asset = Some("goldens-v0.2.tar.gz".to_string());

    let error = install(&source, cache.path(), "goldens-v0.2")
        .unwrap_err()
        .to_string();

    assert!(error.contains("fetching golden fixture archive"), "{error}");
    assert_no_unpublished_state(cache.path());
}

#[test]
fn downloader_rejects_missing_undeclared_duplicate_and_unsorted_archive_entries() {
    assert_archive_rejected(
        archive_bytes(&[("canonical_01.transformers.safetensors", b"reference")]),
        "missing declared fixtures",
    );
    assert_archive_rejected(
        archive_bytes(&[
            ("canonical_01.transformers.safetensors", b"reference"),
            ("canonical_01.vllm.safetensors", b"baseline"),
            ("undeclared.safetensors", b"unexpected"),
        ]),
        "undeclared fixture",
    );
    assert_archive_rejected(
        archive_bytes(&[
            ("canonical_01.transformers.safetensors", b"reference"),
            ("canonical_01.transformers.safetensors", b"reference"),
            ("canonical_01.vllm.safetensors", b"baseline"),
        ]),
        "duplicate fixture entry",
    );
    assert_archive_rejected(
        archive_bytes(&[
            ("canonical_01.vllm.safetensors", b"baseline"),
            ("canonical_01.transformers.safetensors", b"reference"),
        ]),
        "canonical ASCII order",
    );
}

#[test]
fn downloader_rejects_unsafe_and_ambiguous_archive_paths() {
    for raw_name in [
        b"../escape.safetensors".as_slice(),
        b"/absolute.safetensors".as_slice(),
        b"nested/fixture.safetensors".as_slice(),
        b"nested\\fixture.safetensors".as_slice(),
        b".".as_slice(),
        b"..".as_slice(),
    ] {
        assert_archive_rejected(
            archive_bytes_with_raw_name(raw_name),
            "unsafe archive entry path",
        );
    }
    assert_archive_rejected(
        archive_bytes_with_raw_name(b"canonical_01.Transformers.safetensors"),
        "undeclared fixture",
    );
    assert_archive_rejected(
        archive_bytes_with_raw_name(b"canonical_01.transf\x80rmers.safetensors"),
        "not ASCII",
    );
}

#[test]
fn downloader_rejects_links_directories_devices_fifos_and_non_ustar_entries() {
    for entry_type in [
        EntryType::Link,
        EntryType::Symlink,
        EntryType::Directory,
        EntryType::Char,
        EntryType::Block,
        EntryType::Fifo,
        EntryType::Continuous,
    ] {
        assert_archive_rejected(
            archive_bytes_with_options(
                &[("canonical_01.transformers.safetensors", b"reference")],
                entry_type,
                0o644,
                false,
                false,
            ),
            "non-regular or non-USTAR",
        );
    }
    assert_archive_rejected(
        archive_bytes_with_options(
            &[("canonical_01.transformers.safetensors", b"reference")],
            EntryType::Regular,
            0o644,
            true,
            false,
        ),
        "non-regular or non-USTAR",
    );
}

#[test]
fn downloader_rejects_nondeterministic_headers_and_fixture_checksum_mismatch() {
    assert_archive_rejected(
        archive_bytes_with_options(
            &[("canonical_01.transformers.safetensors", b"reference")],
            EntryType::Regular,
            0o600,
            false,
            false,
        ),
        "metadata is not normalized",
    );
    assert_archive_rejected(
        archive_bytes_with_options(
            &[("canonical_01.transformers.safetensors", b"reference")],
            EntryType::Regular,
            0o644,
            false,
            true,
        ),
        "gzip header is not normalized",
    );
    assert_archive_rejected(
        archive_bytes(&[
            ("canonical_01.transformers.safetensors", b"tampered"),
            ("canonical_01.vllm.safetensors", b"baseline"),
        ]),
        "fixture checksum mismatch",
    );
}

#[test]
fn downloader_rejects_malformed_archive_after_partial_extraction_and_cleans_stage() {
    let mut archive = archive_bytes(&[
        ("canonical_01.transformers.safetensors", b"reference"),
        ("canonical_01.vllm.safetensors", b"baseline"),
    ]);
    archive.truncate(archive.len() - 12);

    assert_archive_rejected(archive, "archive");
}

#[test]
fn downloader_rejects_trailing_or_multiple_gzip_members_and_extraction_bombs() {
    let valid = archive_bytes(&[
        ("canonical_01.transformers.safetensors", b"reference"),
        ("canonical_01.vllm.safetensors", b"baseline"),
    ]);
    let mut trailing = valid.clone();
    trailing.extend_from_slice(b"trailing");
    assert_archive_rejected(trailing, "trailing data or multiple gzip members");

    let mut multiple = valid.clone();
    multiple.extend_from_slice(&valid);
    assert_archive_rejected(multiple, "trailing data or multiple gzip members");

    let oversized = vec![0_u8; 2 * 1024 * 1024];
    assert_archive_rejected(
        archive_bytes(&[(
            "canonical_01.transformers.safetensors",
            oversized.as_slice(),
        )]),
        "exceeds extraction limit",
    );
}

#[test]
fn downloader_reuses_only_a_complete_existing_install_and_never_overwrites_an_invalid_one() {
    let cache = TempDir::new().unwrap();
    let (source, _) = release_source();
    let (_, installed) = download_release_with(
        &source,
        "RedHeartSecretMan",
        "vllm-oxide",
        "goldens-v0.2",
        cache.path(),
    )
    .unwrap();

    let (_, reused) = download_release_with(
        &source,
        "RedHeartSecretMan",
        "vllm-oxide",
        "goldens-v0.2",
        cache.path(),
    )
    .unwrap();
    assert_eq!(reused, installed);

    let fixture = installed.join("canonical_01.transformers.safetensors");
    std::fs::write(&fixture, b"old invalid install").unwrap();
    let error = install(&source, cache.path(), "goldens-v0.2")
        .unwrap_err()
        .to_string();

    assert!(error.contains("existing content-addressed"), "{error}");
    assert_eq!(std::fs::read(fixture).unwrap(), b"old invalid install");
    assert_no_staging_directories(cache.path());
}

#[test]
fn concurrent_installs_share_the_verified_rename_winner() {
    let cache = TempDir::new().unwrap();
    let (source, _) = release_source();

    let installed = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            download_release_with(
                &source,
                "RedHeartSecretMan",
                "vllm-oxide",
                "goldens-v0.2",
                cache.path(),
            )
            .unwrap()
            .1
        });
        let second = scope.spawn(|| {
            download_release_with(
                &source,
                "RedHeartSecretMan",
                "vllm-oxide",
                "goldens-v0.2",
                cache.path(),
            )
            .unwrap()
            .1
        });
        [first.join().unwrap(), second.join().unwrap()]
    });

    assert_eq!(installed[0], installed[1]);
    assert!(installed[0].is_dir());
    assert_no_staging_directories(cache.path());
}

#[cfg(unix)]
#[test]
fn downloader_does_not_replace_a_dangling_final_symlink() {
    use std::os::unix::fs::symlink;

    let cache = TempDir::new().unwrap();
    let (source, archive_sha256) = release_source();
    let version_root = cache.path().join("goldens-v0.2");
    std::fs::create_dir(&version_root).unwrap();
    let final_path = version_root.join(archive_sha256);
    symlink("missing-target", &final_path).unwrap();

    let error = install(&source, cache.path(), "goldens-v0.2")
        .unwrap_err()
        .to_string();

    assert!(error.contains("existing content-addressed"), "{error}");
    assert_eq!(
        std::fs::read_link(&final_path).unwrap(),
        Path::new("missing-target")
    );
    assert_no_staging_directories(cache.path());
}
