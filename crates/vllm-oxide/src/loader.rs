//! Model-agnostic weight loader (ADR-0002).
//!
//! One public function — [`load_weights`] — resolves a [`Source`] to a list
//! of `*.safetensors` files, then mmaps them into a candle
//! [`ShardedVarBuilder`]. The loader does NOT do any model-specific work:
//! no QKV fusion, no gate/up fusion, no per-rank slicing. All of that lives
//! in `Linear::<P>::from_vb` (T3) and `ParallelStyle::slice_for_rank`
//! (v0.2). HF checkpoint tensor names map 1:1 with what the model expects.
//!
//! # Resolution
//!
//! - [`Source::Local`] walks the fallback chain: `model.safetensors.index.json`
//!   (multi-shard) → single `model.safetensors` → glob `*.safetensors`.
//! - [`Source::Hub`] uses hf-hub 0.5 sync [`Api`]. When `HF_HUB_OFFLINE=1`
//!   is set, switches to the local [`Cache`] lookup so air-gapped / CI runs
//!   don't hang on the network (shimmed from mistral.rs because hf-hub 0.5
//!   lacks a native offline switch).
//!
//! # Lazy mmap
//!
//! Tensors are not materialised until `vb.get(..)` is called. Loading a
//! multi-GB checkpoint is fast; only the touched tensors cost memory.
//!
//! # `unsafe` boundary
//!
//! This module is the only `vllm_oxide` module that calls `unsafe` code at
//! T15. The single unsafe call site is [`ShardedSafeTensors::var_builder`],
//! whose unsafe is inherited from [`memmap2::MmapOptions`] (a file mapped
//! from disk can produce UB if mutated externally while mapped). We accept
//! this risk the same way upstream candle / mistral.rs / HF tooling do:
//! checkpoint files are read-only after download, and the VarBuilder's
//! lifetime is bounded by the caller's framing of a single model load.
//!
//! [`Api`]: hf_hub::api::sync::Api
//! [`Cache`]: hf_hub::Cache
//! [`ShardedVarBuilder`]: candle_nn::var_builder::ShardedVarBuilder
//! [`ShardedSafeTensors::var_builder`]: candle_nn::var_builder::ShardedSafeTensors::var_builder

#![allow(unsafe_code)]
// `unsafe_code = "deny"` at workspace level is relaxed specifically for this
// module because the mmap-based safetensors loader is the only candle-native
// way to get lazy on-disk tensor access. See module docs above.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use candle_nn::var_builder::{ShardedSafeTensors, ShardedVarBuilder};
use serde::Deserialize;

use crate::config::{is_hf_hub_offline, Source};

/// Mmap one or more `*.safetensors` files and return a candle
/// [`ShardedVarBuilder`] over them.
///
/// `dtype` is the canonical dtype for the returned builder: each tensor is
/// cast-on-`get` to this dtype (a no-op when the checkpoint already matches).
/// The caller picks the dtype (typically via [`crate::config::default_dtype`]
/// from `config.json`'s `torch_dtype`).
///
/// Returns `Err` if no shards resolve, if a multi-shard index references
/// missing files, or if the underlying mmap fails.
pub fn load_weights(
    source: Source,
    dtype: DType,
    device: &Device,
) -> Result<ShardedVarBuilder<'static>> {
    let paths = match source {
        Source::Local(dir) => resolve_local_shards(&dir)
            .with_context(|| format!("resolving local shards under {}", dir.display()))?,
        Source::Hub { repo, revision } => {
            resolve_hub_shards(&repo, revision.as_deref())
                .with_context(|| format!("resolving Hub shards for {repo}"))?
        }
    };
    if paths.is_empty() {
        return Err(anyhow!(
            "resolved zero safetensors shards from the requested source"
        ));
    }
    tracing::debug!(
        count = paths.len(),
        first = ?paths.first().map(|p| p.display().to_string()),
        "loading safetensors shards via ShardedSafeTensors::var_builder"
    );
    // SAFETY: see module docs — `ShardedSafeTensors::var_builder` mmaps the
    // paths; the unsafe is inherited from memmap2's MmapOptions. Checkpoint
    // files are treated as read-only by convention; same risk surface as
    // upstream candle, mistral.rs, and HF tooling.
    let vb = unsafe { ShardedSafeTensors::var_builder(&paths, dtype, device)? };
    Ok(vb)
}

/// Local-dir fallback chain: `model.safetensors.index.json` → single
/// `model.safetensors` → glob `*.safetensors`. Errors with a "looked in"
/// diagnostic when nothing resolves.
fn resolve_local_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        return parse_index_shards(&index, dir);
    }

    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }

    let dir_entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading checkpoint dir {}", dir.display()))?;
    let mut shards: Vec<PathBuf> = dir_entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();

    if shards.is_empty() {
        return Err(anyhow!(
            "no safetensors shards in {} — looked for \
             `model.safetensors.index.json`, `model.safetensors`, `*.safetensors`",
            dir.display()
        ));
    }
    Ok(shards)
}

/// Minimal subset of HF's `model.safetensors.index.json` schema. Only
/// `weight_map` is consumed; `metadata.total_size` etc. are ignored.
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Parse a `model.safetensors.index.json`, dedupe the shard filenames
/// referenced by `weight_map`, and return absolute paths rooted at `dir`.
/// Errors if `weight_map` is empty or any referenced shard is missing.
fn parse_index_shards(index_path: &Path, dir: &Path) -> Result<Vec<PathBuf>> {
    let bytes = std::fs::read(index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let parsed: SafetensorsIndex = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as safetensors index", index_path.display()))?;

    let mut unique: Vec<String> = parsed.weight_map.into_values().collect();
    unique.sort();
    unique.dedup();

    if unique.is_empty() {
        return Err(anyhow!(
            "{} has an empty `weight_map` — nothing to load",
            index_path.display()
        ));
    }

    let paths: Vec<PathBuf> = unique.into_iter().map(|name| dir.join(name)).collect();
    for p in &paths {
        if !p.is_file() {
            return Err(anyhow!(
                "shard {} is referenced by {} but is not present on disk \
                 (partial download?)",
                p.display(),
                index_path.display()
            ));
        }
    }
    Ok(paths)
}

/// Hub-path shard resolution. Honours `HF_HUB_OFFLINE` (online path will hang
/// on the network otherwise; the offline branch uses the local cache only).
fn resolve_hub_shards(repo: &str, revision: Option<&str>) -> Result<Vec<PathBuf>> {
    let rev = revision.unwrap_or("main");
    if is_hf_hub_offline() {
        return resolve_hub_shards_offline(repo, rev);
    }
    resolve_hub_shards_online(repo, rev)
}

fn resolve_hub_shards_online(repo: &str, revision: &str) -> Result<Vec<PathBuf>> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("building hf-hub sync Api")?;
    let repo_handle = api.repo(hf_hub::Repo::with_revision(
        repo.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    match repo_handle.get("model.safetensors.index.json") {
        Ok(index_path) => {
            let dir = index_path
                .parent()
                .ok_or_else(|| anyhow!("cached index path has no parent dir"))?
                .to_path_buf();
            parse_index_shards(&index_path, &dir)
        }
        Err(_) => {
            let p = repo_handle
                .get("model.safetensors")
                .context("downloading single-file `model.safetensors` from Hub")?;
            Ok(vec![p])
        }
    }
}

fn resolve_hub_shards_offline(repo: &str, revision: &str) -> Result<Vec<PathBuf>> {
    let cache = hf_hub::Cache::from_env();
    let repo_handle = cache.repo(hf_hub::Repo::with_revision(
        repo.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    if let Some(index_path) = repo_handle.get("model.safetensors.index.json") {
        let dir = index_path
            .parent()
            .ok_or_else(|| anyhow!("cached index path has no parent dir"))?
            .to_path_buf();
        return parse_index_shards(&index_path, &dir);
    }
    if let Some(p) = repo_handle.get("model.safetensors") {
        return Ok(vec![p]);
    }

    Err(anyhow!(
        "`HF_HUB_OFFLINE=1` is set and no local snapshot of `{repo}` (revision `{revision}`) \
         was found in the HF cache. Pre-download with `huggingface-cli download {repo}` or \
         unset `HF_HUB_OFFLINE` to allow network access."
    ))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::collapsible_match,
    clippy::needless_range_loop,
    clippy::panic
)]
mod tests {
    use super::*;

    /// Helper: write a single-tensor safetensors file at `path` with the
    /// given name + flat f32 values. Uses the same on-disk format candle
    /// reads back, so loaded tensors round-trip exactly.
    fn write_safetensors_fixture(path: &Path, tensor_name: &str, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![values.len()],
            &bytes,
        )
        .unwrap_or_else(|e| panic!("building TensorView: {e}"));
        safetensors::tensor::serialize_to_file(
            std::iter::once((tensor_name.to_string(), view)),
            &None,
            path,
        )
        .unwrap_or_else(|e| panic!("serializing safetensors fixture: {e}"));
    }

    /// Build a tiny fixture checkpoint under `dir`: an index.json + N shards,
    /// each containing one uniquely-named tensor.
    fn write_multishard_fixture(dir: &Path, shard_names: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        let mut weight_map: HashMap<String, String> = HashMap::new();
        for (i, shard) in shard_names.iter().enumerate() {
            let tensor_name = format!("layer.{i}.weight");
            let values = vec![i as f32; 4];
            write_safetensors_fixture(&dir.join(shard), &tensor_name, &values);
            weight_map.insert(tensor_name, (*shard).to_string());
        }
        let index = serde_json::json!({ "weight_map": weight_map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&index).unwrap(),
        )
        .unwrap();
    }

    mod resolve_local_shards {
        use super::*;

        #[test]
        fn index_json_drives_multishard_resolution() {
            let tmp = tempfile::tempdir().unwrap();
            write_multishard_fixture(tmp.path(), &["shard0.safetensors", "shard1.safetensors"]);

            let paths = resolve_local_shards(tmp.path()).unwrap();
            assert_eq!(paths.len(), 2);
            assert!(paths[0].ends_with("shard0.safetensors"));
            assert!(paths[1].ends_with("shard1.safetensors"));
        }

        #[test]
        fn single_model_safetensors_when_no_index() {
            let tmp = tempfile::tempdir().unwrap();
            write_safetensors_fixture(
                &tmp.path().join("model.safetensors"),
                "embeddings.weight",
                &[1.0, 2.0, 3.0],
            );

            let paths = resolve_local_shards(tmp.path()).unwrap();
            assert_eq!(paths.len(), 1);
            assert!(paths[0].ends_with("model.safetensors"));
        }

        #[test]
        fn bare_directory_glob_sorts_shards() {
            let tmp = tempfile::tempdir().unwrap();
            // write out-of-order to prove the sort happens
            write_safetensors_fixture(&tmp.path().join("zzz.safetensors"), "z", &[0.0]);
            write_safetensors_fixture(&tmp.path().join("aaa.safetensors"), "a", &[0.0]);

            let paths = resolve_local_shards(tmp.path()).unwrap();
            assert_eq!(paths.len(), 2);
            assert!(paths[0].ends_with("aaa.safetensors"));
            assert!(paths[1].ends_with("zzz.safetensors"));
        }

        #[test]
        fn empty_dir_errors_with_looked_in_message() {
            let tmp = tempfile::tempdir().unwrap();
            let err = resolve_local_shards(tmp.path()).unwrap_err();
            let msg = format!("{err:#}");
            assert!(msg.contains("no safetensors shards"), "got: {msg}");
            assert!(msg.contains("model.safetensors.index.json"), "got: {msg}");
        }

        #[test]
        fn dir_with_non_safetensors_files_errors() {
            let tmp = tempfile::tempdir().unwrap();
            std::fs::write(tmp.path().join("config.json"), b"{}").unwrap();
            std::fs::write(tmp.path().join("README.md"), b"hi").unwrap();
            assert!(resolve_local_shards(tmp.path()).is_err());
        }

        #[test]
        fn missing_dir_errors() {
            let bogus = Path::new("/definitely/does/not/exist/xyzzy");
            assert!(resolve_local_shards(bogus).is_err());
        }

        #[test]
        fn glob_ignores_index_json_file_extension_overlap() {
            let tmp = tempfile::tempdir().unwrap();
            write_multishard_fixture(tmp.path(), &["model-00001-of-00002.safetensors"]);
            assert!(resolve_local_shards(tmp.path()).is_ok());
            assert!(
                !resolve_local_shards(tmp.path())
                    .unwrap()
                    .iter()
                    .any(|p| p.to_string_lossy().contains("index.json"))
            );
        }

        #[test]
        fn index_present_but_shard_missing_errors() {
            let tmp = tempfile::tempdir().unwrap();
            write_multishard_fixture(tmp.path(), &["model-00001-of-00002.safetensors"]);
            std::fs::remove_file(tmp.path().join("model-00001-of-00002.safetensors")).unwrap();

            let err = resolve_local_shards(tmp.path()).unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.contains("referenced by") && msg.contains("model-00001-of-00002"),
                "got: {msg}"
            );
        }
    }

    mod parse_index_shards {
        use super::*;

        fn write_index(dir: &Path, weight_map: HashMap<String, String>) -> PathBuf {
            let index = serde_json::json!({ "weight_map": weight_map });
            let path = dir.join("model.safetensors.index.json");
            std::fs::write(&path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
            path
        }

        #[test]
        fn dedupes_shard_names() {
            let tmp = tempfile::tempdir().unwrap();
            // 1000 tensors across 3 shards → must collapse to 3 paths.
            let mut wm = HashMap::new();
            for i in 0..1000 {
                let shard = format!("shard-{}.safetensors", i % 3);
                wm.insert(format!("t{i}"), shard);
            }
            for s in ["shard-0.safetensors", "shard-1.safetensors", "shard-2.safetensors"] {
                write_safetensors_fixture(&tmp.path().join(s), "t", &[0.0]);
            }
            let index_path = write_index(tmp.path(), wm);

            let paths = parse_index_shards(&index_path, tmp.path()).unwrap();
            assert_eq!(paths.len(), 3);
            assert!(paths.iter().all(|p| p.is_file()));
        }

        #[test]
        fn empty_weight_map_errors() {
            let tmp = tempfile::tempdir().unwrap();
            let index_path = write_index(tmp.path(), HashMap::new());
            let err = parse_index_shards(&index_path, tmp.path()).unwrap_err();
            assert!(format!("{err:#}").contains("empty"), "got: {err:#}");
        }

        #[test]
        fn missing_shard_referenced_by_index_errors() {
            let tmp = tempfile::tempdir().unwrap();
            let mut wm = HashMap::new();
            wm.insert("t0".to_string(), "shard-0.safetensors".to_string());
            wm.insert("t1".to_string(), "shard-1.safetensors".to_string());
            // Only create shard-0 on disk.
            write_safetensors_fixture(&tmp.path().join("shard-0.safetensors"), "t0", &[0.0]);
            let index_path = write_index(tmp.path(), wm);

            let err = parse_index_shards(&index_path, tmp.path()).unwrap_err();
            let msg = format!("{err:#}");
            assert!(msg.contains("shard-1.safetensors"), "got: {msg}");
            assert!(msg.contains("partial download"), "got: {msg}");
        }

        #[test]
        fn malformed_json_errors() {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("model.safetensors.index.json");
            std::fs::write(&path, b"{not json").unwrap();
            assert!(parse_index_shards(&path, tmp.path()).is_err());
        }

        #[test]
        fn missing_index_file_errors() {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("model.safetensors.index.json");
            assert!(parse_index_shards(&path, tmp.path()).is_err());
        }
    }

    mod load_weights_local {
        use super::*;
        use candle_core::Device;

        #[test]
        fn single_file_load_round_trip() {
            let tmp = tempfile::tempdir().unwrap();
            write_safetensors_fixture(
                &tmp.path().join("model.safetensors"),
                "embed.weight",
                &[1.0_f32, 2.0, 3.0, 4.0],
            );

            let device = Device::Cpu;
            let vb =
                load_weights(Source::Local(tmp.path().to_path_buf()), DType::F32, &device).unwrap();

            // The VarBuilder's prefix starts empty; HF tensor names map 1:1.
            let t = vb.get((4,), "embed.weight").unwrap();
            let got: Vec<f32> = t.to_vec1().unwrap();
            assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn multi_shard_load_round_trip() {
            let tmp = tempfile::tempdir().unwrap();
            write_multishard_fixture(
                tmp.path(),
                &["shard-a.safetensors", "shard-b.safetensors"],
            );

            let device = Device::Cpu;
            let vb =
                load_weights(Source::Local(tmp.path().to_path_buf()), DType::F32, &device).unwrap();

            // write_multishard_fixture puts `layer.<i>.weight` in shard i.
            let t0 = vb.get((4,), "layer.0.weight").unwrap();
            let t1 = vb.get((4,), "layer.1.weight").unwrap();
            assert_eq!(t0.to_vec1::<f32>().unwrap(), vec![0.0; 4]);
            assert_eq!(t1.to_vec1::<f32>().unwrap(), vec![1.0; 4]);
        }

        #[test]
        fn dtype_cast_on_get_when_checkpoint_mismatches() {
            let tmp = tempfile::tempdir().unwrap();
            write_safetensors_fixture(
                &tmp.path().join("model.safetensors"),
                "w",
                &[1.0_f32, 2.0],
            );

            let device = Device::Cpu;
            // Request F16 even though the file is F32 — candle casts lazily
            // on `get`. Verifies the `dtype` parameter threads all the way
            // through ShardedSafeTensors::var_builder.
            let vb =
                load_weights(Source::Local(tmp.path().to_path_buf()), DType::F16, &device).unwrap();
            let t = vb.get((2,), "w").unwrap();
            assert_eq!(t.dtype(), DType::F16);
        }

        #[test]
        fn missing_tensor_errors_cannot_find_tensor() {
            let tmp = tempfile::tempdir().unwrap();
            write_safetensors_fixture(
                &tmp.path().join("model.safetensors"),
                "present",
                &[0.0_f32],
            );

            let device = Device::Cpu;
            let vb =
                load_weights(Source::Local(tmp.path().to_path_buf()), DType::F32, &device).unwrap();
            let err = vb.get((1,), "absent").unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("absent") && msg.contains("find"),
                "expected missing-tensor error mentioning `absent`, got: {msg}"
            );
        }

        #[test]
        fn empty_dir_errors_with_looked_in_context() {
            let tmp = tempfile::tempdir().unwrap();
            let err = load_weights(
                Source::Local(tmp.path().to_path_buf()),
                DType::F32,
                &Device::Cpu,
            )
            .err()
            .expect("expected Err from empty dir");
            let msg = format!("{err:#}");
            assert!(msg.contains("no safetensors shards"), "got: {msg}");
        }

        #[test]
        fn pp_prefix_paths_join_with_dots() {
            // Locks in candle's dotted-prefix convention so HF tensor names
            // like `model.layers.0.self_attn.q_proj.weight` work without a
            // remap table (ADR-0002 contract).
            let tmp = tempfile::tempdir().unwrap();
            write_safetensors_fixture(
                &tmp.path().join("model.safetensors"),
                "model.layers.0.self_attn.q_proj.weight",
                &[0.0_f32; 2],
            );

            let device = Device::Cpu;
            let vb =
                load_weights(Source::Local(tmp.path().to_path_buf()), DType::F32, &device).unwrap();
            let t = vb
                .pp("model")
                .pp("layers")
                .pp(0)
                .pp("self_attn")
                .pp("q_proj")
                .get((2,), "weight")
                .unwrap();
            assert_eq!(t.to_vec1::<f32>().unwrap(), vec![0.0; 2]);
        }
    }
}
