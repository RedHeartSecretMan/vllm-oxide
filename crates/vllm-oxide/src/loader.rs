//! Internal model-agnostic weight loader (ADR-0002, ADR-0011).
//!
//! The composition root resolves a source into one immutable
//! [`ResolvedModel`], then this module mmaps that exact weight set into a
//! candle [`VarBuilder`]. The loader does NOT do any model-specific work:
//! no QKV fusion, no gate/up fusion, no per-rank slicing. All of that lives
//! in `Linear::<P>::from_vb` (T3) and `ParallelStyle::slice_for_rank`
//! (v0.2). HF checkpoint tensor names map 1:1 with what the model expects.
//!
//! # Lazy mmap
//!
//! Tensors are not materialised until `vb.get(..)` is called. Loading a
//! multi-GB checkpoint is fast; only the touched tensors cost memory.
//!
//! # `unsafe` boundary
//!
//! This module is the only `vllm_oxide` module that calls `unsafe` code at
//! the weight-loading seam. The single unsafe call maps the already-resolved
//! safetensors through `MmapedSafetensors::multi`; its risk is inherited from
//! `memmap2::MmapOptions` (a file mapped
//! from disk can produce UB if mutated externally while mapped). We accept
//! this risk the same way upstream candle / mistral.rs / HF tooling do:
//! checkpoint files are read-only after download, and the VarBuilder's
//! lifetime is bounded by the caller's framing of a single model load.
//!
//! [`VarBuilder`]: candle_nn::VarBuilder

#![allow(unsafe_code)]
// `unsafe_code = "deny"` at workspace level is relaxed specifically for this
// module because the mmap-based safetensors loader is the only candle-native
// way to get lazy on-disk tensor access. See module docs above.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device};
use candle_nn::var_builder::{SimpleBackend, VarBuilder, VarBuilderArgs};
use serde::Deserialize;

pub(crate) mod model_identity;
use model_identity::ResolvedModel;

fn var_builder_from_paths(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
    description: &str,
) -> Result<VarBuilder<'static>> {
    if paths.is_empty() {
        return Err(anyhow!("{description} resolved zero safetensors shards"));
    }
    let tensors = unsafe { MmapedSafetensors::multi(paths) }
        .with_context(|| format!("mapping safetensors for {description}"))?;
    let backend: Box<dyn SimpleBackend + 'static> = Box::new(tensors);
    Ok(VarBuilderArgs::new_with_args(backend, dtype, device))
}

/// Build a candle [`VarBuilder`] from the already-resolved artifact paths and
/// dtype. This is the construction path used by model factories: it cannot
/// re-resolve a local path or follow a moving Hub revision.
pub(crate) fn load_resolved_weights_vb(
    resolved: &ResolvedModel,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    var_builder_from_paths(
        resolved.weight_paths(),
        resolved.dtype(),
        device,
        &format!("model identity `{}`", resolved.identity()),
    )
}

pub(crate) fn validate_model_dtype(dtype: DType, origin: &str) -> Result<DType> {
    if matches!(dtype, DType::BF16 | DType::F16) {
        Ok(dtype)
    } else {
        Err(anyhow!(
            "{origin} {dtype:?} is unsupported; supported model and KV-cache dtypes: BF16, F16"
        ))
    }
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
    let unique = parse_index_shard_names(index_path)?;
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

fn parse_index_shard_names(index_path: &Path) -> Result<Vec<String>> {
    let bytes =
        std::fs::read(index_path).with_context(|| format!("reading {}", index_path.display()))?;
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
    for name in &unique {
        let path = Path::new(name);
        if name.is_empty()
            || path
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(anyhow!(
                "safetensors shard `{name}` from {} points outside resolved model identity",
                index_path.display()
            ));
        }
    }
    Ok(unique)
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
            assert!(!resolve_local_shards(tmp.path())
                .unwrap()
                .iter()
                .any(|p| p.to_string_lossy().contains("index.json")));
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
            for s in [
                "shard-0.safetensors",
                "shard-1.safetensors",
                "shard-2.safetensors",
            ] {
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

    mod load_resolved_weights {
        use super::*;

        #[test]
        fn resolved_model_dtype_reaches_weight_builder() {
            let tmp = tempfile::tempdir().unwrap();
            write_safetensors_fixture(&tmp.path().join("model.safetensors"), "w", &[1.0_f32, 2.0]);
            std::fs::write(
                tmp.path().join("config.json"),
                br#"{"torch_dtype":"bfloat16","eos_token_id":1}"#,
            )
            .unwrap();
            std::fs::write(tmp.path().join("tokenizer.json"), b"{}").unwrap();
            let resolved = ResolvedModel::resolve(
                crate::Source::Local(tmp.path().to_path_buf()),
                Some(DType::F16),
            )
            .unwrap();

            let vb = load_resolved_weights_vb(&resolved, &Device::Cpu).unwrap();
            let tensor = vb.get((2,), "w").unwrap();

            assert_eq!(tensor.dtype(), DType::F16);
        }
    }
}
