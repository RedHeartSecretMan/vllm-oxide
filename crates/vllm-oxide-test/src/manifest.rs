use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;

use crate::types::{FixtureData, FixtureMetadata, Manifest};

/// Parse a `manifest.json` file.
pub fn parse_manifest(path: &Path) -> Result<Manifest> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading manifest from {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("parsing manifest from {}", path.display()))
}

/// Load a single `.safetensors` fixture file into a `FixtureData`.
pub fn load_fixture(path: &Path, meta: &FixtureMetadata) -> Result<FixtureData> {
    let file_bytes =
        std::fs::read(path).with_context(|| format!("reading fixture from {}", path.display()))?;

    let tensors = SafeTensors::deserialize(&file_bytes)
        .with_context(|| format!("deserializing safetensors from {}", path.display()))?;

    if meta.num_tokens == 0 {
        anyhow::bail!("unsupported empty fixture shape for {}", meta.filename);
    }
    validate_tensor_shape(
        &tensors,
        "token_ids",
        safetensors::Dtype::I64,
        &[meta.num_tokens as usize],
    )?;
    validate_tensor_shape(&tensors, "n_prompt_tokens", safetensors::Dtype::I64, &[])?;

    let actual_names: HashSet<_> = tensors.names().into_iter().map(String::as_str).collect();
    match meta.category {
        crate::types::PromptCategory::Canonical => {
            let expected_names = HashSet::from(["token_ids", "n_prompt_tokens", "logits"]);
            if actual_names != expected_names {
                anyhow::bail!("unsupported canonical tensor set for {}", meta.filename);
            }
            if meta.logits_shape.0 != meta.num_tokens as usize || meta.logits_shape.1 == 0 {
                anyhow::bail!("unsupported logits shape for {}", meta.filename);
            }
            validate_tensor_shape(
                &tensors,
                "logits",
                safetensors::Dtype::F32,
                &[meta.logits_shape.0, meta.logits_shape.1],
            )?;
        }
        crate::types::PromptCategory::Regression => {
            let expected_names = HashSet::from([
                "token_ids",
                "n_prompt_tokens",
                "top5_indices",
                "top5_logits",
            ]);
            if actual_names != expected_names || meta.logits_shape != (0, 0) {
                anyhow::bail!("unsupported regression fixture shape for {}", meta.filename);
            }
            validate_tensor_shape(
                &tensors,
                "top5_indices",
                safetensors::Dtype::I64,
                &[meta.num_tokens as usize, 5],
            )?;
            validate_tensor_shape(
                &tensors,
                "top5_logits",
                safetensors::Dtype::F32,
                &[meta.num_tokens as usize, 5],
            )?;
        }
    }

    let token_ids: Vec<i64> = read_named_tensor_as_i64(&tensors, "token_ids")?;
    let n_prompt_tokens: i64 = read_named_scalar_as_i64(&tensors, "n_prompt_tokens")?;

    let (logits, top5_indices, top5_logits) = match meta.category {
        crate::types::PromptCategory::Canonical => {
            let logits_vec: Vec<f32> = read_named_tensor_as_f32(&tensors, "logits")?;
            (Some(logits_vec), None, None)
        }
        crate::types::PromptCategory::Regression => {
            let indices: Vec<i64> = read_named_tensor_as_i64(&tensors, "top5_indices")?;
            let logits_top5: Vec<f32> = read_named_tensor_as_f32(&tensors, "top5_logits")?;
            (None, Some(indices), Some(logits_top5))
        }
    };

    Ok(FixtureData {
        prompt_id: meta.prompt_id.clone(),
        category: meta.category.clone(),
        oracle: meta.oracle.clone(),
        num_tokens: meta.num_tokens,
        token_ids,
        n_prompt_tokens,
        logits,
        logits_shape: meta.logits_shape,
        top5_indices,
        top5_logits,
    })
}

fn validate_tensor_shape(
    tensors: &SafeTensors<'_>,
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[usize],
) -> Result<()> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor '{name}' not found in safetensors"))?;
    if view.dtype() != dtype || view.shape() != shape {
        anyhow::bail!(
            "unsupported {name} shape or dtype: got {:?} {:?}, expected {:?} {:?}",
            view.dtype(),
            view.shape(),
            dtype,
            shape
        );
    }
    Ok(())
}

/// Compute SHA-256 hex digest of file bytes.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn read_named_tensor_as_i64(tensors: &SafeTensors, name: &str) -> Result<Vec<i64>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor '{}' not found in safetensors", name))?;
    match view.dtype() {
        safetensors::Dtype::I64 => {
            // Safetensors stores little-endian bytes and does not require each
            // tensor's payload address to be aligned for a typed slice.
            Ok(view
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| i64::from_le_bytes(*chunk))
                .collect())
        }
        other => anyhow::bail!("tensor '{}' has dtype {:?}, expected I64", name, other),
    }
}

fn read_named_scalar_as_i64(tensors: &SafeTensors, name: &str) -> Result<i64> {
    let data = read_named_tensor_as_i64(tensors, name)?;
    data.first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("tensor '{}' is empty (expected scalar)", name))
}

fn read_named_tensor_as_f32(tensors: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor '{}' not found in safetensors", name))?;
    match view.dtype() {
        safetensors::Dtype::F32 => {
            // Safetensors stores little-endian bytes and does not require each
            // tensor's payload address to be aligned for a typed slice.
            Ok(view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect())
        }
        other => anyhow::bail!("tensor '{}' has dtype {:?}, expected F32", name, other),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod fixture_reader_tests {
    use safetensors::SafeTensors;
    use serde_json::json;

    use super::{load_fixture, read_named_tensor_as_f32, read_named_tensor_as_i64};
    use crate::types::{FixtureMetadata, LogitsDtype, OracleName, PromptCategory};

    struct WireTensor {
        name: &'static str,
        dtype: &'static str,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    // The safetensors format allows contiguous payloads in any order. The normal
    // serializer sorts by dtype size, so write the header explicitly to cover
    // files whose integer tensors follow floating-point or byte tensors.
    fn wire_fixture(tensors: &[WireTensor]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for tensor in tensors {
            let start = payload.len();
            payload.extend_from_slice(&tensor.data);
            header.insert(
                tensor.name.to_string(),
                json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [start, payload.len()],
                }),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn regression_tensors() -> Vec<WireTensor> {
        vec![
            WireTensor {
                name: "top5_logits",
                dtype: "F32",
                shape: vec![1, 5],
                data: [1.25_f32, -2.5, 0.0, 8.0, 0.125]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            },
            WireTensor {
                name: "token_ids",
                dtype: "I64",
                shape: vec![1],
                data: 42_i64.to_le_bytes().to_vec(),
            },
            WireTensor {
                name: "n_prompt_tokens",
                dtype: "I64",
                shape: vec![],
                data: 3_i64.to_le_bytes().to_vec(),
            },
            WireTensor {
                name: "top5_indices",
                dtype: "I64",
                shape: vec![1, 5],
                data: [42_i64, 17, 0, 9, 8]
                    .into_iter()
                    .flat_map(i64::to_le_bytes)
                    .collect(),
            },
        ]
    }

    fn metadata(category: PromptCategory) -> FixtureMetadata {
        let logits_shape = match category {
            PromptCategory::Canonical => (1, 5),
            PromptCategory::Regression => (0, 0),
        };
        FixtureMetadata {
            prompt_id: "wire_order".to_string(),
            category,
            oracle: OracleName::Transformers,
            num_tokens: 1,
            logits_dtype: LogitsDtype::Float32,
            logits_shape,
            sha256: "unused".to_string(),
            filename: "wire_order.transformers.safetensors".to_string(),
        }
    }

    #[test]
    fn loads_regression_with_float_payload_before_unaligned_integers() {
        let bytes = wire_fixture(&regression_tensors());
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let token_view = tensors.tensor("token_ids").unwrap();
        assert_ne!(token_view.data().as_ptr().align_offset(8), 0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("regression.safetensors");
        std::fs::write(&path, bytes).unwrap();

        let fixture = load_fixture(&path, &metadata(PromptCategory::Regression)).unwrap();

        assert_eq!(fixture.token_ids, [42]);
        assert_eq!(fixture.n_prompt_tokens, 3);
        assert_eq!(fixture.top5_indices.unwrap(), [42, 17, 0, 9, 8]);
        assert_eq!(fixture.top5_logits.unwrap(), [1.25, -2.5, 0.0, 8.0, 0.125]);
        assert!(fixture.logits.is_none());
    }

    #[test]
    fn reads_unaligned_little_endian_float_and_signed_integer_values() {
        // Explicit wire bytes include signed extremes and negative zero. This
        // checks the file's little-endian representation, not native endianness.
        let bytes = wire_fixture(&[
            WireTensor {
                name: "prefix",
                dtype: "U8",
                shape: vec![1],
                data: vec![0xaa],
            },
            WireTensor {
                name: "integers",
                dtype: "I64",
                shape: vec![4],
                data: vec![
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0x80,
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 4, 3, 2, 1, 0, 0, 0, 0,
                ],
            },
            WireTensor {
                name: "floats",
                dtype: "F32",
                shape: vec![4],
                data: vec![
                    0, 0, 0, 0x80, 0, 0, 0xa0, 0x3f, 0, 0, 0x20, 0xc0, 0, 0, 0x80, 0x7f,
                ],
            },
        ]);
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        assert_ne!(
            tensors
                .tensor("integers")
                .unwrap()
                .data()
                .as_ptr()
                .align_offset(8),
            0
        );
        assert_ne!(
            tensors
                .tensor("floats")
                .unwrap()
                .data()
                .as_ptr()
                .align_offset(4),
            0
        );

        assert_eq!(
            read_named_tensor_as_i64(&tensors, "integers").unwrap(),
            [-1, i64::MIN, i64::MAX, 16_909_060]
        );
        let float_bits: Vec<u32> = read_named_tensor_as_f32(&tensors, "floats")
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(
            float_bits,
            [0x8000_0000, 0x3fa0_0000, 0xc020_0000, 0x7f80_0000]
        );
    }

    #[test]
    fn rejects_wrong_wire_dtypes_without_reinterpreting_values() {
        let bytes = wire_fixture(&[
            WireTensor {
                name: "integers",
                dtype: "U64",
                shape: vec![1],
                data: u64::MAX.to_le_bytes().to_vec(),
            },
            WireTensor {
                name: "floats",
                dtype: "F64",
                shape: vec![1],
                data: 1.25_f64.to_le_bytes().to_vec(),
            },
        ]);
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        assert!(read_named_tensor_as_i64(&tensors, "integers")
            .unwrap_err()
            .to_string()
            .contains("expected I64"));
        assert!(read_named_tensor_as_f32(&tensors, "floats")
            .unwrap_err()
            .to_string()
            .contains("expected F32"));
    }

    #[test]
    fn rejects_regression_shapes_inconsistent_with_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wrong_shape.safetensors");
        for (name, wrong_shape) in [
            ("token_ids", vec![1, 1]),
            ("n_prompt_tokens", vec![1]),
            ("top5_indices", vec![5]),
            ("top5_logits", vec![5]),
        ] {
            let mut tensors = regression_tensors();
            tensors
                .iter_mut()
                .find(|tensor| tensor.name == name)
                .unwrap()
                .shape = wrong_shape;
            std::fs::write(&path, wire_fixture(&tensors)).unwrap();
            let error = load_fixture(&path, &metadata(PromptCategory::Regression))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(name) && error.contains("shape"),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn rejects_canonical_logits_shape_inconsistent_with_metadata() {
        let mut tensors = regression_tensors();
        tensors[0].name = "logits";
        tensors[0].shape = vec![5];
        tensors.pop();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wrong_logits_shape.safetensors");
        std::fs::write(&path, wire_fixture(&tensors)).unwrap();
        let error = load_fixture(&path, &metadata(PromptCategory::Canonical))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("logits") && error.contains("shape"),
            "{error}"
        );
    }
}
