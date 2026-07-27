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
    let file_bytes = std::fs::read(path)
        .with_context(|| format!("reading fixture from {}", path.display()))?;

    let tensors = SafeTensors::deserialize(&file_bytes)
        .with_context(|| format!("deserializing safetensors from {}", path.display()))?;

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
            let len = view.data().len() / 8;
            let data: &[i64] = bytemuck::cast_slice(view.data());
            Ok(data[..len].to_vec())
        }
        other => anyhow::bail!(
            "tensor '{}' has dtype {:?}, expected I64",
            name,
            other
        ),
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
            let len = view.data().len() / 4;
            let data: &[f32] = bytemuck::cast_slice(view.data());
            Ok(data[..len].to_vec())
        }
        other => anyhow::bail!(
            "tensor '{}' has dtype {:?}, expected F32",
            name,
            other
        ),
    }
}
