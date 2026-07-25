//! Thin wrappers around `candle-flash-attn`'s two entry points.
//!
//! Prefill calls `flash_attn_varlen` (unpaged — reads projection K/V).
//! Decode calls `flash_attn_varlen_paged_windowed` (paged — reads the KV cache).
//! Model code calls these directly — no trait (T4 YAGNI decision).

#![cfg(feature = "cuda")]
#![allow(dead_code)]

use candle_core::{Device, Result, Tensor};
use candle_flash_attn::{flash_attn_varlen, flash_attn_varlen_paged_windowed};

use super::metadata::AttnMetadata;

pub fn prefill_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    meta: &AttnMetadata,
    softmax_scale: f32,
) -> Result<Tensor> {
    let dev = q.device();
    let cu_seqlens_q = cu_seqlens_tensor(&meta.cu_seqlens_q, dev)?;
    let cu_seqlens_k = cu_seqlens_tensor(&meta.cu_seqlens_k, dev)?;

    flash_attn_varlen(
        q, k, v,
        &cu_seqlens_q, &cu_seqlens_k,
        meta.max_seqlen_q, meta.max_seqlen_k,
        softmax_scale, true,
    )
}

pub fn decode_attn(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    meta: &AttnMetadata,
    softmax_scale: f32,
    page_block_size: usize,
) -> Result<Tensor> {
    let dev = q.device();
    let cu_seqlens_q = cu_seqlens_tensor(&meta.cu_seqlens_q, dev)?;
    let cu_seqlens_k = cu_seqlens_tensor(&meta.cu_seqlens_k, dev)?;
    let block_table = block_table_tensor(&meta.block_table, dev)?;

    flash_attn_varlen_paged_windowed(
        q, k_cache, v_cache,
        &cu_seqlens_q, &cu_seqlens_k,
        &block_table,
        None,
        meta.max_seqlen_q, meta.max_seqlen_k,
        softmax_scale,
        None,
        Some(0),
        page_block_size,
        None,
    )
}

fn cu_seqlens_tensor(seqlens: &[u32], dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(seqlens.to_vec(), (seqlens.len(),), dev)
}

fn block_table_tensor(block_table: &[Vec<i32>], dev: &Device) -> Result<Tensor> {
    if block_table.is_empty() {
        candle_core::bail!("block_table must be non-empty for decode attention");
    }
    let max_blocks = block_table.iter().map(|r| r.len()).max().unwrap_or(0);
    let batch = block_table.len();
    let mut flat: Vec<i32> = Vec::with_capacity(batch * max_blocks);
    for row in block_table {
        flat.extend_from_slice(row);
        flat.resize(flat.len() + max_blocks - row.len(), 0);
    }
    Tensor::from_vec(flat, (batch, max_blocks), dev)
}
