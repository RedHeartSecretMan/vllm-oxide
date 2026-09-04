//! Thin wrappers around `candle-flash-attn`'s two entry points.
//!
//! Prefill calls `flash_attn_varlen` (unpaged — reads projection K/V).
//! Continued prefill and decode call `flash_attn_varlen_paged_windowed`
//! (paged — reads the KV cache).
//! Model code calls these directly — no trait (T4 YAGNI decision).

#![cfg(feature = "cuda")]
#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_flash_attn::{flash_attn_varlen, flash_attn_varlen_paged_windowed};

use super::PreparedAttention;

pub fn prefill_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prepared: &PreparedAttention,
    softmax_scale: f32,
) -> Result<Tensor> {
    let logical = prepared.logical();

    flash_attn_varlen(
        q,
        k,
        v,
        prepared.cu_seqlens_q(),
        prepared.cu_seqlens_k(),
        logical.max_seqlen_q,
        logical.max_seqlen_k,
        softmax_scale,
        true,
    )
}

pub fn paged_attn(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    prepared: &PreparedAttention,
    softmax_scale: f32,
    page_block_size: usize,
) -> Result<Tensor> {
    let logical = prepared.logical();

    flash_attn_varlen_paged_windowed(
        q,
        k_cache,
        v_cache,
        prepared.cu_seqlens_q(),
        prepared.cu_seqlens_k(),
        prepared.block_table()?,
        None,
        logical.max_seqlen_q,
        logical.max_seqlen_k,
        softmax_scale,
        None,
        Some(0),
        page_block_size,
        None,
    )
}
