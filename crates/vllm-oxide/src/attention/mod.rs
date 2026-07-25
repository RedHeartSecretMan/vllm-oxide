//! `attention/` — T4 paged-attention contract.
//!
//! Model code calls `flash_attn_varlen` (prefill) / `flash_attn_varlen_paged_windowed`
//! (decode) directly — NO `AttentionBackend` trait for v0.1 (YAGNI). The
//! `engine ↔ attention` cycle is broken by `attention/` never importing
//! `engine/`; EngineCore holds `Arc<Mutex<PagedKVCache>>` and builds
//! `AttnMetadata` from scheduler state.

#![allow(dead_code)]

pub(crate) mod flash_attn;
pub(crate) mod metadata;

#[cfg(feature = "cuda")]
pub(crate) mod kernels;

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use crate::utils::kv_cache_layout_shape;

pub use metadata::{AttnMetadata, build_decode_metadata, build_prefill_metadata};

/// Physical GPU buffer for the paged KV cache.
///
/// Shaped `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
/// (nano-vllm parity layout). The leading `2` is the K/V stack: dim 0 = keys,
/// dim 1 = values.
///
/// Shared across decoder layers via `Arc<Mutex<PagedKVCache>>` (constructed by
/// `EngineCore`, cloned into each `Qwen3Attention`).
pub struct PagedKVCache {
    buffer: Tensor,
    num_layers: usize,
    num_blocks: usize,
    block_size: usize,
}

impl PagedKVCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = kv_cache_layout_shape(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
        );
        let buffer = Tensor::zeros(&shape, dtype, device)?;
        Ok(Self { buffer, num_layers, num_blocks, block_size })
    }

    /// Per-layer K cache view: `[num_blocks, block_size, num_kv_heads, head_dim]`.
    pub fn k_cache(&self, layer_id: usize) -> Result<Tensor> {
        self.buffer.i((0, layer_id))
    }

    /// Per-layer V cache view: `[num_blocks, block_size, num_kv_heads, head_dim]`.
    pub fn v_cache(&self, layer_id: usize) -> Result<Tensor> {
        self.buffer.i((1, layer_id))
    }

    /// Write per-step K/V into the paged cache via the custom CUDA kernel.
    #[cfg(feature = "cuda")]
    pub fn reshape_and_cache(
        &self, layer_id: usize, key: &Tensor, value: &Tensor, slot_mapping: &Tensor,
    ) -> Result<()> {
        let k_cache = self.k_cache(layer_id)?;
        let v_cache = self.v_cache(layer_id)?;
        kernels::reshape_and_cache(key, value, &k_cache, &v_cache, slot_mapping)
    }

    pub fn num_blocks(&self) -> usize { self.num_blocks }
    pub fn block_size(&self) -> usize { self.block_size }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_correct_shape() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(28, 100, 256, 4, 128, DType::BF16, &dev).unwrap();
        assert_eq!(cache.buffer.shape().dims(), &[2, 28, 100, 256, 4, 128]);
    }

    #[test]
    fn k_cache_slice_shape() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(4, 10, 256, 4, 128, DType::BF16, &dev).unwrap();
        let k = cache.k_cache(2).unwrap();
        assert_eq!(k.shape().dims(), &[10, 256, 4, 128]);
    }

    #[test]
    fn v_cache_slice_shape() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(4, 10, 256, 4, 128, DType::BF16, &dev).unwrap();
        let v = cache.v_cache(1).unwrap();
        assert_eq!(v.shape().dims(), &[10, 256, 4, 128]);
    }

    #[test]
    fn accessors() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(1, 42, 256, 1, 1, DType::F32, &dev).unwrap();
        assert_eq!(cache.block_size(), 256);
        assert_eq!(cache.num_blocks(), 42);
    }
}
