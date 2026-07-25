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

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use super::*;

    fn cuda_device() -> Device {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    }

    #[test]
    fn reshape_and_cache_writes_correct_slots() {
        let dev = cuda_device();
        if !dev.is_cuda() {
            eprintln!("[gpu_tests] skipping — no CUDA device");
            return;
        }

        let num_kv_heads = 2;
        let head_dim = 64;
        let cache = PagedKVCache::new(1, 4, 256, num_kv_heads, head_dim, DType::BF16, &dev).unwrap();

        let num_tokens = 3;
        let key = Tensor::arange(0f32, (num_tokens * num_kv_heads * head_dim) as f32, &dev)
            .unwrap()
            .reshape((num_tokens, num_kv_heads, head_dim))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let value = Tensor::arange(1000f32, 1000f32 + (num_tokens * num_kv_heads * head_dim) as f32, &dev)
            .unwrap()
            .reshape((num_tokens, num_kv_heads, head_dim))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let slot_mapping = Tensor::from_vec(vec![0i64, 256, 512], (num_tokens,), &dev).unwrap();

        cache.reshape_and_cache(0, &key, &value, &slot_mapping).unwrap();

        let k_cache = cache.k_cache(0).unwrap();
        let v_cache = cache.v_cache(0).unwrap();

        let expected_k_0 = key.i(0).unwrap();
        let written_k_0 = k_cache.i((0, 0)).unwrap();
        let diff = (written_k_0.to_dtype(DType::F32).unwrap() - expected_k_0.to_dtype(DType::F32).unwrap()).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
        assert_eq!(max_diff, 0.0, "K slot 0 (block 0, offset 0) mismatch");

        let expected_k_1 = key.i(1).unwrap();
        let written_k_1 = k_cache.i((1, 0)).unwrap();
        let diff = (written_k_1.to_dtype(DType::F32).unwrap() - expected_k_1.to_dtype(DType::F32).unwrap()).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
        assert_eq!(max_diff, 0.0, "K slot 256 (block 1, offset 0) mismatch");

        let expected_v_2 = value.i(2).unwrap();
        let written_v_2 = v_cache.i((2, 0)).unwrap();
        let diff = (written_v_2.to_dtype(DType::F32).unwrap() - expected_v_2.to_dtype(DType::F32).unwrap()).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
        assert_eq!(max_diff, 0.0, "V slot 512 (block 2, offset 0) mismatch");
    }

    #[test]
    fn flash_attn_prefill_runs() {
        let dev = cuda_device();
        if !dev.is_cuda() {
            eprintln!("[gpu_tests] skipping — no CUDA device");
            return;
        }

        let num_heads = 4;
        let head_dim = 64;
        let seq_len = 8;

        let q = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        let k = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        let v = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &dev).unwrap().to_dtype(DType::BF16).unwrap();

        let meta = build_prefill_metadata(&[seq_len as u32], &[seq_len as u32], &(0..seq_len as i64).collect::<Vec<_>>());
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out = super::flash_attn::prefill_attn(&q, &k, &v, &meta, scale).unwrap();

        assert_eq!(out.shape().dims(), &[seq_len, num_heads, head_dim]);
        let out_f32 = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let has_nan = (0..out_f32.elem_count())
            .step_by(out_f32.elem_count() / 16 + 1)
            .try_fold(false, |acc, i| Ok::<_, candle_core::Error>(acc | out_f32.get(i)?.to_vec0::<f32>()?.is_nan()))
            .unwrap();
        assert!(!has_nan, "prefill output contains NaN");
    }

    #[test]
    fn flash_attn_decode_runs() {
        let dev = cuda_device();
        if !dev.is_cuda() {
            eprintln!("[gpu_tests] skipping — no CUDA device");
            return;
        }

        let num_heads = 4;
        let kv_heads = 4;
        let head_dim = 64;
        let ctx_len = 32;

        let cache = PagedKVCache::new(1, 1, 256, kv_heads, head_dim, DType::BF16, &dev).unwrap();

        let key = Tensor::randn(0f32, 1f32, (ctx_len, kv_heads, head_dim), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        let value = Tensor::randn(0f32, 1f32, (ctx_len, kv_heads, head_dim), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        let slots: Vec<i64> = (0..ctx_len as i64).collect();
        let slot_mapping = Tensor::from_vec(slots, (ctx_len,), &dev).unwrap();
        cache.reshape_and_cache(0, &key, &value, &slot_mapping).unwrap();

        let q = Tensor::randn(0f32, 1f32, (1, num_heads, head_dim), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        let k_cache = cache.k_cache(0).unwrap();
        let v_cache = cache.v_cache(0).unwrap();

        let meta = build_decode_metadata(&[ctx_len as u32], &[vec![0]], &[ctx_len as i64]);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out = super::flash_attn::decode_attn(&q, &k_cache, &v_cache, &meta, scale, 256).unwrap();

        assert_eq!(out.shape().dims(), &[1, num_heads, head_dim]);
        let out_f32 = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let first = out_f32.get(0).unwrap().to_vec0::<f32>().unwrap();
        assert!(!first.is_nan(), "decode output contains NaN");
    }
}
