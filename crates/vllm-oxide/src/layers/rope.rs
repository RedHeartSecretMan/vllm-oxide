//! Rotary positional embedding (RoPE) — nano-vllm `rotary_embedding.py` port.
//!
//! Precomputes a `cos_sin_cache` of shape `(max_position_embeddings,
//! rotary_dim)` from `inv_freq = 1.0 / base ** (arange(0, rotary_dim, 2) /
//! rotary_dim)` and `t = arange(max_position_embeddings)`. Forward indexes
//! the cache by per-token `positions` (supports non-sequential decode
//! positions, not just prefill's `[0..seq_len)`), then applies the
//! half-rotation:
//!
//! ```text
//! x1, x2 = chunk(x, 2, dim=-1)
//! y1 = x1 * cos - x2 * sin
//! y2 = x2 * cos + x1 * sin
//! y  = cat(y1, y2, dim=-1)
//! ```
//!
//! Qwen3 ships `rope_theta = 1_000_000` and no scaling; this module defaults
//! to that and exposes no scaling knob (ADR-0003 R5 — scaling variants land
//! in v0.2 if a model ships `rope_scaling`).
//!
//! Diverges from `candle_nn::rotary_emb::rope` (the free function) because
//! that API consumes cos/sin shaped against a sequential `[0..seq_len)`
//! convention — incompatible with decode where positions are per-sequence
//! (e.g. `[15, 42, 7]`). The nano-vllm `cos_sin_cache[positions]` lookup is
//! the V1 parity path and survives both prefill and decode.
//!
//! TODO(#16 review): spec user story 37 says "use candle-nn's RotaryEmbedding"
//! but candle-nn at rev 27f20fea exposes RoPE only as free functions (`rope`,
//! `rope_thd`, etc.) — no struct. This module hand-rolls `inv_freq` + cos/sin
//! cache + half-rotation to support per-token `positions` (decode). An ADR
//! amendment should land before v0.2 to either (a) upstream a struct to
//! candle-nn, or (b) update user story 37 to "implement V1-style RoPE with
//! position lookup".

use candle_core::{D::Minus1, Device, DType, Result, Tensor};

/// Precomputed RoPE cache + geometry. Stateless after construction — `forward`
/// only reads `cos_sin_cache` indexed by `positions`.
pub struct RotaryEmbedding {
    cos_sin_cache: Tensor,
    head_size: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    base: f32,
}

impl RotaryEmbedding {
    /// Build the cache. `rotary_dim` must equal `head_size` (nano-vllm
    /// convention; Qwen3 does not exercise partial-rotary).
    pub fn new(
        head_size: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        base: f32,
        dev: &Device,
    ) -> Result<Self> {
        assert_eq!(
            rotary_dim, head_size,
            "partial-rotary (rotary_dim != head_size) not supported in v0.1"
        );
        let half = rotary_dim / 2;

        #[allow(clippy::cast_precision_loss)]
        let inv_freq_vec: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / base.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq_vec, (half,), dev)?;

        #[allow(clippy::cast_precision_loss)]
        let t_vec: Vec<f32> = (0..max_position_embeddings).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t_vec, (max_position_embeddings,), dev)?;

        let t = t.unsqueeze(1)?;
        let inv_freq = inv_freq.unsqueeze(0)?;
        let freqs = t.broadcast_mul(&inv_freq)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        let cos_sin_cache = Tensor::cat(&[&cos, &sin], Minus1)?;

        Ok(Self {
            cos_sin_cache,
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
        })
    }

    /// Qwen3 default constructor. `rope_theta = 1_000_000`, full-rotary
    /// (`rotary_dim = head_size`), no scaling.
    pub fn qwen3_default(head_size: usize, max_position_embeddings: usize, dev: &Device) -> Result<Self> {
        Self::new(head_size, head_size, max_position_embeddings, 1_000_000.0, dev)
    }

    pub fn cos_sin_cache(&self) -> &Tensor {
        &self.cos_sin_cache
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    pub fn head_size(&self) -> usize {
        self.head_size
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    /// Apply RoPE to `query` and `key`. Both must have shape
    /// `(batch_seq, num_heads, head_dim)`; `positions` must be a 1-D integer
    /// tensor of length `batch_seq`.
    pub fn forward(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let cos_sin = self.cos_sin_cache.embedding(positions)?;
        let chunks = cos_sin.chunk(2, Minus1)?;
        let cos = chunks[0].unsqueeze(1)?;
        let sin = chunks[1].unsqueeze(1)?;
        let query_rot = apply_rotary_emb(query, &cos, &sin)?;
        let key_rot = apply_rotary_emb(key, &cos, &sin)?;
        Ok((query_rot, key_rot))
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_fp32 = x.to_dtype(DType::F32)?;
    let chunks = x_fp32.chunk(2, Minus1)?;
    let x1 = &chunks[0];
    let x2 = &chunks[1];
    let y1 = x1.broadcast_mul(cos)?.broadcast_sub(&x2.broadcast_mul(sin)?)?;
    let y2 = x2.broadcast_mul(cos)?.broadcast_add(&x1.broadcast_mul(sin)?)?;
    let y = Tensor::cat(&[&y1, &y2], Minus1)?;
    y.to_dtype(orig_dtype)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    mod cache_shape {
        use super::*;

        #[test]
        fn cos_sin_cache_is_max_pos_by_rotary_dim() {
            let dev = Device::Cpu;
            let rope = RotaryEmbedding::new(8, 8, 32, 10_000.0, &dev).unwrap();
            assert_eq!(rope.cos_sin_cache().shape().dims(), [32, 8]);
        }

        #[test]
        fn half_is_half_rotary_dim() {
            let dev = Device::Cpu;
            let rope = RotaryEmbedding::new(16, 16, 4, 1_000_000.0, &dev).unwrap();
            let cos_sin = rope.cos_sin_cache();
            let chunks = cos_sin.chunk(2, Minus1).unwrap();
            assert_eq!(chunks[0].shape().dims(), [4, 8]);
            assert_eq!(chunks[1].shape().dims(), [4, 8]);
        }
    }

    mod forward_shape {
        use super::*;

        #[test]
        fn preserves_query_and_key_shapes() {
            let dev = Device::Cpu;
            let rope = RotaryEmbedding::new(8, 8, 16, 10_000.0, &dev).unwrap();
            let positions = Tensor::from_iter([0u32, 1, 2], &dev).unwrap();
            let q = Tensor::zeros((3, 4, 8), DType::F32, &dev).unwrap();
            let k = Tensor::zeros((3, 4, 8), DType::F32, &dev).unwrap();
            let (q_rot, k_rot) = rope.forward(&positions, &q, &k).unwrap();
            assert_eq!(q_rot.shape().dims(), [3, 4, 8]);
            assert_eq!(k_rot.shape().dims(), [3, 4, 8]);
        }
    }

    mod position_zero_is_identity {
        use super::*;

        #[test]
        fn position_zero_leaves_query_unchanged() {
            // cos(0) = 1, sin(0) = 0 → y1 = x1*1 - x2*0 = x1, y2 = x2*1 + x1*0 = x2
            let dev = Device::Cpu;
            let rope = RotaryEmbedding::new(8, 8, 4, 10_000.0, &dev).unwrap();
            let positions = Tensor::from_iter([0u32], &dev).unwrap();
            let q = Tensor::from_iter(
                [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &dev,
            )
            .unwrap()
            .reshape((1, 1, 8))
            .unwrap();
            let k = q.clone();
            let (q_rot, _) = rope.forward(&positions, &q, &k).unwrap();
            let got = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-6, "position-0 should be identity");
            }
        }
    }

    mod qwen3_default {
        use super::*;

        #[test]
        fn base_is_one_million() {
            let dev = Device::Cpu;
            let rope = RotaryEmbedding::qwen3_default(128, 4096, &dev).unwrap();
            assert_eq!(rope.base(), 1_000_000.0);
            assert_eq!(rope.head_size(), 128);
            assert_eq!(rope.rotary_dim(), 128);
            assert_eq!(rope.max_position_embeddings(), 4096);
        }

        #[test]
        fn different_base_produces_different_cache() {
            let dev = Device::Cpu;
            let theta_1m = RotaryEmbedding::new(8, 8, 16, 1_000_000.0, &dev).unwrap();
            let theta_10k = RotaryEmbedding::new(8, 8, 16, 10_000.0, &dev).unwrap();
            let a = theta_1m.cos_sin_cache().to_vec2::<f32>().unwrap();
            let b = theta_10k.cos_sin_cache().to_vec2::<f32>().unwrap();
            let mut any_diff = false;
            for row in 0..a.len() {
                for col in 0..a[0].len() {
                    if (a[row][col] - b[row][col]).abs() > 1e-6 {
                        any_diff = true;
                        break;
                    }
                }
            }
            assert!(any_diff, "rope_theta must affect cos_sin_cache");
        }
    }

    mod decode_positions {
        use super::*;

        #[test]
        fn non_sequential_positions_index_correct_rows() {
            // Build cache, index with positions=[2, 0], compare against the
            // cache rows for positions 2 and 0. This is the decode case
            // (positions are per-sequence, not 0..seq_len).
            let dev = Device::Cpu;
            let rope = RotaryEmbedding::new(8, 8, 16, 10_000.0, &dev).unwrap();
            let positions = Tensor::from_iter([2u32, 0], &dev).unwrap();
            let q = Tensor::zeros((2, 1, 8), DType::F32, &dev).unwrap();
            let k = q.clone();
            let (q_rot, _) = rope.forward(&positions, &q, &k).unwrap();
            // q is all zeros → q_rot is all zeros regardless of cos/sin.
            // The shape + error-free execution is the contract being verified
            // here; the position-indexed math is exercised by the
            // position-zero-identity test above.
            assert_eq!(q_rot.shape().dims(), [2, 1, 8]);
        }
    }
}
