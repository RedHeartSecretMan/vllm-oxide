//! `silu_and_mul` activation — SwiGLU's element-wise half (nano-vllm
//! `activation.py`).
//!
//! nano-vllm: `x, y = x.chunk(2, -1); return F.silu(x) * y`. The Rust port
//! uses candle's [`candle_nn::ops::silu`] for the SILU half and broadcasts
//! the multiplication against the second half along the last dim.
//!
//! Used in `Qwen3Mlp` as the activation between the fused gate-up projection
//! and the row-parallel down projection (nano-vllm `qwen3.py:147-153`):
//! `down_proj(silu_and_mul(gate_up_proj(x)))`.

use candle_core::{Result, Tensor, D::Minus1};

/// `silu(x[..., :d/2]) * x[..., d/2:]`. A stateless element-wise activation
/// exposed as a free function. Chunks the last dimension in two, applies
/// SiLU to the first half, then broadcasts-multiplies by the second half.
pub fn silu_and_mul(x: &Tensor) -> Result<Tensor> {
    let chunks = x.chunk(2, Minus1)?;
    let silu_gate = candle_nn::ops::silu(&chunks[0])?;
    silu_gate.broadcast_mul(&chunks[1])
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    mod shape {
        use super::*;

        #[test]
        fn output_last_dim_is_half_input_last_dim() {
            let dev = candle_core::Device::Cpu;
            let x = Tensor::zeros((2, 3, 8), DType::F32, &dev).unwrap();
            let out = silu_and_mul(&x).unwrap();
            assert_eq!(out.shape().dims(), [2, 3, 4]);
        }

        #[test]
        fn preserves_leading_dims() {
            let dev = candle_core::Device::Cpu;
            let x = Tensor::zeros((5, 10), DType::F32, &dev).unwrap();
            let out = silu_and_mul(&x).unwrap();
            assert_eq!(out.shape().dims(), [5, 5]);
        }
    }

    mod behaviour {
        use super::*;

        #[test]
        fn matches_silu_gate_times_up_manual_formula() {
            // gate = [1, 2]; up = [3, 4]; out = [silu(1)*3, silu(2)*4]
            let dev = candle_core::Device::Cpu;
            let x = Tensor::from_iter([1.0f32, 2.0, 3.0, 4.0], &dev).unwrap();
            let out = silu_and_mul(&x).unwrap();
            let v = out.to_vec1::<f32>().unwrap();
            let expected = [silu(1.0) * 3.0, silu(2.0) * 4.0];
            assert!((v[0] - expected[0]).abs() < 1e-6);
            assert!((v[1] - expected[1]).abs() < 1e-6);
        }

        #[test]
        fn zero_gate_zeros_output_regardless_of_up() {
            // gate = [0]; up = [99]; silu(0) = 0; out = 0 * 99 = 0
            let dev = candle_core::Device::Cpu;
            let x = Tensor::from_iter([0.0f32, 99.0], &dev).unwrap();
            let out = silu_and_mul(&x).unwrap();
            assert!(out.to_vec1::<f32>().unwrap()[0].abs() < 1e-7);
        }

        #[test]
        fn large_negative_gate_zeros_output() {
            // silu(-x) → 0 as x → +inf; gate = [-50] should be effectively 0.
            let dev = candle_core::Device::Cpu;
            let x = Tensor::from_iter([-50.0f32, 7.0], &dev).unwrap();
            let out = silu_and_mul(&x).unwrap();
            assert!(out.to_vec1::<f32>().unwrap()[0].abs() < 1e-6);
        }
    }
}
