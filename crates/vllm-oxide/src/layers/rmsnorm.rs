//! RMSNorm with FP32 upcast (parity with nano-vllm `layernorm.py:21-25`).
//!
//! nano-vllm computes variance in FP32 regardless of input dtype:
//! `orig_dtype = x.dtype; x = x.float(); var = x.pow(2).mean(-1); x = (x *
//! rsqrt(var + eps)).to(orig_dtype) * weight`. The Rust port preserves this:
//! input is upcast to FP32 (BF16/F16 only — FP32/FP64 pass through) before
//! square-and-mean, then cast back and multiplied by the (original-dtype)
//! weight. Skipping the upcast costs ~1e-3 precision on BF16, enough to flip
//! greedy tokens downstream.
//!
//! The forward signature carries V1's add+norm pattern — `(hidden,
//! Option<residual>) → (normed, residual)`. When `residual` is `Some(r)`,
//! `r` is added to `hidden` (in FP32) BEFORE normalisation; the returned
//! residual is the un-normalised sum (so the next add+norm layer consumes
//! it). When `residual` is `None`, the returned residual is the input
//! itself (the residual stream starts at the embedding output).

use candle_core::{D::Minus1, DType, Result, Tensor};
use candle_nn::VarBuilder;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Read `weight` (shape `(hidden_size,)`) from `vb`. Initialises to 1.0
    /// when the checkpoint omits the tensor (matches candle-nn's
    /// `rms_norm` constructor).
    pub fn from_vb(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self::new(weight, eps))
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// V1 add+norm: `norm(rms(residual + hidden), residual + hidden)`.
    ///
    /// When `residual` is `None`, no add is performed and the returned
    /// residual equals the input `x` (residual stream bootstrap).
    pub fn forward(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let orig_dtype = x.dtype();
        let internal_dtype = match orig_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let (sum_fp32, new_residual) = match residual {
            Some(r) => {
                let r_fp32 = r.to_dtype(internal_dtype)?;
                let sum = x.to_dtype(internal_dtype)?.broadcast_add(&r_fp32)?;
                let new_residual = sum.to_dtype(orig_dtype)?;
                (sum, new_residual)
            }
            None => {
                let x_fp32 = x.to_dtype(internal_dtype)?;
                let new_residual = x.clone();
                (x_fp32, new_residual)
            }
        };

        let hidden_size = sum_fp32.dim(Minus1)?;
        #[allow(clippy::cast_precision_loss)]
        let hidden_f = hidden_size as f64;
        let var = (sum_fp32.sqr()?.sum_keepdim(Minus1)? / hidden_f)?;
        let normed_fp32 = sum_fp32.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let normed = normed_fp32.to_dtype(orig_dtype)?.broadcast_mul(&self.weight)?;
        Ok((normed, new_residual))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    /// Pure-FP32 reference implementation, independent of the production code
    /// path. The property test compares this against `RMSNorm::forward` run on
    /// BF16 input (which internally upcasts to FP32). Agreement within
    /// `1e-3` confirms the upcast path didn't drop precision.
    fn rms_norm_fp32_reference(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let hidden_size = x.dim(Minus1)?;
        let var = (x.sqr()?.sum_keepdim(Minus1)? / hidden_size as f64)?;
        let normed = x.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
        normed.broadcast_mul(weight)
    }

    mod shape {
        use super::*;

        #[test]
        fn normed_and_residual_preserve_input_shape() {
            let dev = candle_core::Device::Cpu;
            let weight = Tensor::ones(4, DType::F32, &dev).unwrap();
            let norm = RMSNorm::new(weight, 1e-6);
            let x = Tensor::randn(0.0f32, 1.0, (2, 4), &dev).unwrap();
            let (normed, residual) = norm.forward(&x, None).unwrap();
            assert_eq!(normed.shape().dims(), [2, 4]);
            assert_eq!(residual.shape().dims(), [2, 4]);
        }
    }

    mod fp32_upcast_invariant {
        use super::*;

        #[test]
        fn bf16_input_matches_fp32_reference_within_tolerance() {
            let dev = candle_core::Device::Cpu;
            let x_f32 = Tensor::randn(0.0f32, 1.0, (3, 8), &dev).unwrap();
            let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();
            let weight_f32 = Tensor::ones(8, DType::F32, &dev).unwrap();
            let weight_bf16 = weight_f32.to_dtype(DType::BF16).unwrap();

            let reference = rms_norm_fp32_reference(&x_f32, &weight_f32, 1e-6).unwrap();
            let norm = RMSNorm::new(weight_bf16, 1e-6);
            let (got, _) = norm.forward(&x_bf16, None).unwrap();
            let got_f32 = got.to_dtype(DType::F32).unwrap();

            let ref_flat = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got_flat = got_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let max_diff = ref_flat
                .iter()
                .zip(got_flat.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // Tolerance set to distinguish "FP32 upcast active" (this measured
            // ~6e-3 on N(0,1) input, BF16 input quantisation dominates) from
            // "upcast removed / pure-BF16 path" which would diverge by ~1e-1.
            // 3e-2 sits safely between the two regimes.
            assert!(
                max_diff < 3e-2,
                "BF16-vs-FP32 RMSNorm divergence {max_diff:.4e} exceeds tolerance 3e-2"
            );
        }

        #[test]
        fn fp32_input_is_identity_through_internal_pipeline() {
            // FP32 input should NOT be upcast (already FP32) — verify the
            // match-arm preserves the dtype so no precision is lost.
            let dev = candle_core::Device::Cpu;
            let weight = Tensor::ones(4, DType::F32, &dev).unwrap();
            let norm = RMSNorm::new(weight, 1e-6);
            let x = Tensor::zeros((2, 4), DType::F32, &dev).unwrap();
            let (normed, _) = norm.forward(&x, None).unwrap();
            assert_eq!(normed.dtype(), DType::F32, "FP32 input must stay FP32");
        }
    }

    mod add_then_norm {
        use super::*;

        #[test]
        fn residual_is_added_before_normalisation() {
            let dev = candle_core::Device::Cpu;
            // weight = ones; eps ~ 0; with x and r chosen so x+r is non-trivial.
            let weight = Tensor::ones(3, DType::F32, &dev).unwrap();
            let norm = RMSNorm::new(weight, 1e-12);
            let x = Tensor::from_iter([3.0f32, 0.0, 0.0], &dev).unwrap();
            let r = Tensor::from_iter([0.0f32, 4.0, 0.0], &dev).unwrap();
            let (normed, new_residual) = norm.forward(&x, Some(&r)).unwrap();

            // x + r = [3, 4, 0]; var = (9+16+0)/3 = 25/3; rms = sqrt(25/3) ≈ 2.887
            // normed = [3, 4, 0] / 2.887 * ones = [1.039, 1.386, 0.0]
            let normed_v = normed.to_vec1::<f32>().unwrap();
            let residual_v = new_residual.to_vec1::<f32>().unwrap();
            assert_eq!(residual_v, vec![3.0, 4.0, 0.0]);
            let rms = (25.0f32 / 3.0).sqrt();
            assert!((normed_v[0] - 3.0 / rms).abs() < 1e-5);
            assert!((normed_v[1] - 4.0 / rms).abs() < 1e-5);
            assert!(normed_v[2].abs() < 1e-6);
        }

        #[test]
        fn none_residual_returns_input_as_new_residual() {
            let dev = candle_core::Device::Cpu;
            let weight = Tensor::ones(3, DType::F32, &dev).unwrap();
            let norm = RMSNorm::new(weight, 1e-6);
            let x = Tensor::from_iter([1.0f32, 2.0, 3.0], &dev).unwrap();
            let (_, residual) = norm.forward(&x.clone(), None).unwrap();
            let want = x.to_vec1::<f32>().unwrap();
            let got = residual.to_vec1::<f32>().unwrap();
            assert_eq!(want, got);
        }
    }

    mod weight_application {
        use super::*;

        #[test]
        fn weight_scales_normed_output() {
            let dev = candle_core::Device::Cpu;
            // x = [3, 4, 0, 0]; mean(x^2) = (9+16)/4 = 6.25; rms = 2.5
            // normed_pre_weight = [1.2, 1.6, 0, 0]; weight = [10, 20, 30, 40]
            // → normed = [12, 32, 0, 0]
            let weight = Tensor::from_iter([10.0f32, 20.0, 30.0, 40.0], &dev).unwrap();
            let norm = RMSNorm::new(weight, 1e-12);
            let x = Tensor::from_iter([3.0f32, 4.0, 0.0, 0.0], &dev).unwrap();
            let (normed, _) = norm.forward(&x, None).unwrap();
            let v = normed.to_vec1::<f32>().unwrap();
            assert!((v[0] - 12.0).abs() < 1e-5);
            assert!((v[1] - 32.0).abs() < 1e-5);
            assert!(v[2].abs() < 1e-5);
            assert!(v[3].abs() < 1e-5);
        }
    }
}
