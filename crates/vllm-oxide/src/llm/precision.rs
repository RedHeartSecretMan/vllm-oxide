//! GEMM precision for each freshly created inference device.

use anyhow::{bail, Result};
use candle_core::Device;

// Call before sharing this fresh device with any model work. BF16/F16 storage
// and tensor-core multiplication are retained; intermediate reductions use FP32.
#[allow(unsafe_code)]
pub(super) fn configure_fp32_reduction(device: &Device) -> Result<()> {
    use candle_core::cuda::cudarc::cublas::sys;
    let Device::Cuda(device) = device else {
        bail!("FP32 reduction requires CUDA");
    };
    let handle = device.cublas_handle();
    // SAFETY: this live cuBLAS handle has not been shared with model execution.
    let status = unsafe {
        sys::cublasSetMathMode(
            *handle.handle(),
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        )
    };
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        bail!("configuring FP32 GEMM reduction failed: {status:?}");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use half::bf16;

    #[test]
    #[ignore = "requires a guarded CUDA owner"]
    #[allow(clippy::cast_precision_loss)] // Integer bounds below prove these casts exact.
    fn cuda_bf16_gemm_matches_exact_dot_products_across_batch_shapes() {
        const K: usize = 1024;
        const N: usize = 3072;
        let device = Device::new_cuda(0).unwrap();
        configure_fp32_reduction(&device).unwrap();
        let input: Vec<i32> = (0..2 * K)
            .map(|i| i32::try_from(i % 113).unwrap() - 56)
            .collect();
        let weights: Vec<i32> = (0..N * K)
            .map(|i| i32::try_from((i * 37) % 257).unwrap() - 128)
            .collect();
        // Inputs use units of 1/64 and 1/128. Every product and any partial
        // sum have integer numerators below 2^24, so FP32 can accumulate them
        // exactly in any order. Round only the final integer dot product to
        // BF16; intermediate BF16 truncation is observable on this fixture.
        let expected: Vec<bf16> = (0..2 * N)
            .map(|i| {
                let row = i / N;
                let column = i % N;
                let sum: i32 = (0..K)
                    .map(|k| input[row * K + k] * weights[column * K + k])
                    .sum();
                bf16::from_f32(sum as f32 / 8192.0)
            })
            .collect();
        let weight_values: Vec<bf16> = weights
            .iter()
            .map(|&v| bf16::from_f32(v as f32 / 128.0))
            .collect();
        let weight = Tensor::from_vec(weight_values, (N, K), &device).unwrap();
        for rows in [2, 771] {
            let values: Vec<bf16> = (0..rows * K)
                .map(|i| bf16::from_f32(input[i % (2 * K)] as f32 / 64.0))
                .collect();
            let x = Tensor::from_vec(values, (rows, K), &device).unwrap();
            let actual = x
                .matmul(&weight.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<bf16>()
                .unwrap();
            for (i, value) in actual.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    expected[i % (2 * N)].to_bits(),
                    "GEMM rows={rows}, row={}, column={}",
                    i / N,
                    i % N
                );
            }
        }
    }
}
