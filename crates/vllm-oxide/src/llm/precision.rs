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

    #[test]
    #[ignore = "requires a guarded CUDA owner"]
    fn cuda_generation_projection_layout_preserves_small_batch_rounding() {
        use crate::layers::linear::{Linear, ProjectionLayout};
        use crate::layers::parallel::Row;

        const K: usize = 3072;
        const N: usize = 1024;
        let decode = |bytes: &[u8]| {
            assert_eq!(bytes.len(), 2 * K);
            bytes
                .chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
                .collect::<Vec<_>>()
        };
        let input = decode(include_bytes!("test_data/projection-input.bf16"));
        let column = decode(include_bytes!("test_data/projection-weight.bf16"));
        let device = Device::new_cuda(0).unwrap();
        configure_fp32_reduction(&device).unwrap();
        let weight = Tensor::from_vec(column.repeat(N), (N, K), &device).unwrap();
        let make_input = |rows| {
            let mut values = vec![bf16::ZERO; rows * K];
            values[..K].copy_from_slice(&input);
            Tensor::from_vec(values, (rows, K), &device).unwrap()
        };
        let bits = |x: &Tensor| x.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
        let reference = make_input(771).matmul(&weight.t().unwrap()).unwrap();
        let small = make_input(2).matmul(&weight.t().unwrap()).unwrap();
        assert_ne!(
            bits(&small),
            bits(&reference.narrow(0, 0, 2).unwrap()),
            "fixture must expose shape-dependent rounding on the validated GPU"
        );
        let linear = Linear::<Row>::from_weight(weight.clone());
        for rows in [1, 2] {
            let x = make_input(rows);
            let original = x.matmul(&weight.t().unwrap()).unwrap();
            assert_eq!(
                bits(&linear.forward(&x, ProjectionLayout::default()).unwrap()),
                bits(&original)
            );
            for count in [3, 8, usize::MAX] {
                let actual = linear
                    .forward(&x, ProjectionLayout::for_batch(count))
                    .unwrap();
                assert_eq!(actual.dims(), [rows, N]);
                assert_eq!(bits(&actual), bits(&reference.narrow(0, 0, rows).unwrap()));
            }
        }
        // Packed checkpoint storage must use the same geometry for an output
        // range and add the corresponding bias only after real rows are kept.
        let bias = Tensor::full(bf16::from_f32(0.125), (2 * N,), &device).unwrap();
        let packed = Linear::<Row>::from_parts(
            Tensor::cat(&[&weight, &weight], 0).unwrap(),
            Some(bias.clone()),
        );
        let actual = packed
            .forward_range(&make_input(2), N, N, ProjectionLayout::for_batch(3))
            .unwrap();
        let expected = reference
            .narrow(0, 0, 2)
            .unwrap()
            .broadcast_add(&bias.narrow(0, N, N).unwrap())
            .unwrap();
        assert_eq!(actual.dims(), [2, N]);
        assert_eq!(bits(&actual), bits(&expected));
    }
    #[test]
    #[ignore = "requires a guarded CUDA owner"]
    fn cuda_batched_workspace_preserves_reference_and_restores_after_error() {
        use crate::layers::linear::{Linear, ProjectionLayout};
        use crate::layers::parallel::Row;
        const K: usize = 3072;
        const N: usize = 1024;
        let decode = |bytes: &[u8]| {
            assert_eq!(bytes.len(), 2 * K);
            bytes
                .chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
                .collect::<Vec<_>>()
        };
        let input = decode(include_bytes!("test_data/workspace-input.bf16"));
        let column = decode(include_bytes!("test_data/workspace-weight.bf16"));
        let device = Device::new_cuda(0).unwrap();
        configure_fp32_reduction(&device).unwrap();
        let mut weights = vec![bf16::ZERO; N * K];
        weights[50 * K..51 * K].copy_from_slice(&column);
        let weight = Tensor::from_vec(weights, (N, K), &device).unwrap();
        let make_input = |rows| {
            let mut values = vec![bf16::ZERO; rows * K];
            values[..K].copy_from_slice(&input);
            Tensor::from_vec(values, (rows, K), &device).unwrap()
        };
        let value = |output: Tensor| {
            output
                .narrow(0, 0, 1)
                .unwrap()
                .narrow(1, 50, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<bf16>()
                .unwrap()[0]
                .to_bits()
        };
        let x = make_input(258);
        let transposed = weight.t().unwrap();
        assert_eq!(value(x.matmul(&transposed).unwrap()), 0xb954);
        assert_eq!(value(make_input(514).matmul(&transposed).unwrap()), 0xb955);
        let linear = Linear::<Row>::from_parts(weight, None);
        assert_eq!(
            value(linear.forward(&x, ProjectionLayout::default()).unwrap()),
            0xb954
        );
        assert_eq!(
            value(
                linear
                    .forward(&make_input(2), ProjectionLayout::for_batch(2))
                    .unwrap()
            ),
            0xb954
        );
        assert_eq!(
            value(linear.forward(&x, ProjectionLayout::for_batch(2)).unwrap()),
            0xb955
        );
        // A following unconfigured GEMM must regain its original reduction path.
        assert_eq!(value(x.matmul(&transposed).unwrap()), 0xb954);
        let invalid = Linear::<Row>::from_parts(
            Tensor::zeros((N, K - 1), candle_core::DType::BF16, &device).unwrap(),
            None,
        );
        assert!(invalid.forward(&x, ProjectionLayout::for_batch(2)).is_err());
        assert_eq!(value(x.matmul(&transposed).unwrap()), 0xb954);
    }
}
