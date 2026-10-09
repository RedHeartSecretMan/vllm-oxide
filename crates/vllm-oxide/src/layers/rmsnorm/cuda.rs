//! Ordered F32 statistics for the pinned Qwen3 normalization widths.
//!
//! CUDA allocation and kernel submission require unsafe code. Contiguous F32
//! storage, validated dimensions and live stream guards bound all pointer use.
#![allow(unsafe_code)]

use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor, D::Minus1};

enum Statistics {
    Mean,
    InverseSqrt,
}

extern "C" {
    fn rms_mean_f32(
        input: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn rms_rsqrt_f32(
        input: *const f32,
        output: *mut f32,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

impl CustomOp1 for Statistics {
    fn name(&self) -> &'static str {
        match self {
            Self::Mean => "rms-ordered-mean",
            Self::InverseSqrt => "rms-rsqrt",
        }
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA RMS statistics require CUDA storage")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::msg("CUDA RMS statistics require contiguous input")
        })?;
        let count = layout.shape().elem_count();
        if count == 0 {
            candle_core::bail!("CUDA RMS statistics require nonempty input")
        }
        let mut shape = layout.dims().to_vec();
        let columns = *shape
            .last()
            .ok_or_else(|| candle_core::Error::msg("scalar RMS input"))?;
        let output_count = match self {
            Self::Mean => {
                if !matches!(columns, 128 | 1024) {
                    candle_core::bail!("ordered RMS mean requires width 128 or 1024")
                }
                let last = shape.len() - 1;
                shape[last] = 1;
                count / columns
            }
            Self::InverseSqrt => count,
        };
        let device = storage.device();
        let source = storage.as_cuda_slice::<f32>()?.slice(start..end);
        // SAFETY: each kernel writes every element of its validated output.
        let mut output = unsafe { device.alloc::<f32>(output_count)? };
        let stream = device.cuda_stream();
        let status = {
            let (src, _input_guard) = source.device_ptr(&stream);
            let (dst, _output_guard) = output.device_ptr_mut(&stream);
            // SAFETY: both allocations cover their declared F32 counts. The
            // owning storage and guards survive submission on this stream.
            unsafe {
                match self {
                    Self::Mean => rms_mean_f32(
                        src as *const f32,
                        dst as *mut f32,
                        output_count,
                        columns,
                        stream.cu_stream().cast(),
                    ),
                    Self::InverseSqrt => rms_rsqrt_f32(
                        src as *const f32,
                        dst as *mut f32,
                        count,
                        stream.cu_stream().cast(),
                    ),
                }
            }
        };
        if status != 0 {
            candle_core::bail!("{} failed with CUDA status {status}", self.name())
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from(shape),
        ))
    }
}

pub(super) fn normalize(input: &Tensor, epsilon: f64) -> Result<Tensor> {
    let width = input.dim(Minus1)?;
    let squares = input.sqr()?;
    let variance = if matches!(width, 128 | 1024) {
        squares.apply_op1_no_bwd(&Statistics::Mean)?
    } else {
        let divisor = u32::try_from(width).map_err(candle_core::Error::wrap)?;
        if divisor == 0 {
            candle_core::bail!("RMSNorm width must be positive")
        }
        (squares.sum_keepdim(Minus1)? / f64::from(divisor))?
    };
    let inverse = (variance + epsilon)?.apply_op1_no_bwd(&Statistics::InverseSqrt)?;
    input.broadcast_mul(&inverse)
}
