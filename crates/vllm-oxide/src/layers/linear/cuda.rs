//! Scoped cuBLAS workspace control for the private, synchronous inference path.
//!
//! The unsafe boundary changes handle configuration only. Each LLM creates its
//! own device/handle, and public generation requires exclusive access to the LLM.
//! No user buffer is installed; restoring the same stream returns to cuBLAS's
//! default workspace pool. Restoration is attempted on errors and unwinding.
#![allow(unsafe_code)]

use candle_core::cuda::cudarc::cublas::sys;
use candle_core::{CudaDevice, Result};

struct DefaultWorkspace<'a> {
    device: &'a CudaDevice,
    restored: bool,
}

impl DefaultWorkspace<'_> {
    fn restore(&mut self) -> Result<()> {
        let blas = self.device.cublas_handle();
        // SAFETY: the owning device keeps the handle and its unchanged stream alive.
        // cuBLAS documents that SetStream unconditionally restores its default pool.
        let status = unsafe {
            sys::cublasSetStream_v2(*blas.handle(), self.device.cuda_stream().cu_stream().cast())
        };
        if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            candle_core::bail!("restoring cuBLAS workspace failed: {status:?}");
        }
        self.restored = true;
        Ok(())
    }
}

impl Drop for DefaultWorkspace<'_> {
    fn drop(&mut self) {
        if !self.restored {
            // Normal return propagates restoration errors; unwinding must not panic.
            let _ = self.restore();
        }
    }
}

pub(super) fn without_workspace<T>(
    device: &CudaDevice,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let blas = device.cublas_handle();
    // SAFETY: this private synchronous model owns the live handle. A zero-length
    // workspace accepts a null pointer and retains no asynchronous user buffer.
    let status = unsafe { sys::cublasSetWorkspace_v2(*blas.handle(), std::ptr::null_mut(), 0) };
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        candle_core::bail!("selecting cuBLAS workspace failed: {status:?}");
    }
    let mut workspace = DefaultWorkspace {
        device,
        restored: false,
    };
    let result = operation();
    match (result, workspace.restore()) {
        (Err(operation), Err(restoration)) => candle_core::bail!(
            "GEMM failed: {operation}; workspace restoration also failed: {restoration}"
        ),
        (_, Err(restoration)) => Err(restoration),
        (result, Ok(())) => result,
    }
}
