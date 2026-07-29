#![allow(unsafe_code)]

use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::sys::CUstream;
use candle_core::cuda::cudarc::driver::{DevicePtr, DeviceSlice, SyncOnDrop};
use candle_core::{DType, Result, Storage, Tensor};
use core::ffi::{c_long, c_void};
use std::sync::RwLockReadGuard;

mod ffi {
    use super::CUstream;
    use core::ffi::{c_long, c_void};

    extern "C" {
        pub fn reshape_and_cache(
            key: *const c_void,
            value: *const c_void,
            key_cache: *mut c_void,
            value_cache: *mut c_void,
            slot_mapping: *const c_long,
            num_tokens: c_long,
            num_kv_heads: c_long,
            head_dim: c_long,
            dtype: i32,
            stream: CUstream,
        );

        pub fn copy_blocks(
            key_cache: *mut c_void,
            value_cache: *mut c_void,
            block_mapping: *const c_long,
            num_pairs: c_long,
            block_stride: c_long,
            dtype: i32,
            stream: CUstream,
        );
    }
}

struct TensorGuard {
    _guard: SyncOnDrop<'static>,
    #[allow(dead_code)]
    _storage: RwLockReadGuard<'static, Storage>,
}

macro_rules! extract_ptr {
    ($storage:expr, $layout:expr, $ty:ty) => {{
        let slice = $storage.as_cuda_slice::<$ty>()?;
        let slice = slice.slice($layout.start_offset()..);
        let stream = slice.stream();
        let (ptr, guard) = slice.device_ptr(stream);
        let guard: SyncOnDrop<'static> = unsafe { std::mem::transmute(guard) };
        (ptr, guard)
    }};
}

fn slice_ptr(tensor: &Tensor) -> Result<(u64, TensorGuard)> {
    let (storage, layout) = tensor.storage_and_layout();
    let cuda_storage = match &*storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("expected CUDA storage"),
    };

    let (ptr, guard): (u64, SyncOnDrop<'static>) = match tensor.dtype() {
        DType::BF16 => extract_ptr!(cuda_storage, layout, half::bf16),
        DType::F16 => extract_ptr!(cuda_storage, layout, half::f16),
        DType::F32 => extract_ptr!(cuda_storage, layout, f32),
        DType::I64 => extract_ptr!(cuda_storage, layout, i64),
        DType::U8 => extract_ptr!(cuda_storage, layout, u8),
        DType::U32 => extract_ptr!(cuda_storage, layout, u32),
        other => candle_core::bail!("unsupported tensor dtype for FFI: {other:?}"),
    };

    // SAFETY: SyncOnDrop borrows from the CudaSlice, which is owned by storage.
    // We keep the RwLockReadGuard alive in TensorGuard, so the device handle and
    // memory remain valid. Caller must drop TensorGuard before the source Tensor.
    let storage: RwLockReadGuard<'static, Storage> = unsafe { std::mem::transmute(storage) };

    Ok((
        ptr,
        TensorGuard {
            _guard: guard,
            _storage: storage,
        },
    ))
}

fn get_stream(tensor: &Tensor) -> Result<CUstream> {
    let (storage, _) = tensor.storage_and_layout();
    let cuda_storage = match &*storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("expected CUDA storage"),
    };
    Ok(cuda_storage.device().cuda_stream().cu_stream())
}

fn dtype_code(dt: DType) -> Result<i32> {
    match dt {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("unsupported KV cache dtype: {other:?}"),
    }
}

pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let dtype = dtype_code(key.dtype())?;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (key_ptr, _g1) = slice_ptr(key)?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (val_ptr, _g2) = slice_ptr(value)?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (kc_ptr, _g3) = slice_ptr(key_cache)?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (vc_ptr, _g4) = slice_ptr(value_cache)?;
    let (sm_ptr, _g5) = slice_ptr(slot_mapping)?;

    let stream = get_stream(key)?;

    let dims = key.dims();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let num_tokens = dims[0] as c_long;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let num_kv_heads = dims[1] as c_long;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let head_dim = dims[2] as c_long;

    unsafe {
        ffi::reshape_and_cache(
            key_ptr as *const c_void,
            val_ptr as *const c_void,
            kc_ptr as *mut c_void,
            vc_ptr as *mut c_void,
            sm_ptr as *const c_long,
            num_tokens,
            num_kv_heads,
            head_dim,
            dtype,
            stream,
        );
    }

    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn copy_blocks(key_cache: &Tensor, value_cache: &Tensor, block_mapping: &Tensor) -> Result<()> {
    let dtype = dtype_code(key_cache.dtype())?;

    let (kc_ptr, _g1) = slice_ptr(key_cache)?;
    let (vc_ptr, _g2) = slice_ptr(value_cache)?;
    let (bm_ptr, _g3) = slice_ptr(block_mapping)?;

    let stream = get_stream(key_cache)?;

    let num_pairs = block_mapping.dims()[0] as c_long;
    let kc_dims = key_cache.dims();
    let block_stride = (kc_dims[1] * kc_dims[2] * kc_dims[3]) as c_long;

    unsafe {
        ffi::copy_blocks(
            kc_ptr as *mut c_void,
            vc_ptr as *mut c_void,
            bm_ptr as *const c_long,
            num_pairs,
            block_stride,
            dtype,
            stream,
        );
    }

    Ok(())
}
