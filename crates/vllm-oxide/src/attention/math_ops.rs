//! CUDA-only mask/softmax and paged gather operations.
//!
//! The unsafe boundary is native kernel submission. Tensor owners, storage
//! locks and stream guards remain live through each launch; validated shapes
//! and contiguous offsets bound every pointer access.
#![allow(unsafe_code)]

use super::PreparedAttention;
use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, DType, Device, Layout, Result, Shape, Storage, Tensor,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Mask {
    pub queries: usize,
    pub keys: usize,
    pub heads: usize,
    pub query_start: usize,
    pub chunk_rows: usize,
    pub member_offset: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GatherShape {
    batch: usize,
    keys: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    table_width: usize,
    num_blocks: usize,
    member_offset: usize,
}
extern "C" {
    fn attention_rows_f32(
        input: *const f32,
        output: *mut f32,
        cu_q: *const u32,
        cu_k: *const u32,
        rows: usize,
        mask: Mask,
        operation: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn attention_gather(
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        cu_k: *const u32,
        table: *const u32,
        shape: GatherShape,
        dtype: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}
fn offsets(layout: &Layout) -> Result<std::ops::Range<usize>> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::msg("attention kernel requires contiguous storage"))?;
    Ok(start..end)
}
fn cuda(storage: &Storage) -> Result<&CudaStorage> {
    match storage {
        Storage::Cuda(storage) => Ok(storage),
        _ => candle_core::bail!("attention requires CUDA metadata"),
    }
}
fn launch_count(shape: &[usize]) -> Result<usize> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| candle_core::Error::msg("attention shape overflows"))?;
    if count == 0 || count.div_ceil(256) > i32::MAX as usize {
        candle_core::bail!("attention shape exceeds CUDA launch capacity")
    }
    Ok(count)
}
fn on_device(tensor: &Tensor, storage: &CudaStorage) -> Result<()> {
    if !tensor
        .device()
        .same_device(&Device::Cuda(storage.device().clone()))
    {
        candle_core::bail!("attention metadata and input must use the same CUDA device")
    }
    Ok(())
}
struct Rows {
    q: Tensor,
    k: Tensor,
    mask: Mask,
    operation: i32,
}
impl CustomOp1 for Rows {
    fn name(&self) -> &'static str {
        "attention-rows"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("attention rows require CUDA")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        on_device(&self.q, storage)?;
        on_device(&self.k, storage)?;
        let count = launch_count(layout.dims())?;
        let source = storage.as_cuda_slice::<f32>()?.slice(offsets(layout)?);
        let (qs, ql) = self.q.storage_and_layout();
        let (ks, kl) = self.k.storage_and_layout();
        let q = cuda(&qs)?.as_cuda_slice::<u32>()?.slice(offsets(ql)?);
        let k = cuda(&ks)?.as_cuda_slice::<u32>()?.slice(offsets(kl)?);
        let device = storage.device();
        // SAFETY: the native operation writes all count output elements.
        let mut output = unsafe { device.alloc::<f32>(count)? };
        let stream = device.cuda_stream();
        let status = {
            let (src, _src_guard) = source.device_ptr(&stream);
            let (qp, _q_guard) = q.device_ptr(&stream);
            let (kp, _k_guard) = k.device_ptr(&stream);
            let (dst, _dst_guard) = output.device_ptr_mut(&stream);
            // SAFETY: softmax() validates member/row extents; all contiguous
            // allocations and their stream guards remain alive for submission.
            unsafe {
                attention_rows_f32(
                    src as *const f32,
                    dst as *mut f32,
                    qp as *const u32,
                    kp as *const u32,
                    count / self.mask.keys,
                    self.mask,
                    self.operation,
                    stream.cu_stream().cast(),
                )
            }
        };
        if status != 0 {
            candle_core::bail!("attention rows CUDA status {status}")
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }
}
pub(super) fn softmax(scores: &Tensor, prepared: &PreparedAttention, mask: Mask) -> Result<Tensor> {
    let (batch_heads, rows, keys) = scores.dims3()?;
    let logical = prepared.logical();
    if mask.heads == 0
        || !batch_heads.is_multiple_of(mask.heads)
        || rows != mask.chunk_rows
        || keys != mask.keys
        || keys < mask.queries
        || mask
            .query_start
            .checked_add(rows)
            .is_none_or(|end| end > mask.queries)
        || mask
            .member_offset
            .checked_add(batch_heads / mask.heads)
            .is_none_or(|end| end >= logical.cu_seqlens_q.len())
    {
        candle_core::bail!("invalid attention mask extent")
    }
    for member in mask.member_offset..mask.member_offset + batch_heads / mask.heads {
        if (logical.cu_seqlens_q[member + 1] - logical.cu_seqlens_q[member]) as usize > mask.queries
            || (logical.cu_seqlens_k[member + 1] - logical.cu_seqlens_k[member]) as usize > keys
        {
            candle_core::bail!("attention mask is smaller than logical lengths")
        }
    }
    let op = |operation| Rows {
        q: prepared.cu_seqlens_q().clone(),
        k: prepared.cu_seqlens_k().clone(),
        mask,
        operation,
    };
    if keys <= 1024 {
        scores.apply_op1_no_bwd(&op(0))
    } else {
        let masked = scores.apply_op1_no_bwd(&op(1))?;
        candle_nn::ops::softmax(&masked, 2)?.apply_op1_no_bwd(&op(2))
    }
}
struct Gather {
    k: Tensor,
    table: Tensor,
    shape: GatherShape,
}
impl CustomOp1 for Gather {
    fn name(&self) -> &'static str {
        "attention-cache-gather"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("attention gather requires CUDA")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        on_device(&self.k, storage)?;
        on_device(&self.table, storage)?;
        let shape = Shape::from((
            self.shape.batch,
            self.shape.keys,
            self.shape.kv_heads,
            self.shape.head_dim,
        ));
        let count = launch_count(shape.dims())?;
        let (ks, kl) = self.k.storage_and_layout();
        let (ts, tl) = self.table.storage_and_layout();
        let k = cuda(&ks)?.as_cuda_slice::<u32>()?.slice(offsets(kl)?);
        let table = cuda(&ts)?.as_cuda_slice::<u32>()?.slice(offsets(tl)?);
        let device = storage.device();
        let stream = device.cuda_stream();
        macro_rules! gather {
            ($ty:ty, $dtype:expr) => {{
                let source = storage.as_cuda_slice::<$ty>()?.slice(offsets(layout)?);
                // SAFETY: the gather writes every validated output element.
                let mut output = unsafe { device.alloc::<$ty>(count)? };
                let status = {
                    let (src, _src_guard) = source.device_ptr(&stream);
                    let (kp, _k_guard) = k.device_ptr(&stream);
                    let (tp, _table_guard) = table.device_ptr(&stream);
                    let (dst, _dst_guard) = output.device_ptr_mut(&stream);
                    // SAFETY: gather_cache validates physical block coverage
                    // against this cache; storage and guards outlive submission.
                    unsafe {
                        attention_gather(
                            src as *const std::ffi::c_void,
                            dst as *mut std::ffi::c_void,
                            kp as *const u32,
                            tp as *const u32,
                            self.shape,
                            $dtype,
                            stream.cu_stream().cast(),
                        )
                    }
                };
                if status != 0 {
                    candle_core::bail!("attention gather CUDA status {status}")
                }
                CudaStorage::wrap_cuda_slice(output, device.clone())
            }};
        }
        let output = match storage.dtype() {
            DType::F16 => gather!(half::f16, 0),
            DType::BF16 => gather!(half::bf16, 1),
            DType::F32 => gather!(f32, 2),
            _ => candle_core::bail!("attention cache requires floating point storage"),
        };
        Ok((output, shape))
    }
}
pub(super) fn gather_cache(cache: &Tensor, prepared: &PreparedAttention) -> Result<Tensor> {
    gather(cache, prepared, None)
}

pub(super) fn gather_member(
    cache: &Tensor,
    prepared: &PreparedAttention,
    member: usize,
) -> Result<Tensor> {
    gather(cache, prepared, Some(member))?.squeeze(0)
}

fn gather(cache: &Tensor, prepared: &PreparedAttention, member: Option<usize>) -> Result<Tensor> {
    let (num_blocks, block_size, kv_heads, head_dim) = cache.dims4()?;
    prepared.validate_cache(cache.dims())?;
    let (table_batch, table_width) = prepared.block_table()?.dims2()?;
    let (batch, keys, member_offset) = match member {
        Some(member) if member < table_batch => {
            let lengths = &prepared.logical().cu_seqlens_k;
            (1, (lengths[member + 1] - lengths[member]) as usize, member)
        }
        Some(_) => candle_core::bail!("paged attention member exceeds batch"),
        None => (table_batch, prepared.logical().max_seqlen_k, 0),
    };
    let shape = GatherShape {
        batch,
        keys,
        kv_heads,
        head_dim,
        block_size,
        table_width,
        num_blocks,
        member_offset,
    };
    cache.apply_op1_no_bwd(&Gather {
        k: prepared.cu_seqlens_k().clone(),
        table: prepared.block_table()?.clone(),
        shape,
    })
}
