//! Throwaway prototype — validates T3 "Stack A" assumption (a):
//! the candle Tensor → cudarc CudaSlice → extern "C" kernel seam.
//!
//! Pattern lifted verbatim from mistral.rs `mistralrs-paged-attn`
//! (commit e124932, `src/cuda/backend/paged_attention.rs` + `mod.rs`).
//!
//! Success criterion: allocate a candle CUDA tensor, hand its raw device
//! pointer + CUstream to a hand-written `.cu` kernel, read back, confirm
//! the kernel ran. If this works, T3's riskiest assumption holds.

use candle_core as candle;
use candle::backend::BackendStorage; // .as_cuda_slice(), .device()
use candle::cuda_backend::cudarc::driver::DevicePtr; // .device_ptr()
use candle_core::cuda::cudarc::driver::DeviceSlice; // .slice(range)
use candle_core::cuda::cudarc::driver::sys::CUstream;
use candle_core::{Device, Result, Storage, Tensor};
use core::ffi::c_void;

extern "C" {
    pub fn fill_const_f32_launch(ptr: *mut c_void, n: i64, val: f32, stream: CUstream);
}

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    println!("[proto] device: {:?}", dev);

    // Allocate a 1024-element f32 tensor on GPU, zero-initialized.
    let n: usize = 1024;
    let t = Tensor::zeros(n, candle::DType::F32, &dev)?;

    // --- The mistral.rs unwrap chain (verbatim shape) ---
    // storage_and_layout() -> (Arc<Storage>, Layout)
    let (storage_arc, layout) = t.storage_and_layout();
    let cuda_storage = match &*storage_arc {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("expected CUDA storage"),
    };
    // as_cuda_slice::<T>() -> &CudaSlice<T>  (needs BackendStorage trait)
    let slice = cuda_storage.as_cuda_slice::<f32>()?;
    // .slice(range) applies the layout offset (needs DeviceSlice trait)
    let slice = slice.slice(layout.start_offset()..);

    // dev handle for stream extraction
    let cuda_dev = cuda_storage.device();

    // --- Pointer + stream extraction (mirrors `slice_ptr` in mod.rs) ---
    // device_ptr(stream) -> (u64, SyncOnDrop). The guard MUST outlive the kernel call.
    let (ptr, _guard) = slice.device_ptr(slice.stream());
    let stream = cuda_dev.cuda_stream().cu_stream();

    // --- The extern "C" call ---
    let val: f32 = 42.0;
    unsafe {
        fill_const_f32_launch(ptr as *mut c_void, n as i64, val, stream);
    }
    // _guard dropped here (after the call) — correct ordering.

    // --- Verify: read back via candle ---
    let sum = t.sum_all()?.to_vec0::<f32>()?;
    let expected = val * n as f32;
    let max = t.max(0)?.to_vec0::<f32>()?;
    let min = t.min(0)?.to_vec0::<f32>()?;
    println!("[proto] n={}, val={}", n, val);
    println!("[proto] sum  = {}  (expected {})", sum, expected);
    println!("[proto] min  = {}, max = {}", min, max);

    let ok = (sum - expected).abs() < 1e-3 && (max - val).abs() < 1e-6 && (min - val).abs() < 1e-6;
    if !ok {
        println!("[proto][A] FAIL — values don't match. Kernel did not write expected data.");
        std::process::exit(1);
    }
    println!("[proto][A] PASS — cudarc seam works; kernel ran on GPU through candle Tensor.");

    phase_b_paged_attn(&dev)
}

/// Phase B — validate the `candle-flash-attn` paged kernel (PR #3655).
///
/// Calls `flash_attn_varlen_paged_windowed` with a minimal batch=1 config and
/// confirms it runs on sm_89. This is the API the engine will actually use for
/// the hot path — so the question is "does it run at all", not "is the output
/// numerically correct" (T8 covers correctness separately).
#[allow(clippy::too_many_arguments)]
fn phase_b_paged_attn(dev: &Device) -> Result<()> {
    use candle_flash_attn::flash_attn_varlen_paged_windowed;

    let batch_size = 1usize;
    let page_block_size = 32usize; // must be multiple of 32
    let num_blocks = 4usize; // 4 pages × 32 = 128 token slots (> seq_k)
    let head_size = 64usize; // smallest supported head dim
    let num_heads = 4usize;
    let num_heads_kv = 4usize; // MHA (num_heads % num_heads_kv == 0)

    let seq_q = 16usize;
    let seq_k = 64usize; // exactly 2 pages
    let total_q = seq_q * batch_size;

    // q: (total_q, num_heads, head_size) F16, rank-3 (varlen-packed).
    let q = Tensor::randn(0.0f32, 1.0, (total_q, num_heads, head_size), dev)?
        .to_dtype(candle::DType::F16)?;
    // k, v: (num_blocks, page_block_size, num_heads_kv, head_size) F16, rank-4 (paged).
    let kv_shape = (num_blocks, page_block_size, num_heads_kv, head_size);
    let k = Tensor::randn(0.0f32, 1.0, kv_shape, dev)?.to_dtype(candle::DType::F16)?;
    let v = Tensor::randn(0.0f32, 1.0, kv_shape, dev)?.to_dtype(candle::DType::F16)?;

    // cu_seqlens (cumulative prefix-sum, NOT vLLM's per-seq context_lens).
    // batch 0: queries [0..16), kv [0..64).
    let cu_seqlens_q = Tensor::from_vec(vec![0u32, seq_q as u32], (batch_size + 1,), dev)?;
    let cu_seqlens_k = Tensor::from_vec(vec![0u32, seq_k as u32], (batch_size + 1,), dev)?;

    // block_table: (batch_size, max_blocks) — vLLM-style physical block indices.
    let block_table = Tensor::from_vec(
        (0..num_blocks as u32).collect::<Vec<_>>(),
        (batch_size, num_blocks),
        dev,
    )?;

    let softmax_scale = 1.0 / (head_size as f32).sqrt();

    let out = flash_attn_varlen_paged_windowed(
        &q,
        &k,
        &v,
        &cu_seqlens_q,
        &cu_seqlens_k,
        &block_table,
        None,            // mm_prefix_ranges
        seq_q,           // max_seqlen_q
        seq_k,           // max_seqlen_k
        softmax_scale,
        None,            // window_size_left (None = unbounded lookback)
        Some(0),         // window_size_right (Some(0) = causal)
        page_block_size,
        None,            // softcap
    )?;

    let dims = out.shape().dims();
    println!(
        "[proto][B] output shape = {:?} (expected [{}, {}, {}])",
        dims, total_q, num_heads, head_size
    );
    let out_f32 = out.to_dtype(candle::DType::F32)?.flatten_all()?;
    let n_elem = out_f32.elem_count();
    let first = out_f32.get(0)?.to_vec0::<f32>()?;
    let has_nan = (0..n_elem)
        .step_by(n_elem / 16 + 1)
        .map(|i| out_f32.get(i)?.to_vec0::<f32>().map(|v| v.is_nan()))
        .try_fold(false, |a, b| Ok::<_, candle::Error>(a | b?))?;
    println!(
        "[proto][B] first elem = {:.6}, any_nan(sampled) = {}, n_elem = {}",
        first, has_nan, n_elem
    );

    let shape_ok = dims == [total_q, num_heads, head_size];
    if !shape_ok || has_nan {
        println!("[proto][B] FAIL — shape wrong or NaN in output.");
        std::process::exit(1);
    }
    println!("[proto][B] PASS — paged flash-attn ran on sm_89 via the PR #3655 API.");
    println!("[proto]    signature confirmed: flash_attn_varlen_paged_windowed(q[F16], k[F16], v[F16], cu_seqlens_q[u32], cu_seqlens_k[u32], block_table[u32], mm_prefix_ranges[Opt], max_sq, max_sk, scale, win_l[Opt], win_r[Opt], page_block_size, softcap[Opt])");
    println!("[proto]    NOTE: cu_seqlens (cumulative) NOT context_lens — T4 must convert.");
    Ok(())
}
