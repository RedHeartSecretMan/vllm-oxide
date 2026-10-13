//! Throwaway prototype — validates T3 "Stack A" assumption (a):
//! the candle Tensor → cudarc CudaSlice → extern "C" kernel seam.
//!
//! Pattern adapted from mistral.rs `mistralrs-paged-attn`
//! (commit e124932, `src/cuda/backend/paged_attention.rs` + `mod.rs`).
//!
//! Success criterion: allocate a candle CUDA tensor, hand its raw device
//! pointer + CUstream to a hand-written `.cu` kernel, read back, confirm
//! the kernel ran. If this works, T3's riskiest assumption holds.

use candle::cuda_backend::cudarc::driver::DevicePtr; // .device_ptr()
use candle_core as candle;
use candle_core::cuda::cudarc::driver::sys::CUstream;
use candle_core::cuda::cudarc::driver::DeviceSlice; // .slice(range)
use candle_core::{Device, Result, Storage, Tensor};
use core::ffi::c_void;

mod validation;

extern "C" {
    fn fill_const_f32_launch(ptr: *mut c_void, n: i64, val: f32, stream: CUstream) -> i32;
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("[proto] device: {:?}", dev);

    // Allocate a 1024-element f32 tensor on GPU, zero-initialized.
    let n: usize = 1024;
    let t = Tensor::zeros(n, candle::DType::F32, &dev)?;

    let val: f32 = 42.0;
    {
        // Keep storage and the pointer guard in a scope ending before readback.
        // This tensor has no other users; the fill runs on its own Candle stream.
        let (storage_lock, layout) = t.storage_and_layout();
        let cuda_storage = match &*storage_lock {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("expected CUDA storage"),
        };
        // The pinned CudaStorage exposes a typed CUDA slice.
        let slice = cuda_storage.as_cuda_slice::<f32>()?;
        // .slice(range) applies the layout offset (needs DeviceSlice trait)
        let slice = slice.slice(layout.start_offset()..);

        // --- Pointer + stream extraction (mirrors `slice_ptr` in mod.rs) ---
        // device_ptr(stream) -> (u64, SyncOnDrop). The guard MUST outlive the kernel call.
        let (ptr, _guard) = slice.device_ptr(slice.stream());
        let stream = slice.stream().cu_stream();

        // --- The extern "C" call ---
        // SAFETY: the fresh contiguous F32 tensor contains n writable elements,
        // its allocation/storage lock and pointer guard remain alive, and the
        // kernel uses that same stream. Synchronize before releasing these borrows
        // because device_ptr tracks reads, while this narrow experiment writes.
        let status = unsafe { fill_const_f32_launch(ptr as *mut c_void, n as i64, val, stream) };
        if status != 0 {
            candle_core::bail!("[proto][A] CUDA launch failed with runtime status {status}");
        }
        // Also surface asynchronous execution errors; launch success is not completion.
        dev.synchronize()?;
    }

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
        candle_core::bail!("[proto][A] FAIL — kernel values don't match");
    }
    println!("[proto][A] PASS — cudarc seam works; kernel ran on GPU through candle Tensor.");

    phase_b_paged_attn(&dev)
}

/// Phase B — validate the `candle-flash-attn` paged kernel (PR #3655).
///
/// Calls `flash_attn_varlen_paged_windowed` with a minimal batch=1 config and
/// checks its shape and all finite output values. This experiment validates the
/// selected API and launch; it has no independent attention numerical oracle.
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

    // Seed the synthetic input stream for replay within the same pinned stack.
    // A seed does not promise bitwise output identity across CUDA versions/GPUs.
    let seed = 11u64;
    dev.set_seed(seed)?;
    println!("[proto][B] input seed = {seed}");

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
        None,  // mm_prefix_ranges
        seq_q, // max_seqlen_q
        seq_k, // max_seqlen_k
        softmax_scale,
        None,    // window_size_left (None = unbounded lookback)
        Some(0), // window_size_right (Some(0) = causal)
        page_block_size,
        None, // softcap
    )?;

    let dims = out.shape().dims();
    println!(
        "[proto][B] output shape = {:?} (expected [{}, {}, {}])",
        dims, total_q, num_heads, head_size
    );
    let out_f32 = out.to_dtype(candle::DType::F32)?.flatten_all()?;
    // One synchronous readback checks every element, including Inf and positions
    // the old sampling stride never visited.
    let values = out_f32.to_vec1::<f32>()?;
    if let Err(error) = validation::validate_output(dims, [total_q, num_heads, head_size], &values)
    {
        candle_core::bail!("[proto][B] FAIL — {error}");
    }
    println!(
        "[proto][B] first elem = {:.6}, all_finite = true, n_elem = {}",
        values[0],
        values.len()
    );
    println!("[proto][B] PASS — paged flash-attn ran via the PR #3655 API.");
    println!("[proto]    signature confirmed: flash_attn_varlen_paged_windowed(q[F16], k[F16], v[F16], cu_seqlens_q[u32], cu_seqlens_k[u32], block_table[u32], mm_prefix_ranges[Opt], max_sq, max_sk, scale, win_l[Opt], win_r[Opt], page_block_size, softcap[Opt])");
    println!("[proto]    NOTE: cu_seqlens (cumulative) NOT context_lens — T4 must convert.");
    Ok(())
}
