//! CUDA adapter behind the `Sampler` seam.
//!
//! # Unsafe boundary
//!
//! Candle exposes CUDA allocations through guarded raw device pointers, while
//! the project kernels are linked through C FFI. The launch scope below keeps
//! every owning tensor, storage guard, CUDA slice view, and `SyncOnDrop` guard
//! alive without extending lifetimes, requires contiguous tensors, and keeps
//! all mutable destinations private to the reusable `Workspace`. The CUDA
//! wrapper synchronizes its stream before returning, so host metadata used by
//! H2D history copies remains alive and asynchronous kernel failures are
//! reported at this seam.

#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

use candle_core::cuda::cudarc::driver::sys::CUstream;
use candle_core::cuda::cudarc::driver::{DevicePtr, DeviceSlice};
use candle_core::{DType, Device, Error, Result, Storage, Tensor};

use super::SamplingParams;

mod ffi {
    use super::{c_char, c_int, c_void, CUstream};

    extern "C" {
        pub fn vllm_oxide_sampling_workspace_bytes(
            vocab_size: u32,
            temp_storage_bytes: *mut usize,
        ) -> c_int;

        #[allow(clippy::too_many_arguments)]
        pub fn vllm_oxide_sample_f32(
            logits: *const f32,
            selected_tokens: *mut u32,
            temperatures: *const f32,
            top_ks: *const u32,
            top_ps: *const f32,
            presence_penalties: *const f32,
            frequency_penalties: *const f32,
            repetition_penalties: *const f32,
            row_seeds: *const u64,
            history_tokens_host: *const u32,
            history_counts_host: *const u32,
            history_offsets_host: *const u32,
            history_tokens_device: *mut u32,
            history_counts_device: *mut u32,
            keys_in: *mut f32,
            keys_out: *mut f32,
            ids_in: *mut u32,
            ids_out: *mut u32,
            limits: *mut u32,
            temp_storage: *mut c_void,
            temp_storage_bytes: usize,
            batch_size: u32,
            vocab_size: u32,
            failed_stage: *mut c_int,
            failed_row: *mut c_int,
            stream: CUstream,
        ) -> c_int;

        pub fn vllm_oxide_sampling_cuda_error_string(status: c_int) -> *const c_char;
    }
}

macro_rules! cuda_ptr {
    ($tensor:expr, $ty:ty, $label:literal, $storage:ident, $view:ident, $ptr:ident, $guard:ident) => {
        let ($storage, layout) = $tensor.storage_and_layout();
        let cuda_storage = match &*$storage {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!(concat!(
                "sampler CUDA adapter expected CUDA storage for ",
                $label
            )),
        };
        let offsets = layout.contiguous_offsets().ok_or_else(|| {
            Error::msg(concat!(
                "sampler CUDA adapter requires contiguous storage for ",
                $label
            ))
        })?;
        let $view = cuda_storage
            .as_cuda_slice::<$ty>()?
            .slice(offsets.0..offsets.1);
        let ($ptr, $guard) = $view.device_ptr($view.stream());
    };
}

/// Reused device memory for one row at a time. Its footprint is O(vocab) plus
/// O(unique history), independent of the number of rows in a batch.
pub(super) struct Workspace {
    device: Device,
    vocab_size: usize,
    history_capacity: usize,
    keys_in: Tensor,
    keys_out: Tensor,
    ids_in: Tensor,
    ids_out: Tensor,
    limits: Tensor,
    temp_storage: Tensor,
    history_tokens: Tensor,
    history_counts: Tensor,
}

impl Workspace {
    fn new(device: &Device, vocab_size: usize, history_len: usize) -> Result<Self> {
        let vocab_u32 = u32::try_from(vocab_size).map_err(|_| {
            Error::msg(format!(
                "sampler CUDA workspace: vocabulary size {vocab_size} exceeds u32"
            ))
        })?;
        let mut temp_storage_bytes = 0usize;
        let status =
            unsafe { ffi::vllm_oxide_sampling_workspace_bytes(vocab_u32, &mut temp_storage_bytes) };
        check_status(status, "workspace-size query")?;

        let history_capacity = next_capacity(history_len)?;
        let alloc = |shape, dtype, label: &str| {
            Tensor::zeros(shape, dtype, device).map_err(|error| {
                Error::msg(format!(
                    "sampler CUDA workspace allocation failed for {label}: {error}"
                ))
            })
        };

        Ok(Self {
            device: device.clone(),
            vocab_size,
            history_capacity,
            keys_in: alloc(vocab_size, DType::F32, "keys_in")?,
            keys_out: alloc(vocab_size, DType::F32, "keys_out")?,
            ids_in: alloc(vocab_size, DType::U32, "ids_in")?,
            ids_out: alloc(vocab_size, DType::U32, "ids_out")?,
            limits: alloc(2, DType::U32, "limits")?,
            temp_storage: alloc(temp_storage_bytes.max(1), DType::U8, "CUB temp storage")?,
            history_tokens: alloc(history_capacity, DType::U32, "history tokens")?,
            history_counts: alloc(history_capacity, DType::U32, "history counts")?,
        })
    }

    fn ensure_history_capacity(&mut self, required: usize) -> Result<()> {
        if required <= self.history_capacity {
            return Ok(());
        }
        let new_capacity = next_capacity(required)?;
        let history_tokens =
            Tensor::zeros(new_capacity, DType::U32, &self.device).map_err(|error| {
                Error::msg(format!(
                    "sampler CUDA workspace growth failed for history tokens ({new_capacity}): {error}"
                ))
            })?;
        let history_counts =
            Tensor::zeros(new_capacity, DType::U32, &self.device).map_err(|error| {
                Error::msg(format!(
                    "sampler CUDA workspace growth failed for history counts ({new_capacity}): {error}"
                ))
            })?;
        // Commit the grown pair atomically only after both allocations
        // succeed. A failed second allocation leaves the reusable workspace
        // and its recorded capacity internally consistent for retry.
        self.history_tokens = history_tokens;
        self.history_counts = history_counts;
        self.history_capacity = new_capacity;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn allocation_identity(&self) -> candle_core::TensorId {
        self.keys_in.id()
    }

    #[cfg(test)]
    pub(super) fn allocated_bytes(&self) -> usize {
        (self.vocab_size * std::mem::size_of::<f32>() * 2)
            + (self.vocab_size * std::mem::size_of::<u32>() * 2)
            + (2 * std::mem::size_of::<u32>())
            + self.temp_storage.elem_count()
            + (self.history_capacity * std::mem::size_of::<u32>() * 2)
    }

    #[cfg(test)]
    pub(super) fn history_capacity(&self) -> usize {
        self.history_capacity
    }
}

fn next_capacity(required: usize) -> Result<usize> {
    required.max(1).checked_next_power_of_two().ok_or_else(|| {
        Error::msg(format!(
            "sampler CUDA workspace history capacity overflow for {required} entries"
        ))
    })
}

struct PreparedBatch {
    temperatures: Vec<f32>,
    top_ks: Vec<u32>,
    top_ps: Vec<f32>,
    presence_penalties: Vec<f32>,
    frequency_penalties: Vec<f32>,
    repetition_penalties: Vec<f32>,
    history_tokens: Vec<u32>,
    history_counts: Vec<u32>,
    history_offsets: Vec<u32>,
}

fn prepare_batch(
    params: &[SamplingParams],
    token_history: &[Vec<u32>],
    vocab_size: usize,
) -> Result<PreparedBatch> {
    let vocab_u32 = u32::try_from(vocab_size).map_err(|_| {
        Error::msg(format!(
            "sampler CUDA adapter: vocabulary size {vocab_size} exceeds u32"
        ))
    })?;
    let mut prepared = PreparedBatch {
        temperatures: Vec::with_capacity(params.len()),
        top_ks: Vec::with_capacity(params.len()),
        top_ps: Vec::with_capacity(params.len()),
        presence_penalties: Vec::with_capacity(params.len()),
        frequency_penalties: Vec::with_capacity(params.len()),
        repetition_penalties: Vec::with_capacity(params.len()),
        history_tokens: Vec::new(),
        history_counts: Vec::new(),
        history_offsets: Vec::with_capacity(params.len() + 1),
    };
    prepared.history_offsets.push(0);

    for (row, (params, history)) in params.iter().zip(token_history).enumerate() {
        params.validate().map_err(|error| {
            Error::msg(format!(
                "sampler CUDA adapter: sampling params at row {row}{error}"
            ))
        })?;
        prepared.temperatures.push(params.temperature);
        let top_k = params.top_k.unwrap_or(vocab_size).min(vocab_size);
        prepared.top_ks.push(u32::try_from(top_k).map_err(|_| {
            Error::msg(format!(
                "sampler CUDA adapter: row {row} top_k {top_k} exceeds u32"
            ))
        })?);
        prepared.top_ps.push(params.top_p.unwrap_or(1.0));
        prepared.presence_penalties.push(params.presence_penalty);
        prepared.frequency_penalties.push(params.frequency_penalty);
        prepared
            .repetition_penalties
            .push(params.repetition_penalty);

        if params.has_penalties() && !history.is_empty() {
            let mut counts = BTreeMap::<u32, u32>::new();
            for &token in history {
                if token < vocab_u32 {
                    let count = counts.entry(token).or_default();
                    *count = count.checked_add(1).ok_or_else(|| {
                        Error::msg(format!(
                            "sampler CUDA adapter: row {row} history count overflow for token {token}"
                        ))
                    })?;
                }
            }
            for (token, count) in counts {
                prepared.history_tokens.push(token);
                prepared.history_counts.push(count);
            }
        }
        prepared
            .history_offsets
            .push(u32::try_from(prepared.history_tokens.len()).map_err(|_| {
                Error::msg(format!(
                    "sampler CUDA adapter: flattened history exceeds u32 after row {row}"
                ))
            })?);
    }
    Ok(prepared)
}

fn ensure_workspace<'a>(
    slot: &'a mut Option<Workspace>,
    device: &Device,
    vocab_size: usize,
    history_len: usize,
) -> Result<&'a mut Workspace> {
    let must_rebuild = match slot.as_ref() {
        None => true,
        Some(workspace) => {
            workspace.vocab_size != vocab_size || !workspace.device.same_device(device)
        }
    };
    if must_rebuild {
        *slot = Some(Workspace::new(device, vocab_size, history_len)?);
    }
    let workspace = slot
        .as_mut()
        .ok_or_else(|| Error::msg("sampler CUDA workspace was not initialized"))?;
    workspace.ensure_history_capacity(history_len)?;
    Ok(workspace)
}

pub(super) fn sample(
    logits: &Tensor,
    params: &[SamplingParams],
    token_history: &[Vec<u32>],
    row_seeds: &[u64],
    workspace_slot: &mut Option<Workspace>,
) -> Result<Tensor> {
    let batch_size = logits.dim(0)?;
    let vocab_size = logits.dim(1)?;
    if vocab_size == 0 {
        candle_core::bail!("sampler CUDA adapter: vocabulary must not be empty")
    }
    let batch_u32 = u32::try_from(batch_size).map_err(|_| {
        Error::msg(format!(
            "sampler CUDA adapter: batch size {batch_size} exceeds u32"
        ))
    })?;
    let vocab_u32 = u32::try_from(vocab_size).map_err(|_| {
        Error::msg(format!(
            "sampler CUDA adapter: vocabulary size {vocab_size} exceeds u32"
        ))
    })?;
    i32::try_from(vocab_size).map_err(|_| {
        Error::msg(format!(
            "sampler CUDA adapter: vocabulary size {vocab_size} exceeds i32"
        ))
    })?;
    if row_seeds.len() != batch_size {
        candle_core::bail!(
            "sampler CUDA adapter: row_seeds.len() = {} but batch = {batch_size}",
            row_seeds.len()
        )
    }

    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|error| Error::msg(format!("sampler CUDA FP32 upcast failed: {error}")))?
        .contiguous()
        .map_err(|error| Error::msg(format!("sampler CUDA contiguous copy failed: {error}")))?;
    let prepared = prepare_batch(params, token_history, vocab_size)?;
    let workspace = ensure_workspace(
        workspace_slot,
        logits.device(),
        vocab_size,
        prepared.history_tokens.len(),
    )?;
    let selected_tokens =
        Tensor::zeros(batch_size, DType::U32, logits.device()).map_err(|error| {
            Error::msg(format!(
                "sampler CUDA selected-token allocation failed: {error}"
            ))
        })?;

    let mut failed_stage = 0;
    let mut failed_row = -1;
    let status = {
        cuda_ptr!(
            logits,
            f32,
            "logits",
            logits_storage,
            logits_view,
            logits_ptr,
            _g_logits
        );
        cuda_ptr!(
            selected_tokens,
            u32,
            "selected tokens",
            selected_storage,
            selected_view,
            selected_ptr,
            _g_selected
        );
        cuda_ptr!(
            workspace.history_tokens,
            u32,
            "history tokens",
            history_tokens_storage,
            history_tokens_view,
            history_tokens_ptr,
            _g_history_tokens
        );
        cuda_ptr!(
            workspace.history_counts,
            u32,
            "history counts",
            history_counts_storage,
            history_counts_view,
            history_counts_ptr,
            _g_history_counts
        );
        cuda_ptr!(
            workspace.keys_in,
            f32,
            "input keys",
            keys_in_storage,
            keys_in_view,
            keys_in_ptr,
            _g_keys_in
        );
        cuda_ptr!(
            workspace.keys_out,
            f32,
            "output keys",
            keys_out_storage,
            keys_out_view,
            keys_out_ptr,
            _g_keys_out
        );
        cuda_ptr!(
            workspace.ids_in,
            u32,
            "input token indices",
            ids_in_storage,
            ids_in_view,
            ids_in_ptr,
            _g_ids_in
        );
        cuda_ptr!(
            workspace.ids_out,
            u32,
            "output token indices",
            ids_out_storage,
            ids_out_view,
            ids_out_ptr,
            _g_ids_out
        );
        cuda_ptr!(
            workspace.limits,
            u32,
            "filter limits",
            limits_storage,
            limits_view,
            limits_ptr,
            _g_limits
        );
        cuda_ptr!(
            workspace.temp_storage,
            u8,
            "CUB temp storage",
            temp_storage_storage,
            temp_storage_view,
            temp_storage_ptr,
            _g_temp_storage
        );
        let stream = match logits.device() {
            Device::Cuda(device) => device.cuda_stream().cu_stream(),
            _ => candle_core::bail!("sampler CUDA adapter expected a CUDA device"),
        };

        unsafe {
            ffi::vllm_oxide_sample_f32(
                logits_ptr as *const f32,
                selected_ptr as *mut u32,
                prepared.temperatures.as_ptr(),
                prepared.top_ks.as_ptr(),
                prepared.top_ps.as_ptr(),
                prepared.presence_penalties.as_ptr(),
                prepared.frequency_penalties.as_ptr(),
                prepared.repetition_penalties.as_ptr(),
                row_seeds.as_ptr(),
                prepared.history_tokens.as_ptr(),
                prepared.history_counts.as_ptr(),
                prepared.history_offsets.as_ptr(),
                history_tokens_ptr as *mut u32,
                history_counts_ptr as *mut u32,
                keys_in_ptr as *mut f32,
                keys_out_ptr as *mut f32,
                ids_in_ptr as *mut u32,
                ids_out_ptr as *mut u32,
                limits_ptr as *mut u32,
                temp_storage_ptr as *mut c_void,
                workspace.temp_storage.elem_count(),
                batch_u32,
                vocab_u32,
                &mut failed_stage,
                &mut failed_row,
                stream,
            )
        }
    };
    if status != 0 && failed_row != -1 {
        return Err(Error::msg(format!(
            "sampler CUDA adapter violated error attribution contract: runtime status {status} reported origin row {failed_row}"
        )));
    }
    check_status(status, stage_name(failed_stage))?;
    Ok(selected_tokens)
}

fn stage_name(stage: c_int) -> &'static str {
    match stage {
        1 => "history H2D",
        2 => "row D2D copy",
        3 => "penalty kernel",
        4 => "greedy argmax kernel",
        5 => "temperature kernel",
        6 => "token-index kernel",
        7 => "CUB radix sort",
        8 => "top-k threshold kernel",
        9 => "top-p cutoff kernel",
        10 => "categorical selection kernel",
        11 => "CUDA stream synchronization",
        _ => "CUDA sampling",
    }
}

fn check_status(status: c_int, stage: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let description = unsafe {
        let ptr = ffi::vllm_oxide_sampling_cuda_error_string(status);
        if ptr.is_null() {
            "unknown CUDA runtime error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(Error::msg(format!(
        "sampler CUDA error observed during {stage}: {description} (status {status})"
    )))
}

#[cfg(test)]
pub(super) fn injected_error_observation(stage: c_int) -> Result<String> {
    match check_status(1, stage_name(stage)) {
        Ok(()) => Err(Error::msg("injected CUDA status unexpectedly succeeded")),
        Err(error) => Ok(error.to_string()),
    }
}
