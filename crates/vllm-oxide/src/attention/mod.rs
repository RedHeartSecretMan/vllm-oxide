//! `attention/` — T4 paged-attention contract.
//!
//! Model code calls `flash_attn_varlen` (initial prefill) /
//! `flash_attn_varlen_paged_windowed` (continued prefill and decode) directly —
//! NO `AttentionBackend` trait for v0.1 (YAGNI). The
//! `engine ↔ attention` cycle is broken by `attention/` never importing
//! `engine/`; EngineCore prepares scheduler-owned `AttnMetadata` through the
//! shared `AttentionContext` before model execution.

#![allow(dead_code)]

pub(crate) mod flash_attn;
pub(crate) mod metadata;

#[cfg(feature = "cuda")]
pub(crate) mod kernels;

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use crate::utils::kv_cache_layout_shape;

pub(crate) use metadata::{build_continued_prefill_metadata, PreparedAttention};
pub(crate) use metadata::{build_decode_metadata, build_prefill_metadata, AttnMetadata};

/// Shared attention state crossing the `engine ↔ model` seam.
///
/// The cache initially carries only immutable geometry; the composition root
/// binds its backing tensor exactly once after device-memory sizing. Before a
/// model forward, `EngineCore` transactionally prepares one logical
/// [`AttnMetadata`] value into device tensors and binds it to that StepPlan's
/// epoch. Every layer then borrows the same immutable prepared value. The
/// active value is released after plan execution, including error and unwind
/// paths, so a later step cannot silently consume stale metadata. The context
/// owns that lifecycle directly; neither its fields nor the physical cache are
/// part of the public generation interface (ADR-0011).
#[derive(Clone)]
pub(crate) struct AttentionContext {
    pub(crate) paged_kv: Arc<Mutex<PagedKVCache>>,
    runtime: Arc<Mutex<AttentionState>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionEpoch {
    StepPlan(u64),
    WarmupPrefill,
    WarmupDecode,
}

impl std::fmt::Display for AttentionEpoch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepPlan(plan_id) => write!(formatter, "StepPlan({plan_id})"),
            Self::WarmupPrefill => formatter.write_str("WarmupPrefill"),
            Self::WarmupDecode => formatter.write_str("WarmupDecode"),
        }
    }
}

#[derive(Debug)]
enum AttentionLifecycle {
    Idle,
    Preparing(AttentionEpoch),
    Ready(Arc<PreparedAttention>),
}

#[derive(Debug)]
struct AttentionState {
    lifecycle: AttentionLifecycle,
    bound_consumer: Option<AttentionEpoch>,
    prepare_calls: usize,
    device_tensor_uploads: usize,
}

impl Default for AttentionState {
    fn default() -> Self {
        Self {
            lifecycle: AttentionLifecycle::Idle,
            bound_consumer: None,
            prepare_calls: 0,
            device_tensor_uploads: 0,
        }
    }
}

impl AttentionContext {
    pub(crate) fn new(paged_kv: Arc<Mutex<PagedKVCache>>) -> Self {
        Self {
            paged_kv,
            runtime: Arc::new(Mutex::new(AttentionState::default())),
        }
    }

    /// Prepare and publish one complete metadata value for `epoch`.
    ///
    /// The lifecycle is reserved before uploads begin, but the active value is
    /// replaced only after every tensor succeeds. A failed preparation from an
    /// idle context returns it to idle; a conflicting preparation leaves the
    /// existing active value untouched.
    pub(crate) fn prepare(
        &self,
        epoch: AttentionEpoch,
        logical: AttnMetadata,
        device: &Device,
    ) -> Result<AttentionStepGuard> {
        self.reserve(epoch)?;
        self.complete_preparation(epoch, logical, device)
    }

    /// Bind the only model consumer allowed to use the active epoch.
    pub(crate) fn bind_consumer(&self, expected: AttentionEpoch) -> Result<AttentionConsumerGuard> {
        let runtime = self.runtime_state();
        let mut state = lock_attention_state(runtime)?;
        match &state.lifecycle {
            AttentionLifecycle::Ready(active) if active.epoch() != expected => {
                candle_core::bail!(
                    "stale attention consumer expected {expected}, active epoch is {}",
                    active.epoch()
                )
            }
            AttentionLifecycle::Ready(_) => {
                if let Some(bound) = state.bound_consumer {
                    candle_core::bail!(
                        "attention epoch {expected} already has bound consumer {bound}"
                    )
                }
                state.bound_consumer = Some(expected);
            }
            AttentionLifecycle::Preparing(active_epoch) => {
                candle_core::bail!(
                    "cannot bind attention consumer {expected}; {active_epoch} is still preparing"
                )
            }
            AttentionLifecycle::Idle => {
                candle_core::bail!("cannot bind attention consumer {expected}; context is idle")
            }
        }
        Ok(AttentionConsumerGuard {
            context: self.clone(),
            epoch: expected,
            armed: true,
        })
    }

    /// Borrow only when a consumer is explicitly bound to the ready epoch.
    pub(crate) fn prepared_for_bound_consumer(&self) -> Result<Arc<PreparedAttention>> {
        let runtime = self.runtime_state();
        let state = lock_attention_state(runtime)?;
        match (&state.lifecycle, state.bound_consumer) {
            (AttentionLifecycle::Ready(prepared), Some(bound)) if prepared.epoch() == bound => {
                Ok(prepared.clone())
            }
            (AttentionLifecycle::Ready(prepared), Some(bound)) => candle_core::bail!(
                "bound attention consumer {bound} does not match active epoch {}",
                prepared.epoch()
            ),
            (AttentionLifecycle::Ready(prepared), None) => {
                candle_core::bail!("attention epoch {} has no bound consumer", prepared.epoch())
            }
            (AttentionLifecycle::Preparing(epoch), _) => {
                candle_core::bail!("attention metadata for {epoch} is not ready")
            }
            (AttentionLifecycle::Idle, _) => {
                candle_core::bail!("attention context has no prepared metadata")
            }
        }
    }

    fn reserve(&self, epoch: AttentionEpoch) -> Result<()> {
        {
            let runtime = self.runtime_state();
            let mut state = lock_attention_state(runtime)?;
            if let Some(bound) = state.bound_consumer {
                candle_core::bail!(
                    "attention context has bound consumer {bound}; cannot prepare {epoch}"
                )
            }
            match &state.lifecycle {
                AttentionLifecycle::Idle => {
                    state.lifecycle = AttentionLifecycle::Preparing(epoch);
                }
                AttentionLifecycle::Preparing(active_epoch) => {
                    candle_core::bail!(
                        "attention context is preparing {active_epoch}; cannot prepare {epoch}"
                    )
                }
                AttentionLifecycle::Ready(active) => {
                    candle_core::bail!(
                        "attention context is active for {}; cannot prepare {epoch}",
                        active.epoch()
                    )
                }
            }
        }
        Ok(())
    }

    fn complete_preparation(
        &self,
        epoch: AttentionEpoch,
        logical: AttnMetadata,
        device: &Device,
    ) -> Result<AttentionStepGuard> {
        let mut reservation = AttentionPreparationReservation {
            context: self.clone(),
            epoch,
            armed: true,
        };
        let prepared = PreparedAttention::prepare(epoch, logical, device)?;
        let device_tensor_uploads = prepared.device_tensor_uploads();
        let prepared = Arc::new(prepared);

        {
            let runtime = self.runtime_state();
            let mut state = lock_attention_state(runtime)?;
            match state.lifecycle {
                AttentionLifecycle::Preparing(current)
                    if current == epoch && state.bound_consumer.is_none() =>
                {
                    state.lifecycle = AttentionLifecycle::Ready(prepared);
                    state.prepare_calls += 1;
                    state.device_tensor_uploads += device_tensor_uploads;
                }
                _ => {
                    candle_core::bail!("attention context lost preparation reservation for {epoch}")
                }
            }
        }
        reservation.armed = false;

        Ok(AttentionStepGuard {
            context: self.clone(),
            epoch,
            armed: true,
        })
    }

    fn runtime_state(&self) -> &Mutex<AttentionState> {
        &self.runtime
    }

    fn cancel_preparation(&self, epoch: AttentionEpoch) {
        let runtime = self.runtime_state();
        if let Ok(mut state) = runtime.lock() {
            if matches!(state.lifecycle, AttentionLifecycle::Preparing(current) if current == epoch)
            {
                state.lifecycle = AttentionLifecycle::Idle;
            }
        };
    }

    fn release_consumer(&self, epoch: AttentionEpoch) -> Result<()> {
        let runtime = self.runtime_state();
        let mut state = lock_attention_state(runtime)?;
        match state.bound_consumer {
            Some(bound) if bound == epoch => {
                state.bound_consumer = None;
                Ok(())
            }
            Some(bound) => candle_core::bail!(
                "cannot release attention consumer {epoch}; bound consumer is {bound}"
            ),
            None => candle_core::bail!(
                "cannot release attention consumer {epoch}; no consumer is bound"
            ),
        }
    }

    fn release(&self, epoch: AttentionEpoch) -> Result<()> {
        let runtime = self.runtime_state();
        let mut state = lock_attention_state(runtime)?;
        if let Some(bound) = state.bound_consumer {
            candle_core::bail!(
                "cannot release attention epoch {epoch}; consumer {bound} is still bound"
            )
        }
        match &state.lifecycle {
            AttentionLifecycle::Ready(active) if active.epoch() == epoch => {
                state.lifecycle = AttentionLifecycle::Idle;
                Ok(())
            }
            AttentionLifecycle::Ready(active) => candle_core::bail!(
                "cannot release attention epoch {epoch}; active epoch is {}",
                active.epoch()
            ),
            AttentionLifecycle::Preparing(active_epoch) => candle_core::bail!(
                "cannot release attention epoch {epoch}; {active_epoch} is still preparing"
            ),
            AttentionLifecycle::Idle => {
                candle_core::bail!("cannot release attention epoch {epoch}; context is idle")
            }
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used)]
    pub(crate) fn instrumentation(&self) -> AttentionInstrumentation {
        let state = self.runtime.lock().unwrap();
        AttentionInstrumentation {
            prepare_calls: state.prepare_calls,
            device_tensor_uploads: state.device_tensor_uploads,
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used)]
    pub(crate) fn is_idle(&self) -> bool {
        let state = self.runtime.lock().unwrap();
        matches!(state.lifecycle, AttentionLifecycle::Idle) && state.bound_consumer.is_none()
    }
}

fn lock_attention_state(
    runtime: &Mutex<AttentionState>,
) -> Result<std::sync::MutexGuard<'_, AttentionState>> {
    runtime.lock().map_err(|error| {
        candle_core::Error::Msg(format!("attention runtime state lock poisoned: {error}"))
    })
}

pub(crate) struct AttentionStepGuard {
    context: AttentionContext,
    epoch: AttentionEpoch,
    armed: bool,
}

pub(crate) struct AttentionConsumerGuard {
    context: AttentionContext,
    epoch: AttentionEpoch,
    armed: bool,
}

impl AttentionConsumerGuard {
    pub(crate) fn finish(mut self) -> Result<()> {
        self.context.release_consumer(self.epoch)?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for AttentionConsumerGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.context.release_consumer(self.epoch);
        }
    }
}

impl AttentionStepGuard {
    pub(crate) fn finish(mut self) -> Result<()> {
        self.context.release(self.epoch)?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for AttentionStepGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.context.release(self.epoch);
        }
    }
}

struct AttentionPreparationReservation {
    context: AttentionContext,
    epoch: AttentionEpoch,
    armed: bool,
}

impl Drop for AttentionPreparationReservation {
    fn drop(&mut self) {
        if self.armed {
            self.context.cancel_preparation(self.epoch);
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionInstrumentation {
    pub(crate) prepare_calls: usize,
    pub(crate) device_tensor_uploads: usize,
}

/// Immutable dimensions and dtype needed to size a paged KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PagedKVCacheGeometry {
    pub(crate) num_layers: usize,
    pub(crate) block_size: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) dtype: DType,
}

/// Model-owned paged KV-cache geometry and its one-time physical GPU buffer.
///
/// Shaped `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
/// (nano-vllm parity layout). The leading `2` is the K/V stack: dim 0 = keys,
/// dim 1 = values.
///
/// The model factory creates the deferred owner without allocating a tensor.
/// `LLM` allocates the final backing buffer once, then shares it across decoder
/// layers and `EngineCore` through `Arc<Mutex<PagedKVCache>>`.
pub struct PagedKVCache {
    buffer: Option<Tensor>,
    geometry: PagedKVCacheGeometry,
    num_blocks: usize,
}

impl PagedKVCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut cache = Self::deferred(PagedKVCacheGeometry {
            num_layers,
            block_size,
            num_kv_heads,
            head_dim,
            dtype,
        });
        cache.allocate(num_blocks, device)?;
        Ok(cache)
    }

    pub(crate) fn deferred(geometry: PagedKVCacheGeometry) -> Self {
        Self {
            buffer: None,
            geometry,
            num_blocks: 0,
        }
    }

    pub(crate) fn allocate(&mut self, num_blocks: usize, device: &Device) -> Result<()> {
        if self.buffer.is_some() {
            candle_core::bail!(
                "PagedKVCache backing buffer is already allocated with {} blocks",
                self.num_blocks
            );
        }
        if num_blocks == 0 {
            candle_core::bail!("PagedKVCache allocation requires at least one block");
        }
        let shape = kv_cache_layout_shape(
            self.geometry.num_layers,
            num_blocks,
            self.geometry.block_size,
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
        );
        let buffer = Tensor::zeros(&shape, self.geometry.dtype, device)?;
        self.buffer = Some(buffer);
        self.num_blocks = num_blocks;
        Ok(())
    }

    fn buffer(&self) -> Result<&Tensor> {
        self.buffer.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("PagedKVCache backing buffer is not allocated".to_string())
        })
    }

    /// Per-layer K cache view: `[num_blocks, block_size, num_kv_heads, head_dim]`.
    pub fn k_cache(&self, layer_id: usize) -> Result<Tensor> {
        self.buffer()?.i((0, layer_id))
    }

    /// Per-layer V cache view: `[num_blocks, block_size, num_kv_heads, head_dim]`.
    pub fn v_cache(&self, layer_id: usize) -> Result<Tensor> {
        self.buffer()?.i((1, layer_id))
    }

    /// Write per-step K/V into the paged cache via the custom CUDA kernel.
    #[cfg(feature = "cuda")]
    pub fn reshape_and_cache(
        &self,
        layer_id: usize,
        key: &Tensor,
        value: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        let k_cache = self.k_cache(layer_id)?;
        let v_cache = self.v_cache(layer_id)?;
        kernels::reshape_and_cache(key, value, &k_cache, &v_cache, slot_mapping)
    }

    /// Return the full buffer shape `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`.
    pub fn buffer_shape(&self) -> Vec<usize> {
        kv_cache_layout_shape(
            self.geometry.num_layers,
            self.num_blocks,
            self.geometry.block_size,
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
        )
        .to_vec()
    }

    /// Storage dtype shared by every layer in this cache.
    pub(crate) fn dtype(&self) -> DType {
        self.geometry.dtype
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    pub fn block_size(&self) -> usize {
        self.geometry.block_size
    }

    pub(crate) fn geometry(&self) -> PagedKVCacheGeometry {
        self.geometry
    }

    #[cfg(test)]
    pub(crate) fn allocation_count(&self) -> usize {
        usize::from(self.buffer.is_some())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
// All test unwraps: PagedKVCache::new with hardcoded small dimensions always
// succeeds on CPU; k_cache/v_cache indices are within allocated layer count.
mod tests {
    use super::*;

    fn context() -> AttentionContext {
        AttentionContext::new(Arc::new(Mutex::new(
            PagedKVCache::new(1, 4, 4, 1, 8, DType::F32, &Device::Cpu).unwrap(),
        )))
    }

    #[test]
    fn allocates_correct_shape() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(28, 100, 256, 4, 128, DType::BF16, &dev).unwrap();
        assert_eq!(cache.buffer_shape(), vec![2, 28, 100, 256, 4, 128]);
    }

    #[test]
    fn k_cache_slice_shape() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(4, 10, 256, 4, 128, DType::BF16, &dev).unwrap();
        let k = cache.k_cache(2).unwrap();
        assert_eq!(k.shape().dims(), &[10, 256, 4, 128]);
    }

    #[test]
    fn v_cache_slice_shape() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(4, 10, 256, 4, 128, DType::BF16, &dev).unwrap();
        let v = cache.v_cache(1).unwrap();
        assert_eq!(v.shape().dims(), &[10, 256, 4, 128]);
    }

    #[test]
    fn accessors() {
        let dev = Device::Cpu;
        let cache = PagedKVCache::new(1, 42, 256, 1, 1, DType::F32, &dev).unwrap();
        assert_eq!(cache.block_size(), 256);
        assert_eq!(cache.num_blocks(), 42);
    }

    #[test]
    fn deferred_cache_allocates_its_backing_buffer_once() {
        let dev = Device::Cpu;
        let mut cache = PagedKVCache::deferred(PagedKVCacheGeometry {
            num_layers: 2,
            block_size: 4,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F16,
        });

        assert_eq!(cache.allocation_count(), 0);
        cache.allocate(4, &dev).unwrap();
        assert_eq!(cache.allocation_count(), 1);
        assert_eq!(cache.buffer_shape(), vec![2, 2, 4, 4, 2, 8]);

        let error = cache.allocate(8, &dev).unwrap_err();
        assert!(error.to_string().contains("already allocated"));
        assert_eq!(cache.allocation_count(), 1);
        assert_eq!(cache.num_blocks(), 4);
    }

    #[test]
    fn failed_preparation_from_idle_is_transactional() {
        let context = context();
        let malformed = AttnMetadata {
            is_prefill: true,
            cu_seqlens_q: vec![0, 2],
            cu_seqlens_k: vec![0, 2],
            max_seqlen_q: 2,
            max_seqlen_k: 2,
            slot_mapping: vec![0],
            block_table: Vec::new(),
        };

        let error = context
            .prepare(AttentionEpoch::StepPlan(7), malformed, &Device::Cpu)
            .err()
            .unwrap();

        assert!(error
            .to_string()
            .contains("slot mapping has 1 entries for 2 query tokens"));
        assert!(context.is_idle());
        assert_eq!(
            context.instrumentation(),
            AttentionInstrumentation {
                prepare_calls: 0,
                device_tensor_uploads: 0,
            }
        );
    }

    #[test]
    fn conflicting_preparation_preserves_the_active_epoch() {
        let context = context();
        let first = context
            .prepare(
                AttentionEpoch::StepPlan(7),
                build_prefill_metadata(&[1], &[1], &[0]),
                &Device::Cpu,
            )
            .unwrap();
        let consumer = context.bind_consumer(AttentionEpoch::StepPlan(7)).unwrap();
        let slot_mapping_id = context
            .prepared_for_bound_consumer()
            .unwrap()
            .slot_mapping()
            .id();
        consumer.finish().unwrap();

        let error = context
            .prepare(
                AttentionEpoch::StepPlan(8),
                build_prefill_metadata(&[1], &[1], &[1]),
                &Device::Cpu,
            )
            .err()
            .unwrap();

        assert!(error.to_string().contains("active for StepPlan(7)"));
        assert_eq!(
            {
                let consumer = context.bind_consumer(AttentionEpoch::StepPlan(7)).unwrap();
                let current = context
                    .prepared_for_bound_consumer()
                    .unwrap()
                    .slot_mapping()
                    .id();
                consumer.finish().unwrap();
                current
            },
            slot_mapping_id
        );
        assert_eq!(context.instrumentation().prepare_calls, 1);
        first.finish().unwrap();
        assert!(context.is_idle());
    }

    #[test]
    fn stale_consumer_is_rejected_with_both_epochs() {
        let context = context();
        let step = context
            .prepare(
                AttentionEpoch::StepPlan(9),
                build_prefill_metadata(&[1], &[1], &[0]),
                &Device::Cpu,
            )
            .unwrap();

        let error = context
            .bind_consumer(AttentionEpoch::StepPlan(8))
            .err()
            .unwrap();

        assert_eq!(
            error.to_string(),
            "stale attention consumer expected StepPlan(8), active epoch is StepPlan(9)"
        );
        let consumer = context.bind_consumer(AttentionEpoch::StepPlan(9)).unwrap();
        assert_eq!(
            context.prepared_for_bound_consumer().unwrap().epoch(),
            AttentionEpoch::StepPlan(9)
        );
        consumer.finish().unwrap();
        step.finish().unwrap();
    }

    #[test]
    fn ready_metadata_requires_a_matching_consumer_binding() {
        let context = context();
        let step = context
            .prepare(
                AttentionEpoch::StepPlan(9),
                build_prefill_metadata(&[1], &[1], &[0]),
                &Device::Cpu,
            )
            .unwrap();

        let unbound = context.prepared_for_bound_consumer().unwrap_err();
        assert!(unbound.to_string().contains("has no bound consumer"));
        let stale = context
            .bind_consumer(AttentionEpoch::StepPlan(8))
            .err()
            .unwrap();
        assert_eq!(
            stale.to_string(),
            "stale attention consumer expected StepPlan(8), active epoch is StepPlan(9)"
        );

        let consumer = context.bind_consumer(AttentionEpoch::StepPlan(9)).unwrap();
        assert_eq!(
            context.prepared_for_bound_consumer().unwrap().epoch(),
            AttentionEpoch::StepPlan(9)
        );
        consumer.finish().unwrap();
        step.finish().unwrap();
        assert!(context.is_idle());
    }

    #[test]
    fn guard_drop_releases_metadata_during_unwind() {
        let context = context();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let context = context.clone();
            move || {
                let _step = context
                    .prepare(
                        AttentionEpoch::StepPlan(11),
                        build_prefill_metadata(&[1], &[1], &[0]),
                        &Device::Cpu,
                    )
                    .unwrap();
                panic!("injected model panic");
            }
        }));

        assert!(unwind.is_err());
        assert!(context.is_idle());
        assert!(context
            .prepared_for_bound_consumer()
            .unwrap_err()
            .to_string()
            .contains("no prepared"));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use super::*;

    fn cuda_device() -> Device {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    }

    #[test]
    fn prepared_context_uploads_ragged_metadata_once_on_cuda() {
        let dev = Device::new_cuda(0).unwrap();
        let context = AttentionContext::new(Arc::new(Mutex::new(
            PagedKVCache::new(1, 2, 256, 1, 64, DType::BF16, &dev).unwrap(),
        )));
        let step = context
            .prepare(
                AttentionEpoch::StepPlan(77),
                build_decode_metadata(&[1, 257], &[vec![0], vec![0, 1]], &[0, 256]),
                &dev,
            )
            .unwrap();
        let consumer = context.bind_consumer(AttentionEpoch::StepPlan(77)).unwrap();
        let prepared = context.prepared_for_bound_consumer().unwrap();

        assert!(prepared.cu_seqlens_q().device().is_cuda());
        assert!(prepared.cu_seqlens_k().device().is_cuda());
        assert!(prepared.slot_mapping().device().is_cuda());
        assert!(prepared.block_table().unwrap().device().is_cuda());
        assert_eq!(
            context.instrumentation(),
            AttentionInstrumentation {
                prepare_calls: 1,
                device_tensor_uploads: 4,
            }
        );

        consumer.finish().unwrap();
        step.finish().unwrap();
        assert!(context.is_idle());
    }

    #[test]
    fn reshape_and_cache_writes_correct_slots() {
        let dev = cuda_device();
        if !dev.is_cuda() {
            eprintln!("[gpu_tests] skipping — no CUDA device");
            return;
        }

        let num_kv_heads = 2;
        let head_dim = 64;
        let cache =
            PagedKVCache::new(1, 4, 256, num_kv_heads, head_dim, DType::BF16, &dev).unwrap();

        let num_tokens = 3;
        let key = Tensor::arange(0f32, (num_tokens * num_kv_heads * head_dim) as f32, &dev)
            .unwrap()
            .reshape((num_tokens, num_kv_heads, head_dim))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let value = Tensor::arange(
            1000f32,
            1000f32 + (num_tokens * num_kv_heads * head_dim) as f32,
            &dev,
        )
        .unwrap()
        .reshape((num_tokens, num_kv_heads, head_dim))
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let slot_mapping = Tensor::from_vec(vec![0i64, 256, 512], (num_tokens,), &dev).unwrap();

        cache
            .reshape_and_cache(0, &key, &value, &slot_mapping)
            .unwrap();

        let k_cache = cache.k_cache(0).unwrap();
        let v_cache = cache.v_cache(0).unwrap();

        let expected_k_0 = key.i(0).unwrap();
        let written_k_0 = k_cache.i((0, 0)).unwrap();
        let diff = (written_k_0.to_dtype(DType::F32).unwrap()
            - expected_k_0.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap();
        let max_diff = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
        assert_eq!(max_diff, 0.0, "K slot 0 (block 0, offset 0) mismatch");

        let expected_k_1 = key.i(1).unwrap();
        let written_k_1 = k_cache.i((1, 0)).unwrap();
        let diff = (written_k_1.to_dtype(DType::F32).unwrap()
            - expected_k_1.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap();
        let max_diff = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
        assert_eq!(max_diff, 0.0, "K slot 256 (block 1, offset 0) mismatch");

        let expected_v_2 = value.i(2).unwrap();
        let written_v_2 = v_cache.i((2, 0)).unwrap();
        let diff = (written_v_2.to_dtype(DType::F32).unwrap()
            - expected_v_2.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap();
        let max_diff = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
        assert_eq!(max_diff, 0.0, "V slot 512 (block 2, offset 0) mismatch");
    }

    #[test]
    fn flash_attn_prefill_runs() {
        let dev = cuda_device();
        if !dev.is_cuda() {
            eprintln!("[gpu_tests] skipping — no CUDA device");
            return;
        }

        let num_heads = 4;
        let head_dim = 64;
        let seq_len = 8;

        let q = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let meta = build_prefill_metadata(
            &[seq_len as u32],
            &[seq_len as u32],
            &(0..seq_len as i64).collect::<Vec<_>>(),
        );
        let context = AttentionContext::new(Arc::new(Mutex::new(
            PagedKVCache::new(1, 1, 256, num_heads, head_dim, DType::BF16, &dev).unwrap(),
        )));
        let step = context
            .prepare(AttentionEpoch::StepPlan(0), meta, &dev)
            .unwrap();
        let consumer = context.bind_consumer(AttentionEpoch::StepPlan(0)).unwrap();
        let prepared = context.prepared_for_bound_consumer().unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out = super::flash_attn::prefill_attn(&q, &k, &v, &prepared, scale).unwrap();
        consumer.finish().unwrap();
        step.finish().unwrap();

        assert_eq!(out.shape().dims(), &[seq_len, num_heads, head_dim]);
        let out_f32 = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let has_nan = (0..out_f32.elem_count())
            .step_by(out_f32.elem_count() / 16 + 1)
            .try_fold(false, |acc, i| {
                Ok::<_, candle_core::Error>(acc | out_f32.get(i)?.to_vec0::<f32>()?.is_nan())
            })
            .unwrap();
        assert!(!has_nan, "prefill output contains NaN");
    }

    #[test]
    fn flash_attn_decode_runs() {
        let dev = cuda_device();
        if !dev.is_cuda() {
            eprintln!("[gpu_tests] skipping — no CUDA device");
            return;
        }

        let num_heads = 4;
        let kv_heads = 4;
        let head_dim = 64;
        let ctx_len = 32;

        let cache = Arc::new(Mutex::new(
            PagedKVCache::new(1, 1, 256, kv_heads, head_dim, DType::BF16, &dev).unwrap(),
        ));

        let key = Tensor::randn(0f32, 1f32, (ctx_len, kv_heads, head_dim), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let value = Tensor::randn(0f32, 1f32, (ctx_len, kv_heads, head_dim), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let slots: Vec<i64> = (0..ctx_len as i64).collect();
        let slot_mapping = Tensor::from_vec(slots, (ctx_len,), &dev).unwrap();
        cache
            .lock()
            .unwrap()
            .reshape_and_cache(0, &key, &value, &slot_mapping)
            .unwrap();

        let q = Tensor::randn(0f32, 1f32, (1, num_heads, head_dim), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k_cache = cache.lock().unwrap().k_cache(0).unwrap();
        let v_cache = cache.lock().unwrap().v_cache(0).unwrap();

        let meta = build_decode_metadata(&[ctx_len as u32], &[vec![0]], &[ctx_len as i64]);
        let context = AttentionContext::new(cache);
        let step = context
            .prepare(AttentionEpoch::StepPlan(0), meta, &dev)
            .unwrap();
        let consumer = context.bind_consumer(AttentionEpoch::StepPlan(0)).unwrap();
        let prepared = context.prepared_for_bound_consumer().unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out =
            super::flash_attn::paged_attn(&q, &k_cache, &v_cache, &prepared, scale, 256).unwrap();
        consumer.finish().unwrap();
        step.finish().unwrap();

        assert_eq!(out.shape().dims(), &[1, num_heads, head_dim]);
        let out_f32 = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let first = out_f32.get(0).unwrap().to_vec0::<f32>().unwrap();
        assert!(!first.is_nan(), "decode output contains NaN");
    }
}
