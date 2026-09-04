//! KV-cache capacity, one-time allocation, and representative model warmup.
//!
//! This is an internal module of the `LLM` composition root. It keeps device
//! memory accounting and pre-admission validation out of model factories while
//! exposing only the two initialization operations the composition root needs.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::attention::{
    build_prefill_metadata, AttentionContext, PagedKVCache, PagedKVCacheGeometry,
};
use crate::causal_lm::CausalLM;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KvCacheAllocation {
    pub(super) free_bytes: usize,
    pub(super) total_bytes: usize,
    pub(super) bytes_per_block: usize,
    pub(super) pool_bytes: usize,
    pub(super) num_blocks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KvCachePlan {
    bytes_per_block: usize,
    pool_bytes: usize,
    num_blocks: usize,
}

impl KvCachePlan {
    fn from_budget(
        geometry: PagedKVCacheGeometry,
        available_bytes: usize,
        gpu_memory_utilization: f32,
        max_model_len: usize,
    ) -> Result<Self> {
        if !gpu_memory_utilization.is_finite()
            || !(0.0..=1.0).contains(&gpu_memory_utilization)
            || gpu_memory_utilization == 0.0
        {
            bail!(
                "gpu_memory_utilization must be finite and in (0, 1], got \
                 {gpu_memory_utilization}"
            );
        }
        if !matches!(geometry.dtype, DType::BF16 | DType::F16) {
            bail!(
                "KV cache dtype {:?} is unsupported by the CUDA attention path; \
                 expected BF16 or F16",
                geometry.dtype
            );
        }
        for (name, value) in [
            ("num_layers", geometry.num_layers),
            ("block_size", geometry.block_size),
            ("num_kv_heads", geometry.num_kv_heads),
            ("head_dim", geometry.head_dim),
        ] {
            if value == 0 {
                bail!("KV cache model geometry `{name}` must be greater than zero");
            }
        }
        let bytes_per_block = [
            2,
            geometry.num_layers,
            geometry.block_size,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.dtype.size_in_bytes(),
        ]
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| anyhow!("KV cache model geometry overflows bytes_per_block"))?;
        // GPU address spaces are far below f64's exact-integer range (2^53).
        // Keep the existing EngineOptions contract: utilization is a fraction
        // of currently available device memory, rounded down to whole bytes.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss
        )]
        let pool_bytes = (available_bytes as f64 * f64::from(gpu_memory_utilization)) as usize;
        let num_blocks = pool_bytes / bytes_per_block;
        let required_blocks = max_model_len.div_ceil(geometry.block_size);
        if num_blocks < required_blocks {
            bail!(
                "KV cache memory budget is limiting: {pool_bytes} bytes fit {num_blocks} blocks, \
                 but max_model_len {max_model_len} requires at least {required_blocks} blocks of \
                 {bytes_per_block} bytes"
            );
        }
        Ok(Self {
            bytes_per_block,
            pool_bytes,
            num_blocks,
        })
    }
}

pub(super) fn allocate_kv_cache(
    paged_kv: &Arc<Mutex<PagedKVCache>>,
    device: &Device,
    gpu_memory_utilization: f32,
    max_model_len: usize,
) -> Result<KvCacheAllocation> {
    device
        .synchronize()
        .context("synchronizing CUDA before KV cache sizing")?;
    let (free_bytes, total_bytes) = cuda_mem_info()?;
    let plan = allocate_kv_cache_with_available_memory(
        paged_kv,
        device,
        free_bytes,
        gpu_memory_utilization,
        max_model_len,
    )?;
    Ok(KvCacheAllocation {
        free_bytes,
        total_bytes,
        bytes_per_block: plan.bytes_per_block,
        pool_bytes: plan.pool_bytes,
        num_blocks: plan.num_blocks,
    })
}

fn allocate_kv_cache_with_available_memory(
    paged_kv: &Arc<Mutex<PagedKVCache>>,
    device: &Device,
    available_bytes: usize,
    gpu_memory_utilization: f32,
    max_model_len: usize,
) -> Result<KvCachePlan> {
    let mut cache = paged_kv
        .lock()
        .map_err(|error| anyhow!("paged_kv lock: {error}"))?;
    let plan = KvCachePlan::from_budget(
        cache.geometry(),
        available_bytes,
        gpu_memory_utilization,
        max_model_len,
    )?;
    let allocation_bytes = plan.num_blocks * plan.bytes_per_block;
    cache.allocate(plan.num_blocks, device).map_err(|error| {
        anyhow!(
            "KV cache GPU allocation failed for {} blocks ({allocation_bytes} bytes) within the \
             {}-byte memory budget: {error}",
            plan.num_blocks,
            plan.pool_bytes
        )
    })?;
    Ok(plan)
}

pub(super) fn warmup_model(
    model: &mut dyn CausalLM,
    attn_ctx: &AttentionContext,
    device: &Device,
    warmup_tokens: usize,
) -> Result<()> {
    if warmup_tokens == 0 {
        bail!("warmup shape is limiting: at least one token is required");
    }
    let warmup_tokens_u32 = u32::try_from(warmup_tokens)
        .map_err(|_| anyhow!("warmup shape is limiting: {warmup_tokens} tokens do not fit u32"))?;
    let positions = (0..warmup_tokens)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow!("warmup positions do not fit u32"))?;
    let slot_mapping = (0..warmup_tokens)
        .map(i64::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow!("warmup slot mapping does not fit i64"))?;

    let (cache_dtype, cache_slots) = {
        let cache = attn_ctx
            .paged_kv
            .lock()
            .map_err(|error| anyhow!("paged_kv lock: {error}"))?;
        let slots = cache
            .num_blocks()
            .checked_mul(cache.block_size())
            .ok_or_else(|| anyhow!("KV cache capacity overflows addressable slots"))?;
        (cache.dtype(), slots)
    };
    if warmup_tokens > cache_slots {
        bail!(
            "KV cache capacity is limiting: warmup requires {warmup_tokens} slots, \
             but the allocated cache has {cache_slots}"
        );
    }

    let input_ids = Tensor::zeros(warmup_tokens, DType::U32, device)?;
    let positions = Tensor::from_vec(positions, warmup_tokens, device)?;
    let warmup_meta =
        build_prefill_metadata(&[warmup_tokens_u32], &[warmup_tokens_u32], &slot_mapping);
    let previous_meta = {
        let mut metadata = attn_ctx
            .attn_meta
            .lock()
            .map_err(|error| anyhow!("attn_meta lock: {error}"))?;
        std::mem::replace(&mut *metadata, warmup_meta)
    };

    let warmup_result = (|| -> Result<()> {
        let hidden = model
            .forward(&input_ids, &positions)
            .context("warmup model forward")?;
        if hidden.dim(0)? != warmup_tokens {
            bail!(
                "warmup model shape contract failed: expected {warmup_tokens} hidden rows, got {}",
                hidden.dim(0)?
            );
        }
        if hidden.dtype() != cache_dtype {
            bail!(
                "warmup dtype contract failed: model produced {:?}, KV cache uses {cache_dtype:?}",
                hidden.dtype()
            );
        }
        let final_hidden = hidden.get(warmup_tokens - 1)?.unsqueeze(0)?;
        let logits = model
            .compute_logits(&final_hidden)
            .context("warmup logits projection")?;
        if logits.dims() != [1, model.vocab_size()] {
            bail!(
                "warmup logits shape contract failed: expected [1, {}], got {:?}",
                model.vocab_size(),
                logits.dims()
            );
        }
        device
            .synchronize()
            .context("synchronizing representative warmup")?;
        Ok(())
    })();

    {
        let mut metadata = attn_ctx
            .attn_meta
            .lock()
            .map_err(|error| anyhow!("restoring attn_meta after warmup: {error}"))?;
        *metadata = previous_meta;
    }

    warmup_result.context("representative model warmup failed")
}

/// Query CUDA free and total memory (in bytes) via the CUDA driver API.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn cuda_mem_info() -> Result<(usize, usize)> {
    use candle_core::cuda::cudarc::driver::sys;

    let mut free: usize = 0;
    let mut total: usize = 0;
    let result = unsafe { sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize) };
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("cuMemGetInfo_v2 failed with error code {}", result as i32);
    }
    Ok((free, total))
}

#[cfg(not(feature = "cuda"))]
fn cuda_mem_info() -> Result<(usize, usize)> {
    bail!("CUDA memory information requires the `cuda` feature")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::attention::{build_decode_metadata, AttnMetadata};

    fn geometry(dtype: DType) -> PagedKVCacheGeometry {
        PagedKVCacheGeometry {
            num_layers: 2,
            block_size: 4,
            num_kv_heads: 2,
            head_dim: 8,
            dtype,
        }
    }

    mod capacity {
        use super::*;

        #[test]
        fn sizes_whole_blocks_from_geometry_dtype_and_budget() {
            let plan = KvCachePlan::from_budget(geometry(DType::F16), 4096, 0.5, 8).unwrap();

            assert_eq!(plan.bytes_per_block, 512);
            assert_eq!(plan.pool_bytes, 2048);
            assert_eq!(plan.num_blocks, 4);
        }

        #[test]
        fn reports_memory_budget_when_one_max_length_request_cannot_fit() {
            let error = KvCachePlan::from_budget(geometry(DType::F16), 2048, 0.5, 9).unwrap_err();

            assert_eq!(
                error.to_string(),
                "KV cache memory budget is limiting: 1024 bytes fit 2 blocks, but \
                 max_model_len 9 requires at least 3 blocks of 512 bytes"
            );
        }

        #[test]
        fn rejects_an_unsafe_configured_memory_fraction() {
            for utilization in [0.0, -0.1, 1.1, f32::NAN] {
                let error = KvCachePlan::from_budget(geometry(DType::F16), 4096, utilization, 8)
                    .unwrap_err();
                assert!(
                    error.to_string().contains("gpu_memory_utilization"),
                    "unexpected error for {utilization}: {error}"
                );
            }
        }

        #[test]
        fn reports_dtype_when_the_cuda_attention_path_cannot_use_it() {
            for dtype in [DType::U8, DType::F32] {
                let error = KvCachePlan::from_budget(geometry(dtype), 4096, 0.5, 8).unwrap_err();

                assert_eq!(
                    error.to_string(),
                    format!(
                        "KV cache dtype {dtype:?} is unsupported by the CUDA attention path; \
                         expected BF16 or F16"
                    )
                );
            }
        }

        #[test]
        fn reports_the_zero_model_geometry_dimension() {
            let valid = geometry(DType::F16);
            let cases = [
                (
                    "num_layers",
                    PagedKVCacheGeometry {
                        num_layers: 0,
                        ..valid
                    },
                ),
                (
                    "block_size",
                    PagedKVCacheGeometry {
                        block_size: 0,
                        ..valid
                    },
                ),
                (
                    "num_kv_heads",
                    PagedKVCacheGeometry {
                        num_kv_heads: 0,
                        ..valid
                    },
                ),
                (
                    "head_dim",
                    PagedKVCacheGeometry {
                        head_dim: 0,
                        ..valid
                    },
                ),
            ];

            for (limiting_dimension, geometry) in cases {
                let error = KvCachePlan::from_budget(geometry, 4096, 0.5, 8).unwrap_err();
                assert!(
                    error.to_string().contains(limiting_dimension),
                    "unexpected error for {limiting_dimension}: {error}"
                );
            }
        }

        #[test]
        fn reports_geometry_when_bytes_per_block_overflows() {
            let overflowing = PagedKVCacheGeometry {
                num_layers: usize::MAX,
                block_size: 2,
                num_kv_heads: 2,
                head_dim: 8,
                dtype: DType::F16,
            };

            let error = KvCachePlan::from_budget(overflowing, 4096, 0.5, 8).unwrap_err();

            assert_eq!(
                error.to_string(),
                "KV cache model geometry overflows bytes_per_block"
            );
        }
    }

    #[test]
    fn allocation_happens_once_with_the_planned_capacity() {
        let device = Device::Cpu;
        let cache = Arc::new(Mutex::new(PagedKVCache::deferred(geometry(DType::F16))));

        let plan = allocate_kv_cache_with_available_memory(&cache, &device, 4096, 0.5, 8).unwrap();

        assert_eq!(plan.num_blocks, 4);
        assert_eq!(cache.lock().unwrap().allocation_count(), 1);

        let error =
            allocate_kv_cache_with_available_memory(&cache, &device, 4096, 0.5, 8).unwrap_err();
        assert!(error.to_string().contains("already allocated"));
        assert_eq!(cache.lock().unwrap().allocation_count(), 1);
    }

    #[derive(Default)]
    struct WarmupObservation {
        forward_calls: usize,
        logits_calls: usize,
        input_ids: Vec<u32>,
        positions: Vec<u32>,
        attention: Option<AttnMetadata>,
    }

    struct RecordingModel {
        device: Device,
        dtype: DType,
        attn_ctx: AttentionContext,
        observation: Arc<Mutex<WarmupObservation>>,
    }

    impl CausalLM for RecordingModel {
        fn forward(
            &mut self,
            input_ids: &Tensor,
            positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            let mut observation = self.observation.lock().unwrap();
            observation.forward_calls += 1;
            observation.input_ids = input_ids.to_vec1()?;
            observation.positions = positions.to_vec1()?;
            observation.attention = Some(self.attn_ctx.attn_meta.lock().unwrap().clone());
            Tensor::zeros((input_ids.dim(0)?, 4), self.dtype, &self.device)
        }

        fn compute_logits(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            self.observation.lock().unwrap().logits_calls += 1;
            Tensor::zeros((hidden_states.dim(0)?, 10), self.dtype, &self.device)
        }

        fn vocab_size(&self) -> usize {
            10
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    struct FailingModel {
        device: Device,
    }

    impl CausalLM for FailingModel {
        fn forward(
            &mut self,
            _input_ids: &Tensor,
            _positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            Err(candle_core::Error::Msg(
                "injected model execution failure".to_string(),
            ))
        }

        fn compute_logits(&self, _hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            unreachable!("forward failure must stop warmup")
        }

        fn vocab_size(&self) -> usize {
            10
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    #[test]
    fn warmup_executes_representative_prefill_and_restores_attention_metadata() {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(
            PagedKVCache::new(2, 4, 4, 2, 8, DType::F16, &device).unwrap(),
        ));
        let original_meta = build_decode_metadata(&[1], &[vec![3]], &[12]);
        let attn_ctx = AttentionContext {
            paged_kv,
            attn_meta: Arc::new(Mutex::new(original_meta.clone())),
        };
        let observation = Arc::new(Mutex::new(WarmupObservation::default()));
        let mut model = RecordingModel {
            device: device.clone(),
            dtype: DType::F16,
            attn_ctx: attn_ctx.clone(),
            observation: observation.clone(),
        };

        warmup_model(&mut model, &attn_ctx, &device, 6).unwrap();

        let observation = observation.lock().unwrap();
        assert_eq!(observation.forward_calls, 1);
        assert_eq!(observation.logits_calls, 1);
        assert_eq!(observation.input_ids, vec![0; 6]);
        assert_eq!(observation.positions, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            observation.attention,
            Some(build_prefill_metadata(&[6], &[6], &[0, 1, 2, 3, 4, 5]))
        );
        drop(observation);
        assert_eq!(*attn_ctx.attn_meta.lock().unwrap(), original_meta);
    }

    #[test]
    fn warmup_propagates_model_failure_and_still_restores_attention_metadata() {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(
            PagedKVCache::new(1, 4, 4, 1, 1, DType::F32, &device).unwrap(),
        ));
        let original_meta = build_decode_metadata(&[1], &[vec![2]], &[8]);
        let attn_ctx = AttentionContext {
            paged_kv,
            attn_meta: Arc::new(Mutex::new(original_meta.clone())),
        };
        let mut model = FailingModel {
            device: device.clone(),
        };

        let error = warmup_model(&mut model, &attn_ctx, &device, 2).unwrap_err();

        assert!(format!("{error:#}").contains("injected model execution failure"));
        assert!(error
            .to_string()
            .contains("representative model warmup failed"));
        assert_eq!(*attn_ctx.attn_meta.lock().unwrap(), original_meta);
    }
}
