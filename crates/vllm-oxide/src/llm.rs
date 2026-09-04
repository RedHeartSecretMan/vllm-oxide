//! Composition root — `LLM` public API surface (ADR-0004).
//!
//! The ONLY module that simultaneously imports `engine`, `models::registry`,
//! `loader`, `sampler`, and `attention`. Port of nano-vllm `llm.py` /
//! `llm_engine.py`.

mod initialization;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer as HFTokenizer;

use crate::attention::PagedKVCache;
use crate::config::Source;
use crate::engine::{
    scheduler::{
        DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS,
    },
    EngineCore, KvCacheManager, RequestOutput, Scheduler,
};
use crate::loader::model_identity::ResolvedModel;
use crate::models::registry::{resolved_factory, BuiltModel};
use crate::sampler::{Sampler, SamplingParams};

use initialization::initialize_model;

/// Construction-time configuration for `LLM::new`.
/// Mirrors nano-vllm's `Config` with v0.1 scope.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// Maximum number of tokens processed in one prefill step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of concurrently-running sequences.
    pub max_num_seqs: usize,
    /// Maximum model context length (prompt + completion).
    pub max_model_len: usize,
    /// Fraction of free GPU memory to allocate to the KV cache pool (0.0–1.0).
    pub gpu_memory_utilization: f32,
    /// Always `true` in v0.1 — CUDA graph capture is deferred to v0.2.
    pub enforce_eager: bool,
    /// Override the dtype read from `config.json`'s `torch_dtype`. `None`
    /// means "use the checkpoint's dtype" (BF16 for Qwen3).
    pub dtype: Option<DType>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_model_len: 4096,
            gpu_memory_utilization: DEFAULT_GPU_MEMORY_UTILIZATION,
            enforce_eager: true,
            dtype: None,
        }
    }
}

/// Input prompt for `LLM::generate`. Supports both natural-language text
/// and pre-tokenized fixtures (useful for golden comparison tests).
#[derive(Debug, Clone)]
pub enum Prompt {
    Text(String),
    TokenIds(Vec<u32>),
}

/// The composition root: owns the engine core, tokenizer, and shared
/// `PagedKVCache`. Constructed via `LLM::new`; generate via `LLM::generate`.
pub struct LLM {
    engine: EngineCore,
    tokenizer: HFTokenizer,
    _resolved_model: ResolvedModel,
    _paged_kv: Arc<Mutex<PagedKVCache>>,
    device: Device,
}

/// Build a registered model from one `ResolvedModel`.
pub fn build_model(source: Source, device: &Device, max_model_len: usize) -> Result<BuiltModel> {
    let resolved_model = ResolvedModel::resolve(source, None)?;
    let built = build_resolved_model(&resolved_model, device, max_model_len)?;
    built
        .attn_ctx
        .paged_kv
        .lock()
        .map_err(|error| anyhow!("paged_kv lock: {error}"))?
        .allocate(100, device)?;
    Ok(built)
}

fn build_resolved_model(
    resolved_model: &ResolvedModel,
    device: &Device,
    max_model_len: usize,
) -> Result<BuiltModel> {
    let model_factory = resolved_factory(resolved_model.config_json())?;
    model_factory(resolved_model, device, max_model_len)
}

impl LLM {
    /// Build the full inference stack — model, tokenizer, engine — from a
    /// model source (`Local` directory or `Hub` repo) and `EngineOptions`.
    ///
    /// Side effects: resolves architecture via registry, loads weights, sizes
    /// and allocates one KV cache from resolved geometry and remaining free GPU
    /// memory, runs representative prefill and paged-decode forwards with logits,
    /// and validates the CUDA device compute capability (≥ sm_89). No request is
    /// admitted until all initialization work succeeds.
    pub fn new(model: impl Into<Source>, options: EngineOptions) -> Result<Self> {
        let source: Source = model.into();

        let device = Device::cuda_if_available(0).map_err(|e| {
            anyhow!("CUDA device unavailable: {e}. vllm-oxide v0.1 requires a GPU.")
        })?;
        if !device.is_cuda() {
            bail!("vllm-oxide v0.1 requires a CUDA device. CPU-only inference is not supported.");
        }

        #[cfg(feature = "cuda")]
        validate_sm_version(&device)?;

        let resolved_model = ResolvedModel::resolve(source, options.dtype)?;
        let config_bytes = resolved_model.config_json();

        let max_pos = read_max_position_embeddings(config_bytes)?;
        if options.max_model_len > max_pos {
            bail!(
                "max_model_len ({}) exceeds the model's max_position_embeddings ({}) — \
                 RoPE positions would be out of range",
                options.max_model_len,
                max_pos,
            );
        }

        let dtype = resolved_model.dtype();

        tracing::info!(?dtype, identity = %resolved_model.identity(), "loading model");

        let tokenizer = resolved_model.load_tokenizer()?;
        let special_token_ids = resolved_model.special_token_ids(&tokenizer)?;
        tracing::debug!(
            config = %resolved_model.config_path().display(),
            tokenizer = %resolved_model.tokenizer_path().display(),
            bos_token_id = ?special_token_ids.bos(),
            pad_token_id = ?special_token_ids.pad(),
            "validated resolved model artifact contract"
        );

        let BuiltModel {
            mut model,
            attn_ctx,
        } = build_resolved_model(&resolved_model, &device, options.max_model_len)?;

        let cache_allocation = initialize_model(model.as_mut(), &attn_ctx, &device, &options)?;
        let num_gpu_blocks = cache_allocation.num_blocks;

        tracing::info!(
            free_mb = cache_allocation.free_bytes / (1024 * 1024),
            total_mb = cache_allocation.total_bytes / (1024 * 1024),
            kv_pool_mb = cache_allocation.pool_bytes / (1024 * 1024),
            bytes_per_block = cache_allocation.bytes_per_block,
            num_gpu_blocks,
            warmup_prefill_tokens = cache_allocation.warmup_prefill_tokens,
            "allocated and warmed KV cache pool"
        );

        let scheduler = Scheduler::new_with_eos_token_ids(
            options.max_num_batched_tokens,
            options.max_num_seqs,
            options.gpu_memory_utilization,
            special_token_ids.eos().to_vec(),
        );

        let kv_cache_manager = KvCacheManager::new(num_gpu_blocks, 256, attn_ctx.paged_kv.clone());

        let sampler = Sampler::new_with_seed(0);

        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            sampler,
            attn_ctx.clone(),
            device.clone(),
        );

        tracing::info!(
            vocab_size = tokenizer.get_vocab_size(true),
            "tokenizer loaded"
        );

        Ok(Self {
            engine,
            tokenizer,
            _resolved_model: resolved_model,
            _paged_kv: attn_ctx.paged_kv,
            device,
        })
    }

    /// Run inference on a batch of prompts with per-prompt sampling parameters.
    ///
    /// Text prompts (`Prompt::Text`) are tokenized using the loaded tokenizer;
    /// pre-tokenized prompts (`Prompt::TokenIds`) are used directly. All
    /// prompts in the batch are continuous-batched together.
    ///
    /// Returns one `RequestOutput` per prompt, in the same order, with both
    /// decoded text and raw token IDs.
    pub fn generate(
        &mut self,
        prompts: &[Prompt],
        sampling_params: &[SamplingParams],
    ) -> Result<Vec<RequestOutput>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        if sampling_params.len() != prompts.len() {
            bail!(
                "generate: expected {} sampling_params, got {}",
                prompts.len(),
                sampling_params.len(),
            );
        }

        let mut prompt_lens: Vec<usize> = Vec::with_capacity(prompts.len());
        let mut request_ids = Vec::with_capacity(prompts.len());
        for (prompt, params) in prompts.iter().zip(sampling_params.iter()) {
            let token_ids = tokenize_prompt(prompt, &self.tokenizer)?;
            let len = token_ids.len();
            prompt_lens.push(len);
            let request_id = self.engine.add_request(token_ids, params.clone());
            request_ids.push(request_id);
        }

        let start = Instant::now();
        let mut step_count: usize = 0;

        let mut completed_outputs = Vec::with_capacity(prompts.len());

        while self.engine.is_running() {
            let outputs = self.engine.step()?;
            step_count += 1;

            completed_outputs.extend(outputs);
        }

        let elapsed = start.elapsed();
        let mut results = order_request_outputs(&request_ids, completed_outputs)?;
        for output in &mut results {
            output.text = self
                .tokenizer
                .decode(&output.token_ids, true)
                .map_err(|e| anyhow!("detokenization failed: {e}"))?;
        }
        let total_tokens: usize = prompt_lens.iter().sum::<usize>()
            + results.iter().map(|o| o.token_ids.len()).sum::<usize>();

        if total_tokens > 0 {
            // total_tokens ≤ max_model_len (4096 default) ≪ 2^53 f64 mantissa
            #[allow(clippy::cast_precision_loss)]
            let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
            tracing::info!(
                total_tokens,
                tok_per_sec = format!("{tok_per_sec:.1}"),
                elapsed_ms = elapsed.as_millis(),
                steps = step_count,
                "generate complete"
            );
        }

        Ok(results)
    }

    /// Run greedy generation and return the pre-sampling logits at every step.
    ///
    /// This runs the full generation loop (prefill + `max_tokens` decode steps)
    /// with greedy sampling (temperature=0), collecting the raw logits tensor
    /// `[batch, vocab_size]` before each sampling step. The logits are stacked
    /// into a single tensor of shape `[total_steps, vocab_size]` (FP32).
    ///
    /// Used for L2 golden comparison (#23). The caller controls `max_tokens`
    /// to match the fixture's `num_tokens`.
    pub fn generate_logits(
        &mut self,
        prompt: &Prompt,
        max_tokens: usize,
    ) -> Result<candle_core::Tensor> {
        let token_ids = tokenize_prompt(prompt, &self.tokenizer)?;
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens,
            ignore_eos: true,
            ..SamplingParams::default()
        };
        self.engine.add_request(token_ids, params);

        let mut logits_list: Vec<candle_core::Tensor> = Vec::new();

        while self.engine.is_running() {
            let (_outputs, step_logits) = self.engine.step_with_logits()?;
            if step_logits.dims().iter().all(|&d| d == 0) {
                continue;
            }
            logits_list.push(step_logits);
        }

        if logits_list.is_empty() {
            return Ok(candle_core::Tensor::zeros(
                (0, 0),
                DType::F32,
                &self.device,
            )?);
        }

        let refs: Vec<&candle_core::Tensor> = logits_list.iter().collect();
        Ok(candle_core::Tensor::cat(&refs, 0)?)
    }
}

fn order_request_outputs(
    request_ids: &[usize],
    completed_outputs: Vec<RequestOutput>,
) -> Result<Vec<RequestOutput>> {
    let mut input_positions = HashMap::with_capacity(request_ids.len());
    for (input_position, &request_id) in request_ids.iter().enumerate() {
        if input_positions.insert(request_id, input_position).is_some() {
            bail!("generate: request {request_id} was accepted more than once");
        }
    }

    let mut ordered_outputs = vec![None; request_ids.len()];
    for output in completed_outputs {
        let input_position = input_positions
            .get(&output.request_id)
            .copied()
            .ok_or_else(|| anyhow!("generate: completed unknown request {}", output.request_id))?;
        if ordered_outputs[input_position].replace(output).is_some() {
            bail!(
                "generate: request {} completed more than once",
                request_ids[input_position]
            );
        }
    }

    ordered_outputs
        .into_iter()
        .enumerate()
        .map(|(input_position, output)| {
            output.ok_or_else(|| {
                anyhow!(
                    "generate: request {} at input position {input_position} did not complete",
                    request_ids[input_position]
                )
            })
        })
        .collect()
}

impl Drop for LLM {
    fn drop(&mut self) {
        if self.device.is_cuda() {
            if let Err(e) = self.device.synchronize() {
                tracing::warn!("CUDA synchronize on drop failed: {e}");
            }
        }
    }
}

/// Tokenize a `Prompt` into `Vec<u32>` token IDs.
fn tokenize_prompt(prompt: &Prompt, tokenizer: &HFTokenizer) -> Result<Vec<u32>> {
    match prompt {
        Prompt::Text(text) => {
            let encoding = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
            Ok(encoding.get_ids().to_vec())
        }
        Prompt::TokenIds(ids) => Ok(ids.clone()),
    }
}

/// Extract `max_position_embeddings` from config.json bytes.
fn read_max_position_embeddings(config_json: &[u8]) -> Result<usize> {
    #[derive(serde::Deserialize)]
    struct PosCheck {
        max_position_embeddings: Option<usize>,
    }
    let parsed: PosCheck = serde_json::from_slice(config_json)
        .context("parsing config.json for max_position_embeddings")?;
    parsed
        .max_position_embeddings
        .ok_or_else(|| anyhow!("config.json has no `max_position_embeddings` field"))
}

/// Validate CUDA device compute capability ≥ sm_89.
///
/// Flash-attention paged kernels and Qwen3 BF16 matmuls require Hopper (sm_90)
/// or Ada Lovelace (sm_89) or newer. Older GPUs will hit opaque kernel launch
/// failures — this check gives a clear error before loading weights.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn validate_sm_version(_device: &Device) -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let mut cu_device: sys::CUdevice = 0;
    let result = unsafe { sys::cuDeviceGet(&mut cu_device, 0) };
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!(
            "cuDeviceGet(0) failed with error code {}. Is the CUDA driver installed?",
            result as i32
        );
    }

    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    let result = unsafe { sys::cuDeviceComputeCapability(&mut major, &mut minor, cu_device) };
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!(
            "cuDeviceComputeCapability failed with error code {}. \
             Cannot determine GPU compute capability.",
            result as i32
        );
    }

    let sm = major * 10 + minor;
    if sm < 89 {
        bail!(
            "GPU compute capability sm_{major}{minor} (sm_{sm}) is below the minimum \
             required sm_89. vllm-oxide v0.1 flash-attention kernels and Qwen3 BF16 \
             matmuls require Ada Lovelace (sm_89) or Hopper (sm_90) architecture. \
             Supported GPUs: RTX 40-series (Ada), H100/H200 (Hopper), and newer."
        );
    }

    tracing::info!(sm_version = sm, "CUDA device compute capability validated");
    Ok(())
}

/// Stub: SM validation is only meaningful with `--features cuda`. Without
/// CUDA features, the program would have already failed at
/// `Device::cuda_if_available(0)`.
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn validate_sm_version(_device: &Device) {}

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
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::attention::{build_prefill_metadata, AttentionContext, PagedKVCacheGeometry};
    use crate::causal_lm::CausalLM;
    use crate::engine::sequence::BLOCK_SIZE;
    use candle_core::Tensor;
    use tokenizers::models::bpe::{Vocab, BPE};

    struct MockModel {
        device: Device,
        fail_forward: bool,
    }

    struct CacheAwareModel {
        device: Device,
        attn_ctx: AttentionContext,
        physical_tokens: Arc<Mutex<HashMap<usize, u32>>>,
    }

    impl CausalLM for CacheAwareModel {
        #[allow(clippy::cast_precision_loss)]
        fn forward(
            &mut self,
            input_ids: &Tensor,
            positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            let input_ids = input_ids.to_vec1::<u32>()?;
            let positions = positions.to_vec1::<u32>()?;
            let metadata = self.attn_ctx.attn_meta.lock().unwrap().clone();
            if input_ids.len() != positions.len() || input_ids.len() != metadata.slot_mapping.len()
            {
                candle_core::bail!("cache-aware model received inconsistent step metadata");
            }
            let block_size = self.attn_ctx.paged_kv.lock().unwrap().block_size();
            let mut physical_tokens = self.physical_tokens.lock().unwrap();
            for ((&token_id, &position), &slot) in
                input_ids.iter().zip(&positions).zip(&metadata.slot_mapping)
            {
                let slot = usize::try_from(slot).map_err(|_| {
                    candle_core::Error::Msg("cache-aware model received a negative slot".into())
                })?;
                physical_tokens.insert(slot, token_id.wrapping_add(position));
            }

            let hidden = if metadata.is_prefill {
                input_ids
                    .iter()
                    .zip(&positions)
                    .map(|(&token_id, &position)| token_id.wrapping_add(position) as f32)
                    .collect::<Vec<_>>()
            } else {
                let context_len = metadata
                    .cu_seqlens_k
                    .last()
                    .copied()
                    .ok_or_else(|| candle_core::Error::Msg("missing decode context".into()))?
                    as usize;
                let block_table = metadata
                    .block_table
                    .first()
                    .ok_or_else(|| candle_core::Error::Msg("missing decode block table".into()))?;
                let mut fingerprint = 0u32;
                for logical_position in 0..context_len {
                    let block_id =
                        *block_table
                            .get(logical_position / block_size)
                            .ok_or_else(|| {
                                candle_core::Error::Msg("decode block table is too short".into())
                            })?;
                    let block_id = usize::try_from(block_id).map_err(|_| {
                        candle_core::Error::Msg("decode block id is negative".into())
                    })?;
                    let physical_slot = block_id
                        .checked_mul(block_size)
                        .and_then(|base| base.checked_add(logical_position % block_size))
                        .ok_or_else(|| {
                            candle_core::Error::Msg("decode physical slot overflow".into())
                        })?;
                    fingerprint = fingerprint.wrapping_add(
                        *physical_tokens.get(&physical_slot).ok_or_else(|| {
                            candle_core::Error::Msg("decode read an unwritten cache slot".into())
                        })?,
                    );
                }
                vec![(fingerprint % 100) as f32]
            };
            Tensor::from_vec(hidden, (input_ids.len(), 1), &self.device)?.to_dtype(DType::F16)
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        fn compute_logits(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            let rows = hidden_states.dim(0)?;
            let hidden = hidden_states
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut logits = vec![-100.0_f32; rows * 100];
            for (row, value) in hidden.into_iter().enumerate() {
                let target = value.round() as usize % 100;
                logits[row * 100 + target] = 100.0;
            }
            Tensor::from_vec(logits, (rows, 100), &self.device)
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    impl CausalLM for MockModel {
        fn forward(
            &mut self,
            input_ids: &Tensor,
            _positions: &Tensor,
        ) -> candle_core::Result<Tensor> {
            if self.fail_forward {
                return Err(candle_core::Error::Msg(
                    "injected model execution failure".to_string(),
                ));
            }
            Tensor::zeros((input_ids.dim(0)?, 1), DType::F32, &self.device)
        }

        fn compute_logits(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            let rows = hidden_states.dim(0)?;
            let mut logits = vec![-100.0_f32; rows * 100];
            for row in 0..rows {
                logits[row * 100 + 42] = 100.0;
            }
            Tensor::from_vec(logits, (rows, 100), &self.device)
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn finish_test_llm(
        engine: EngineCore,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        device: Device,
    ) -> LLM {
        let vocab = (0..100_u32)
            .map(|token_id| (format!("token-{token_id}"), token_id))
            .collect::<Vocab>();
        let tokenizer = HFTokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, Vec::new())
                .build()
                .unwrap(),
        );
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            br#"{"torch_dtype":"bfloat16"}"#,
        )
        .unwrap();
        tokenizer
            .save(model_dir.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(model_dir.path().join("model.safetensors"), b"").unwrap();
        let resolved_model =
            ResolvedModel::resolve(Source::Local(model_dir.path().to_path_buf()), None).unwrap();

        LLM {
            engine,
            tokenizer,
            _resolved_model: resolved_model,
            _paged_kv: paged_kv,
            device,
        }
    }

    fn test_llm_with_configuration(fail_forward: bool) -> LLM {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(
            PagedKVCache::new(1, 32, BLOCK_SIZE, 1, 1, DType::F32, &device).unwrap(),
        ));
        let attn_ctx = AttentionContext {
            paged_kv: paged_kv.clone(),
            attn_meta: Arc::new(Mutex::new(build_prefill_metadata(&[], &[], &[]))),
        };
        let scheduler = Scheduler::new(16, 16, 0.9);
        let kv_cache_manager = KvCacheManager::new(32, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        let model: Box<dyn CausalLM> = Box::new(MockModel {
            device: device.clone(),
            fail_forward,
        });
        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            Sampler::new_with_seed(0),
            attn_ctx,
            device.clone(),
        );
        finish_test_llm(engine, paged_kv, device)
    }

    fn test_llm_with_forward_failure(fail_forward: bool) -> LLM {
        test_llm_with_configuration(fail_forward)
    }

    fn test_llm() -> LLM {
        test_llm_with_forward_failure(false)
    }

    fn cache_aware_test_llm(run_warmup: bool) -> LLM {
        let device = Device::Cpu;
        let paged_kv = Arc::new(Mutex::new(PagedKVCache::deferred(PagedKVCacheGeometry {
            num_layers: 1,
            block_size: BLOCK_SIZE,
            num_kv_heads: 1,
            head_dim: 1,
            dtype: DType::F16,
        })));
        let attn_ctx = AttentionContext {
            paged_kv: paged_kv.clone(),
            attn_meta: Arc::new(Mutex::new(build_prefill_metadata(&[], &[], &[]))),
        };
        let physical_tokens = Arc::new(Mutex::new(HashMap::new()));
        let mut model: Box<dyn CausalLM> = Box::new(CacheAwareModel {
            device: device.clone(),
            attn_ctx: attn_ctx.clone(),
            physical_tokens: physical_tokens.clone(),
        });
        let options = EngineOptions {
            max_num_batched_tokens: 2,
            max_num_seqs: 1,
            max_model_len: 8,
            gpu_memory_utilization: 0.5,
            enforce_eager: true,
            dtype: Some(DType::F16),
        };

        let num_blocks = if run_warmup {
            let allocation = initialization::initialize_model_with_available_memory(
                model.as_mut(),
                &attn_ctx,
                &device,
                initialization::DeviceMemory::new(4096, 8192),
                &options,
            )
            .unwrap();
            assert_eq!(physical_tokens.lock().unwrap().len(), 3);
            allocation.num_blocks
        } else {
            paged_kv.lock().unwrap().allocate(2, &device).unwrap();
            assert!(physical_tokens.lock().unwrap().is_empty());
            2
        };
        assert_eq!(paged_kv.lock().unwrap().allocation_count(), 1);

        let scheduler = Scheduler::new(
            options.max_num_batched_tokens,
            options.max_num_seqs,
            options.gpu_memory_utilization,
        );
        let kv_cache_manager =
            KvCacheManager::new(num_blocks, BLOCK_SIZE, attn_ctx.paged_kv.clone());
        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            Sampler::new_with_seed(0),
            attn_ctx,
            device.clone(),
        );
        finish_test_llm(engine, paged_kv, device)
    }

    mod engine_options {
        use super::*;

        #[test]
        fn default_has_sane_values() {
            let opts = EngineOptions::default();
            assert!(opts.max_num_batched_tokens > 0);
            assert!(opts.max_num_seqs > 0);
            assert!(opts.max_model_len > 0);
            assert!(opts.gpu_memory_utilization > 0.0 && opts.gpu_memory_utilization <= 1.0);
            assert!(opts.enforce_eager);
            assert!(opts.dtype.is_none());
        }

        #[test]
        fn custom_override() {
            let opts = EngineOptions {
                max_num_batched_tokens: 100,
                max_num_seqs: 10,
                max_model_len: 2048,
                gpu_memory_utilization: 0.5,
                enforce_eager: true,
                dtype: Some(DType::F16),
            };
            assert_eq!(opts.max_num_batched_tokens, 100);
            assert_eq!(opts.max_num_seqs, 10);
            assert_eq!(opts.max_model_len, 2048);
            assert_eq!(opts.gpu_memory_utilization, 0.5);
            assert_eq!(opts.dtype, Some(DType::F16));
        }
    }

    mod model_warmup {
        use super::*;

        #[test]
        fn first_generation_matches_the_non_mutating_reference_scenario() {
            let mut warmed = cache_aware_test_llm(true);
            let mut reference = cache_aware_test_llm(false);
            let prompts = [Prompt::TokenIds(vec![1, 2, 3])];
            let params = [SamplingParams {
                temperature: 0.0,
                max_tokens: 3,
                ignore_eos: true,
                ..SamplingParams::default()
            }];

            let warmed_output = warmed.generate(&prompts, &params).unwrap();
            let reference_output = reference.generate(&prompts, &params).unwrap();

            assert_eq!(warmed_output.len(), 1);
            assert_eq!(warmed_output[0].request_id, 0);
            assert_eq!(warmed_output[0].token_ids, vec![5, 17, 38]);
            assert_eq!(warmed_output[0].request_id, reference_output[0].request_id);
            assert_eq!(warmed_output[0].token_ids, reference_output[0].token_ids);
            assert_eq!(warmed_output[0].text, reference_output[0].text);
            assert_eq!(warmed_output[0].finished, reference_output[0].finished);
        }
    }

    mod prompt {
        use super::*;

        #[test]
        fn text_variant_carries_string() {
            match Prompt::Text("hello".into()) {
                Prompt::Text(s) => assert_eq!(s, "hello"),
                _ => panic!("expected Text variant"),
            }
        }

        #[test]
        fn token_ids_variant_carries_vec() {
            match Prompt::TokenIds(vec![1, 2, 3]) {
                Prompt::TokenIds(ids) => assert_eq!(ids, vec![1, 2, 3]),
                _ => panic!("expected TokenIds variant"),
            }
        }

        #[test]
        fn token_ids_path_bypasses_tokenization_entirely() {
            let ids = vec![42, 99, 151_645];
            let prompt = Prompt::TokenIds(ids.clone());
            match &prompt {
                Prompt::TokenIds(existing) => assert_eq!(existing, &ids),
                Prompt::Text(_) => panic!("expected TokenIds"),
            }
        }
    }

    mod read_max_position_embeddings {
        use super::*;

        #[test]
        fn parses_qwen3_style_config() {
            let json = br#"{"max_position_embeddings": 40960, "hidden_size": 1024}"#;
            assert_eq!(read_max_position_embeddings(json).unwrap(), 40960);
        }

        #[test]
        fn parses_small_config() {
            let json = br#"{"max_position_embeddings": 128}"#;
            assert_eq!(read_max_position_embeddings(json).unwrap(), 128);
        }

        #[test]
        fn missing_field_errors() {
            let json = br#"{"hidden_size": 1024}"#;
            assert!(read_max_position_embeddings(json).is_err());
        }

        #[test]
        fn null_field_errors() {
            let json = br#"{"max_position_embeddings": null}"#;
            assert!(read_max_position_embeddings(json).is_err());
        }

        #[test]
        fn empty_json_errors() {
            let json = br#"{}"#;
            assert!(read_max_position_embeddings(json).is_err());
        }
    }

    mod request_output {
        use crate::engine::RequestOutput;

        #[test]
        fn text_field_defaults_empty() {
            let output = RequestOutput {
                request_id: 0,
                token_ids: vec![1, 2, 3],
                text: String::new(),
                finished: true,
            };
            assert!(output.text.is_empty());
            assert_eq!(output.token_ids.len(), 3);
            assert!(output.finished);
        }

        #[test]
        fn text_field_can_be_populated() {
            let output = RequestOutput {
                request_id: 1,
                token_ids: vec![42],
                text: "hello".into(),
                finished: true,
            };
            assert_eq!(output.text, "hello");
            assert_eq!(output.request_id, 1);
        }
    }

    mod order_request_outputs {
        use super::*;

        fn output(request_id: usize, token_id: u32) -> RequestOutput {
            RequestOutput {
                request_id,
                token_ids: vec![token_id],
                text: String::new(),
                finished: true,
            }
        }

        #[test]
        fn restores_input_order_from_reversed_completion_order() {
            let outputs = order_request_outputs(
                &[10, 20, 30],
                vec![output(30, 3), output(20, 2), output(10, 1)],
            )
            .unwrap();

            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![10, 20, 30]
            );
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.token_ids[0])
                    .collect::<Vec<_>>(),
                vec![1, 2, 3]
            );
        }

        #[test]
        fn rejects_an_unknown_completion() {
            let error = order_request_outputs(&[10], vec![output(99, 1)]).unwrap_err();

            assert_eq!(error.to_string(), "generate: completed unknown request 99");
        }

        #[test]
        fn rejects_a_duplicate_completion() {
            let error =
                order_request_outputs(&[10], vec![output(10, 1), output(10, 2)]).unwrap_err();

            assert_eq!(
                error.to_string(),
                "generate: request 10 completed more than once"
            );
        }

        #[test]
        fn rejects_a_missing_completion() {
            let error = order_request_outputs(&[10, 20], vec![output(10, 1)]).unwrap_err();

            assert_eq!(
                error.to_string(),
                "generate: request 20 at input position 1 did not complete"
            );
        }
    }

    mod generate_empty_batch {
        use super::*;

        #[test]
        fn engine_options_defaults_match_scheduler_constants() {
            let opts = EngineOptions::default();
            assert_eq!(
                opts.max_num_batched_tokens,
                crate::engine::scheduler::DEFAULT_MAX_NUM_BATCHED_TOKENS
            );
            assert_eq!(
                opts.max_num_seqs,
                crate::engine::scheduler::DEFAULT_MAX_NUM_SEQS
            );
            assert_eq!(
                opts.gpu_memory_utilization,
                crate::engine::scheduler::DEFAULT_GPU_MEMORY_UTILIZATION
            );
        }
    }

    mod generate_repeated_calls {
        use super::*;

        #[test]
        fn second_call_returns_a_real_output() {
            let mut llm = test_llm();
            let prompts = [Prompt::TokenIds(vec![1])];
            let params = [SamplingParams {
                max_tokens: 1,
                ..SamplingParams::default()
            }];

            let first = llm.generate(&prompts, &params).unwrap();
            assert_eq!(first[0].token_ids, vec![42]);

            let second = llm.generate(&prompts, &params).unwrap();
            assert_eq!(second[0].token_ids, vec![42]);
            assert!(second[0].finished);
        }

        #[test]
        fn one_instance_returns_complete_outputs_for_one_hundred_calls() {
            let mut llm = test_llm();
            let prompts = [Prompt::TokenIds(vec![1])];
            let params = [SamplingParams {
                max_tokens: 1,
                ..SamplingParams::default()
            }];

            for expected_request_id in 0..100 {
                let outputs = llm.generate(&prompts, &params).unwrap();
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].request_id, expected_request_id);
                assert_eq!(outputs[0].token_ids, vec![42]);
                assert!(outputs[0].finished);
            }
        }
    }

    mod generate_batch_order {
        use super::*;

        #[test]
        fn mixed_batch_stays_in_input_order_when_requests_finish_out_of_order() {
            let mut llm = test_llm();
            let warmup_prompts = [Prompt::TokenIds(vec![99])];
            let warmup_params = [SamplingParams {
                max_tokens: 1,
                ..SamplingParams::default()
            }];
            llm.generate(&warmup_prompts, &warmup_params).unwrap();

            let prompts = [
                Prompt::TokenIds(vec![1]),
                Prompt::TokenIds(vec![2, 3, 4]),
                Prompt::TokenIds(vec![5, 6]),
            ];
            let params = [3, 1, 2].map(|max_tokens| SamplingParams {
                max_tokens,
                ..SamplingParams::default()
            });

            let outputs = llm.generate(&prompts, &params).unwrap();

            assert_eq!(outputs.len(), prompts.len());
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.request_id)
                    .collect::<Vec<_>>(),
                vec![1, 2, 3]
            );
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.token_ids.len())
                    .collect::<Vec<_>>(),
                vec![3, 1, 2]
            );
            assert!(outputs.iter().all(|output| output.finished));
        }
    }

    mod generate_errors {
        use super::*;

        #[test]
        fn model_execution_failure_is_returned_as_an_error() {
            let mut llm = test_llm_with_forward_failure(true);
            let error = llm
                .generate(
                    &[Prompt::TokenIds(vec![1])],
                    &[SamplingParams {
                        max_tokens: 1,
                        ..SamplingParams::default()
                    }],
                )
                .unwrap_err();

            assert!(error
                .to_string()
                .contains("injected model execution failure"));
        }
    }
}
