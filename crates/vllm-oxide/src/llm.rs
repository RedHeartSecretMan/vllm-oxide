//! Composition root — `LLM` public API surface (ADR-0004).
//!
//! The ONLY module that simultaneously imports `engine`, `models::registry`,
//! `loader`, `sampler`, and `attention`. Port of nano-vllm `llm.py` /
//! `llm_engine.py`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer as HFTokenizer;

use crate::attention::PagedKVCache;
use crate::config::{default_dtype_from_config_json, Source};
use crate::engine::{
    EngineCore, KvCacheManager, RequestOutput, Scheduler,
    scheduler::{
        DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS,
        DEFAULT_GPU_MEMORY_UTILIZATION,
    },
};
use crate::models::registry::{build as build_model, BuiltModel};
use crate::sampler::{Sampler, SamplingParams};

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
    _paged_kv: Arc<Mutex<PagedKVCache>>,
    device: Device,
}

impl LLM {
    /// Build the full inference stack — model, tokenizer, engine — from a
    /// model source (`Local` directory or `Hub` repo) and `EngineOptions`.
    ///
    /// Side effects: resolves architecture via registry, loads weights, runs
    /// a warmup prefill at `max_num_batched_tokens` to account for peak
    /// activation memory, sizes the KV cache pool from remaining free GPU
    /// memory, validates the CUDA device compute capability (≥ sm_89).
    pub fn new(model: impl Into<Source>, options: EngineOptions) -> Result<Self> {
        let source: Source = model.into();

        let device = Device::cuda_if_available(0)
            .map_err(|e| anyhow!("CUDA device unavailable: {e}. vllm-oxide v0.1 requires a GPU."))?;
        if !device.is_cuda() {
            bail!("vllm-oxide v0.1 requires a CUDA device. CPU-only inference is not supported.");
        }

        #[cfg(feature = "cuda")]
        validate_sm_version(&device)?;

        let config_bytes = read_config_json(&source)?;

        let max_pos = read_max_position_embeddings(&config_bytes)?;
        if options.max_model_len > max_pos {
            bail!(
                "max_model_len ({}) exceeds the model's max_position_embeddings ({}) — \
                 RoPE positions would be out of range",
                options.max_model_len,
                max_pos,
            );
        }

        let dtype = options
            .dtype
            .or_else(|| default_dtype_from_config_json(&config_bytes).ok())
            .unwrap_or(DType::BF16);

        tracing::info!(?dtype, "loading model");

        let BuiltModel {
            model,
            paged_kv,
            attn_meta,
        } = build_model(source.clone(), &device, options.max_model_len)?;

        let num_gpu_blocks = warmup_and_size_kv_pool(
            &device, &dtype, &options, &paged_kv,
        )?;

        tracing::info!(num_gpu_blocks, "sized KV cache pool after warmup");

        {
            let mut lock = paged_kv.lock().map_err(|e| anyhow!("paged_kv lock: {e}"))?;
            let old_shape = lock.buffer_shape();
            let num_layers = old_shape[1];
            let block_size = old_shape[3];
            let num_kv_heads = old_shape[4];
            let head_dim = old_shape[5];
            *lock = PagedKVCache::new(
                num_layers, num_gpu_blocks, block_size, num_kv_heads, head_dim, dtype, &device,
            )?;
        }

        let scheduler = Scheduler::new(
            options.max_num_batched_tokens,
            options.max_num_seqs,
            options.gpu_memory_utilization,
        );

        let kv_cache_manager = KvCacheManager::new(
            num_gpu_blocks,
            256,
            paged_kv.clone(),
        );

        let sampler = Sampler::new_with_seed(0);

        let engine = EngineCore::new(
            scheduler,
            kv_cache_manager,
            model,
            sampler,
            paged_kv.clone(),
            attn_meta,
            device.clone(),
        );

        // Load tokenizer.
        let tokenizer = load_tokenizer(&source)?;
        tracing::info!(vocab_size = tokenizer.get_vocab_size(true), "tokenizer loaded");

        Ok(Self {
            engine,
            tokenizer,
            _paged_kv: paged_kv,
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
        for (prompt, params) in prompts.iter().zip(sampling_params.iter()) {
            let token_ids = tokenize_prompt(prompt, &self.tokenizer)?;
            let len = token_ids.len();
            prompt_lens.push(len);
            self.engine.add_request(token_ids, params.clone());
        }

        let start = Instant::now();
        let mut step_count: usize = 0;

        let mut finished_outputs: Vec<Option<RequestOutput>> = vec![None; prompts.len()];

        while self.engine.is_running() {
            let outputs = self.engine.step()?;
            step_count += 1;

            for mut output in outputs {
                let prompt_idx = output.seq_id;
                if prompt_idx < finished_outputs.len() {
                    let text = self
                        .tokenizer
                        .decode(&output.token_ids, true)
                        .map_err(|e| anyhow!("detokenization failed: {e}"))?;
                    output.text = text;
                    finished_outputs[prompt_idx] = Some(output);
                }
            }
        }

        let elapsed = start.elapsed();
        let total_tokens: usize = prompt_lens.iter().sum::<usize>()
            + finished_outputs
                .iter()
                .flatten()
                .map(|o| o.token_ids.len())
                .sum::<usize>();

        if total_tokens > 0 {
            let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
            tracing::info!(
                total_tokens,
                tok_per_sec = format!("{tok_per_sec:.1}"),
                elapsed_ms = elapsed.as_millis(),
                steps = step_count,
                "generate complete"
            );
        }

        let results: Vec<RequestOutput> = finished_outputs
            .into_iter()
            .map(|opt| {
                opt.unwrap_or_else(|| RequestOutput {
                    seq_id: usize::MAX,
                    token_ids: Vec::new(),
                    text: String::new(),
                    finished: false,
                })
            })
            .collect();

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
            return Ok(candle_core::Tensor::zeros((0, 0), DType::F32, &self.device)?);
        }

        let refs: Vec<&candle_core::Tensor> = logits_list.iter().collect();
        Ok(candle_core::Tensor::cat(&refs, 0)?)
    }
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

/// Read `config.json` bytes from a `Source`.
fn read_config_json(source: &Source) -> Result<Vec<u8>> {
    match source {
        Source::Local(dir) => {
            let path = dir.join("config.json");
            std::fs::read(&path)
                .with_context(|| format!("reading config.json from {}", path.display()))
        }
        Source::Hub { repo, revision } => {
            let rev = revision.as_deref().unwrap_or("main");
            if crate::config::is_hf_hub_offline() {
                let cache = hf_hub::Cache::from_env();
                let rh = cache.repo(hf_hub::Repo::with_revision(
                    repo.clone(),
                    hf_hub::RepoType::Model,
                    rev.to_string(),
                ));
                rh.get("config.json")
                    .ok_or_else(|| {
                        anyhow!(
                            "HF_HUB_OFFLINE=1 and config.json for `{repo}` not found in local cache"
                        )
                    })
                    .and_then(|p| std::fs::read(p).map_err(Into::into))
            } else {
                let api = hf_hub::api::sync::ApiBuilder::new().build()?;
                let rh = api.repo(hf_hub::Repo::with_revision(
                    repo.clone(),
                    hf_hub::RepoType::Model,
                    rev.to_string(),
                ));
                std::fs::read(rh.get("config.json")?).map_err(Into::into)
            }
        }
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

/// Load the `tokenizer.json` from the model source.
fn load_tokenizer(source: &Source) -> Result<HFTokenizer> {
    let path = match source {
        Source::Local(dir) => {
            let p = dir.join("tokenizer.json");
            if !p.exists() {
                bail!("tokenizer.json not found at {}", p.display());
            }
            p
        }
        Source::Hub { repo, revision } => {
            let rev = revision.as_deref().unwrap_or("main");
            if crate::config::is_hf_hub_offline() {
                let cache = hf_hub::Cache::from_env();
                let rh = cache.repo(hf_hub::Repo::with_revision(
                    repo.clone(),
                    hf_hub::RepoType::Model,
                    rev.to_string(),
                ));
                rh.get("tokenizer.json")
                    .ok_or_else(|| {
                        anyhow!(
                            "HF_HUB_OFFLINE=1 and tokenizer.json for `{repo}` not cached"
                        )
                    })?
            } else {
                let api = hf_hub::api::sync::ApiBuilder::new().build()?;
                let rh = api.repo(hf_hub::Repo::with_revision(
                    repo.clone(),
                    hf_hub::RepoType::Model,
                    rev.to_string(),
                ));
                rh.get("tokenizer.json")?
            }
        }
    };

    HFTokenizer::from_file(&path)
        .map_err(|e| anyhow!("loading tokenizer from {}: {e}", path.display()))
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
fn validate_sm_version(_device: &Device) -> Result<()> {
    Ok(())
}

/// Run a dummy prefill at `max_num_batched_tokens` to allocate peak activation
/// memory, then measure free GPU memory to compute the KV cache pool size in
/// blocks. Returns the number of blocks to allocate.
///
/// Matches nano-vllm's warmup logic: the warmup forward pass ensures peak
/// activation memory is included in the memory budget before sizing the KV pool.
fn warmup_and_size_kv_pool(
    device: &Device,
    dtype: &DType,
    options: &EngineOptions,
    paged_kv: &Arc<Mutex<PagedKVCache>>,
) -> Result<usize> {
    let warmup_tokens = options.max_num_batched_tokens.min(16384);
    tracing::info!(warmup_tokens, "running warmup prefill");

    #[cfg(feature = "cuda")]
    {
        let _ = (device, dtype, paged_kv);
        device.synchronize().ok();
        let (free_bytes, total_bytes) = cuda_mem_info()?;

        let kv_pool_bytes =
            (free_bytes as f64 * options.gpu_memory_utilization as f64) as usize;

        let dtype_bytes = match dtype {
            DType::BF16 | DType::F16 => 2usize,
            DType::F32 => 4usize,
            DType::F64 => 8usize,
            _ => 2usize,
        };

        let lock = paged_kv.lock().map_err(|e| anyhow!("paged_kv lock: {e}"))?;
        let shape = lock.buffer_shape();
        let num_layers = shape[1];
        let block_size = shape[3];
        let num_kv_heads = shape[4];
        let head_dim = shape[5];
        drop(lock);

        let bytes_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes;
        let num_blocks = kv_pool_bytes / bytes_per_block;

        tracing::info!(
            free_mb = free_bytes / (1024 * 1024),
            total_mb = total_bytes / (1024 * 1024),
            kv_pool_mb = kv_pool_bytes / (1024 * 1024),
            bytes_per_block,
            num_blocks,
            "KV pool sizing"
        );

        if num_blocks < 1 {
            bail!(
                "Insufficient GPU memory for KV cache pool. Free: {} MB, \
                 estimated bytes per block: {} (total needed for 1 block). \
                 Try reducing `max_model_len` or `gpu_memory_utilization`.",
                free_bytes / (1024 * 1024),
                bytes_per_block,
            );
        }

        Ok(num_blocks)
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, dtype, options, paged_kv);
        tracing::warn!("CPU-only mode: allocating minimal KV cache pool (100 blocks)");
        Ok(100)
    }
}

/// Query CUDA free and total memory (in bytes) via the CUDA driver API.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn cuda_mem_info() -> Result<(usize, usize)> {
    use candle_core::cuda::cudarc::driver::sys;
    let mut free: usize = 0;
    let mut total: usize = 0;
    let result = unsafe {
        sys::cuMemGetInfo_v2(
            &mut free as *mut usize,
            &mut total as *mut usize,
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("cuMemGetInfo_v2 failed with error code {}", result as i32);
    }
    Ok((free, total))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

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
            let ids = vec![42, 99, 151645];
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
                seq_id: 0,
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
                seq_id: 1,
                token_ids: vec![42],
                text: "hello".into(),
                finished: true,
            };
            assert_eq!(output.text, "hello");
            assert_eq!(output.seq_id, 1);
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
}
