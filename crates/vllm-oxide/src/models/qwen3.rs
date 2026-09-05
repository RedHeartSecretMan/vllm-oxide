#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use candle_core::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;

use crate::attention::{AttentionContext, PagedKVCache, PagedKVCacheGeometry, PreparedAttention};
use crate::layers::activation::silu_and_mul;
use crate::layers::linear::{Linear, LinearSpec};
use crate::layers::parallel::{GateUpMerged, QkvMerged, Row};
use crate::layers::rmsnorm::RMSNorm;
use crate::layers::rope::RotaryEmbedding;
use crate::loader::load_resolved_weights_vb;
use crate::loader::model_identity::ResolvedModel;

use super::registry::{BuiltModel, ModelEntry};
use crate::causal_lm::CausalLM;

fn default_rope_theta() -> f32 {
    1_000_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub attention_bias: Option<bool>,
}

impl Qwen3Config {
    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    fn has_qkv_bias(&self) -> bool {
        self.attention_bias.unwrap_or(true)
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }
}

struct Qwen3Mlp {
    gate_up_proj: Linear<GateUpMerged>,
    down_proj: Linear<Row>,
}

impl Qwen3Mlp {
    fn from_vb(vb: VarBuilder, config: &Qwen3Config, dev: &Device) -> CandleResult<Self> {
        let gu_spec = LinearSpec {
            in_features: config.hidden_size,
            out_features_per_shard: config.intermediate_size,
            bias: false,
        };
        let dn_spec = LinearSpec {
            in_features: config.intermediate_size,
            out_features_per_shard: config.hidden_size,
            bias: false,
        };
        Ok(Self {
            gate_up_proj: Linear::<GateUpMerged>::from_vb(vb.clone(), &gu_spec, dev)?,
            down_proj: Linear::<Row>::from_vb(vb.pp("down_proj"), &dn_spec, dev)?,
        })
    }
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gu = self.gate_up_proj.forward(x)?;
        let act = silu_and_mul(&gu)?;
        self.down_proj.forward(&act)
    }
}

struct Qwen3Attention {
    qkv_proj: Linear<QkvMerged>,
    o_proj: Linear<Row>,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    rotary_emb: RotaryEmbedding,
    attn_ctx: AttentionContext,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    layer_id: usize,
}

impl Qwen3Attention {
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        attn_ctx: AttentionContext,
        layer_id: usize,
    ) -> CandleResult<Self> {
        let hd = config.head_dim();
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let bias = config.has_qkv_bias();
        let qkv_spec = LinearSpec {
            in_features: config.hidden_size,
            out_features_per_shard: nh * hd,
            bias,
        };
        let o_spec = LinearSpec {
            in_features: nh * hd,
            out_features_per_shard: config.hidden_size,
            bias: false,
        };
        let (q_norm, k_norm) = if !bias {
            (
                Some(RMSNorm::from_vb(
                    vb.pp("self_attn").pp("q_norm"),
                    hd,
                    config.rms_norm_eps,
                )?),
                Some(RMSNorm::from_vb(
                    vb.pp("self_attn").pp("k_norm"),
                    hd,
                    config.rms_norm_eps,
                )?),
            )
        } else {
            (None, None)
        };
        let rotary_emb = RotaryEmbedding::new(
            hd,
            hd,
            config.max_position_embeddings,
            config.rope_theta,
            dev,
        )?;
        Ok(Self {
            qkv_proj: Linear::<QkvMerged>::from_vb(vb.pp("self_attn"), &qkv_spec, dev)?,
            o_proj: Linear::<Row>::from_vb(vb.pp("self_attn").pp("o_proj"), &o_spec, dev)?,
            q_norm,
            k_norm,
            rotary_emb,
            attn_ctx,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            layer_id,
        })
    }
    fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        prepared: &PreparedAttention,
    ) -> CandleResult<Tensor> {
        let qkv = self.qkv_proj.forward(hidden)?;
        let qs = self.num_heads * self.head_dim;
        let ks = self.num_kv_heads * self.head_dim;
        let q = qkv.i((.., 0..qs))?;
        let k = qkv.i((.., qs..qs + ks))?;
        let v = qkv.i((.., qs + ks..qs + 2 * ks))?;
        let n = q.dim(0)?;
        let q = q.reshape((n, self.num_heads, self.head_dim))?;
        let k = k.reshape((n, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((n, self.num_kv_heads, self.head_dim))?;
        let q = match &self.q_norm {
            Some(nm) => nm.forward(&q, None)?.0,
            None => q,
        };
        let k = match &self.k_norm {
            Some(nm) => nm.forward(&k, None)?.0,
            None => k,
        };
        let (q, k) = self.rotary_emb.forward(positions, &q, &k)?;
        self.attn_compute(&q, &k, &v, prepared)
    }
    #[cfg(feature = "cuda")]
    fn attn_compute(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        prepared: &PreparedAttention,
    ) -> CandleResult<Tensor> {
        let logical = prepared.logical();
        let pkv = self
            .attn_ctx
            .paged_kv
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("pkv: {e}")))?;
        pkv.reshape_and_cache(self.layer_id, k, v, prepared.slot_mapping())?;
        let kc = pkv.k_cache(self.layer_id)?;
        let vc = pkv.v_cache(self.layer_id)?;
        let bs = pkv.block_size();
        drop(pkv);
        let scale = 1.0_f32 / (self.head_dim as f32).sqrt();
        let out = if logical.is_prefill && !logical.uses_paged_kv() {
            crate::attention::flash_attn::prefill_attn(q, k, v, prepared, scale)?
        } else {
            crate::attention::flash_attn::paged_attn(q, &kc, &vc, prepared, scale, bs)?
        };
        let n = out.dim(0)?;
        self.o_proj
            .forward(&out.reshape((n, self.num_heads * self.head_dim))?)
    }
    #[cfg(not(feature = "cuda"))]
    fn attn_compute(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &PreparedAttention,
    ) -> CandleResult<Tensor> {
        candle_core::bail!("attention requires --features cuda")
    }
}

struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen3DecoderLayer {
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        attn_ctx: AttentionContext,
        layer_id: usize,
    ) -> CandleResult<Self> {
        let eps = config.rms_norm_eps;
        Ok(Self {
            self_attn: Qwen3Attention::from_vb(vb.clone(), config, dev, attn_ctx, layer_id)?,
            mlp: Qwen3Mlp::from_vb(vb.pp("mlp"), config, dev)?,
            input_layernorm: RMSNorm::from_vb(vb.pp("input_layernorm"), config.hidden_size, eps)?,
            post_attention_layernorm: RMSNorm::from_vb(
                vb.pp("post_attention_layernorm"),
                config.hidden_size,
                eps,
            )?,
        })
    }
    fn forward(
        &self,
        positions: &Tensor,
        hidden: &Tensor,
        residual: Option<&Tensor>,
        prepared: &PreparedAttention,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (normed, res) = self.input_layernorm.forward(hidden, residual)?;
        let attn = self.self_attn.forward(&normed, positions, prepared)?;
        let (normed, res) = self.post_attention_layernorm.forward(&attn, Some(&res))?;
        let mlp = self.mlp.forward(&normed)?;
        Ok((mlp, res))
    }
}

struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RMSNorm,
    #[cfg(feature = "internal-golden")]
    layer_trace: Option<crate::golden_capture::layer_trace::LayerTrace>,
}

impl Qwen3Model {
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        attn_ctx: AttentionContext,
    ) -> CandleResult<Self> {
        let ew = vb
            .pp("embed_tokens")
            .get((config.vocab_size, config.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(ew, config.hidden_size);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::from_vb(
                vb.pp("layers").pp(i),
                config,
                dev,
                attn_ctx.clone(),
                i,
            )?);
        }
        let norm = RMSNorm::from_vb(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            #[cfg(feature = "internal-golden")]
            layer_trace: crate::golden_capture::layer_trace::LayerTrace::from_env()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?,
        })
    }
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        prepared: &PreparedAttention,
    ) -> CandleResult<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        #[cfg(feature = "internal-golden")]
        let mut trace = self
            .layer_trace
            .as_ref()
            .map(|trace| trace.begin(input_ids, positions))
            .transpose()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
            .flatten();
        #[cfg(feature = "internal-golden")]
        if let Some(trace) = trace.as_mut() {
            trace
                .record("embedding", &hidden)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        }
        let mut residual: Option<Tensor> = None;
        for (_layer_index, layer) in self.layers.iter().enumerate() {
            let (out, res) = layer.forward(positions, &hidden, residual.as_ref(), prepared)?;
            hidden = out;
            #[cfg(feature = "internal-golden")]
            if let Some(trace) = trace.as_mut() {
                trace
                    .record(&format!("layer_{_layer_index}"), &(&hidden + &res)?)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            }
            residual = Some(res);
        }
        let hidden = self.norm.forward(&hidden, residual.as_ref())?.0;
        #[cfg(feature = "internal-golden")]
        if let Some(mut trace) = trace {
            trace
                .record("final_norm", &hidden)
                .and_then(|()| trace.finish())
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        }
        Ok(hidden)
    }
}

pub struct Qwen3ForCausalLM {
    model: Qwen3Model,
    lm_head: Linear<Row>,
    vocab_size: usize,
    attn_ctx: AttentionContext,
}

impl Qwen3ForCausalLM {
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        attn_ctx: AttentionContext,
    ) -> CandleResult<Self> {
        let model = Qwen3Model::from_vb(vb.pp("model"), config, dev, attn_ctx.clone())?;
        let lm_head = if config.tie_word_embeddings() {
            Linear::<Row>::from_weight(model.embed_tokens.embeddings().clone())
        } else {
            let spec = LinearSpec {
                in_features: config.hidden_size,
                out_features_per_shard: config.vocab_size,
                bias: false,
            };
            Linear::<Row>::from_vb(vb.pp("lm_head"), &spec, dev)?
        };
        Ok(Self {
            model,
            lm_head,
            vocab_size: config.vocab_size,
            attn_ctx,
        })
    }
    pub fn build(
        resolved: &ResolvedModel,
        device: &Device,
        max_model_len: usize,
    ) -> Result<BuiltModel> {
        let config_json = resolved.config_json();
        let dtype = resolved.dtype();
        let vb = load_resolved_weights_vb(resolved, device)?;
        let config: Qwen3Config =
            serde_json::from_slice(config_json).map_err(|e| anyhow!("Qwen3Config: {e}"))?;
        if max_model_len > config.max_position_embeddings {
            anyhow::bail!(
                "max_model_len ({max_model_len}) exceeds the model's max_position_embeddings ({}) — \
                 RoPE would produce garbage from out-of-range positions",
                config.max_position_embeddings
            );
        }
        let paged_kv = Arc::new(Mutex::new(PagedKVCache::deferred(PagedKVCacheGeometry {
            num_layers: config.num_hidden_layers,
            block_size: 256,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim(),
            dtype,
        })));
        let attn_ctx = AttentionContext::new(paged_kv);
        let model = Box::new(Qwen3ForCausalLM::from_vb(
            vb,
            &config,
            device,
            attn_ctx.clone(),
        )?);
        Ok(BuiltModel { model, attn_ctx })
    }
}

impl CausalLM for Qwen3ForCausalLM {
    fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
        let prepared = self.attn_ctx.prepared_for_bound_consumer()?;
        self.model.forward(input_ids, positions, &prepared)
    }
    fn compute_logits(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        self.lm_head.forward(hidden_states)
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

inventory::submit! { ModelEntry { arch: "Qwen3ForCausalLM", factory: Qwen3ForCausalLM::build } }

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn write_minimal_weights(path: &std::path::Path) {
        let embedding_bytes: Vec<u8> = [0.0_f32; 10]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let norm_bytes: Vec<u8> = [1.0_f32; 2]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let embedding = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![5, 2],
            &embedding_bytes,
        )
        .unwrap();
        let norm =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &norm_bytes)
                .unwrap();
        safetensors::tensor::serialize_to_file(
            [
                ("model.embed_tokens.weight", embedding),
                ("model.norm.weight", norm),
            ],
            &None,
            path,
        )
        .unwrap();
    }

    fn build_zero_layer_model() -> BuiltModel {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{
                "architectures":["Qwen3ForCausalLM"],
                "torch_dtype":"bfloat16",
                "eos_token_id":2,
                "hidden_size":2,
                "num_hidden_layers":0,
                "num_attention_heads":1,
                "num_key_value_heads":1,
                "intermediate_size":2,
                "vocab_size":5,
                "rms_norm_eps":0.000001,
                "max_position_embeddings":16,
                "hidden_act":"silu",
                "tie_word_embeddings":true
            }"#,
        )
        .unwrap();
        std::fs::write(tmp.path().join("tokenizer.json"), b"{}").unwrap();
        write_minimal_weights(&tmp.path().join("model.safetensors"));
        let resolved = ResolvedModel::resolve(
            crate::config::Source::Local(tmp.path().to_path_buf()),
            Some(candle_core::DType::F16),
        )
        .unwrap();
        Qwen3ForCausalLM::build(&resolved, &Device::Cpu, 8).unwrap()
    }

    #[test]
    fn deserialises_qwen3_06b_config() {
        let json = r#"{"architectures":["Qwen3ForCausalLM"],"attention_bias":false,"head_dim":128,
            "hidden_act":"silu","hidden_size":1024,"intermediate_size":3072,"max_position_embeddings":40960,
            "num_attention_heads":16,"num_hidden_layers":28,"num_key_value_heads":8,
            "rms_norm_eps":1e-06,"rope_theta":1000000,"tie_word_embeddings":true,"vocab_size":151936}"#;
        let cfg: Qwen3Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.head_dim(), 128);
        assert!(!cfg.has_qkv_bias());
        assert!(cfg.tie_word_embeddings());
    }
    #[test]
    fn head_dim_fallback() {
        let json = r#"{"hidden_size":1024,"num_hidden_layers":2,"num_attention_heads":16,"num_key_value_heads":8,
            "intermediate_size":3072,"vocab_size":100,"rms_norm_eps":1e-6,"max_position_embeddings":4096,"hidden_act":"silu"}"#;
        let cfg: Qwen3Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 64);
    }
    #[test]
    fn accepts_unknown_fields() {
        let json = r#"{"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":4,
            "intermediate_size":128,"vocab_size":100,"rms_norm_eps":1e-6,"max_position_embeddings":128,
            "hidden_act":"silu","future":42}"#;
        assert!(serde_json::from_str::<Qwen3Config>(json).is_ok());
    }
    #[test]
    fn qwen3_registered() {
        assert!(inventory::iter::<ModelEntry>().any(|e| e.arch == "Qwen3ForCausalLM"));
    }

    #[test]
    fn resolved_dtype_reaches_model_without_allocating_a_cache_buffer() {
        let built = build_zero_layer_model();
        let cache = built.attn_ctx.paged_kv.lock().unwrap();

        assert_eq!(cache.dtype(), candle_core::DType::F16);
        assert_eq!(cache.num_blocks(), 0);
        assert!(cache
            .k_cache(0)
            .unwrap_err()
            .to_string()
            .contains("not allocated"));
        drop(cache);
        assert!(built.attn_ctx.is_idle());
    }
}
