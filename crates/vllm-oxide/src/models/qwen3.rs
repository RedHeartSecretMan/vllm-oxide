#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use candle_core::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;

use crate::attention::{build_prefill_metadata, AttnMetadata, PagedKVCache};
use crate::config::{default_dtype_from_config_json, Source};
use crate::layers::activation::SiluAndMul;
use crate::layers::linear::{Linear, LinearSpec};
use crate::layers::parallel::{GateUpMerged, QkvMerged, Row};
use crate::layers::rmsnorm::RMSNorm;
use crate::layers::rope::RotaryEmbedding;
use crate::loader::load_weights_vb;

use super::registry::{BuiltModel, ModelEntry};
use super::CausalLM;

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
    act_fn: SiluAndMul,
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
            act_fn: SiluAndMul::new(),
        })
    }
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gu = self.gate_up_proj.forward(x)?;
        let act = self.act_fn.forward(&gu)?;
        self.down_proj.forward(&act)
    }
}

struct Qwen3Attention {
    qkv_proj: Linear<QkvMerged>,
    o_proj: Linear<Row>,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    rotary_emb: RotaryEmbedding,
    paged_kv: Arc<Mutex<PagedKVCache>>,
    attn_meta: Arc<Mutex<AttnMetadata>>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    layer_id: usize,
}

impl Qwen3Attention {
    #[allow(clippy::too_many_arguments)]
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        attn_meta: Arc<Mutex<AttnMetadata>>,
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
            paged_kv,
            attn_meta,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            layer_id,
        })
    }
    fn forward(&self, hidden: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
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
        self.attn_compute(&q, &k, &v)
    }
    #[cfg(feature = "cuda")]
    fn attn_compute(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
        let meta = self
            .attn_meta
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("attn_meta: {e}")))?;
        let sm = Tensor::from_vec(
            meta.slot_mapping.clone(),
            (meta.slot_mapping.len(),),
            q.device(),
        )?;
        let pkv = self
            .paged_kv
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("pkv: {e}")))?;
        pkv.reshape_and_cache(self.layer_id, k, v, &sm)?;
        let kc = pkv.k_cache(self.layer_id)?;
        let vc = pkv.v_cache(self.layer_id)?;
        let bs = pkv.block_size();
        drop(pkv);
        let scale = 1.0_f32 / (self.head_dim as f32).sqrt();
        let out = if meta.is_prefill {
            crate::attention::flash_attn::prefill_attn(q, k, v, &meta, scale)?
        } else {
            crate::attention::flash_attn::decode_attn(q, &kc, &vc, &meta, scale, bs)?
        };
        let n = out.dim(0)?;
        self.o_proj
            .forward(&out.reshape((n, self.num_heads * self.head_dim))?)
    }
    #[cfg(not(feature = "cuda"))]
    fn attn_compute(&self, _: &Tensor, _: &Tensor, _: &Tensor) -> CandleResult<Tensor> {
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
    #[allow(clippy::too_many_arguments)]
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        attn_meta: Arc<Mutex<AttnMetadata>>,
        layer_id: usize,
    ) -> CandleResult<Self> {
        let eps = config.rms_norm_eps;
        Ok(Self {
            self_attn: Qwen3Attention::from_vb(
                vb.clone(),
                config,
                dev,
                paged_kv,
                attn_meta,
                layer_id,
            )?,
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
    ) -> CandleResult<(Tensor, Tensor)> {
        let (normed, res) = self.input_layernorm.forward(hidden, residual)?;
        let attn = self.self_attn.forward(&normed, positions)?;
        let (normed, res) = self.post_attention_layernorm.forward(&attn, Some(&res))?;
        let mlp = self.mlp.forward(&normed)?;
        Ok((mlp, res))
    }
}

struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RMSNorm,
}

impl Qwen3Model {
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        attn_meta: Arc<Mutex<AttnMetadata>>,
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
                paged_kv.clone(),
                attn_meta.clone(),
                i,
            )?);
        }
        let norm = RMSNorm::from_vb(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
    fn forward(&self, input_ids: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let mut residual: Option<Tensor> = None;
        for layer in &self.layers {
            let (out, res) = layer.forward(positions, &hidden, residual.as_ref())?;
            hidden = out;
            residual = Some(res);
        }
        Ok(self.norm.forward(&hidden, residual.as_ref())?.0)
    }
}

pub struct Qwen3ForCausalLM {
    model: Qwen3Model,
    lm_head: Linear<Row>,
    vocab_size: usize,
    device: Device,
}

impl Qwen3ForCausalLM {
    fn from_vb(
        vb: VarBuilder,
        config: &Qwen3Config,
        dev: &Device,
        paged_kv: Arc<Mutex<PagedKVCache>>,
        attn_meta: Arc<Mutex<AttnMetadata>>,
    ) -> CandleResult<Self> {
        let model = Qwen3Model::from_vb(vb.pp("model"), config, dev, paged_kv, attn_meta)?;
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
            device: dev.clone(),
        })
    }
    pub fn build(
        config_json: &[u8],
        source: Source,
        device: &Device,
        max_model_len: usize,
    ) -> Result<BuiltModel> {
        let config: Qwen3Config =
            serde_json::from_slice(config_json).map_err(|e| anyhow!("Qwen3Config: {e}"))?;
        if max_model_len > config.max_position_embeddings {
            anyhow::bail!(
                "max_model_len ({max_model_len}) exceeds the model's max_position_embeddings ({}) — \
                 RoPE would produce garbage from out-of-range positions",
                config.max_position_embeddings
            );
        }
        let dtype = default_dtype_from_config_json(config_json)?;
        let vb = load_weights_vb(source, dtype, device)?;
        let paged_kv = Arc::new(Mutex::new(PagedKVCache::new(
            config.num_hidden_layers,
            100,
            256,
            config.num_key_value_heads,
            config.head_dim(),
            dtype,
            device,
        )?));
        let attn_meta = Arc::new(Mutex::new(build_prefill_metadata(&[1], &[1], &[0])));
        let model = Box::new(Qwen3ForCausalLM::from_vb(
            vb,
            &config,
            device,
            paged_kv.clone(),
            attn_meta.clone(),
        )?);
        Ok(BuiltModel {
            model,
            paged_kv,
            attn_meta,
        })
    }
}

impl CausalLM for Qwen3ForCausalLM {
    fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
        self.model.forward(input_ids, positions)
    }
    fn compute_logits(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        self.lm_head.forward(hidden_states)
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

inventory::submit! { ModelEntry { arch: "Qwen3ForCausalLM", factory: Qwen3ForCausalLM::build } }

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
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
}
