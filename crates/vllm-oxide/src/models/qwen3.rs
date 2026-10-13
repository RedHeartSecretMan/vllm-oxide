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
    fn head_dim(&self) -> Result<usize> {
        if self.num_attention_heads == 0 {
            anyhow::bail!("Qwen3Config: num_attention_heads must be greater than zero");
        }
        let head_dim = if let Some(head_dim) = self.head_dim {
            head_dim
        } else {
            if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
                anyhow::bail!(
                    "Qwen3Config: hidden_size must be divisible by num_attention_heads when head_dim is absent"
                );
            }
            self.hidden_size / self.num_attention_heads
        };
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            anyhow::bail!(
                "Qwen3Config: head_dim must be positive and even for full rotary embedding"
            );
        }
        Ok(head_dim)
    }

    fn validate(&self, dtype: candle_core::DType) -> Result<()> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 {
                anyhow::bail!("Qwen3Config: {name} must be greater than zero");
            }
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            anyhow::bail!(
                "Qwen3Config: num_attention_heads must be divisible by num_key_value_heads"
            );
        }
        let head_dim = self.head_dim()?;
        self.num_key_value_heads
            .checked_mul(2)
            .and_then(|kv_heads| self.num_attention_heads.checked_add(kv_heads))
            .and_then(|heads| heads.checked_mul(head_dim))
            .ok_or_else(|| anyhow!("Qwen3Config: fused QKV geometry overflows usize"))?;
        self.intermediate_size
            .checked_mul(2)
            .ok_or_else(|| anyhow!("Qwen3Config: fused gate/up geometry overflows usize"))?;
        self.validate_cache_geometry(dtype, 1)?;
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            anyhow::bail!("Qwen3Config: rope_theta must be finite and greater than zero");
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            anyhow::bail!("Qwen3Config: rms_norm_eps must be finite and greater than zero");
        }
        Ok(())
    }
    fn validate_cache_geometry(&self, dtype: candle_core::DType, num_blocks: usize) -> Result<()> {
        let elements = [
            2,
            self.num_hidden_layers,
            num_blocks,
            256,
            self.num_key_value_heads,
            self.head_dim()?,
        ]
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| {
            anyhow!(
            "Qwen3Config: KV cache geometry including num_hidden_layers overflows usize elements"
        )
        })?;
        std::ffi::c_long::try_from(elements).map_err(|_| anyhow!(
            "Qwen3Config: KV cache geometry including num_hidden_layers exceeds the signed cache-index ABI"
        ))?;
        let bytes = elements.checked_mul(dtype.size_in_bytes()).ok_or_else(|| {
            anyhow!(
                "Qwen3Config: KV cache geometry including num_hidden_layers overflows usize bytes"
            )
        })?;
        // Rust allocations/pointer offsets and the signed cache-index ABI
        // cannot address a buffer larger than isize::MAX bytes.
        isize::try_from(bytes).map_err(|_| anyhow!(
            "Qwen3Config: KV cache geometry including num_hidden_layers exceeds signed index/byte bounds"
        ))?;
        Ok(())
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
        let hd = config
            .head_dim()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
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
        #[cfg(feature = "internal-golden")] mut trace: Option<
            &mut crate::golden_capture::layer_trace::StepTrace,
        >,
    ) -> CandleResult<Tensor> {
        let qkv = self.qkv_proj.forward(hidden)?;
        let qs = self.num_heads * self.head_dim;
        let ks = self.num_kv_heads * self.head_dim;
        let q = qkv.i((.., 0..qs))?;
        let k = qkv.i((.., qs..qs + ks))?;
        let v = qkv.i((.., qs + ks..qs + 2 * ks))?;
        #[cfg(feature = "internal-golden")]
        if let Some(trace) = trace.as_deref_mut() {
            for (name, value) in [("layer0_q", &q), ("layer0_k", &k), ("layer0_v", &v)] {
                trace
                    .record(name, value)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            }
        }
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
        #[cfg(feature = "internal-golden")]
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .record("layer0_q_norm", &q.reshape((n, qs))?)
                .and_then(|()| trace.record("layer0_k_norm", &k.reshape((n, ks))?))
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        }
        let (q, k) = self.rotary_emb.forward(positions, &q, &k)?;
        #[cfg(feature = "internal-golden")]
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .record("layer0_q_rope", &q.reshape((n, qs))?)
                .and_then(|()| trace.record("layer0_k_rope", &k.reshape((n, ks))?))
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        }
        self.attn_compute(
            &q,
            &k,
            &v,
            prepared,
            #[cfg(feature = "internal-golden")]
            trace,
        )
    }
    #[cfg(feature = "cuda")]
    fn attn_compute(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        prepared: &PreparedAttention,
        #[cfg(feature = "internal-golden")] mut trace: Option<
            &mut crate::golden_capture::layer_trace::StepTrace,
        >,
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
        #[cfg(feature = "internal-golden")]
        if let Some(trace) = trace.as_deref_mut() {
            use crate::attention::flash_attn::{
                PAGED_WINDOW_LEFT, PAGED_WINDOW_RIGHT, PREFILL_CAUSAL,
            };
            let unpaged = logical.is_prefill && !logical.uses_paged_kv();
            trace
                .record_attention(crate::golden_capture::layer_trace::AttentionCall {
                    cu_q: prepared.cu_seqlens_q().to_vec1::<u32>()?,
                    cu_k: prepared.cu_seqlens_k().to_vec1::<u32>()?,
                    max_q: logical.max_seqlen_q,
                    max_k: logical.max_seqlen_k,
                    heads: (q.dim(1)?, k.dim(1)?, q.dim(2)?),
                    scale,
                    causal: unpaged.then_some(PREFILL_CAUSAL),
                    window: if unpaged {
                        (None, None)
                    } else {
                        (PAGED_WINDOW_LEFT, PAGED_WINDOW_RIGHT)
                    },
                })
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        }
        let out = if logical.is_prefill && !logical.uses_paged_kv() {
            crate::attention::flash_attn::prefill_attn(q, k, v, prepared, scale)?
        } else {
            crate::attention::flash_attn::paged_attn(q, &kc, &vc, prepared, scale, bs)?
        };
        let n = out.dim(0)?;
        #[cfg(feature = "internal-golden")]
        if let Some(trace) = trace {
            trace
                .record(
                    "layer0_attention_context",
                    &out.reshape((n, self.num_heads * self.head_dim))?,
                )
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        }
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
        #[cfg(feature = "internal-golden")] _: Option<
            &mut crate::golden_capture::layer_trace::StepTrace,
        >,
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
        #[cfg(feature = "internal-golden")] mut trace: Option<
            &mut crate::golden_capture::layer_trace::StepTrace,
        >,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (normed, res) = self.input_layernorm.forward(hidden, residual)?;
        #[cfg(feature = "internal-golden")]
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .record("layer0_input_norm", &normed)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        }
        let attn = self.self_attn.forward(
            &normed,
            positions,
            prepared,
            #[cfg(feature = "internal-golden")]
            trace,
        )?;
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
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(|error| {
                candle_core::Error::Msg(format!("Qwen3 layer storage allocation failed: {error}"))
            })?;
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
        // Layer indices are consumed only by the optional golden trace below.
        #[cfg_attr(
            not(feature = "internal-golden"),
            allow(clippy::unused_enumerate_index)
        )]
        for (_layer_index, layer) in self.layers.iter().enumerate() {
            let (out, res) = layer.forward(
                positions,
                &hidden,
                residual.as_ref(),
                prepared,
                #[cfg(feature = "internal-golden")]
                trace.as_mut().filter(|_| _layer_index == 0),
            )?;
            hidden = out;
            #[cfg(feature = "internal-golden")]
            if let Some(trace) = trace.as_mut() {
                trace
                    .record(&format!("layer_{_layer_index}"), &(&hidden + &res)?)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            }
            #[cfg(feature = "internal-golden")]
            if _layer_index == 0 && trace.as_ref().is_some_and(|trace| trace.layer0_only()) {
                if let Some(trace) = trace.take() {
                    trace
                        .finish()
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                }
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
        let config: Qwen3Config =
            serde_json::from_slice(config_json).map_err(|e| anyhow!("Qwen3Config: {e}"))?;
        config.validate(dtype)?;
        if max_model_len > config.max_position_embeddings {
            anyhow::bail!(
                "max_model_len ({max_model_len}) exceeds the model's max_position_embeddings ({}) — \
                 RoPE would produce garbage from out-of-range positions",
                config.max_position_embeddings
            );
        }
        let vb = load_resolved_weights_vb(resolved, device)?;
        let paged_kv = Arc::new(Mutex::new(PagedKVCache::deferred(PagedKVCacheGeometry {
            num_layers: config.num_hidden_layers,
            block_size: 256,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim()?,
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
        let specifications = [
            ("model.embed_tokens.weight", vec![5, 2], 0.0_f32),
            ("model.norm.weight", vec![2], 1.0_f32),
            ("model.layers.0.input_layernorm.weight", vec![2], 1.0_f32),
            (
                "model.layers.0.post_attention_layernorm.weight",
                vec![2],
                1.0_f32,
            ),
            ("model.layers.0.self_attn.q_norm.weight", vec![2], 1.0_f32),
            ("model.layers.0.self_attn.k_norm.weight", vec![2], 1.0_f32),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![2, 2],
                0.0_f32,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![2, 2],
                0.0_f32,
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                vec![2, 2],
                0.0_f32,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                vec![2, 2],
                0.0_f32,
            ),
            ("model.layers.0.mlp.gate_proj.weight", vec![2, 2], 0.0_f32),
            ("model.layers.0.mlp.up_proj.weight", vec![2, 2], 0.0_f32),
            ("model.layers.0.mlp.down_proj.weight", vec![2, 2], 0.0_f32),
        ];
        let bytes: Vec<Vec<u8>> = specifications
            .iter()
            .map(|(_, shape, value)| {
                (0..shape.iter().product::<usize>())
                    .flat_map(|_| value.to_le_bytes())
                    .collect()
            })
            .collect();
        let tensors: Vec<_> = specifications
            .iter()
            .zip(&bytes)
            .map(|((name, shape, _), data)| {
                (
                    *name,
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.clone(),
                        data,
                    )
                    .unwrap(),
                )
            })
            .collect();
        safetensors::tensor::serialize_to_file(tensors, &None, path).unwrap();
    }

    fn build_one_layer_model() -> BuiltModel {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{
                "architectures":["Qwen3ForCausalLM"],
                "torch_dtype":"bfloat16",
                "eos_token_id":2,
                "hidden_size":2,
                "num_hidden_layers":1,
                "attention_bias":false,
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

    fn valid_small_config() -> Qwen3Config {
        serde_json::from_str(
            r#"{"hidden_size":64,"num_hidden_layers":1,"num_attention_heads":4,
                "num_key_value_heads":2,"intermediate_size":128,"vocab_size":100,
                "rms_norm_eps":1e-6,"max_position_embeddings":128,"hidden_act":"silu"}"#,
        )
        .unwrap()
    }

    #[test]
    fn validates_head_geometry_without_assuming_hidden_equals_attention_width() {
        let mut config = valid_small_config();
        config.hidden_size = 65;
        assert!(config
            .validate(candle_core::DType::BF16)
            .unwrap_err()
            .to_string()
            .contains("head_dim is absent"));
        // Explicit head_dim defines projection width independently of hidden_size.
        config.head_dim = Some(32);
        config.validate(candle_core::DType::BF16).unwrap();
        assert_eq!(config.head_dim().unwrap(), 32);
        config.num_attention_heads = 0;
        assert!(config
            .head_dim()
            .unwrap_err()
            .to_string()
            .contains("num_attention_heads"));
        assert!(config
            .validate(candle_core::DType::BF16)
            .unwrap_err()
            .to_string()
            .contains("num_attention_heads"));
        config.num_attention_heads = 4;
        for head_dim in [0, 3] {
            config.head_dim = Some(head_dim);
            assert!(config
                .validate(candle_core::DType::BF16)
                .unwrap_err()
                .to_string()
                .contains("head_dim"));
        }
        config.head_dim = Some(32);
        config.num_key_value_heads = 3;
        assert!(config
            .validate(candle_core::DType::BF16)
            .unwrap_err()
            .to_string()
            .contains("num_key_value_heads"));
        config.num_key_value_heads = 8;
        assert!(config
            .validate(candle_core::DType::BF16)
            .unwrap_err()
            .to_string()
            .contains("num_key_value_heads"));
    }

    #[test]
    fn rejects_invalid_qwen3_scalars_and_projection_overflow() {
        let mut config = valid_small_config();
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            config.rope_theta = value;
            assert!(config
                .validate(candle_core::DType::BF16)
                .unwrap_err()
                .to_string()
                .contains("rope_theta"));
        }
        config.rope_theta = 1_000_000.0;
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            config.rms_norm_eps = value;
            assert!(config
                .validate(candle_core::DType::BF16)
                .unwrap_err()
                .to_string()
                .contains("rms_norm_eps"));
        }
        config.rms_norm_eps = 1e-6;
        config.head_dim = Some(usize::MAX - 1);
        assert!(config
            .validate(candle_core::DType::BF16)
            .unwrap_err()
            .to_string()
            .contains("QKV geometry overflows"));
        config.head_dim = Some(16);
        config.intermediate_size = usize::MAX;
        assert!(config
            .validate(candle_core::DType::BF16)
            .unwrap_err()
            .to_string()
            .contains("gate/up geometry overflows"));
    }

    #[test]
    fn rejects_unaddressable_kv_cache_geometry_without_a_fixed_layer_limit() {
        let mut config = valid_small_config();
        let bytes_per_layer_block = 2 * 256 * 2 * 16 * 2;
        for layers in [
            usize::MAX,
            usize::MAX / (bytes_per_layer_block / 2) + 1,
            usize::MAX / bytes_per_layer_block + 1,
            usize::try_from(isize::MAX).unwrap() / bytes_per_layer_block + 1,
        ] {
            config.num_hidden_layers = layers;
            let error = config
                .validate(candle_core::DType::BF16)
                .unwrap_err()
                .to_string();
            assert!(error.contains("Qwen3Config: KV cache geometry"), "{error}");
            assert!(error.contains("num_hidden_layers"), "{error}");
        }
        // Validity follows representable geometry, not an arbitrary model-size cap.
        config.num_hidden_layers = 1_000;
        config.validate(candle_core::DType::BF16).unwrap();
    }

    #[test]
    fn rejects_invalid_geometry_before_loading_weights() {
        let base = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16",
            "eos_token_id": 2, "hidden_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "intermediate_size": 128, "vocab_size": 100,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 128,
            "head_dim": 16, "hidden_act": "silu"
        });
        let mut cases: Vec<(&str, serde_json::Value)> = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "head_dim",
        ]
        .into_iter()
        .map(|field| (field, serde_json::json!(0)))
        .collect();
        cases.push(("num_hidden_layers", serde_json::json!(usize::MAX)));
        cases.push((
            "num_hidden_layers",
            serde_json::json!(usize::try_from(isize::MAX).unwrap() / (2 * 256 * 2 * 16 * 2) + 1),
        ));
        for (field, value) in cases {
            let tmp = tempfile::tempdir().unwrap();
            let mut config = base.clone();
            config[field] = value;
            let config_json = serde_json::to_vec(&config).unwrap();
            std::fs::write(tmp.path().join("config.json"), &config_json).unwrap();
            std::fs::write(tmp.path().join("tokenizer.json"), b"{}").unwrap();
            // Resolution may inspect the file's name; the model builder must
            // reject config before mapping or parsing this invalid weight file.
            std::fs::write(tmp.path().join("model.safetensors"), b"not weights").unwrap();
            let resolved = ResolvedModel::resolve(
                crate::config::Source::Local(tmp.path().to_path_buf()),
                None,
            )
            .unwrap();
            let result = Qwen3ForCausalLM::build(&resolved, &Device::Cpu, 8);
            let error = result.err().unwrap().to_string();
            assert!(error.contains("Qwen3Config"), "{field}: {error}");
            assert!(error.contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn deserialises_qwen3_06b_config() {
        let json = r#"{"architectures":["Qwen3ForCausalLM"],"attention_bias":false,"head_dim":128,
            "hidden_act":"silu","hidden_size":1024,"intermediate_size":3072,"max_position_embeddings":40960,
            "num_attention_heads":16,"num_hidden_layers":28,"num_key_value_heads":8,
            "rms_norm_eps":1e-06,"rope_theta":1000000,"tie_word_embeddings":true,"vocab_size":151936}"#;
        let cfg: Qwen3Config = serde_json::from_str(json).unwrap();
        cfg.validate(candle_core::DType::BF16).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.head_dim().unwrap(), 128);
        assert!(!cfg.has_qkv_bias());
        assert!(cfg.tie_word_embeddings());
    }
    #[test]
    fn head_dim_fallback() {
        let json = r#"{"hidden_size":1024,"num_hidden_layers":2,"num_attention_heads":16,"num_key_value_heads":8,
            "intermediate_size":3072,"vocab_size":100,"rms_norm_eps":1e-6,"max_position_embeddings":4096,"hidden_act":"silu"}"#;
        let cfg: Qwen3Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim().unwrap(), 64);
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
        let built = build_one_layer_model();
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
