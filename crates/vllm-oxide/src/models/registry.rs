use anyhow::{anyhow, Result};
use candle_core::Device;
use serde::Deserialize;

use crate::attention::AttentionContext;
use crate::config::Source;
use crate::model_identity::ResolvedModel;

use crate::causal_lm::CausalLM;

pub(crate) type ModelFactory =
    fn(resolved: &ResolvedModel, device: &Device, max_model_len: usize) -> Result<BuiltModel>;

pub struct ModelEntry {
    pub arch: &'static str,
    pub(crate) factory: ModelFactory,
}

inventory::collect!(ModelEntry);

pub struct BuiltModel {
    pub model: Box<dyn CausalLM>,
    pub attn_ctx: AttentionContext,
}

pub fn build(source: Source, device: &Device, max_model_len: usize) -> Result<BuiltModel> {
    let resolved = ResolvedModel::resolve(source, None)?;
    build_resolved(&resolved, device, max_model_len)
}

pub(crate) fn build_resolved(
    resolved: &ResolvedModel,
    device: &Device,
    max_model_len: usize,
) -> Result<BuiltModel> {
    let config_bytes = resolved.config_json();
    #[derive(Deserialize)]
    struct ArchCheck {
        architectures: Vec<String>,
    }
    let parsed: ArchCheck = serde_json::from_slice(config_bytes)
        .map_err(|e| anyhow!("parsing config.json architectures: {e}"))?;
    let arch = parsed
        .architectures
        .first()
        .ok_or_else(|| anyhow!("config.json has no `architectures` field"))?;
    let entry = inventory::iter::<ModelEntry>()
        .find(|e| e.arch == arch)
        .ok_or_else(|| {
            let supported: Vec<&str> = inventory::iter::<ModelEntry>().map(|e| e.arch).collect();
            anyhow!(
                "unknown architecture `{arch}`; supported: [{}]",
                supported.join(", ")
            )
        })?;
    (entry.factory)(resolved, device, max_model_len)
}
