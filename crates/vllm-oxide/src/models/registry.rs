use anyhow::{anyhow, Result};
use candle_core::Device;
use serde::Deserialize;

use crate::attention::AttentionContext;
use crate::loader::model_identity::ResolvedModel;

use crate::causal_lm::CausalLM;

pub type ModelFactory =
    fn(resolved: &ResolvedModel, device: &Device, max_model_len: usize) -> Result<BuiltModel>;

pub struct ModelEntry {
    pub arch: &'static str,
    pub factory: ModelFactory,
}

inventory::collect!(ModelEntry);

pub struct BuiltModel {
    pub model: Box<dyn CausalLM>,
    pub attn_ctx: AttentionContext,
}

/// Query the factory for an already-resolved model without performing I/O.
pub(crate) fn resolved_factory(config_json: &[u8]) -> Result<ModelFactory> {
    let arch = read_architecture(config_json)?;
    inventory::iter::<ModelEntry>()
        .find(|entry| entry.arch == arch)
        .map(|entry| entry.factory)
        .ok_or_else(|| {
            unknown_architecture(
                &arch,
                inventory::iter::<ModelEntry>().map(|entry| entry.arch),
            )
        })
}

fn read_architecture(config_json: &[u8]) -> Result<String> {
    #[derive(Deserialize)]
    struct ArchCheck {
        architectures: Vec<String>,
    }

    let parsed: ArchCheck = serde_json::from_slice(config_json)
        .map_err(|error| anyhow!("parsing config.json architectures: {error}"))?;
    parsed
        .architectures
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("config.json has no `architectures` field"))
}

fn unknown_architecture<'a>(
    architecture: &str,
    supported: impl Iterator<Item = &'a str>,
) -> anyhow::Error {
    let supported: Vec<&str> = supported.collect();
    anyhow!(
        "unknown architecture `{architecture}`; supported: [{}]",
        supported.join(", ")
    )
}
