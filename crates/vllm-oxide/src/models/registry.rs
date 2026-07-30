use anyhow::{anyhow, Result};
use candle_core::Device;
use serde::Deserialize;

use crate::attention::AttentionContext;
use crate::config::Source;

use crate::causal_lm::CausalLM;

pub type ModelFactory = fn(
    config_json: &[u8],
    source: Source,
    device: &Device,
    max_model_len: usize,
) -> Result<BuiltModel>;

pub struct ModelEntry {
    pub arch: &'static str,
    pub factory: ModelFactory,
}

inventory::collect!(ModelEntry);

pub struct BuiltModel {
    pub model: Box<dyn CausalLM>,
    pub attn_ctx: AttentionContext,
}

pub fn build(source: Source, device: &Device, max_model_len: usize) -> Result<BuiltModel> {
    let config_bytes = read_config_json(&source)?;
    #[derive(Deserialize)]
    struct ArchCheck {
        architectures: Vec<String>,
    }
    let parsed: ArchCheck = serde_json::from_slice(&config_bytes)
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
    (entry.factory)(&config_bytes, source, device, max_model_len)
}

fn read_config_json(source: &Source) -> Result<Vec<u8>> {
    match source {
        Source::Local(dir) => std::fs::read(dir.join("config.json"))
            .map_err(|e| anyhow!("reading config.json from {}: {e}", dir.display())),
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
                        anyhow!("HF_HUB_OFFLINE=1 and config.json for `{repo}` not cached")
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
