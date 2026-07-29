//! T15 acceptance demo — load a Qwen3 checkpoint into a `ShardedVarBuilder`
//! and retrieve one tensor by name.
//!
//! Usage:
//!   cargo run --example load_qwen3 -- /path/to/Qwen3-0.6B
//!   cargo run --example load_qwen3 -- hub:Qwen/Qwen3-0.6B
//!   cargo run --example load_qwen3 -- hub:Qwen/Qwen3-0.6B@<commit-sha>
//!
//! What you should see: the resolved dtype, the shard count, the loaded
//! `model.embed_tokens.weight` tensor's shape + dtype. The full ~1.2 GB
//! checkpoint is NOT materialised — only the one tensor we ask for.

use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::Device;
use vllm_oxide::{default_dtype_from_config_json, is_hf_hub_offline, load_weights, Source};

fn parse_arg(arg: &str) -> Source {
    if let Some(rest) = arg.strip_prefix("hub:") {
        if let Some((repo, rev)) = rest.split_once('@') {
            Source::Hub {
                repo: repo.to_string(),
                revision: Some(rev.to_string()),
            }
        } else {
            Source::Hub {
                repo: rest.to_string(),
                revision: None,
            }
        }
    } else {
        Source::Local(PathBuf::from(arg))
    }
}

fn read_config_dtype(source: &Source) -> anyhow::Result<candle_core::DType> {
    let bytes = match source {
        Source::Local(dir) => std::fs::read(dir.join("config.json"))?,
        Source::Hub { repo, revision } => {
            let rev = revision.clone().unwrap_or_else(|| "main".to_string());
            let path = hub_get_file(repo, &rev, "config.json")?;
            std::fs::read(path)?
        }
    };
    default_dtype_from_config_json(&bytes)
}

/// Resolve a single file from a Hub repo, honouring `HF_HUB_OFFLINE`.
/// Mirrors the library's `resolve_hub_shards_online` / `_offline` split so
/// the demo's config read can't hang on the network when the user has
/// explicitly opted out (`HF_HUB_OFFLINE=1`).
fn hub_get_file(repo: &str, revision: &str, name: &str) -> anyhow::Result<PathBuf> {
    if is_hf_hub_offline() {
        let cache = hf_hub::Cache::from_env();
        let repo_handle = cache.repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        repo_handle.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "`HF_HUB_OFFLINE=1` is set and `{name}` for `{repo}` (revision `{revision}`) \
                 was not found in the local HF cache. Pre-download with \
                 `huggingface-cli download {repo}` or unset `HF_HUB_OFFLINE`."
            )
        })
    } else {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()?;
        let repo_handle = api.repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        Ok(repo_handle.get(name)?)
    }
}

fn main() -> ExitCode {
    let arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hub:Qwen/Qwen3-0.6B".to_string());
    let source = parse_arg(&arg);

    eprintln!("[demo] source        = {arg}");
    eprintln!("[demo] HF_HUB_OFFLINE = {}", is_hf_hub_offline());

    let dtype = match read_config_dtype(&source) {
        Ok(d) => {
            eprintln!("[demo] torch_dtype    = {d:?}");
            d
        }
        Err(e) => {
            eprintln!("[demo] failed to read torch_dtype from config.json: {e:#}");
            eprintln!("[demo] defaulting to BF16 (Qwen3 checkpoint default)");
            candle_core::DType::BF16
        }
    };

    let device = Device::Cpu;
    let vb = match load_weights(source, dtype, &device) {
        Ok(vb) => vb,
        Err(e) => {
            eprintln!("[demo] load_weights failed: {e:#}");
            return ExitCode::from(1);
        }
    };

    let tensor_name = "model.embed_tokens.weight";
    if !vb.contains_tensor(tensor_name) {
        eprintln!("[demo] {tensor_name} not present in checkpoint");
        return ExitCode::from(2);
    }
    // Qwen3-0.6B's embed is `[vocab_size=151936, hidden_size=1024]`. Other
    // Qwen3 variants have different shapes; adjust if pointing elsewhere.
    let tensor = match vb.get((151_936_usize, 1024), tensor_name) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[demo] retrieving {tensor_name} failed: {e}");
            return ExitCode::from(2);
        }
    };
    eprintln!(
        "[demo] {tensor_name}: shape={:?} dtype={:?}",
        tensor.shape(),
        tensor.dtype()
    );
    ExitCode::SUCCESS
}
