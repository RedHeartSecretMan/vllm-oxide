//! Admission evidence for unforced public-API checks; no logits transfer or forcing.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use serde::Deserialize;
use std::path::PathBuf;

pub(crate) const PLAN_ENV: &str = "VLLM_OXIDE_INTERNAL_BEHAVIOR_PLAN";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    calls: Vec<Call>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    call_id: String,
    prompts: Vec<Vec<u32>>,
    params: Vec<Parameters>,
    #[serde(rename = "expected")]
    _expected: String,
    #[serde(default, rename = "error_contains")]
    _error_contains: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    max_tokens: usize,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    temperature: f32,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    presence_penalty: f32,
    #[serde(default)]
    frequency_penalty: f32,
    #[serde(default)]
    repetition_penalty: f32,
}

/// Calls the unchanged public generate seam. Expected verdicts are consumer-only.
pub(crate) fn run_from_env(
    mut generate: impl FnMut(&[Vec<u32>], &[crate::SamplingParams]) -> Result<Vec<crate::RequestOutput>>,
) -> Result<()> {
    let plan = std::env::var_os(PLAN_ENV).context("behavior plan missing")?;
    let destination = PathBuf::from(
        std::env::var_os("VLLM_OXIDE_INTERNAL_BEHAVIOR_OUTPUT")
            .context("behavior output missing")?,
    );
    let parent = super::validate_private_temp_dir(
        destination
            .parent()
            .context("behavior output parent missing")?,
    )?;
    super::validate_destination_name(
        destination
            .file_name()
            .context("behavior output name missing")?,
    )?;
    let scenario: Scenario = serde_json::from_slice(&std::fs::read(plan)?)?;
    let ids = scenario
        .calls
        .iter()
        .map(|c| c.call_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    if scenario.calls.is_empty()
        || ids.len() != scenario.calls.len()
        || ids.contains("")
        || destination.exists()
        || std::env::var_os("VLLM_OXIDE_INTERNAL_BEHAVIOR_BINDING").is_some()
    {
        bail!("invalid/freshness-violating behavior scenario");
    }
    // Prevent recursion, while leaving every nested call on the production API path.
    std::env::remove_var(PLAN_ENV);
    let mut calls = Vec::new();
    for (index, call) in scenario.calls.into_iter().enumerate() {
        let binding_path = parent.join(format!("behavior-binding-{index}.json"));
        if binding_path.exists() {
            bail!("behavior binding already exists");
        }
        std::env::set_var("VLLM_OXIDE_INTERNAL_BEHAVIOR_BINDING", &binding_path);
        let prompts = call.prompts;
        let params = call
            .params
            .into_iter()
            .map(|p| crate::SamplingParams {
                max_tokens: p.max_tokens,
                ignore_eos: p.ignore_eos,
                temperature: p.temperature,
                top_k: p.top_k,
                top_p: p.top_p,
                presence_penalty: p.presence_penalty,
                frequency_penalty: p.frequency_penalty,
                repetition_penalty: p.repetition_penalty,
            })
            .collect::<Vec<_>>();
        let result = generate(&prompts, &params);
        std::env::remove_var("VLLM_OXIDE_INTERNAL_BEHAVIOR_BINDING");
        let binding: Option<serde_json::Value> = if binding_path.exists() {
            Some(serde_json::from_slice(&std::fs::read(binding_path)?)?)
        } else {
            None
        };
        let (outputs, error) = match result {
            Ok(outputs) => (
                outputs
                    .into_iter()
                    .map(|o| {
                        serde_json::json!({"request_id":o.request_id,
                "token_ids":o.token_ids,"text":o.text,"finished":o.finished})
                    })
                    .collect::<Vec<_>>(),
                None,
            ),
            Err(error) => (Vec::new(), Some(format!("{error:#}"))),
        };
        calls.push(serde_json::json!({"call_id":call.call_id,"outputs":outputs,"error":error,"binding":binding}));
    }
    super::write_atomic_json(
        &destination,
        &serde_json::json!({"protocol":"layered-accuracy-v1",
        "schema_version":1,"mode":"free_generation","calls":calls}),
    )
}

pub(crate) fn record_binding(
    ids: &[usize],
    prompt_lengths: &[usize],
    eos: &[u32],
    max_model_len: usize,
    device: &Device,
) -> Result<()> {
    let Some(destination) = std::env::var_os("VLLM_OXIDE_INTERNAL_BEHAVIOR_BINDING") else {
        return Ok(());
    };
    let destination = PathBuf::from(destination);
    let parent = super::validate_private_temp_dir(
        destination
            .parent()
            .context("behavior binding parent missing")?,
    )?;
    let name = destination
        .file_name()
        .context("behavior binding name missing")?;
    super::validate_destination_name(name)?;
    let destination = parent.join(name);
    super::write_atomic_json(
        &destination,
        &serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
        "mode":"behavior_binding","device":if device.is_cuda(){"cuda:0"}else{"cpu"},
        "request_ids":ids,"prompt_lengths":prompt_lengths,"eos_token_ids":eos,"max_model_len":max_model_len,
        "forcing_enabled":std::env::var_os(super::fixed_prefix::PLAN_ENV).is_some()}),
    )
}
