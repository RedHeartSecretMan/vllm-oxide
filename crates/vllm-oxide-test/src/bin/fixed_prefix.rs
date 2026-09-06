//! Private versioned capture entrypoint, through the unchanged public generate seam.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use vllm_oxide::{EngineOptions, Prompt, SamplingParams, Source, LLM};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    repo_root: PathBuf,
    #[arg(long)]
    measurement_commit: String,
    #[arg(long)]
    measurement_tree: String,
    #[arg(long, required_unless_present_any = ["operator_plan", "behavior_plan"])]
    plan: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    options: PathBuf,
    #[arg(long)]
    setup_plan: Vec<PathBuf>,
    #[arg(long)]
    control: bool,
    #[arg(long, conflicts_with_all = ["plan", "behavior_plan", "control", "setup_plan"])]
    operator_plan: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["plan", "operator_plan", "control", "setup_plan"])]
    behavior_plan: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    case_id: String,
    member_id: String,
    prompt: Vec<u32>,
    continuation: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    protocol: String,
    schema_version: u32,
    execution_group_id: String,
    call_id: String,
    vocab_size: usize,
    members: Vec<Member>,
}

fn load_plan(path: &Path) -> Result<Plan> {
    let plan: Plan = serde_json::from_slice(&std::fs::read(path)?)?;
    if plan.protocol != "layered-accuracy-v1"
        || plan.schema_version != 1
        || plan.vocab_size != 151_936
        || plan.execution_group_id.is_empty()
        || plan.call_id.is_empty()
        || plan.members.is_empty()
        || plan.members.iter().any(|m| {
            m.case_id.is_empty()
                || m.member_id.is_empty()
                || m.prompt.is_empty()
                || m.continuation.is_empty()
        })
    {
        bail!("invalid release fixed-prefix plan");
    }
    Ok(plan)
}

fn engine_options(path: &Path) -> Result<(EngineOptions, Option<usize>)> {
    let values: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let object = values
        .as_object()
        .context("engine options must be an object")?;
    if object.keys().any(|k| {
        !matches!(
            k.as_str(),
            "max_num_batched_tokens"
                | "max_num_seqs"
                | "max_model_len"
                | "gpu_memory_utilization"
                | "enforce_eager"
                | "test_kv_blocks"
                | "baseline_blocks"
        )
    }) {
        bail!("unknown fixed-prefix engine option");
    }
    let mut options = EngineOptions {
        gpu_memory_utilization: 0.5,
        dtype: Some(candle_core::DType::BF16),
        ..EngineOptions::default()
    };
    for (key, target) in [
        (
            "max_num_batched_tokens",
            &mut options.max_num_batched_tokens,
        ),
        ("max_num_seqs", &mut options.max_num_seqs),
        ("max_model_len", &mut options.max_model_len),
    ] {
        if let Some(value) = object.get(key) {
            *target = usize::try_from(value.as_u64().context("invalid positive engine count")?)?;
        }
    }
    if let Some(value) = object.get("gpu_memory_utilization") {
        options.gpu_memory_utilization = serde_json::from_value(value.clone())?;
    }
    if let Some(value) = object.get("enforce_eager") {
        options.enforce_eager = value.as_bool().context("invalid eager flag")?;
    }
    let capacity = object
        .get("test_kv_blocks")
        .map(|v| {
            v.as_u64()
                .context("invalid private KV capacity")
                .and_then(|v| usize::try_from(v).map_err(Into::into))
        })
        .transpose()?;
    Ok((options, capacity))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    vllm_oxide_test::measurement::validate_deterministic_environment()?;
    vllm_oxide_test::measurement::validate_measurement_identity(
        &cli.repo_root,
        &cli.measurement_commit,
        &cli.measurement_tree,
    )?;
    vllm_oxide_test::measurement::validate_running_binary(&cli.repo_root)?;
    let target = cli.plan.as_deref().map(load_plan).transpose()?;
    let setups = cli
        .setup_plan
        .iter()
        .map(|path| load_plan(path))
        .collect::<Result<Vec<_>>>()?;
    let (options, capacity) = engine_options(&cli.options)?;
    for plan in setups.iter().chain(target.iter()) {
        if plan.members.iter().any(|m| {
            m.prompt
                .len()
                .checked_add(m.continuation.len())
                .map_or(true, |n| n > options.max_model_len)
        }) {
            bail!("fixed-prefix prompt plus completion budget exceeds declared context");
        }
    }
    for name in [
        "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN",
        "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_OUTPUT",
        "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CONTROL",
        "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CACHE_BLOCKS",
        "VLLM_OXIDE_INTERNAL_GOLDEN_TEMP_DIR",
        "VLLM_OXIDE_INTERNAL_LAYER_TRACE_DIR",
        "VLLM_OXIDE_INTERNAL_BENCHMARK_TEMP_DIR",
        "VLLM_OXIDE_INTERNAL_OPERATOR_PLAN",
        "VLLM_OXIDE_INTERNAL_OPERATOR_OUTPUT",
        "VLLM_OXIDE_INTERNAL_BEHAVIOR_PLAN",
        "VLLM_OXIDE_INTERNAL_BEHAVIOR_OUTPUT",
        "VLLM_OXIDE_INTERNAL_BEHAVIOR_BINDING",
    ] {
        if std::env::var_os(name).is_some() {
            bail!("stale private execution environment: {name}");
        }
    }
    if let Some(plan) = &cli.plan {
        std::env::set_var("VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN", plan);
    }
    if let Some(capacity) = capacity {
        if cli.plan.is_none() {
            bail!("private KV capacity requires a replay plan");
        }
        std::env::set_var(
            "VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CACHE_BLOCKS",
            capacity.to_string(),
        );
    }
    vllm_oxide_test::measurement::validate_release_model(&cli.model_path)?;
    let mut llm = LLM::new(Source::Local(cli.model_path), options)?;
    if let Some(plan) = cli.operator_plan {
        std::env::remove_var("VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN");
        std::env::remove_var("VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CACHE_BLOCKS");
        std::env::set_var("VLLM_OXIDE_INTERNAL_OPERATOR_PLAN", plan);
        std::env::set_var("VLLM_OXIDE_INTERNAL_OPERATOR_OUTPUT", &cli.output);
        llm.generate(&[], &[])?;
        println!(
            "{}",
            serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
            "source":{"commit":cli.measurement_commit,"tree":cli.measurement_tree},"producer_pid":std::process::id(),
            "build_source_id":env!("VLLM_OXIDE_BUILD_SOURCE_ID"),"cuda_feature_enabled":cfg!(feature="cuda")})
        );
        return Ok(());
    }
    if let Some(plan) = cli.behavior_plan {
        std::env::set_var("VLLM_OXIDE_INTERNAL_BEHAVIOR_PLAN", plan);
        std::env::set_var("VLLM_OXIDE_INTERNAL_BEHAVIOR_OUTPUT", &cli.output);
        llm.generate(&[], &[])?;
        println!(
            "{}",
            serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
            "source":{"commit":cli.measurement_commit,"tree":cli.measurement_tree},"producer_pid":std::process::id(),
            "build_source_id":env!("VLLM_OXIDE_BUILD_SOURCE_ID"),"cuda_feature_enabled":cfg!(feature="cuda")})
        );
        return Ok(());
    }
    let target = target.context("fixed-prefix plan missing")?;
    let target_path = cli.plan.context("fixed-prefix plan path missing")?;
    for (index, (path, plan)) in cli
        .setup_plan
        .iter()
        .zip(&setups)
        .chain(std::iter::once((&target_path, &target)))
        .enumerate()
    {
        let is_target = index == setups.len();
        let output = if is_target {
            cli.output.clone()
        } else {
            cli.output
                .parent()
                .context("output parent missing")?
                .join(format!("setup-{index}.capture.json"))
        };
        std::env::set_var("VLLM_OXIDE_INTERNAL_FIXED_PREFIX_PLAN", path);
        std::env::set_var("VLLM_OXIDE_INTERNAL_FIXED_PREFIX_OUTPUT", output);
        if cli.control {
            std::env::set_var("VLLM_OXIDE_INTERNAL_FIXED_PREFIX_CONTROL", "1");
        }
        let prompts = plan
            .members
            .iter()
            .map(|m| Prompt::TokenIds(m.prompt.clone()))
            .collect::<Vec<_>>();
        let params = plan
            .members
            .iter()
            .map(|m| SamplingParams {
                max_tokens: m.continuation.len(),
                ignore_eos: true,
                ..SamplingParams::default()
            })
            .collect::<Vec<_>>();
        llm.generate(&prompts, &params)?;
    }
    println!(
        "{}",
        serde_json::json!({"protocol":"layered-accuracy-v1","schema_version":1,
        "source":{"commit":cli.measurement_commit,"tree":cli.measurement_tree},
        "producer_pid":std::process::id(),"build_source_id":env!("VLLM_OXIDE_BUILD_SOURCE_ID"),"cuda_feature_enabled":cfg!(feature="cuda")})
    );
    Ok(())
}
