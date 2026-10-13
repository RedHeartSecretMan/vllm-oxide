//! GPU-owning benchmark with a record-only direct NVML memory method.

use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use vllm_oxide::{Source, LLM};

use super::{
    aggregate_workload, approved_workloads, fixed_engine_options, fixed_sampling_params,
    summarize_durations, validate_private_telemetry, workload_prompts, BenchmarkRepetitionEvidence,
    BenchmarkRunEvidence, MemoryMonitorEvidence, MemorySample,
};
use crate::measurement::{validate_measurement_identity, validate_running_binary};
use crate::prompts::PromptEntry;

#[path = "nvml_monitor.rs"]
mod nvml_monitor;
use nvml_monitor::ActiveMemoryMonitor;

fn ensure_no_unrelated_compute_processes() -> Result<()> {
    let output = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        .output()
        .context("querying CUDA compute processes before benchmark")?;
    if !output.status.success() {
        bail!(
            "nvidia-smi compute-process query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let pids = parse_compute_processes(
        &String::from_utf8(output.stdout).context("nvidia-smi process output is not UTF-8")?,
        std::process::id(),
    )?;
    if !pids.is_empty() {
        bail!("unrelated CUDA compute processes are active: {pids:?}");
    }
    Ok(())
}

fn parse_compute_processes(output: &str, own_pid: u32) -> Result<Vec<u32>> {
    let mut others = Vec::new();
    for line in output.lines() {
        let pid = line
            .trim()
            .parse::<u32>()
            .context("malformed CUDA compute-process identity")?;
        if pid == 0 {
            bail!("CUDA compute-process identity must be positive");
        }
        if pid != own_pid {
            others.push(pid);
        }
    }
    Ok(others)
}

fn observed_interval(gap: Duration) -> Result<u64> {
    if gap.is_zero() || gap > Duration::from_millis(50) {
        bail!("actual GPU memory sampling interval must be positive and at most 50 ms");
    }
    u64::try_from(gap.as_nanos().div_ceil(1_000_000)).context("GPU memory interval exceeds u64")
}

struct StageJournal {
    path: PathBuf,
    origin: Instant,
    events: Vec<serde_json::Value>,
    completed: bool,
    saved: bool,
}

impl StageJournal {
    fn new(directory: &Path) -> Result<Self> {
        let path = directory.join("benchmark-stages.json");
        if path.exists() || path.is_symlink() {
            bail!("benchmark stage journal must be fresh");
        }
        Ok(Self {
            path,
            origin: Instant::now(),
            events: Vec::with_capacity(64),
            completed: false,
            saved: false,
        })
    }

    fn mark(&mut self, call: &str, phase: &'static str) -> Result<()> {
        if self.events.len() == 64 {
            bail!("bounded benchmark stage journal exhausted");
        }
        self.events.push(serde_json::json!({"call_id":call,"phase":phase,"elapsed_ns":self.origin.elapsed().as_nanos()}));
        Ok(())
    }

    fn persist(&mut self) -> Result<()> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)?;
        serde_json::to_writer_pretty(
            &mut output,
            &serde_json::json!({
                "protocol":"memory-method-stage-record-v1","schema_version":1,"method":"direct-nvml-v1","accepting":false,"record_only":true,
                "clock":"shared Rust Instant origin with every NVML monitor trace",
                "completed":self.completed,"events":self.events,"build_source_id":env!("VLLM_OXIDE_BUILD_SOURCE_ID")
            }),
        )?;
        output.write_all(b"\n")?;
        output.sync_all()?;
        self.saved = true;
        Ok(())
    }
}

impl Drop for StageJournal {
    fn drop(&mut self) {
        if !self.saved {
            if let Err(error) = self.persist() {
                eprintln!("NVML_MEMORY_STAGE_PRESERVATION_FAILURE {error:#}");
            }
        }
    }
}

const RAW_CAPTURE_ENV: [&str; 3] = [
    "VLLM_OXIDE_INTERNAL_GOLDEN_TEMP_DIR",
    "VLLM_OXIDE_INTERNAL_GOLDEN_DESTINATION",
    "VLLM_OXIDE_INTERNAL_GOLDEN_CALL_ID",
];
const BENCHMARK_ENV: [&str; 3] = [
    "VLLM_OXIDE_INTERNAL_BENCHMARK_TEMP_DIR",
    "VLLM_OXIDE_INTERNAL_BENCHMARK_DESTINATION",
    "VLLM_OXIDE_INTERNAL_BENCHMARK_CALL_ID",
];

struct BenchmarkEnvironmentGuard;

impl BenchmarkEnvironmentGuard {
    fn install(directory: &Path, destination: &str, call_id: &str) -> Result<Self> {
        if RAW_CAPTURE_ENV
            .iter()
            .chain(BENCHMARK_ENV.iter())
            .any(|name| std::env::var_os(name).is_some())
        {
            bail!("benchmark process contains stale diagnostic environment configuration");
        }
        std::env::set_var(BENCHMARK_ENV[0], directory);
        std::env::set_var(BENCHMARK_ENV[1], destination);
        std::env::set_var(BENCHMARK_ENV[2], call_id);
        Ok(Self)
    }
}

impl Drop for BenchmarkEnvironmentGuard {
    fn drop(&mut self) {
        for name in BENCHMARK_ENV {
            std::env::remove_var(name);
        }
    }
}

pub fn run_release_benchmark(
    model_path: &Path,
    prompts: &HashMap<String, PromptEntry>,
    output_path: &Path,
    repo_root: &Path,
    measurement_commit: &str,
    measurement_tree: &str,
) -> Result<BenchmarkRunEvidence> {
    if std::env::vars_os()
        .any(|(name, _)| name.to_string_lossy().starts_with("VLLM_OXIDE_INTERNAL_"))
    {
        bail!("performance requires a fresh process without forcing or capture configuration");
    }
    crate::measurement::validate_deterministic_environment()?;
    let measurement =
        validate_measurement_identity(repo_root, measurement_commit, measurement_tree)?;
    validate_running_binary(repo_root)?;
    crate::measurement::validate_release_model(model_path)?;
    let output_dir = output_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("benchmark output must have a parent directory"))?;
    if output_path.exists() || output_path.is_symlink() {
        bail!("benchmark output must be a fresh non-existing path");
    }
    let mut stages = StageJournal::new(output_dir)?;
    let mut workloads = BTreeMap::new();
    for workload in approved_workloads() {
        let workload_inputs = workload_prompts(&workload, prompts)?;
        let params = fixed_sampling_params(workload_inputs.len());

        ensure_no_unrelated_compute_processes()?;
        stages.mark(workload.id, "discarded_warm_llm_new_entered")?;
        let mut throwaway = LLM::new(
            Source::Local(model_path.to_path_buf()),
            fixed_engine_options(),
        )?;
        stages.mark(workload.id, "discarded_warm_llm_new_returned")?;
        let throwaway_outputs = throwaway.generate(&workload_inputs, &params)?;
        if throwaway_outputs
            .iter()
            .any(|output| !output.finished || output.token_ids.len() != 64)
        {
            bail!("throwaway benchmark run did not produce exactly 64 tokens per request");
        }
        stages.mark(workload.id, "discarded_warm_generate_completed")?;
        drop(throwaway);
        stages.mark(workload.id, "discarded_warm_dropped")?;
        let discarded_warm_outputs = throwaway_outputs
            .iter()
            .map(|o| {
                serde_json::json!({
            "request_id":o.request_id,"token_ids":o.token_ids,"text":o.text,"finished":o.finished})
            })
            .collect();

        let mut repetitions = Vec::new();
        for repetition in 1..=workload.measured_repetitions {
            ensure_no_unrelated_compute_processes()?;
            let call_id = format!("{}-repetition-{repetition}", workload.id);
            stages.mark(&call_id, "measured_llm_new_entered")?;
            let mut llm = LLM::new(
                Source::Local(model_path.to_path_buf()),
                fixed_engine_options(),
            )?;
            let destination = format!("{}-repetition-{repetition}.telemetry.json", workload.id);
            stages.mark(&call_id, "measured_llm_new_returned")?;
            let mut monitor =
                ActiveMemoryMonitor::start(output_dir, &call_id, stages.origin, &measurement)?;
            let guard = BenchmarkEnvironmentGuard::install(output_dir, &destination, &call_id)?;
            stages.mark(&call_id, "measured_generate_entered")?;
            monitor.mark("generate_invoked_ns");
            let outputs = llm.generate(&workload_inputs, &params);
            monitor.mark("generate_returned_ns");
            stages.mark(&call_id, "measured_generate_returned")?;
            if outputs.is_ok() {
                // LLM::generate synchronizes every benchmark step end before returning.
                monitor.mark("generate_succeeded_after_internal_final_step_sync_ns");
                stages.mark(
                    &call_id,
                    "measured_generate_succeeded_after_internal_final_step_sync",
                )?;
            }
            drop(guard);
            let memory = monitor.finish();
            let (outputs, memory) = match (outputs, memory) {
                (Ok(outputs), Ok(memory)) => (outputs, memory),
                (Err(primary), Err(secondary)) => {
                    return Err(primary.context(format!(
                        "generate primary failure; monitor secondary failure: {secondary:#}"
                    )))
                }
                (Err(error), _) | (_, Err(error)) => return Err(error),
            };
            stages.mark(&call_id, "final_memory_sample_and_worker_cleanup_completed")?;
            if outputs.len() != workload_inputs.len()
                || outputs
                    .iter()
                    .any(|output| !output.finished || output.token_ids.len() != 64)
            {
                bail!("measured benchmark run did not produce exactly 64 tokens per request");
            }
            let telemetry_path = output_dir.join(&destination);
            let (telemetry, telemetry_artifact_sha256) =
                validate_private_telemetry(&telemetry_path, &call_id, workload_inputs.len())?;
            let time_to_first_token_ns = telemetry
                .time_to_first_token_ns
                .iter()
                .map(|(_request_id, duration)| *duration)
                .collect();
            let inter_token_samples = telemetry
                .inter_token_latency_ns
                .iter()
                .map(|(_request_id, duration)| *duration)
                .collect::<Vec<_>>();
            repetitions.push(BenchmarkRepetitionEvidence {
                prefill_tokens_per_second: telemetry.prefill_tokens_per_second,
                decode_tokens_per_second: telemetry.decode_tokens_per_second,
                time_to_first_token_ns,
                inter_token_latency_ns: summarize_durations(&inter_token_samples)?,
                memory,
                telemetry_artifact_sha256,
            });
            drop(llm);
        }
        workloads.insert(
            workload.id.to_string(),
            aggregate_workload(repetitions, discarded_warm_outputs)?,
        );
    }
    stages.completed = true;
    stages.persist()?;
    let evidence = BenchmarkRunEvidence {
        protocol: "layered-accuracy-v1",
        schema_version: 1,
        measurement_commit: measurement.commit,
        measurement_tree: measurement.tree,
        build_source_id: env!("VLLM_OXIDE_BUILD_SOURCE_ID"),
        cuda_feature_enabled: cfg!(feature = "cuda"),
        producer_pid: std::process::id(),
        workloads,
    };
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .with_context(|| format!("creating benchmark evidence {}", output_path.display()))?;
    serde_json::to_writer_pretty(&mut output, &evidence)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(evidence)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn malformed_compute_process_output_cannot_hide_another_owner() {
        assert!(parse_compute_processes("N/A\n", 10).is_err());
        assert!(parse_compute_processes("10\nbad PID\n", 10).is_err());
        assert!(parse_compute_processes("0\n", 10).is_err());
        assert_eq!(parse_compute_processes("10\n20\n", 10).unwrap(), [20]);
        assert!(parse_compute_processes("", 10).unwrap().is_empty());
    }

    #[test]
    fn sub_millisecond_polling_overruns_are_rejected_before_rounding() {
        assert_eq!(observed_interval(Duration::from_millis(50)).unwrap(), 50);
        assert_eq!(
            observed_interval(Duration::from_micros(10_434)).unwrap(),
            11
        );
        assert!(observed_interval(Duration::from_micros(50_001)).is_err());
        assert!(observed_interval(Duration::from_micros(50_429)).is_err());
        assert!(observed_interval(Duration::ZERO).is_err());
    }
}
