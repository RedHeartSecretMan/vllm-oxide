//! GPU-owning benchmark orchestration and device-wide memory monitoring.

use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
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
    let pids = String::from_utf8(output.stdout)
        .context("nvidia-smi process output is not UTF-8")?
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter(|pid| *pid != std::process::id())
        .collect::<Vec<_>>();
    if !pids.is_empty() {
        bail!("unrelated CUDA compute processes are active: {pids:?}");
    }
    Ok(())
}

struct ActiveMemoryMonitor {
    child: Child,
    reader: Option<thread::JoinHandle<Result<Vec<MemorySample>>>>,
}

impl ActiveMemoryMonitor {
    fn start() -> Result<Self> {
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
                "--loop-ms=50",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting 50 ms GPU memory monitor")?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("GPU memory monitor stdout is unavailable"))?;
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let reader = thread::spawn(move || -> Result<Vec<MemorySample>> {
            let started = Instant::now();
            let mut samples = Vec::new();
            for line in BufReader::new(stdout).lines() {
                let line = line.context("reading GPU memory monitor output")?;
                let used_mib = line
                    .trim()
                    .parse::<u64>()
                    .with_context(|| format!("parsing GPU memory sample {line:?}"))?;
                let elapsed = if samples.is_empty() {
                    0
                } else {
                    u64::try_from(started.elapsed().as_millis())
                        .context("GPU memory monitor duration exceeds u64")?
                };
                samples.push(MemorySample {
                    elapsed_ms: elapsed,
                    used_mib,
                });
                if samples.len() == 1 {
                    let _ = ready_sender.send(());
                }
            }
            Ok(samples)
        });
        let monitor = Self {
            child,
            reader: Some(reader),
        };
        ready_receiver
            .recv_timeout(Duration::from_secs(5))
            .context("GPU memory monitor did not produce a post-initialization baseline")?;
        Ok(monitor)
    }

    fn finish(mut self) -> Result<MemoryMonitorEvidence> {
        if self.child.try_wait()?.is_some() {
            bail!("GPU memory monitor exited before the final synchronization");
        }
        self.child.kill().context("stopping GPU memory monitor")?;
        self.child
            .wait()
            .context("waiting for GPU memory monitor")?;
        let samples = self
            .reader
            .take()
            .ok_or_else(|| anyhow::anyhow!("GPU memory reader missing"))?
            .join()
            .map_err(|_| anyhow::anyhow!("GPU memory monitor reader panicked"))??;
        MemoryMonitorEvidence::validate(50, false, &[], samples)
    }
}

impl Drop for ActiveMemoryMonitor {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
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
    let measurement =
        validate_measurement_identity(repo_root, measurement_commit, measurement_tree)?;
    validate_running_binary(repo_root)?;
    let output_dir = output_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("benchmark output must have a parent directory"))?;
    if output_path.exists() || output_path.is_symlink() {
        bail!("benchmark output must be a fresh non-existing path");
    }
    let mut workloads = BTreeMap::new();
    for workload in approved_workloads() {
        let workload_inputs = workload_prompts(&workload, prompts)?;
        let params = fixed_sampling_params(workload_inputs.len());

        ensure_no_unrelated_compute_processes()?;
        let mut throwaway = LLM::new(
            Source::Local(model_path.to_path_buf()),
            fixed_engine_options(),
        )?;
        let throwaway_outputs = throwaway.generate(&workload_inputs, &params)?;
        if throwaway_outputs
            .iter()
            .any(|output| !output.finished || output.token_ids.len() != 64)
        {
            bail!("throwaway benchmark run did not produce exactly 64 tokens per request");
        }
        drop(throwaway);

        let mut repetitions = Vec::new();
        for repetition in 1..=workload.measured_repetitions {
            ensure_no_unrelated_compute_processes()?;
            let mut llm = LLM::new(
                Source::Local(model_path.to_path_buf()),
                fixed_engine_options(),
            )?;
            let destination = format!("{}-repetition-{repetition}.telemetry.json", workload.id);
            let call_id = format!("{}-repetition-{repetition}", workload.id);
            let monitor = ActiveMemoryMonitor::start()?;
            let guard = BenchmarkEnvironmentGuard::install(output_dir, &destination, &call_id)?;
            let outputs = llm.generate(&workload_inputs, &params);
            drop(guard);
            let memory = monitor.finish()?;
            let outputs = outputs?;
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
        workloads.insert(workload.id.to_string(), aggregate_workload(repetitions)?);
    }
    let evidence = BenchmarkRunEvidence {
        schema_version: 1,
        measurement_commit: measurement.commit,
        measurement_tree: measurement.tree,
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
