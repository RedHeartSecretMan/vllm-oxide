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

struct MemoryReadings {
    samples: Vec<MemorySample>,
    maximum_gap: Duration,
}

fn observed_interval(gap: Duration) -> Result<u64> {
    if gap.is_zero() || gap > Duration::from_millis(50) {
        bail!("actual GPU memory sampling interval must be positive and at most 50 ms");
    }
    u64::try_from(gap.as_nanos().div_ceil(1_000_000)).context("GPU memory interval exceeds u64")
}

struct ActiveMemoryMonitor {
    child: Child,
    reader: Option<thread::JoinHandle<Result<MemoryReadings>>>,
}

impl ActiveMemoryMonitor {
    fn start() -> Result<Self> {
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
                "--loop-ms=10",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting 10 ms GPU memory monitor")?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("GPU memory monitor stdout is unavailable"))?;
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let reader = thread::spawn(move || -> Result<MemoryReadings> {
            let mut started = None;
            let mut previous = None;
            let mut maximum_gap = Duration::ZERO;
            let mut samples = Vec::new();
            for line in BufReader::new(stdout).lines() {
                let line = line.context("reading GPU memory monitor output")?;
                let used_mib = line
                    .trim()
                    .parse::<u64>()
                    .with_context(|| format!("parsing GPU memory sample {line:?}"))?;
                let observed = Instant::now();
                if let Some(last) = previous.replace(observed) {
                    let gap = observed.duration_since(last);
                    observed_interval(gap)?; // Enforce before any millisecond truncation.
                    maximum_gap = maximum_gap.max(gap);
                }
                let baseline = *started.get_or_insert(observed);
                let elapsed = u64::try_from(observed.duration_since(baseline).as_millis())
                    .context("GPU memory monitor duration exceeds u64")?;
                samples.push(MemorySample {
                    elapsed_ms: elapsed,
                    used_mib,
                });
                if samples.len() == 1 {
                    let _ = ready_sender.send(());
                }
            }
            Ok(MemoryReadings {
                samples,
                maximum_gap,
            })
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
        let readings = self
            .reader
            .take()
            .ok_or_else(|| anyhow::anyhow!("GPU memory reader missing"))?
            .join()
            .map_err(|_| anyhow::anyhow!("GPU memory monitor reader panicked"))??;
        // Round the actual maximum interval upward; the reader already
        // rejected sub-millisecond violations of the unchanged 50 ms ceiling.
        let interval = observed_interval(readings.maximum_gap)?;
        MemoryMonitorEvidence::validate(interval, false, &[], readings.samples)
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
        workloads.insert(
            workload.id.to_string(),
            aggregate_workload(repetitions, discarded_warm_outputs)?,
        );
    }
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

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a guarded NVIDIA telemetry owner"]
    fn actual_memory_monitor_preserves_the_fifty_millisecond_ceiling() {
        let monitor = ActiveMemoryMonitor::start().unwrap();
        std::thread::sleep(Duration::from_millis(500));
        let evidence = monitor.finish().unwrap();
        assert!((1..=50).contains(&evidence.polling_interval_ms));
        assert!(evidence.sample_count >= 2);
        assert_eq!(evidence.samples[0].elapsed_ms, 0);
    }
}
