//! ADR-0012 release benchmark plan, statistics, and memory-monitor evidence.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::DType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vllm_oxide::{EngineOptions, Prompt, SamplingParams};

use crate::prompts::PromptEntry;

mod runner;
pub use runner::run_release_benchmark;

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkWorkload {
    pub id: &'static str,
    pub prompt_ids: Vec<&'static str>,
    pub throwaway_runs: usize,
    pub measured_repetitions: usize,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub gpu_memory_utilization: f32,
    pub max_model_len: usize,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
}

pub fn approved_workloads() -> Vec<BenchmarkWorkload> {
    let fixed = |id, prompt_ids| BenchmarkWorkload {
        id,
        prompt_ids,
        throwaway_runs: 1,
        measured_repetitions: 3,
        max_tokens: 64,
        ignore_eos: true,
        gpu_memory_utilization: 0.50,
        max_model_len: 4096,
        max_num_batched_tokens: 16_384,
        max_num_seqs: 512,
    };
    vec![
        fixed("canonical_04", vec!["canonical_04"]),
        fixed(
            "canonical_05",
            vec![
                "canonical_05a",
                "canonical_05b",
                "canonical_05c",
                "canonical_05d",
            ],
        ),
    ]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DurationSummary {
    pub samples: Vec<u64>,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
}

pub fn summarize_durations(samples: &[u64]) -> Result<DurationSummary> {
    if samples.is_empty() {
        bail!("duration summary requires at least one raw sample");
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    #[allow(clippy::cast_precision_loss)]
    let mean = sorted.iter().map(|value| *value as f64).sum::<f64>() / sorted.len() as f64;
    Ok(DurationSummary {
        samples: samples.to_vec(),
        mean,
        p50: linear_percentile(&sorted, 0.50),
        p95: linear_percentile(&sorted, 0.95),
    })
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn linear_percentile(sorted: &[u64], quantile: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let index = quantile * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    #[allow(clippy::cast_precision_loss)]
    let lower_value = sorted[lower] as f64;
    #[allow(clippy::cast_precision_loss)]
    let upper_value = sorted[upper] as f64;
    lower_value + (upper_value - lower_value) * (index - lower as f64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemorySample {
    pub elapsed_ms: u64,
    pub used_mib: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMonitorEvidence {
    pub polling_interval_ms: u64,
    pub sample_count: usize,
    pub baseline_mib: u64,
    pub peak_mib: u64,
    pub delta_mib: u64,
    pub samples: Vec<MemorySample>,
    pub other_compute_processes: Vec<u32>,
}

impl MemoryMonitorEvidence {
    pub fn validate(
        polling_interval_ms: u64,
        interrupted: bool,
        other_compute_processes: &[u32],
        samples: Vec<MemorySample>,
    ) -> Result<Self> {
        if polling_interval_ms == 0 || polling_interval_ms > 50 {
            bail!("GPU memory polling interval must be in 1..=50 ms");
        }
        if interrupted {
            bail!("GPU memory monitor was interrupted");
        }
        if !other_compute_processes.is_empty() {
            bail!("unrelated CUDA compute process was active at benchmark start");
        }
        if samples.len() < 2 || samples[0].elapsed_ms != 0 {
            bail!("GPU memory monitor requires a post-initialization baseline and peak samples");
        }
        if samples.windows(2).any(|pair| {
            pair[1].elapsed_ms <= pair[0].elapsed_ms
                || pair[1].elapsed_ms - pair[0].elapsed_ms > polling_interval_ms
        }) {
            bail!("GPU memory samples are interrupted or exceed the configured interval");
        }
        let baseline_mib = samples[0].used_mib;
        let peak_mib = samples
            .iter()
            .map(|sample| sample.used_mib)
            .max()
            .ok_or_else(|| anyhow::anyhow!("GPU memory sample set is empty"))?;
        Ok(Self {
            polling_interval_ms,
            sample_count: samples.len(),
            baseline_mib,
            peak_mib,
            delta_mib: peak_mib.saturating_sub(baseline_mib),
            samples,
            other_compute_processes: other_compute_processes.to_vec(),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEmission {
    request_id: usize,
    completion_step: usize,
    sampled_at_ns: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStepSample {
    phase: String,
    started_ns: u64,
    ended_ns: u64,
    prefill_tokens: usize,
    emissions: Vec<RawEmission>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTelemetry {
    prefill_tokens: usize,
    prefill_duration_ns: u64,
    prefill_tokens_per_second: f64,
    decode_tokens: usize,
    decode_duration_ns: u64,
    decode_tokens_per_second: f64,
    time_to_first_token_ns: Vec<(usize, u64)>,
    inter_token_latency_ns: Vec<(usize, u64)>,
    steps: Vec<RawStepSample>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrivateTelemetryArtifact {
    format: String,
    schema_version: u32,
    call_id: String,
    request_ids: Vec<usize>,
    complete: bool,
    telemetry: RawTelemetry,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkRepetitionEvidence {
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub time_to_first_token_ns: Vec<u64>,
    pub inter_token_latency_ns: DurationSummary,
    pub memory: MemoryMonitorEvidence,
    pub telemetry_artifact_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkloadBenchmarkEvidence {
    pub repetitions: Vec<BenchmarkRepetitionEvidence>,
    pub headline_prefill_tokens_per_second: f64,
    pub headline_decode_tokens_per_second: f64,
    pub headline_time_to_first_token_ns: f64,
    pub headline_peak_memory_mib: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkRunEvidence {
    pub schema_version: u32,
    pub measurement_commit: String,
    pub measurement_tree: String,
    pub workloads: BTreeMap<String, WorkloadBenchmarkEvidence>,
}

fn validate_private_telemetry(
    path: &Path,
    expected_call_id: &str,
    expected_request_count: usize,
) -> Result<(RawTelemetry, String)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading benchmark telemetry {}", path.display()))?;
    let artifact: PrivateTelemetryArtifact = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing benchmark telemetry {}", path.display()))?;
    if artifact.format != "vllm-oxide-internal-benchmark-json-v1"
        || artifact.schema_version != 1
        || artifact.call_id != expected_call_id
        || !artifact.complete
        || artifact.request_ids.len() != expected_request_count
        || artifact
            .request_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            != expected_request_count
    {
        bail!("benchmark telemetry header, completion, or request identity mismatch");
    }
    let telemetry = artifact.telemetry;
    let mut completion_steps = artifact
        .request_ids
        .iter()
        .copied()
        .map(|request_id| (request_id, 0usize))
        .collect::<HashMap<_, _>>();
    let mut previous_end = 0u64;
    let mut prefill_tokens = 0usize;
    let mut prefill_duration_ns = 0u64;
    let mut decode_tokens = 0usize;
    let mut decode_duration_ns = 0u64;
    for step in &telemetry.steps {
        if !matches!(step.phase.as_str(), "prefill" | "decode" | "mixed")
            || step.started_ns < previous_end
            || step.ended_ns <= step.started_ns
            || step
                .emissions
                .iter()
                .any(|emission| emission.sampled_at_ns != step.ended_ns)
        {
            bail!("benchmark telemetry contains an invalid synchronized step");
        }
        previous_end = step.ended_ns;
        let duration = step.ended_ns - step.started_ns;
        if step.prefill_tokens > 0 {
            prefill_tokens += step.prefill_tokens;
            prefill_duration_ns += duration;
        }
        if step
            .emissions
            .iter()
            .any(|emission| emission.completion_step > 0)
        {
            decode_duration_ns += duration;
        }
        for emission in &step.emissions {
            let expected = completion_steps
                .get_mut(&emission.request_id)
                .ok_or_else(|| anyhow::anyhow!("telemetry emission has an unknown request"))?;
            if emission.completion_step != *expected {
                bail!("telemetry completion steps are duplicate or non-contiguous");
            }
            *expected += 1;
            decode_tokens += usize::from(emission.completion_step > 0);
        }
    }
    if completion_steps.values().any(|count| *count != 64)
        || prefill_tokens != telemetry.prefill_tokens
        || prefill_duration_ns != telemetry.prefill_duration_ns
        || decode_tokens != telemetry.decode_tokens
        || decode_duration_ns != telemetry.decode_duration_ns
    {
        bail!("benchmark telemetry does not contain exactly 64 tokens per request or recompute");
    }
    #[allow(clippy::cast_precision_loss)]
    let prefill_rate = prefill_tokens as f64 * 1_000_000_000.0 / prefill_duration_ns as f64;
    #[allow(clippy::cast_precision_loss)]
    let decode_rate = decode_tokens as f64 * 1_000_000_000.0 / decode_duration_ns as f64;
    if (prefill_rate - telemetry.prefill_tokens_per_second).abs() > f64::EPSILON
        || (decode_rate - telemetry.decode_tokens_per_second).abs() > f64::EPSILON
        || telemetry.time_to_first_token_ns.len() != expected_request_count
        || telemetry.inter_token_latency_ns.len() != expected_request_count * 63
    {
        bail!("benchmark telemetry derived metrics do not match raw synchronized samples");
    }
    Ok((telemetry, format!("{:x}", Sha256::digest(&bytes))))
}

pub(crate) fn fixed_engine_options() -> EngineOptions {
    EngineOptions {
        max_num_batched_tokens: 16_384,
        max_num_seqs: 512,
        max_model_len: 4096,
        gpu_memory_utilization: 0.50,
        enforce_eager: true,
        dtype: Some(DType::BF16),
    }
}

fn fixed_sampling_params(count: usize) -> Vec<SamplingParams> {
    vec![
        SamplingParams {
            max_tokens: 64,
            ignore_eos: true,
            ..SamplingParams::default()
        };
        count
    ]
}

fn workload_prompts(
    workload: &BenchmarkWorkload,
    prompts: &HashMap<String, PromptEntry>,
) -> Result<Vec<Prompt>> {
    workload
        .prompt_ids
        .iter()
        .map(|prompt_id| {
            prompts
                .get(*prompt_id)
                .map(|entry| Prompt::Text(entry.prompt.clone()))
                .ok_or_else(|| anyhow::anyhow!("benchmark prompt is missing: {prompt_id}"))
        })
        .collect()
}

fn median_f64(mut values: Vec<f64>) -> Result<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("benchmark headline requires finite samples");
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

#[allow(clippy::cast_precision_loss)]
fn aggregate_workload(
    repetitions: Vec<BenchmarkRepetitionEvidence>,
) -> Result<WorkloadBenchmarkEvidence> {
    if repetitions.len() != 3 {
        bail!("benchmark workload requires exactly three measured repetitions");
    }
    let headline_prefill_tokens_per_second = median_f64(
        repetitions
            .iter()
            .map(|repetition| repetition.prefill_tokens_per_second)
            .collect(),
    )?;
    let headline_decode_tokens_per_second = median_f64(
        repetitions
            .iter()
            .map(|repetition| repetition.decode_tokens_per_second)
            .collect(),
    )?;
    let headline_time_to_first_token_ns = median_f64(
        repetitions
            .iter()
            .flat_map(|repetition| repetition.time_to_first_token_ns.iter())
            .map(|value| *value as f64)
            .collect(),
    )?;
    let headline_peak_memory_mib = median_f64(
        repetitions
            .iter()
            .map(|repetition| repetition.memory.peak_mib as f64)
            .collect(),
    )?;
    Ok(WorkloadBenchmarkEvidence {
        repetitions,
        headline_prefill_tokens_per_second,
        headline_decode_tokens_per_second,
        headline_time_to_first_token_ns,
        headline_peak_memory_mib,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{approved_workloads, summarize_durations, MemoryMonitorEvidence, MemorySample};

    #[test]
    fn workloads_fix_single_and_four_request_batches_with_three_fresh_repetitions() {
        let workloads = approved_workloads();

        assert_eq!(workloads.len(), 2);
        assert_eq!(workloads[0].id, "canonical_04");
        assert_eq!(workloads[0].prompt_ids, ["canonical_04"]);
        assert_eq!(workloads[1].id, "canonical_05");
        assert_eq!(
            workloads[1].prompt_ids,
            [
                "canonical_05a",
                "canonical_05b",
                "canonical_05c",
                "canonical_05d"
            ]
        );
        for workload in workloads {
            assert_eq!(workload.throwaway_runs, 1);
            assert_eq!(workload.measured_repetitions, 3);
            assert_eq!(workload.max_tokens, 64);
            assert!(workload.ignore_eos);
            assert_eq!(workload.gpu_memory_utilization, 0.50);
            assert_eq!(workload.max_model_len, 4096);
            assert_eq!(workload.max_num_batched_tokens, 16384);
            assert_eq!(workload.max_num_seqs, 512);
        }
    }

    #[test]
    fn duration_summary_keeps_all_samples_and_reports_mean_p50_p95() {
        let summary = summarize_durations(&[1, 2, 3, 4, 5]).unwrap();

        assert_eq!(summary.samples, [1, 2, 3, 4, 5]);
        assert_eq!(summary.mean, 3.0);
        assert_eq!(summary.p50, 3.0);
        assert_eq!(summary.p95, 4.8);
    }

    #[test]
    fn memory_monitor_records_baseline_peak_delta_and_rejects_invalid_monitoring() {
        let evidence = MemoryMonitorEvidence::validate(
            50,
            false,
            &[],
            vec![
                MemorySample {
                    elapsed_ms: 0,
                    used_mib: 2_000,
                },
                MemorySample {
                    elapsed_ms: 50,
                    used_mib: 3_250,
                },
            ],
        )
        .unwrap();

        assert_eq!(evidence.baseline_mib, 2_000);
        assert_eq!(evidence.peak_mib, 3_250);
        assert_eq!(evidence.delta_mib, 1_250);
        assert_eq!(evidence.sample_count, 2);

        assert!(MemoryMonitorEvidence::validate(51, false, &[], vec![]).is_err());
        assert!(MemoryMonitorEvidence::validate(50, true, &[], vec![]).is_err());
        assert!(MemoryMonitorEvidence::validate(50, false, &[1234], vec![]).is_err());
    }
}
