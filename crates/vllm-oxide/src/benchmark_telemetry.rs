//! Private release-benchmark timing calculations from ADR-0012.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StepPhase {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Emission {
    pub(crate) request_id: usize,
    pub(crate) completion_step: usize,
    pub(crate) sampled_at_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StepSample {
    pub(crate) phase: StepPhase,
    pub(crate) started_ns: u64,
    pub(crate) ended_ns: u64,
    pub(crate) prefill_tokens: usize,
    pub(crate) emissions: Vec<Emission>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct BenchmarkTelemetry {
    pub(crate) prefill_tokens: usize,
    pub(crate) prefill_duration_ns: u64,
    pub(crate) prefill_tokens_per_second: f64,
    pub(crate) decode_tokens: usize,
    pub(crate) decode_duration_ns: u64,
    pub(crate) decode_tokens_per_second: f64,
    pub(crate) time_to_first_token_ns: Vec<(usize, u64)>,
    pub(crate) inter_token_latency_ns: Vec<(usize, u64)>,
    pub(crate) steps: Vec<StepSample>,
}

impl BenchmarkTelemetry {
    pub(crate) fn from_samples(request_ids: &[usize], steps: Vec<StepSample>) -> Result<Self> {
        if request_ids.is_empty() || steps.is_empty() {
            bail!("benchmark telemetry requires requests and synchronized step samples");
        }
        let request_set = request_ids.iter().copied().collect::<HashSet<_>>();
        if request_set.len() != request_ids.len() {
            bail!("benchmark telemetry received duplicate request ids");
        }

        let mut previous_end = 0u64;
        let mut next_completion = request_ids
            .iter()
            .copied()
            .map(|request_id| (request_id, 0usize))
            .collect::<HashMap<_, _>>();
        let mut last_sampled_at = HashMap::new();
        let mut time_to_first = HashMap::new();
        let mut inter_token_latency_ns = Vec::new();
        let mut prefill_tokens = 0usize;
        let mut prefill_duration_ns = 0u64;
        let mut decode_tokens = 0usize;
        let mut decode_duration_ns = 0u64;

        for step in &steps {
            if step
                .emissions
                .iter()
                .map(|e| e.request_id)
                .collect::<HashSet<_>>()
                .len()
                != step.emissions.len()
            {
                bail!("benchmark permits only one token per request in a step");
            }
            if step.started_ns < previous_end
                || step.ended_ns <= step.started_ns
                || step
                    .emissions
                    .iter()
                    .any(|emission| emission.sampled_at_ns != step.ended_ns)
            {
                bail!("benchmark telemetry steps are not synchronized and monotonic");
            }
            previous_end = step.ended_ns;
            let duration = step.ended_ns - step.started_ns;
            if step.prefill_tokens > 0 {
                if step.phase == StepPhase::Decode {
                    bail!("decode-only telemetry step cannot contain prefill tokens");
                }
                prefill_tokens = prefill_tokens
                    .checked_add(step.prefill_tokens)
                    .ok_or_else(|| anyhow::anyhow!("prefill token count overflow"))?;
                prefill_duration_ns = prefill_duration_ns
                    .checked_add(duration)
                    .ok_or_else(|| anyhow::anyhow!("prefill duration overflow"))?;
            }
            if step
                .emissions
                .iter()
                .any(|emission| emission.completion_step > 0)
            {
                decode_duration_ns = decode_duration_ns
                    .checked_add(duration)
                    .ok_or_else(|| anyhow::anyhow!("decode duration overflow"))?;
            }
            for emission in &step.emissions {
                let expected = next_completion
                    .get_mut(&emission.request_id)
                    .ok_or_else(|| anyhow::anyhow!("telemetry references an unknown request"))?;
                if emission.completion_step != *expected {
                    bail!("telemetry completion steps are duplicate or non-contiguous");
                }
                *expected += 1;
                if emission.completion_step == 0 {
                    time_to_first.insert(emission.request_id, emission.sampled_at_ns);
                } else {
                    decode_tokens += 1;
                    let previous = last_sampled_at[&emission.request_id];
                    inter_token_latency_ns
                        .push((emission.request_id, emission.sampled_at_ns - previous));
                }
                last_sampled_at.insert(emission.request_id, emission.sampled_at_ns);
            }
        }
        if prefill_tokens == 0
            || prefill_duration_ns == 0
            || decode_tokens == 0
            || decode_duration_ns == 0
        {
            bail!("benchmark telemetry is missing required prefill or post-first decode samples");
        }
        let time_to_first_token_ns = request_ids
            .iter()
            .map(|request_id| {
                time_to_first
                    .get(request_id)
                    .copied()
                    .map(|duration| (*request_id, duration))
                    .ok_or_else(|| anyhow::anyhow!("benchmark request has no first-token sample"))
            })
            .collect::<Result<Vec<_>>>()?;

        #[allow(clippy::cast_precision_loss)]
        let prefill_tokens_per_second =
            prefill_tokens as f64 * 1_000_000_000.0 / prefill_duration_ns as f64;
        #[allow(clippy::cast_precision_loss)]
        let decode_tokens_per_second =
            decode_tokens as f64 * 1_000_000_000.0 / decode_duration_ns as f64;
        Ok(Self {
            prefill_tokens,
            prefill_duration_ns,
            prefill_tokens_per_second,
            decode_tokens,
            decode_duration_ns,
            decode_tokens_per_second,
            time_to_first_token_ns,
            inter_token_latency_ns,
            steps,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{BenchmarkTelemetry, Emission, StepPhase, StepSample};

    #[test]
    fn one_request_cannot_emit_multiple_tokens_in_one_step() {
        let step = StepSample {
            phase: StepPhase::Decode,
            started_ns: 0,
            ended_ns: 10,
            prefill_tokens: 0,
            emissions: (0..2)
                .map(|completion_step| Emission {
                    request_id: 0,
                    completion_step,
                    sampled_at_ns: 10,
                })
                .collect(),
        };
        let error = BenchmarkTelemetry::from_samples(&[0], vec![step]).unwrap_err();
        assert!(error.to_string().contains("one token per request"));
    }

    #[test]
    fn fixed_step_samples_produce_the_approved_metric_formulas() {
        let samples = vec![
            StepSample {
                phase: StepPhase::Prefill,
                started_ns: 0,
                ended_ns: 10_000_000,
                prefill_tokens: 6,
                emissions: vec![Emission {
                    request_id: 7,
                    completion_step: 0,
                    sampled_at_ns: 10_000_000,
                }],
            },
            StepSample {
                phase: StepPhase::Decode,
                started_ns: 10_000_000,
                ended_ns: 14_000_000,
                prefill_tokens: 0,
                emissions: vec![
                    Emission {
                        request_id: 7,
                        completion_step: 1,
                        sampled_at_ns: 14_000_000,
                    },
                    Emission {
                        request_id: 9,
                        completion_step: 0,
                        sampled_at_ns: 14_000_000,
                    },
                ],
            },
            StepSample {
                phase: StepPhase::Decode,
                started_ns: 14_000_000,
                ended_ns: 20_000_000,
                prefill_tokens: 0,
                emissions: vec![
                    Emission {
                        request_id: 7,
                        completion_step: 2,
                        sampled_at_ns: 20_000_000,
                    },
                    Emission {
                        request_id: 9,
                        completion_step: 1,
                        sampled_at_ns: 20_000_000,
                    },
                ],
            },
        ];

        let report = BenchmarkTelemetry::from_samples(&[7, 9], samples).unwrap();

        assert_eq!(report.prefill_tokens, 6);
        assert_eq!(report.prefill_duration_ns, 10_000_000);
        assert_eq!(report.prefill_tokens_per_second, 600.0);
        assert_eq!(report.decode_tokens, 3);
        assert_eq!(report.decode_duration_ns, 10_000_000);
        assert_eq!(report.decode_tokens_per_second, 300.0);
        assert_eq!(
            report.time_to_first_token_ns,
            vec![(7, 10_000_000), (9, 14_000_000)]
        );
        assert_eq!(
            report.inter_token_latency_ns,
            vec![(7, 4_000_000), (7, 6_000_000), (9, 6_000_000)]
        );
    }
}
