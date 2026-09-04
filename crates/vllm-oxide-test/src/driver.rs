//! Golden comparison driver — single loop over all fixtures, dispatching to
//! the appropriate comparison layer per fixture category.
//!
//! This module owns the "ceremony" that [`main`](crate) previously duplicated
//! across canonical and regression loops: prompt lookup, fixture loading,
//! engine init, `generate_logits`, logits flattening, and greedy-token
//! extraction. Adding a new fixture category is one new `match` arm here —
//! zero changes to the CLI entrypoint.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::DType;
use vllm_oxide::{EngineOptions, Prompt, Source, LLM};

use crate::l1::{compare_l1, compare_l1_regression};
use crate::l2::compare_l2;
use crate::l3::compare_l3;
use crate::lifecycle::{preflight, LifecyclePreflight, ReferenceCase};
use crate::prompts::PromptEntry;
use crate::report::ComparisonReport;
use crate::types::{Manifest, PromptCategory, RequiredComparison};

/// Flags that control which comparison layers run.
#[derive(Debug, Clone, Copy)]
pub struct DriverOptions {
    pub l1_only: bool,
    pub l2_only: bool,
    pub debug: bool,
    pub epsilon: Option<f64>,
}

/// Run all golden comparisons described by `manifest`.
///
/// Iterates every fixture once, loading the engine per fixture, generating
/// logits, extracting greedy tokens, and dispatching to the comparison layer
/// indicated by `FixtureMetadata::category`.
///
/// Returns a [`ComparisonReport`] with all L1/L2/L3 results collected.
/// The caller is responsible for printing or serialising the report.
pub fn run_comparison(
    manifest: &Manifest,
    fixture_dir: &Path,
    model_path: &Path,
    prompts: &HashMap<String, PromptEntry>,
    opts: &DriverOptions,
) -> Result<ComparisonReport> {
    let prepared = preflight(manifest, fixture_dir, prompts);
    Ok(run_prepared_comparisons(prepared, opts, |case| {
        compare_reference_case(case, manifest, fixture_dir, model_path, opts)
    }))
}

pub(crate) fn run_prepared_comparisons<F>(
    prepared: LifecyclePreflight,
    opts: &DriverOptions,
    mut compare: F,
) -> ComparisonReport
where
    F: FnMut(&ReferenceCase) -> Result<CaseComparison>,
{
    let mut tracker = prepared.tracker;
    let mut report = ComparisonReport {
        failures: prepared.errors,
        ..ComparisonReport::default()
    };

    for case in prepared.reference_cases {
        let fixture_id = case.expected.fixture_id.clone();
        if tracker.was_skipped(&fixture_id)
            || !required_layers_enabled(&case.expected.required_comparison, opts)
        {
            tracker.record_skipped(&fixture_id);
            continue;
        }
        match compare(&case) {
            Ok(comparison) => {
                let passed = comparison.l1.as_ref().map_or(true, |result| result.passed)
                    && comparison.l2.as_ref().map_or(true, |result| result.passed);
                if let Some(result) = comparison.l1 {
                    report.l1_results.push(result);
                }
                if let Some(result) = comparison.l2 {
                    report.l2_results.push(result);
                }
                if let Some(result) = comparison.l3 {
                    report.l3_results.push(result);
                }
                tracker.record_compared(&fixture_id);
                if !passed {
                    tracker.record_failed(&fixture_id);
                }
            }
            Err(error) => {
                tracker.record_failed(&fixture_id);
                report
                    .failures
                    .push(format!("fixture {fixture_id} comparison failed: {error:#}"));
            }
        }
    }

    report.lifecycle = tracker.totals();
    report
}

pub(crate) struct CaseComparison {
    pub(crate) l1: Option<crate::l1::L1Result>,
    pub(crate) l2: Option<crate::l2::L2Result>,
    pub(crate) l3: Option<crate::l3::L3Result>,
}

fn required_layers_enabled(required: &RequiredComparison, opts: &DriverOptions) -> bool {
    match required {
        RequiredComparison::L1 => !opts.l2_only,
        RequiredComparison::L1L2 => !opts.l1_only && !opts.l2_only,
        RequiredComparison::Calibration => false,
    }
}

fn compare_reference_case(
    case: &ReferenceCase,
    manifest: &Manifest,
    fixture_dir: &Path,
    model_path: &Path,
    opts: &DriverOptions,
) -> Result<CaseComparison> {
    let prompt = Prompt::Text(case.prompt.prompt.clone());
    let max_tokens = case.metadata.num_tokens as usize;
    let layer_label = match case.metadata.category {
        PromptCategory::Canonical => "L1+L2",
        PromptCategory::Regression => "L1",
    };
    tracing::info!(
        "[{}/{}] loading engine",
        case.metadata.prompt_id,
        layer_label
    );
    let mut llm = LLM::new(
        Source::Local(model_path.to_path_buf()),
        EngineOptions::default(),
    )?;
    let logits = llm.generate_logits(&prompt, max_tokens)?;
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vals = logits_f32.flatten_all()?.to_vec1::<f32>()?;
    let (n_steps, vocab_size) =
        generated_logits_geometry(logits.dims(), logits_vals.len(), manifest.model.vocab_size)?;
    let generated_tokens = extract_greedy_tokens(&logits_vals, n_steps, vocab_size);

    let (l1, l2) = match case.metadata.category {
        PromptCategory::Canonical => (
            Some(compare_l1(
                &case.fixture,
                &generated_tokens,
                Some(&logits),
                &manifest.tolerance,
                opts.epsilon,
            )?),
            Some(compare_l2(
                &case.fixture,
                &logits_vals,
                &generated_tokens,
                &manifest.tolerance,
            )?),
        ),
        PromptCategory::Regression => (
            Some(compare_l1_regression(
                &case.fixture,
                &generated_tokens,
                &manifest.regression_skip_map,
            )?),
            None,
        ),
    };
    let l3 = if opts.debug {
        Some(compare_l3(manifest, fixture_dir, &case.metadata.prompt_id)?)
    } else {
        None
    };
    Ok(CaseComparison { l1, l2, l3 })
}

fn generated_logits_geometry(
    dims: &[usize],
    value_count: usize,
    expected_vocab_size: usize,
) -> Result<(usize, usize)> {
    let [n_steps, vocab_size] = dims else {
        anyhow::bail!("generated logits must have shape [steps, vocab], got {dims:?}");
    };
    if *n_steps == 0 || *vocab_size == 0 {
        anyhow::bail!("generated logits must have non-zero steps and vocabulary width");
    }
    if *vocab_size != expected_vocab_size {
        anyhow::bail!(
            "generated logits vocabulary width {} does not match manifest vocabulary width {}",
            vocab_size,
            expected_vocab_size
        );
    }
    if n_steps.checked_mul(*vocab_size) != Some(value_count) {
        anyhow::bail!(
            "generated logits shape {dims:?} does not match flattened value count {value_count}"
        );
    }
    Ok((*n_steps, *vocab_size))
}

/// Extract greedy tokens from flat F32 logits via per-step argmax.
///
/// `logits_vals` is a row-major flat array of shape `[n_steps * vocab_size]`.
/// Returns one token id per step.
fn extract_greedy_tokens(logits_vals: &[f32], n_steps: usize, vocab_size: usize) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let start = step * vocab_size;
        let end = start + vocab_size;
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0u32;
        // vocab size ≤ 200k; truncation impossible
        #[allow(clippy::cast_possible_truncation)]
        for (j, &val) in logits_vals[start..end].iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = j as u32;
            }
        }
        tokens.push(max_idx);
    }
    tokens
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn argmax_extracts_greedy_tokens() {
        // 2 steps × 4 vocab: row 0 → argmax at idx 2 (val 0.9),
        // row 1 → argmax at idx 0 (val 0.7).
        let logits: Vec<f32> = vec![0.1, 0.2, 0.9, 0.3, 0.7, 0.5, 0.1, 0.4];
        let tokens = extract_greedy_tokens(&logits, 2, 4);
        assert_eq!(tokens, vec![2, 0]);
    }

    #[test]
    fn argmax_single_step() {
        let logits: Vec<f32> = vec![0.1, 0.8, 0.3];
        let tokens = extract_greedy_tokens(&logits, 1, 3);
        assert_eq!(tokens, vec![1]);
    }

    #[test]
    fn runtime_shape_supplies_vocab_width_for_regression_fixture() {
        let (n_steps, vocab_size) = generated_logits_geometry(&[2, 3], 6, 3).unwrap();

        assert_eq!(n_steps, 2);
        assert_eq!(vocab_size, 3);
    }

    #[test]
    fn runtime_vocab_width_must_match_manifest() {
        let error = generated_logits_geometry(&[2, 4], 8, 3).unwrap_err();

        assert!(error.to_string().contains("manifest vocabulary width"));
    }

    #[test]
    fn batch_and_regression_declared_layers_control_release_comparison() {
        let all_layers = DriverOptions {
            l1_only: false,
            l2_only: false,
            debug: false,
            epsilon: None,
        };
        let l1_only = DriverOptions {
            l1_only: true,
            ..all_layers
        };
        let l2_only = DriverOptions {
            l1_only: false,
            l2_only: true,
            ..all_layers
        };

        assert!(required_layers_enabled(
            &RequiredComparison::L1L2,
            &all_layers
        ));
        assert!(!required_layers_enabled(
            &RequiredComparison::L1L2,
            &l1_only
        ));
        assert!(required_layers_enabled(&RequiredComparison::L1, &l1_only));
        assert!(!required_layers_enabled(&RequiredComparison::L1, &l2_only));
    }
}
