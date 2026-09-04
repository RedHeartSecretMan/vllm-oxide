//! Golden comparison driver — single loop over all fixtures, dispatching to
//! the appropriate comparison layer per fixture category.
//!
//! This module owns the "ceremony" that [`main`](crate) previously duplicated
//! across canonical and regression loops: prompt lookup, fixture loading,
//! engine init, private diagnostic capture, artifact validation, and
//! comparison dispatch. Adding a new fixture category is one new `match` arm here —
//! zero changes to the CLI entrypoint.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use vllm_oxide::{Prompt, Source, LLM};

use crate::benchmark::fixed_engine_options;
use crate::capture::{generate_with_capture, generate_with_preserved_capture};
use crate::l1::{compare_l1, compare_l1_tokens_only};
use crate::l2::compare_l2;
use crate::l3::compare_l3;
use crate::lifecycle::{preflight, LifecyclePreflight, ReferenceCase};
use crate::prompts::PromptEntry;
use crate::report::ComparisonReport;
use crate::types::{Manifest, PromptCategory, RequiredComparison};

/// Flags that control which comparison layers run.
#[derive(Debug, Clone)]
pub struct DriverOptions {
    pub l1_only: bool,
    pub l2_only: bool,
    pub debug: bool,
    pub capture_dir: Option<std::path::PathBuf>,
}

/// Run all golden comparisons described by `manifest`.
///
/// Iterates every fixture once, loading the engine per fixture, generating
/// captured logits and selected tokens, and dispatching to the comparison layer
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
        fixed_engine_options(),
    )?;
    let captured = if let Some(capture_dir) = &opts.capture_dir {
        let destination = capture_dir.join(format!("{}.candidate.jsonl", case.metadata.prompt_id));
        generate_with_preserved_capture(
            &mut llm,
            prompt,
            max_tokens,
            &case.expected.fixture_id,
            &destination,
        )?
        .0
    } else {
        generate_with_capture(&mut llm, prompt, max_tokens, &case.expected.fixture_id)?
    };
    let (captured_steps, captured_vocab_size) = captured.tensor_shape;
    let (_n_steps, vocab_size) = generated_logits_geometry(
        &[captured_steps, captured_vocab_size],
        captured.logits.len(),
        manifest.model.vocab_size,
    )?;
    let generated_tokens = captured
        .tokens_by_input
        .first()
        .ok_or_else(|| anyhow::anyhow!("diagnostic capture has no request rows"))?;

    let (l1, l2) = match case.metadata.category {
        PromptCategory::Canonical => (
            Some(compare_l1(
                &case.fixture,
                generated_tokens,
                Some((&captured.logits, vocab_size)),
                &manifest.tolerance_policy,
            )?),
            Some(compare_l2(
                &case.fixture,
                &captured.logits,
                generated_tokens,
                &manifest.tolerance_policy,
            )?),
        ),
        PromptCategory::Regression => (
            Some(compare_l1_tokens_only(
                &case.fixture,
                generated_tokens,
                &manifest.tolerance_policy,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

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
            capture_dir: None,
        };
        let l1_only = DriverOptions {
            l1_only: true,
            ..all_layers.clone()
        };
        let l2_only = DriverOptions {
            l1_only: false,
            l2_only: true,
            ..all_layers.clone()
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
