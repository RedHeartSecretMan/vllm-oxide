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
use crate::manifest;
use crate::prompts::PromptEntry;
use crate::report::ComparisonReport;
use crate::types::{Manifest, PromptCategory};

/// Flags that control which comparison layers run.
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
    canonical_prompts: &HashMap<String, PromptEntry>,
    opts: &DriverOptions,
) -> Result<ComparisonReport> {
    let mut report = ComparisonReport::default();

    for meta in &manifest.fixtures {
        let Some(prompt_entry) = canonical_prompts.get(&meta.prompt_id) else {
            tracing::warn!(
                "prompt_id '{}' not found in canonical.jsonl — skipping",
                meta.prompt_id
            );
            continue;
        };

        // Batch prompts (canonical_05) have sub_prompts — skip in v0.1.
        if prompt_entry.sub_prompts.is_some() {
            tracing::info!(
                "[{}] skipping batch prompt (not supported in v0.1 generate_logits)",
                meta.prompt_id
            );
            continue;
        }

        let fixture = manifest::load_fixture(&fixture_dir.join(&meta.filename), meta)?;
        let prompt = Prompt::Text(prompt_entry.prompt.clone());
        let max_tokens = meta.num_tokens as usize;

        let layer_label = match meta.category {
            PromptCategory::Canonical => "L1+L2",
            PromptCategory::Regression => "L1",
        };
        tracing::info!("[{}/{}] loading engine", meta.prompt_id, layer_label);
        let mut llm = LLM::new(Source::Local(model_path.to_path_buf()), EngineOptions::default())?;

        let logits = match llm.generate_logits(&prompt, max_tokens) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("generate_logits failed for {}: {e}", meta.prompt_id);
                continue;
            }
        };

        // Flatten to F32 once — shared by argmax extraction and L2 comparison.
        let logits_f32 = match logits.to_dtype(DType::F32) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("logits to_dtype failed: {e}");
                continue;
            }
        };
        let logits_vals = logits_f32.flatten_all()?.to_vec1::<f32>()?;
        let vocab_size = fixture.model_vocab_size(&logits_vals);
        let n_steps = logits.dims()[0];
        let generated_tokens = extract_greedy_tokens(&logits_vals, n_steps, vocab_size);

        match meta.category {
            PromptCategory::Canonical => {
                if !opts.l2_only {
                    let l1_result = compare_l1(
                        &fixture,
                        &generated_tokens,
                        Some(&logits),
                        &manifest.tolerance,
                        opts.epsilon,
                    )?;
                    report.l1_results.push(l1_result);
                }

                if !opts.l1_only {
                    // ADR-0005: L2 uses same-prefix comparison (skips divergent steps)
                    let l2_result =
                        compare_l2(&fixture, &logits_vals, &generated_tokens, &manifest.tolerance)?;
                    report.l2_results.push(l2_result);
                }

                if opts.debug {
                    let l3_result = compare_l3(manifest, fixture_dir, &meta.prompt_id)?;
                    report.l3_results.push(l3_result);
                }
            }
            PromptCategory::Regression => {
                if !opts.l2_only {
                    let l1_result = compare_l1_regression(
                        &fixture,
                        &generated_tokens,
                        &manifest.regression_skip_map,
                    )?;
                    report.l1_results.push(l1_result);
                }
            }
        }
    }

    Ok(report)
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
}
