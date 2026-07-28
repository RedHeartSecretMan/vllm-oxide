use std::collections::{HashMap, HashSet};

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::types::{FixtureData, ToleranceCalibration};

/// Result of an L1 token-sequence comparison.
#[derive(Debug)]
pub struct L1Result {
    pub prompt_id: String,
    pub passed: bool,
    pub total_positions: usize,
    pub exact_matches: usize,
    pub near_tie_skips: usize,
    /// Number of positions skipped via `regression_skip_map` (only non-zero
    /// during regression comparison). Stored separately from `near_tie_skips`
    /// to avoid semantic conflation — near-tie skips and regression skips
    /// have different causes.
    pub regression_skips: usize,
    pub mismatches: usize,
    /// First mismatch position (if any), 0-indexed in completion tokens.
    pub first_mismatch: Option<usize>,
    /// The epsilon used for near-tie detection.
    pub epsilon: f64,
    pub details: Vec<L1PositionDetail>,
}

#[derive(Debug)]
pub enum L1PositionDetail {
    Match { position: usize, token_id: i64 },
    NearTieSkip { position: usize, token_id: i64, gap: f64 },
    Mismatch { position: usize, expected: i64, actual: i64 },
    RegressionSkip { position: usize, token_id: i64 },
}

/// Compare generated token IDs against golden token IDs with near-tie
/// skipping. At positions where the top-2 logit gap < ε, the comparison
/// is skipped — these are inherently non-deterministic under BF16/FP16.
///
/// `generated_tokens` are the output from `LLM::generate` (greedy, temp=0).
/// `fixture` holds the golden token_ids.
/// `generated_logits` provides the raw logits for near-tie detection — shape
/// `[n, vocab_size]` (from `generate_logits`).
/// `tolerance` provides the calibrated `atol` which is used as the base for ε
/// (multiplied by the calibration factor for a safety margin, per T8 Q8.2).
pub fn compare_l1(
    fixture: &FixtureData,
    generated_tokens: &[u32],
    generated_logits: Option<&Tensor>,
    tolerance: &ToleranceCalibration,
    epsilon: Option<f64>,
) -> Result<L1Result> {
    let eps = epsilon.unwrap_or(tolerance.atol * 2.0);

    compare_tokens_loop(fixture, generated_tokens, eps, |i, expected, actual| {
        if expected == actual {
            return Ok(None);
        }
        let logits = match generated_logits {
            Some(l) => l,
            None => return Ok(Some(MismatchKind::Deterministic)),
        };
        let gap = top2_logit_gap(logits, i, expected as usize)?;
        if gap < eps {
            Ok(Some(MismatchKind::NearTie(gap)))
        } else {
            Ok(Some(MismatchKind::Deterministic))
        }
    })
}

/// Run L1 comparison without logits (no near-tie detection).
pub fn compare_l1_tokens_only(
    fixture: &FixtureData,
    generated_tokens: &[u32],
) -> Result<L1Result> {
    compare_tokens_loop(fixture, generated_tokens, 0.0, |_, expected, actual| {
        if expected == actual {
            Ok(None)
        } else {
            Ok(Some(MismatchKind::Deterministic))
        }
    })
}

/// Compare generated token IDs against golden token IDs for regression
/// fixtures, using a `skip_map` to skip positions where vLLM also disagrees
/// with transformers.
///
/// Regression fixtures do not have full logits, so near-tie ε-based gap
/// detection is not available. Instead, this function uses a pre-computed
/// `skip_map` where the key is a prompt_id and the value is a list of token
/// positions to skip. The `skip_map` records positions where vLLM (the
/// reference BF16 engine) disagrees with transformers; all other positions
/// must match the golden token_ids exactly.
///
/// This delegates to `compare_tokens_loop` with a classifier that consults
/// the skip_map, rather than duplicating the token-loop logic.
pub fn compare_l1_regression(
    fixture: &FixtureData,
    generated_tokens: &[u32],
    skip_map: &HashMap<String, Vec<usize>>,
) -> Result<L1Result> {
    let skips: HashSet<usize> = skip_map
        .get(&fixture.prompt_id)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();

    compare_tokens_loop(fixture, generated_tokens, 0.0, |i, expected, actual| {
        if skips.contains(&i) {
            return Ok(Some(MismatchKind::RegressionSkip));
        }
        if expected == actual {
            Ok(None)
        } else {
            Ok(Some(MismatchKind::Deterministic))
        }
    })
}

enum MismatchKind {
    Deterministic,
    NearTie(f64),
    RegressionSkip,
}

fn compare_tokens_loop<F>(
    fixture: &FixtureData,
    generated_tokens: &[u32],
    eps: f64,
    classify: F,
) -> Result<L1Result>
where
    F: Fn(usize, i64, i64) -> Result<Option<MismatchKind>>,
{
    let n = fixture.token_ids.len().min(generated_tokens.len());
    let mut details = Vec::with_capacity(n);
    let mut exact_matches = 0usize;
    let mut near_tie_skips = 0usize;
    let mut regression_skips = 0usize;
    let mut mismatches = 0usize;
    let mut first_mismatch: Option<usize> = None;

    for i in 0..n {
        let expected = fixture.token_ids[i];
        let actual = generated_tokens[i] as i64;

        match classify(i, expected, actual)? {
            None => {
                exact_matches += 1;
                details.push(L1PositionDetail::Match { position: i, token_id: expected });
            }
            Some(MismatchKind::NearTie(gap)) => {
                near_tie_skips += 1;
                details.push(L1PositionDetail::NearTieSkip { position: i, token_id: expected, gap });
            }
            Some(MismatchKind::Deterministic) => {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
                details.push(L1PositionDetail::Mismatch { position: i, expected, actual });
            }
            Some(MismatchKind::RegressionSkip) => {
                regression_skips += 1;
                details.push(L1PositionDetail::RegressionSkip { position: i, token_id: expected });
            }
        }
    }

    mismatches += (generated_tokens.len().saturating_sub(n))
        + (fixture.token_ids.len().saturating_sub(n));

    Ok(L1Result {
        prompt_id: fixture.prompt_id.clone(),
        passed: mismatches == 0,
        total_positions: fixture.token_ids.len().max(generated_tokens.len()),
        exact_matches,
        near_tie_skips,
        regression_skips,
        mismatches,
        first_mismatch,
        epsilon: eps,
        details,
    })
}

/// Compute the gap between the top-2 logits at a given position.
///
/// Extracts row `position` from the logits tensor `[n, vocab_size]`, finds
/// the top two values, and returns their difference. If the top-2 includes
/// the expected token, returns the gap from the expected token to the next
/// highest. Otherwise returns infinity (hard mismatch).
fn top2_logit_gap(logits: &Tensor, position: usize, expected_token: usize) -> Result<f64> {
    let row = logits.get(position)?;
    let _vocab_size = row.dims()[0];
    let row_f32 = row.to_dtype(DType::F32)?;
    let values = row_f32.to_vec1::<f32>()?;

    // Find top-2 values.
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    let mut top1_idx = 0usize;

    for (idx, &val) in values.iter().enumerate() {
        if val > top1 {
            top2 = top1;
            top1 = val;
            top1_idx = idx;
        } else if val > top2 {
            top2 = val;
        }
    }

    if top1_idx == expected_token {
        Ok((top1 - top2) as f64)
    } else {
        // Expected token not in top-2 — hard mismatch, no near-tie.
        Ok(f64::INFINITY)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_tolerance() -> ToleranceCalibration {
        ToleranceCalibration {
            atol: 1e-5,
            observed_max_abs_diff: 0.1,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        }
    }

    #[test]
    fn exact_match_all_tokens() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 3,
            token_ids: vec![1, 2, 3],
            n_prompt_tokens: 5,
            logits: None,
            logits_shape: (3, 10),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![1, 2, 3];
        let tolerance = make_tolerance();

        let result = compare_l1(&fixture, &generated, None, &tolerance, None).unwrap();
        assert!(result.passed);
        assert_eq!(result.exact_matches, 3);
        assert_eq!(result.mismatches, 0);
        assert_eq!(result.near_tie_skips, 0);
    }

    #[test]
    fn token_mismatch_detected() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 3,
            token_ids: vec![1, 2, 3],
            n_prompt_tokens: 5,
            logits: None,
            logits_shape: (3, 10),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![1, 99, 3];
        let tolerance = make_tolerance();

        let result = compare_l1(&fixture, &generated, None, &tolerance, None).unwrap();
        assert!(!result.passed);
        assert_eq!(result.exact_matches, 2);
        assert_eq!(result.mismatches, 1);
    }

    #[test]
    fn length_mismatch_detected() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Regression,
            oracle: crate::types::OracleName::Vllm,
            num_tokens: 5,
            token_ids: vec![1, 2, 3, 4, 5],
            n_prompt_tokens: 3,
            logits: None,
            logits_shape: (0, 0),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![1, 2, 3];

        let tolerance = make_tolerance();

        let result = compare_l1(&fixture, &generated, None, &tolerance, None).unwrap();
        assert!(!result.passed);
        assert_eq!(result.mismatches, 2); // 2 extra in fixture
    }

    #[test]
    fn regression_skip_map_skips_known_positions() {
        let fixture = FixtureData {
            prompt_id: "reg-test".into(),
            category: crate::types::PromptCategory::Regression,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 4,
            token_ids: vec![10, 20, 30, 40],
            n_prompt_tokens: 3,
            logits: None,
            logits_shape: (0, 0),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![10, 99, 30, 40];

        let mut skip_map = HashMap::new();
        skip_map.insert("reg-test".to_string(), vec![1]); // position 1 is where vLLM also disagrees

        let result = compare_l1_regression(&fixture, &generated, &skip_map).unwrap();
        assert!(result.passed);
        assert_eq!(result.exact_matches, 3);
        assert_eq!(result.regression_skips, 1);
        assert_eq!(result.mismatches, 0);
    }

    #[test]
    fn regression_skip_map_hard_mismatch() {
        let fixture = FixtureData {
            prompt_id: "reg-test".into(),
            category: crate::types::PromptCategory::Regression,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 4,
            token_ids: vec![10, 20, 30, 40],
            n_prompt_tokens: 3,
            logits: None,
            logits_shape: (0, 0),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![10, 99, 30, 40];

        // Empty skip_map — position 1 should be a hard mismatch.
        let skip_map = HashMap::new();
        let result = compare_l1_regression(&fixture, &generated, &skip_map).unwrap();
        assert!(!result.passed);
        assert_eq!(result.exact_matches, 3);
        assert_eq!(result.regression_skips, 0);
        assert_eq!(result.mismatches, 1);
    }
}
