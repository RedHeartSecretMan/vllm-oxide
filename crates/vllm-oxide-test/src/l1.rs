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
    let eps = epsilon.unwrap_or_else(|| {
        // Use 2× the calibrated atol as near-tie epsilon (T8 Q8.2 guidance).
        tolerance.atol * 2.0
    });

    let n = fixture.token_ids.len().min(generated_tokens.len());
    let mut details = Vec::with_capacity(n);
    let mut exact_matches = 0usize;
    let mut near_tie_skips = 0usize;
    let mut mismatches = 0usize;
    let mut first_mismatch: Option<usize> = None;

    for i in 0..n {
        let expected = fixture.token_ids[i];
        let actual = generated_tokens[i] as i64;

        if expected == actual {
            exact_matches += 1;
            details.push(L1PositionDetail::Match {
                position: i,
                token_id: expected,
            });
        } else if let Some(ref logits) = generated_logits {
            // Check near-tie: if top-2 gap < ε, skip.
            let gap = top2_logit_gap(logits, i, expected as usize)?;
            if gap < eps {
                near_tie_skips += 1;
                details.push(L1PositionDetail::NearTieSkip {
                    position: i,
                    token_id: expected,
                    gap,
                });
            } else {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
                details.push(L1PositionDetail::Mismatch {
                    position: i,
                    expected,
                    actual,
                });
            }
        } else {
            // No logits available — treat as hard mismatch.
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
            details.push(L1PositionDetail::Mismatch {
                position: i,
                expected,
                actual,
            });
        }
    }

    // Positions beyond min(n, generated.len()) are also mismatches.
    if generated_tokens.len() > n {
        for i in n..generated_tokens.len() {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
        }
    }
    if n < fixture.token_ids.len() {
        for i in n..fixture.token_ids.len() {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
        }
    }

    Ok(L1Result {
        prompt_id: fixture.prompt_id.clone(),
        passed: mismatches == 0,
        total_positions: fixture.token_ids.len().max(generated_tokens.len()),
        exact_matches,
        near_tie_skips,
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

/// Run L1 comparison without logits (no near-tie detection).
///
/// This is for regression fixtures where full logits aren't stored.
pub fn compare_l1_tokens_only(
    fixture: &FixtureData,
    generated_tokens: &[u32],
) -> L1Result {
    let n = fixture.token_ids.len().min(generated_tokens.len());
    let mut details = Vec::with_capacity(n);
    let mut exact_matches = 0usize;
    let mut mismatches = 0usize;
    let mut first_mismatch: Option<usize> = None;

    for i in 0..n {
        let expected = fixture.token_ids[i];
        let actual = generated_tokens[i] as i64;

        if expected == actual {
            exact_matches += 1;
            details.push(L1PositionDetail::Match {
                position: i,
                token_id: expected,
            });
        } else {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
            details.push(L1PositionDetail::Mismatch {
                position: i,
                expected,
                actual,
            });
        }
    }

    // Handle length mismatches.
    let extra_mismatches = (generated_tokens.len().saturating_sub(n))
        + (fixture.token_ids.len().saturating_sub(n));
    mismatches += extra_mismatches;

    L1Result {
        prompt_id: fixture.prompt_id.clone(),
        passed: mismatches == 0,
        total_positions: fixture.token_ids.len().max(generated_tokens.len()),
        exact_matches,
        near_tie_skips: 0,
        mismatches,
        first_mismatch,
        epsilon: 0.0,
        details,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

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
        let tolerance = ToleranceCalibration {
            atol: 1e-5,
            rtol: 1e-3,
            observed_max_l2: 0.1,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        };

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
        let tolerance = ToleranceCalibration {
            atol: 1e-5,
            rtol: 1e-3,
            observed_max_l2: 0.1,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        };

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
            oracle: crate::types::OracleName::VllmV1,
            num_tokens: 5,
            token_ids: vec![1, 2, 3, 4, 5],
            n_prompt_tokens: 3,
            logits: None,
            logits_shape: (0, 0),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![1, 2, 3];

        let tolerance = ToleranceCalibration {
            atol: 1e-5,
            rtol: 1e-3,
            observed_max_l2: 0.1,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        };

        let result = compare_l1(&fixture, &generated, None, &tolerance, None).unwrap();
        assert!(!result.passed);
        assert_eq!(result.mismatches, 2); // 2 extra in fixture
    }
}
