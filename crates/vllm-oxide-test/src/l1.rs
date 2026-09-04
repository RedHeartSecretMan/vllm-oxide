use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::types::{FixtureData, TolerancePolicy};

/// Result of an L1 token-sequence comparison.
#[derive(Debug)]
pub struct L1Result {
    pub prompt_id: String,
    pub passed: bool,
    pub total_positions: usize,
    pub compared_positions: usize,
    pub exact_matches: usize,
    pub near_ties: usize,
    pub mismatches: usize,
    /// First divergent token position, including an accepted near tie.
    pub first_divergence: Option<usize>,
    /// Token positions excluded after the causal histories diverged.
    pub excluded_positions: usize,
    pub policy_version: String,
    pub near_tie_max_abs_logit_gap: f64,
    pub details: Vec<L1PositionDetail>,
}

#[derive(Debug)]
pub enum L1PositionDetail {
    Match {
        position: usize,
        token_id: i64,
    },
    NearTie {
        position: usize,
        expected: i64,
        actual: i64,
        candidate_gap: f64,
    },
    Mismatch {
        position: usize,
        expected: i64,
        actual: i64,
    },
}

/// Compare generated token IDs against the reference oracle under one
/// versioned candidate-gap policy.
///
/// `generated_tokens` are the output from `LLM::generate` (greedy, temp=0).
/// `fixture` holds the golden token_ids.
/// `generated_logits` provides the raw logits for near-tie detection — shape
/// `[n, vocab_size]` (from `generate_logits`).
pub fn compare_l1(
    fixture: &FixtureData,
    generated_tokens: &[u32],
    generated_logits: Option<&Tensor>,
    policy: &TolerancePolicy,
) -> Result<L1Result> {
    compare_tokens_loop(fixture, generated_tokens, policy, |i, expected, actual| {
        if expected == actual {
            return Ok(None);
        }
        let Some(logits) = generated_logits else {
            return Ok(Some(MismatchKind::Deterministic));
        };
        // Expected and actual are validated token ids bounded by vocab_size.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let gap = candidate_logit_gap(logits, i, expected as usize, actual as usize)?;
        if gap <= policy.l1_near_tie_max_abs_logit_gap {
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
    policy: &TolerancePolicy,
) -> Result<L1Result> {
    compare_tokens_loop(fixture, generated_tokens, policy, |_, expected, actual| {
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
}

fn compare_tokens_loop<F>(
    fixture: &FixtureData,
    generated_tokens: &[u32],
    policy: &TolerancePolicy,
    classify: F,
) -> Result<L1Result>
where
    F: Fn(usize, i64, i64) -> Result<Option<MismatchKind>>,
{
    let n = fixture.token_ids.len().min(generated_tokens.len());
    let mut details = Vec::with_capacity(n);
    let mut exact_matches = 0usize;
    let mut near_ties = 0usize;
    let mut mismatches = 0usize;
    let mut first_divergence = None;

    for (i, (&expected, &actual)) in fixture.token_ids[..n]
        .iter()
        .zip(generated_tokens[..n].iter())
        .enumerate()
    {
        let actual = actual as i64;

        match classify(i, expected, actual)? {
            None => {
                exact_matches += 1;
                details.push(L1PositionDetail::Match {
                    position: i,
                    token_id: expected,
                });
            }
            Some(MismatchKind::NearTie(gap)) => {
                near_ties += 1;
                first_divergence = Some(i);
                details.push(L1PositionDetail::NearTie {
                    position: i,
                    expected,
                    actual,
                    candidate_gap: gap,
                });
                break;
            }
            Some(MismatchKind::Deterministic) => {
                mismatches += 1;
                first_divergence = Some(i);
                details.push(L1PositionDetail::Mismatch {
                    position: i,
                    expected,
                    actual,
                });
                break;
            }
        }
    }

    let total_positions = fixture.token_ids.len().max(generated_tokens.len());
    if first_divergence.is_none() && fixture.token_ids.len() != generated_tokens.len() {
        first_divergence = Some(n);
        mismatches += 1;
    }
    let compared_positions = details.len();
    let excluded_positions = total_positions.saturating_sub(compared_positions);

    Ok(L1Result {
        prompt_id: fixture.prompt_id.clone(),
        passed: mismatches == 0,
        total_positions,
        compared_positions,
        exact_matches,
        near_ties,
        mismatches,
        first_divergence,
        excluded_positions,
        policy_version: policy.version.clone(),
        near_tie_max_abs_logit_gap: policy.l1_near_tie_max_abs_logit_gap,
        details,
    })
}

/// Compute the absolute gap between the expected and actual candidate logits.
fn candidate_logit_gap(
    logits: &Tensor,
    position: usize,
    expected_token: usize,
    actual_token: usize,
) -> Result<f64> {
    let row = logits.get(position)?;
    let row_f32 = row.to_dtype(DType::F32)?;
    let values = row_f32.to_vec1::<f32>()?;
    let expected = values.get(expected_token).ok_or_else(|| {
        anyhow::anyhow!("expected token {expected_token} is outside candidate logits")
    })?;
    let actual = values.get(actual_token).ok_or_else(|| {
        anyhow::anyhow!("actual token {actual_token} is outside candidate logits")
    })?;
    Ok(f64::from((actual - expected).abs()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_policy() -> TolerancePolicy {
        TolerancePolicy {
            version: "same-prefix-v1".into(),
            dtype: "bfloat16".into(),
            kernel: "sdpa".into(),
            l1_near_tie_max_abs_logit_gap: 0.02,
            l2_atol: 1e-5,
            rationale: "Reviewed synthetic policy".into(),
            evidence: vec!["synthetic:l1".into()],
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
        let policy = make_policy();

        let result = compare_l1(&fixture, &generated, None, &policy).unwrap();
        assert!(result.passed);
        assert_eq!(result.exact_matches, 3);
        assert_eq!(result.mismatches, 0);
        assert_eq!(result.near_ties, 0);
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
        let policy = make_policy();

        let result = compare_l1(&fixture, &generated, None, &policy).unwrap();
        assert!(!result.passed);
        assert_eq!(result.exact_matches, 1);
        assert_eq!(result.mismatches, 1);
    }

    #[test]
    fn genuine_top1_mismatch_within_candidate_gap_is_near_tie() {
        let fixture = FixtureData {
            prompt_id: "near-tie".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 1,
            token_ids: vec![1],
            n_prompt_tokens: 5,
            logits: None,
            logits_shape: (1, 3),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![2];
        let logits = Tensor::new(&[[0.0_f32, 9.99, 10.0]], &candle_core::Device::Cpu).unwrap();
        let policy = make_policy();

        let result = compare_l1(&fixture, &generated, Some(&logits), &policy).unwrap();

        assert!(result.passed);
        assert_eq!(result.near_ties, 1);
        assert_eq!(result.mismatches, 0);
    }

    #[test]
    fn near_tie_stops_before_different_prefix_candidate_logits() {
        let fixture = FixtureData {
            prompt_id: "near-tie-prefix".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 2,
            token_ids: vec![1, 1],
            n_prompt_tokens: 5,
            logits: None,
            logits_shape: (2, 3),
            top5_indices: None,
            top5_logits: None,
        };
        let generated: Vec<u32> = vec![2, 2];
        let logits = Tensor::new(
            &[[0.0_f32, 9.99, 10.0], [0.0, 0.0, 10.0]],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let result = compare_l1(&fixture, &generated, Some(&logits), &make_policy()).unwrap();

        assert!(result.passed);
        assert_eq!(result.near_ties, 1);
        assert_eq!(result.mismatches, 0);
    }

    #[test]
    fn accepted_near_tie_excludes_the_different_prefix_suffix_length() {
        let fixture = FixtureData {
            prompt_id: "near-tie-short".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 2,
            token_ids: vec![1, 1],
            n_prompt_tokens: 5,
            logits: None,
            logits_shape: (2, 3),
            top5_indices: None,
            top5_logits: None,
        };
        let generated = vec![2_u32];
        let logits = Tensor::new(&[[0.0_f32, 9.99, 10.0]], &candle_core::Device::Cpu).unwrap();

        let result = compare_l1(&fixture, &generated, Some(&logits), &make_policy()).unwrap();

        assert!(result.passed);
        assert_eq!(result.near_ties, 1);
        assert_eq!(result.mismatches, 0);
        assert_eq!(result.excluded_positions, 1);
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

        let policy = make_policy();

        let result = compare_l1(&fixture, &generated, None, &policy).unwrap();
        assert!(!result.passed);
        assert_eq!(result.mismatches, 1);
        assert_eq!(result.first_divergence, Some(3));
        assert_eq!(result.excluded_positions, 2);
    }

    #[test]
    fn regression_tokens_require_an_exact_match() {
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

        let result = compare_l1_tokens_only(&fixture, &generated, &make_policy()).unwrap();
        assert!(!result.passed);
        assert_eq!(result.exact_matches, 1);
        assert_eq!(result.mismatches, 1);
    }
}
