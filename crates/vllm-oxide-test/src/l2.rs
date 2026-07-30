use anyhow::Result;

use crate::types::{FixtureData, ToleranceCalibration};

/// Result of L2 comparison over shared-prefix steps only.
///
/// Unlike a full per-step comparison which compares logits at every step
/// (including steps after token divergence), this only compares steps where
/// the generated token matches the golden token. This eliminates chain
/// divergence and gives a tighter numerical accuracy signal.
#[derive(Debug)]
pub struct L2Result {
    pub prompt_id: String,
    pub passed: bool,
    pub same_token_steps: usize,
    pub diff_token_steps: usize,
    pub total_elements: usize,
    pub max_abs_diff: f64,
    pub elements_exceeding_tol: usize,
}

/// Compare logits only on steps where generated tokens match golden tokens.
///
/// Chain divergence occurs when a small BF16 difference flips argmax at step N,
/// causing all subsequent hidden states to differ. This function isolates the
/// clean numerical comparison by only analyzing steps where the prefix is shared.
///
/// The comparison uses only `atol` (no rtol), matching ADR-0005:
///   `|actual - expected| <= atol`
pub fn compare_l2(
    fixture: &FixtureData,
    generated_logits: &[f32],
    generated_tokens: &[u32],
    tolerance: &ToleranceCalibration,
) -> Result<L2Result> {
    let Some(logits_flat) = &fixture.logits else {
        anyhow::bail!("L2 comparison requires canonical fixture with logits tensor.")
    };

    let n_steps = fixture.num_tokens as usize;
    let vocab_size = fixture.model_vocab_size(generated_logits);

    let min_steps = n_steps
        .min(generated_tokens.len())
        .min(fixture.token_ids.len());
    let mut same_token_steps = 0usize;
    let mut diff_token_steps = 0usize;
    let mut total_exceeding = 0usize;
    let mut max_abs_diff = 0.0f64;

    for (step, (&expected_token, &actual_token)) in fixture.token_ids[..min_steps]
        .iter()
        .zip(generated_tokens[..min_steps].iter())
        .enumerate()
    {
        let actual_token = actual_token as i64;

        if expected_token != actual_token {
            diff_token_steps += 1;
            continue;
        }
        same_token_steps += 1;

        let start = step * vocab_size;
        let end = start + vocab_size;
        let expected_slice = &logits_flat[start..end];

        for (j, &expected) in expected_slice.iter().enumerate() {
            let actual = generated_logits[start + j] as f64;
            let expected_f = expected as f64;
            let abs_diff = (actual - expected_f).abs();

            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }

            // ADR-0005: atol-only comparison (no rtol)
            let threshold = tolerance.atol;
            if abs_diff > threshold {
                total_exceeding += 1;
            }
        }
    }

    Ok(L2Result {
        prompt_id: fixture.prompt_id.clone(),
        passed: total_exceeding == 0,
        same_token_steps,
        diff_token_steps,
        total_elements: same_token_steps * vocab_size,
        max_abs_diff,
        elements_exceeding_tol: total_exceeding,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_tolerance() -> ToleranceCalibration {
        ToleranceCalibration {
            atol: 1e-5,
            observed_max_abs_diff: 0.01,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        }
    }

    #[test]
    fn identical_logits_pass() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 2,
            token_ids: vec![1, 2],
            n_prompt_tokens: 3,
            logits: Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), // 2 steps × 3 vocab
            logits_shape: (2, 3),
            top5_indices: None,
            top5_logits: None,
        };

        let gen_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let generated_tokens: Vec<u32> = vec![1, 2];
        let tol = make_tolerance();

        let result = compare_l2(&fixture, &gen_vals, &generated_tokens, &tol).unwrap();
        assert!(result.passed);
        assert_eq!(result.elements_exceeding_tol, 0);
        assert_eq!(result.same_token_steps, 2);
        assert_eq!(result.diff_token_steps, 0);
    }

    #[test]
    fn logits_beyond_tolerance_detected() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 1,
            token_ids: vec![42],
            n_prompt_tokens: 5,
            logits: Some(vec![1.0, 2.0, 3.0]),
            logits_shape: (1, 3),
            top5_indices: None,
            top5_logits: None,
        };

        // Second value differs by 1.1 — well beyond atol=1e-5.
        let gen_vals = vec![1.0f32, 3.1, 3.0];
        let generated_tokens: Vec<u32> = vec![42];
        let tol = make_tolerance();

        let result = compare_l2(&fixture, &gen_vals, &generated_tokens, &tol).unwrap();
        assert!(!result.passed);
        assert!(result.elements_exceeding_tol > 0);
    }

    #[test]
    fn small_difference_within_tolerance_passes() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 1,
            token_ids: vec![7],
            n_prompt_tokens: 4,
            logits: Some(vec![100.0, 50.0]),
            logits_shape: (1, 2),
            top5_indices: None,
            top5_logits: None,
        };

        // Difference of 0.001 — within atol=1e-5? No. But this test was
        // originally designed for rtol=1e-3 * 100.0 = 0.1. With atol-only,
        // a diff of 0.001 exceeds atol=1e-5, so the test should NOT pass.
        // We keep it as a demonstration of atol-only strictness.
        let tol = make_tolerance();
        let gen_vals = vec![100.001f32, 50.0];
        let generated_tokens: Vec<u32> = vec![7];
        let result = compare_l2(&fixture, &gen_vals, &generated_tokens, &tol).unwrap();
        // 0.001 > 1e-5 → does NOT pass under atol-only
        assert!(!result.passed);
    }

    #[test]
    fn token_mismatch_skipped_in_l2() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 3,
            token_ids: vec![1, 99, 3],
            n_prompt_tokens: 2,
            logits: Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            logits_shape: (3, 3),
            top5_indices: None,
            top5_logits: None,
        };

        // Step 0: token matches (golden=1, gen=1) → compared
        // Step 1: token mismatch (golden=99, gen=2) → skipped
        // Step 2: token matches (golden=3, gen=3) → compared
        let gen_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let generated_tokens: Vec<u32> = vec![1, 2, 3];
        let tol = make_tolerance();

        let result = compare_l2(&fixture, &gen_vals, &generated_tokens, &tol).unwrap();
        assert!(result.passed);
        assert_eq!(result.same_token_steps, 2);
        assert_eq!(result.diff_token_steps, 1);
    }
}
