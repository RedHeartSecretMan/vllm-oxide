use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::types::{FixtureData, ToleranceCalibration};

/// Result of L2 comparison over shared-prefix steps only.
///
/// Compare every row produced from the complete shared causal prefix,
/// including the row selecting the first divergent token. All later rows
/// are excluded because their causal histories differ.
#[derive(Debug)]
pub struct L2Result {
    pub prompt_id: String,
    pub passed: bool,
    /// Legacy name: number of compared shared-prefix rows.
    pub same_token_steps: usize,
    /// Legacy name: number of rows excluded after the first divergence.
    pub diff_token_steps: usize,
    pub total_elements: usize,
    pub max_abs_diff: f64,
    pub elements_exceeding_tol: usize,
}

/// Compare logits through the first divergent token, then stop.
///
/// Chain divergence occurs when a small BF16 difference flips argmax at step N,
/// causing all subsequent hidden states to differ. This function isolates the
/// clean numerical comparison by only analyzing steps where the prefix is shared.
///
/// The comparison uses only `atol` (no rtol), matching ADR-0005:
///   `|actual - expected| <= atol`
pub fn compare_l2(
    fixture: &FixtureData,
    generated_logits: &Tensor,
    generated_tokens: &[u32],
    tolerance: &ToleranceCalibration,
) -> Result<L2Result> {
    let logits_flat = match &fixture.logits {
        Some(l) => l,
        None => anyhow::bail!("L2 comparison requires canonical fixture with logits tensor."),
    };

    let n_steps = fixture.num_tokens as usize;
    let vocab_size = fixture.logits_shape.1;

    let gen_f32 = generated_logits.to_dtype(DType::F32)?.flatten_all()?;
    let gen_vals = gen_f32.to_vec1::<f32>()?;

    if n_steps == 0
        || vocab_size == 0
        || fixture.logits_shape.0 != n_steps
        || fixture.token_ids.len() != n_steps
        || n_steps.checked_mul(vocab_size) != Some(logits_flat.len())
        || generated_tokens.is_empty()
        || generated_tokens.len().checked_mul(vocab_size) != Some(gen_vals.len())
        || !tolerance.atol.is_finite()
        || tolerance.atol < 0.0
    {
        anyhow::bail!(
            "L2 requires non-empty, consistent logit geometry and a finite non-negative tolerance"
        );
    }

    let min_steps = n_steps
        .min(generated_tokens.len())
        .min(fixture.token_ids.len());
    let mut same_token_steps = 0usize;
    let mut diff_token_steps = 0usize;
    let mut total_exceeding = 0usize;
    let mut max_abs_diff = 0.0f64;

    for (step, &actual_token) in generated_tokens.iter().take(min_steps).enumerate() {
        let expected_token = fixture.token_ids[step];
        let actual_token = i64::from(actual_token);

        same_token_steps += 1;

        let start = step * vocab_size;
        let end = start + vocab_size;
        let expected_slice = &logits_flat[start..end];

        for (j, &expected) in expected_slice.iter().enumerate() {
            let actual = gen_vals[start + j] as f64;
            let expected_f = expected as f64;
            if !actual.is_finite() || !expected_f.is_finite() {
                max_abs_diff = f64::INFINITY;
                total_exceeding += 1;
                continue;
            }
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
        if expected_token != actual_token {
            diff_token_steps = min_steps - same_token_steps;
            break;
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
        let gen = Tensor::from_vec(gen_vals, (2, 3), &candle_core::Device::Cpu).unwrap();
        let generated_tokens: Vec<u32> = vec![1, 2];
        let tol = make_tolerance();

        let result = compare_l2(&fixture, &gen, &generated_tokens, &tol).unwrap();
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

        // Second value differs by 1.0 — well beyond atol=1e-5.
        let gen =
            Tensor::from_vec(vec![1.0f32, 3.1, 3.0], (1, 3), &candle_core::Device::Cpu).unwrap();
        let generated_tokens: Vec<u32> = vec![42];
        let tol = make_tolerance();

        let result = compare_l2(&fixture, &gen, &generated_tokens, &tol).unwrap();
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
        let gen =
            Tensor::from_vec(vec![100.001f32, 50.0], (1, 2), &candle_core::Device::Cpu).unwrap();
        let generated_tokens: Vec<u32> = vec![7];
        let result = compare_l2(&fixture, &gen, &generated_tokens, &tol).unwrap();
        // 0.001 > 1e-5 → does NOT pass under atol-only
        assert!(!result.passed);
    }

    #[test]
    fn first_divergence_row_is_compared_and_suffix_is_excluded() {
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

        // Rows 0 and 1 share the complete causal prefix; row 2 is excluded
        // even though its selected token happens to match again.
        let gen = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 700.0, 800.0, 900.0],
            (3, 3),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let generated_tokens: Vec<u32> = vec![1, 2, 3];
        let tol = make_tolerance();

        let result = compare_l2(&fixture, &gen, &generated_tokens, &tol).unwrap();
        assert!(result.passed);
        assert_eq!(result.same_token_steps, 2);
        assert_eq!(result.diff_token_steps, 1);
    }
    #[test]
    fn malformed_empty_or_non_finite_logits_fail_closed() {
        let fixture = FixtureData {
            prompt_id: "invalid".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 1,
            token_ids: vec![1],
            n_prompt_tokens: 1,
            logits: Some(vec![0.0, 1.0]),
            logits_shape: (1, 2),
            top5_indices: None,
            top5_logits: None,
        };
        let device = &candle_core::Device::Cpu;
        let empty = Tensor::from_vec(Vec::<f32>::new(), (0, 2), device).unwrap();
        assert!(compare_l2(&fixture, &empty, &[], &make_tolerance()).is_err());
        let short = Tensor::from_vec(vec![0.0f32], (1, 1), device).unwrap();
        assert!(compare_l2(&fixture, &short, &[1], &make_tolerance()).is_err());
        let non_finite = Tensor::from_vec(vec![0.0f32, f32::NAN], (1, 2), device).unwrap();
        assert!(
            !compare_l2(&fixture, &non_finite, &[1], &make_tolerance())
                .unwrap()
                .passed
        );
        let mut invalid_tolerance = make_tolerance();
        invalid_tolerance.atol = f64::NAN;
        assert!(compare_l2(&fixture, &non_finite, &[1], &invalid_tolerance).is_err());
    }
    #[test]
    fn the_first_divergent_row_cannot_be_skipped() {
        let fixture = FixtureData {
            prompt_id: "first-divergence".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Transformers,
            num_tokens: 1,
            token_ids: vec![1],
            n_prompt_tokens: 1,
            logits: Some(vec![0.0, 1.0]),
            logits_shape: (1, 2),
            top5_indices: None,
            top5_logits: None,
        };
        let logits = Tensor::new(&[[10.0_f32, 1.0]], &candle_core::Device::Cpu).unwrap();
        let result = compare_l2(&fixture, &logits, &[0], &make_tolerance()).unwrap();
        assert!(!result.passed);
        assert_eq!(result.same_token_steps, 1);
        assert_eq!(result.diff_token_steps, 0);
        assert_eq!(result.total_elements, 2);
    }
}
