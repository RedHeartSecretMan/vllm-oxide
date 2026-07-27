use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::types::{FixtureData, ToleranceCalibration};

/// Result of an L2 per-step logits tensor comparison.
#[derive(Debug)]
pub struct L2Result {
    pub prompt_id: String,
    pub passed: bool,
    pub total_steps: usize,
    pub total_elements: usize,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    pub elements_exceeding_tol: usize,
    /// Step index (0-based) where the max absolute difference occurred.
    pub max_abs_step: Option<usize>,
    /// Per-step statistics.
    pub step_details: Vec<L2StepDetail>,
}

#[derive(Debug, Clone)]
pub struct L2StepDetail {
    pub step: usize,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    pub elements_exceeding: usize,
}

/// Compare generated logits against golden logits using `atol + rtol`.
///
/// `generated_logits`: shape `[n, vocab_size]` FP32 from `generate_logits`.
/// `fixture.logits`: flat `Vec<f32>` of the same shape.
/// `tolerance`: `atol` and `rtol` from the manifest's calibration.
///
/// The comparison formula (matching numpy's `allclose` behavior):
///   `|actual - expected| <= atol + rtol * |expected|`
///
/// Returns `L2Result` with per-step and aggregate statistics.
pub fn compare_l2(
    fixture: &FixtureData,
    generated_logits: &Tensor,
    tolerance: &ToleranceCalibration,
) -> Result<L2Result> {
    let logits_flat = match &fixture.logits {
        Some(l) => l,
        None => anyhow::bail!(
            "L2 comparison requires canonical fixture with logits tensor. \
             Fixture '{}' has no logits data (regression fixture).",
            fixture.prompt_id
        ),
    };

    let n_steps = fixture.num_tokens as usize;
    let vocab_size = fixture.model_vocab_size();

    let gen_shape = generated_logits.dims();
    if gen_shape.len() != 2 || gen_shape[0] != n_steps || gen_shape[1] != vocab_size {
        anyhow::bail!(
            "logits shape mismatch for '{}': generated {:?}, expected [{}, {}]",
            fixture.prompt_id,
            gen_shape,
            n_steps,
            vocab_size,
        );
    }

    let gen_f32 = generated_logits.to_dtype(DType::F32)?.flatten_all()?;
    let gen_vals = gen_f32.to_vec1::<f32>()?;

    let mut max_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    let mut max_abs_step: Option<usize> = None;
    let mut total_exceeding = 0usize;
    let mut step_details = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let start = step * vocab_size;
        let end = start + vocab_size;
        let expected_slice = &logits_flat[start..end];

        let mut step_max_abs = 0.0f64;
        let mut step_max_rel = 0.0f64;
        let mut step_exceeding = 0usize;

        for (j, &expected) in expected_slice.iter().enumerate() {
            let actual = gen_vals[start + j] as f64;
            let expected_f = expected as f64;
            let abs_diff = (actual - expected_f).abs();
            let threshold = tolerance.atol + tolerance.rtol * expected_f.abs();

            if abs_diff > step_max_abs {
                step_max_abs = abs_diff;
            }
            let rel_diff = if expected_f.abs() > 1e-30 {
                abs_diff / expected_f.abs()
            } else if actual.abs() > 1e-30 {
                f64::INFINITY
            } else {
                0.0
            };
            if rel_diff > step_max_rel {
                step_max_rel = rel_diff;
            }

            if abs_diff > threshold {
                step_exceeding += 1;
            }
        }

        if step_max_abs > max_abs_diff {
            max_abs_diff = step_max_abs;
            max_abs_step = Some(step);
        }
        if step_max_rel > max_rel_diff {
            max_rel_diff = step_max_rel;
        }
        total_exceeding += step_exceeding;

        step_details.push(L2StepDetail {
            step,
            max_abs_diff: step_max_abs,
            max_rel_diff: step_max_rel,
            elements_exceeding: step_exceeding,
        });
    }

    Ok(L2Result {
        prompt_id: fixture.prompt_id.clone(),
        passed: total_exceeding == 0,
        total_steps: n_steps,
        total_elements: n_steps * vocab_size,
        max_abs_diff,
        max_rel_diff,
        elements_exceeding_tol: total_exceeding,
        max_abs_step,
        step_details,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

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

        let tol = ToleranceCalibration {
            atol: 1e-5,
            rtol: 1e-3,
            observed_max_l2: 0.01,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        };

        let result = compare_l2(&fixture, &gen, &tol).unwrap();
        assert!(result.passed);
        assert_eq!(result.elements_exceeding_tol, 0);
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
        let gen = Tensor::from_vec(vec![1.0f32, 3.1, 3.0], (1, 3), &candle_core::Device::Cpu).unwrap();

        let tol = ToleranceCalibration {
            atol: 1e-5,
            rtol: 1e-3,
            observed_max_l2: 0.01,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        };

        let result = compare_l2(&fixture, &gen, &tol).unwrap();
        assert!(!result.passed);
        assert!(result.elements_exceeding_tol > 0);
    }

    #[test]
    fn small_difference_within_tolerance_passes() {
        let fixture = FixtureData {
            prompt_id: "test".into(),
            category: crate::types::PromptCategory::Canonical,
            oracle: crate::types::OracleName::Nanovllm,
            num_tokens: 1,
            token_ids: vec![7],
            n_prompt_tokens: 4,
            logits: Some(vec![100.0, 50.0]),
            logits_shape: (1, 2),
            top5_indices: None,
            top5_logits: None,
        };

        // Difference of 0.001 — within rtol=1e-3 * 100.0 = 0.1 + atol.
        let tol = ToleranceCalibration {
            atol: 1e-5,
            rtol: 1e-3,
            observed_max_l2: 0.01,
            calibration_factor: 2.0,
            method: "pairwise".into(),
        };
        let gen = Tensor::from_vec(vec![100.001f32, 50.0], (1, 2), &candle_core::Device::Cpu).unwrap();
        let result = compare_l2(&fixture, &gen, &tol).unwrap();
        assert!(result.passed);
    }
}
