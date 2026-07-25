import numpy as np
import pytest

from golden_gen.config import VOCAB_SIZE
from golden_gen.cross_validate import (
    calibrate_tolerance,
    count_argmax_mismatches,
    cross_validate_all,
    flag_suspicious_divergence,
    pairwise_l2,
)


class TestPairwiseL2:
    def test_identical(self):
        logits = np.random.default_rng(42).standard_normal((10, VOCAB_SIZE), dtype=np.float32)
        dist = pairwise_l2(logits, logits)
        assert dist == pytest.approx(0.0)

    def test_different(self):
        a = np.zeros((10, VOCAB_SIZE), dtype=np.float32)
        b = np.ones((10, VOCAB_SIZE), dtype=np.float32)
        dist = pairwise_l2(a, b)
        # L2 norm of all-ones vector of length VOCAB_SIZE
        expected = np.sqrt(VOCAB_SIZE)
        assert dist == pytest.approx(expected)

    def test_different_lengths(self):
        a = np.zeros((10, VOCAB_SIZE), dtype=np.float32)
        b = np.ones((5, VOCAB_SIZE), dtype=np.float32)
        dist = pairwise_l2(a, b)
        expected = np.sqrt(VOCAB_SIZE)
        assert dist == pytest.approx(expected)

    def test_single_step(self):
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        dist = pairwise_l2(a, b)
        expected = np.sqrt(2.0)
        assert dist == pytest.approx(expected)


class TestCountArgmaxMismatches:
    def test_identical(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        b = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        assert count_argmax_mismatches(a, b) == 0

    def test_all_different(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([4, 5, 6], dtype=np.int64)
        assert count_argmax_mismatches(a, b) == 3

    def test_partial(self):
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = np.array([1, 9, 3, 9], dtype=np.int64)
        assert count_argmax_mismatches(a, b) == 2

    def test_different_lengths(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        b = np.array([1, 2], dtype=np.int64)
        assert count_argmax_mismatches(a, b) == 0  # only compare shared prefix

    def test_empty(self):
        a = np.array([], dtype=np.int64)
        b = np.array([], dtype=np.int64)
        assert count_argmax_mismatches(a, b) == 0


class TestCalibrateTolerance:
    def test_empty(self):
        tol = calibrate_tolerance({})
        assert tol.observed_max_l2 == 0.0
        assert tol.atol == 0.0
        assert tol.calibration_factor == 2.0

    def test_small_values(self):
        tol = calibrate_tolerance({"canonical_01": 0.001, "canonical_02": 0.005})
        assert tol.observed_max_l2 == 0.005
        assert tol.atol == 0.01  # 2 * 0.005
        assert tol.rtol == 0.01

    def test_large_values(self):
        tol = calibrate_tolerance({"canonical_01": 0.5, "canonical_02": 1.0})
        assert tol.observed_max_l2 == 1.0
        assert tol.atol == 2.0

    def test_method_string(self):
        tol = calibrate_tolerance({"canonical_01": 0.001})
        assert "2.0x" in tol.method or "2x" in tol.method
        assert "pairwise L2" in tol.method


class TestCrossValidateAll:
    def test_no_results(self):
        deviations, per_prompt = cross_validate_all({})
        assert deviations == []
        assert per_prompt == {}

    def test_identical_results(self):
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((4, VOCAB_SIZE), dtype=np.float32)
        token_ids = np.array([1, 2, 3, 4], dtype=np.int64)

        results = {
            ("transformers", "canonical_01"): (token_ids, logits),
            ("nanovllm", "canonical_01"): (token_ids, logits),
            ("vllm_v1", "canonical_01"): (token_ids, logits),
        }
        deviations, per_prompt = cross_validate_all(results)
        assert "canonical_01" in per_prompt
        assert per_prompt["canonical_01"] == pytest.approx(0.0)
        # With identical logits, no deviations expected
        # (all L2 = 0, mismatches = 0)
        for d in deviations:
            assert d.max_l2 < 1e-6 and d.argmax_mismatches == 0

    def test_with_mismatches(self):
        rng = np.random.default_rng(42)
        logits_a = rng.standard_normal((4, VOCAB_SIZE), dtype=np.float32)
        logits_b = logits_a + 0.1
        token_ids_a = np.array([1, 2, 3, 4], dtype=np.int64)
        token_ids_b = np.array([1, 5, 3, 6], dtype=np.int64)

        results = {
            ("transformers", "canonical_01"): (token_ids_a, logits_a),
            ("nanovllm", "canonical_01"): (token_ids_b, logits_b),
        }
        deviations, per_prompt = cross_validate_all(results)
        assert len(deviations) > 0
        assert deviations[0].argmax_mismatches == 2
        assert deviations[0].max_l2 > 0


class TestFlagSuspiciousDivergence:
    def test_empty(self):
        result = flag_suspicious_divergence({})
        assert result == []

    def test_below_threshold(self):
        result = flag_suspicious_divergence({"canonical_01": 0.001, "canonical_02": 0.05})
        assert result == []

    def test_above_threshold(self):
        result = flag_suspicious_divergence({"canonical_01": 0.001, "canonical_02": 0.5})
        assert result == ["canonical_02"]

    def test_mixed(self):
        result = flag_suspicious_divergence(
            {"canonical_01": 0.05, "canonical_02": 0.5, "canonical_03": 0.2}
        )
        assert result == ["canonical_02", "canonical_03"]

    def test_all_above(self):
        result = flag_suspicious_divergence(
            {"canonical_01": 0.5, "canonical_02": 1.0}, threshold=0.1
        )
        assert result == ["canonical_01", "canonical_02"]

    def test_custom_threshold(self):
        result = flag_suspicious_divergence(
            {"canonical_01": 0.51, "canonical_02": 0.05}, threshold=0.5
        )
        assert result == ["canonical_01"]
        rng = np.random.default_rng(42)
        logits_a = rng.standard_normal((4, VOCAB_SIZE), dtype=np.float32)
        logits_b = logits_a + 0.1  # slight shift
        token_ids_a = np.array([1, 2, 3, 4], dtype=np.int64)
        token_ids_b = np.array([1, 5, 3, 6], dtype=np.int64)

        results = {
            ("transformers", "canonical_01"): (token_ids_a, logits_a),
            ("nanovllm", "canonical_01"): (token_ids_b, logits_b),
        }
        deviations, per_prompt = cross_validate_all(results)
        assert len(deviations) > 0
        assert deviations[0].argmax_mismatches == 2
        assert deviations[0].max_l2 > 0
