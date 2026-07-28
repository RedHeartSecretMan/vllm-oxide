import numpy as np
import pytest

from golden_gen.calibrate import (
    calibrate_from_fixtures,
    compute_skip_positions,
    count_argmax_mismatches,
    pairwise_max_abs_diff,
)
from golden_gen.config import VOCAB_SIZE


class TestPairwiseMaxAbsDiff:
    def test_identical(self):
        logits = np.random.default_rng(42).standard_normal((10, VOCAB_SIZE), dtype=np.float32)
        diff = pairwise_max_abs_diff(logits, logits)
        assert diff == pytest.approx(0.0)

    def test_different(self):
        a = np.zeros((10, VOCAB_SIZE), dtype=np.float32)
        b = np.ones((10, VOCAB_SIZE), dtype=np.float32)
        diff = pairwise_max_abs_diff(a, b)
        assert diff == pytest.approx(1.0)

    def test_different_lengths(self):
        a = np.zeros((10, VOCAB_SIZE), dtype=np.float32)
        b = np.ones((5, VOCAB_SIZE), dtype=np.float32)
        diff = pairwise_max_abs_diff(a, b)
        assert diff == pytest.approx(1.0)

    def test_single_step(self):
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        diff = pairwise_max_abs_diff(a, b)
        assert diff == pytest.approx(1.0)


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
        assert count_argmax_mismatches(a, b) == 0

    def test_empty(self):
        a = np.array([], dtype=np.int64)
        b = np.array([], dtype=np.int64)
        assert count_argmax_mismatches(a, b) == 0


class TestComputeSkipPositions:
    def test_identical(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([1, 2, 3], dtype=np.int64)
        assert compute_skip_positions(a, b) == []

    def test_some_mismatches(self):
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = np.array([1, 9, 3, 9], dtype=np.int64)
        assert compute_skip_positions(a, b) == [1, 3]

    def test_different_lengths(self):
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = np.array([1, 2], dtype=np.int64)
        assert compute_skip_positions(a, b) == []


class TestCalibrateFromFixtures:
    def test_missing_manifest(self, tmp_path):
        """Should fail gracefully when manifest does not exist."""
        with pytest.raises(FileNotFoundError):
            calibrate_from_fixtures(tmp_path / "nonexistent")


class TestComputeSkipMap:
    def test_missing_manifest(self, tmp_path):
        """Should fail gracefully when manifest does not exist."""
        from golden_gen.calibrate import compute_regression_skip_map

        with pytest.raises(FileNotFoundError):
            compute_regression_skip_map(tmp_path / "nonexistent")
