from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import golden_gen.cli as cli
from golden_gen.calibrate import (
    calibrate_from_fixtures,
    count_argmax_mismatches,
    pairwise_max_abs_diff,
    same_prefix_max_abs_diff,
    validate_calibration_coverage,
)
from golden_gen.config import VOCAB_SIZE
from golden_gen.io import save_fixture
from golden_gen.manifest import build_manifest, read_manifest, write_manifest
from golden_gen.schema import (
    ComparisonPolicy,
    ExpectedFixture,
    FixtureMetadata,
    ToleranceCalibration,
)


def same_prefix_policy() -> ComparisonPolicy:
    return ComparisonPolicy(
        version="same-prefix-v1",
        l1_near_tie_max_abs_logit_gap=0.0,
        l2_atol=0.0,
    )


def expected_pair() -> list[ExpectedFixture]:
    revision = "7e4ae267688d671ddfca3122e4528ee980cf3234"
    return [
        ExpectedFixture(
            fixture_id="canonical_01.transformers",
            prompt_id="canonical_01",
            family="canonical",
            model_revision=revision,
            dtype="bfloat16",
            oracle="transformers",
            oracle_role="reference",
            required_comparison="l1_l2",
            filename="canonical_01.transformers.safetensors",
        ),
        ExpectedFixture(
            fixture_id="canonical_01.vllm",
            prompt_id="canonical_01",
            family="canonical",
            model_revision=revision,
            dtype="bfloat16",
            oracle="vllm",
            oracle_role="baseline",
            required_comparison="calibration",
            filename="canonical_01.vllm.safetensors",
        ),
    ]


def save_canonical_metadata(
    output_dir: Path,
    oracle: Literal["transformers", "vllm"],
    *,
    vocab_size: int = VOCAB_SIZE,
    num_tokens: int = 1,
) -> FixtureMetadata:
    filename = f"canonical_01.{oracle}.safetensors"
    sha256 = save_fixture(
        output_dir / filename,
        token_ids=np.ones(num_tokens, dtype=np.int64),
        logits=np.zeros((num_tokens, vocab_size), dtype=np.float32),
        top5_indices=None,
        top5_logits=None,
        n_prompt_tokens=1,
    )
    return FixtureMetadata(
        prompt_id="canonical_01",
        category="canonical",
        oracle=oracle,
        num_tokens=num_tokens,
        logits_dtype="float32",
        logits_shape=(num_tokens, vocab_size),
        sha256=sha256,
        filename=filename,
    )


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
        with pytest.raises(ValueError, match="identical shapes"):
            pairwise_max_abs_diff(a, b)

    def test_single_step(self):
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        diff = pairwise_max_abs_diff(a, b)
        assert diff == pytest.approx(1.0)

    def test_non_finite_values_fail_closed(self):
        a = np.array([[1.0, np.nan]], dtype=np.float32)
        b = np.array([[1.0, 2.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="finite"):
            pairwise_max_abs_diff(a, b)


class TestSamePrefixMaxAbsDiff:
    def test_first_token_divergence_includes_only_the_shared_prompt_row(self):
        reference_logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        baseline_logits = np.array([[1.25, 2.0], [300.0, 400.0]], dtype=np.float32)
        reference_tokens = np.array([1, 2], dtype=np.int64)
        baseline_tokens = np.array([9, 2], dtype=np.int64)

        max_abs_diff = same_prefix_max_abs_diff(
            reference_logits,
            baseline_logits,
            reference_tokens,
            baseline_tokens,
        )

        assert max_abs_diff == pytest.approx(0.25)

    def test_later_divergence_excludes_following_logit_rows(self):
        reference_logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        baseline_logits = np.array([[1.0, 2.0], [3.0, 4.0], [500.0, 600.0]], dtype=np.float32)
        reference_tokens = np.array([1, 2, 3], dtype=np.int64)
        baseline_tokens = np.array([1, 9, 3], dtype=np.int64)

        max_abs_diff = same_prefix_max_abs_diff(
            reference_logits,
            baseline_logits,
            reference_tokens,
            baseline_tokens,
        )

        assert max_abs_diff == pytest.approx(0.0)


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
        with pytest.raises(ValueError, match="identical lengths"):
            count_argmax_mismatches(a, b)

    def test_empty(self):
        a = np.array([], dtype=np.int64)
        b = np.array([], dtype=np.int64)
        assert count_argmax_mismatches(a, b) == 0


class TestCalibrateFromFixtures:
    def test_missing_manifest(self, tmp_path):
        """Should fail gracefully when manifest does not exist."""
        with pytest.raises(FileNotFoundError):
            calibrate_from_fixtures(tmp_path / "nonexistent")

    def test_empty_comparison_set_fails_closed(self, tmp_path):
        tolerance = ToleranceCalibration(
            atol=0.0,
            observed_max_abs_diff=0.0,
            calibration_factor=2.0,
            method="pending",
        )
        manifest = build_manifest(
            fixtures=[],
            expected_fixtures=expected_pair(),
            comparison_policy=same_prefix_policy(),
            tolerance=tolerance,
        )
        write_manifest(manifest, tmp_path / "manifest.json")

        with pytest.raises(ValueError, match="empty calibration comparison set"):
            calibrate_from_fixtures(tmp_path)

    def test_success_records_every_calibrated_baseline_fixture(self, tmp_path):
        reference_tokens = np.array([1, 2, 3], dtype=np.int64)
        baseline_tokens = np.array([1, 9, 3], dtype=np.int64)
        reference_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
        baseline_logits = reference_logits.copy()
        baseline_logits[0, 0] = 0.001
        baseline_logits[2, 0] = 100.0
        fixtures = []
        for oracle, token_ids, logits in (
            ("transformers", reference_tokens, reference_logits),
            ("vllm", baseline_tokens, baseline_logits),
        ):
            filename = f"canonical_01.{oracle}.safetensors"
            sha256 = save_fixture(
                tmp_path / filename,
                token_ids=token_ids,
                logits=logits,
                top5_indices=None,
                top5_logits=None,
                n_prompt_tokens=1,
            )
            fixtures.append(
                FixtureMetadata(
                    prompt_id="canonical_01",
                    category="canonical",
                    oracle=oracle,
                    num_tokens=3,
                    logits_dtype="float32",
                    logits_shape=(3, VOCAB_SIZE),
                    sha256=sha256,
                    filename=filename,
                )
            )
        tolerance = ToleranceCalibration(
            atol=0.0,
            observed_max_abs_diff=0.0,
            calibration_factor=2.0,
            method="pending",
        )
        manifest = build_manifest(
            fixtures=fixtures,
            expected_fixtures=expected_pair(),
            comparison_policy=same_prefix_policy(),
            tolerance=tolerance,
        )
        write_manifest(manifest, tmp_path / "manifest.json")

        exit_code = cli.main(
            [
                "calibrate",
                "--manifest-dir",
                str(tmp_path),
                "--comparison-policy-version",
                "same-prefix-v1",
                "--l1-near-tie-max-abs-logit-gap",
                "0.125",
                "--l2-atol",
                "0.25",
            ]
        )

        assert exit_code == 0
        calibrated = read_manifest(tmp_path / "manifest.json")
        assert calibrated.calibrated_fixtures == ["canonical_01.vllm"]
        assert calibrated.comparison_policy.version == "same-prefix-v1"
        assert calibrated.tolerance.observed_max_abs_diff == pytest.approx(0.001)
        assert calibrated.comparison_policy.l2_atol == pytest.approx(0.25)
        assert calibrated.comparison_policy.l1_near_tie_max_abs_logit_gap == pytest.approx(0.125)

    def test_missing_baseline_fixture_fails_closed(self, tmp_path):
        manifest = build_manifest(
            fixtures=[save_canonical_metadata(tmp_path, "transformers")],
            expected_fixtures=expected_pair(),
            comparison_policy=same_prefix_policy(),
            tolerance=ToleranceCalibration(
                atol=0.0,
                observed_max_abs_diff=0.0,
                calibration_factor=2.0,
                method="pending",
            ),
        )

        with pytest.raises(ValueError, match="missing oracle pair"):
            validate_calibration_coverage(tmp_path, manifest)

    def test_unsupported_fixture_shape_fails_closed(self, tmp_path):
        manifest = build_manifest(
            fixtures=[
                save_canonical_metadata(tmp_path, "transformers", vocab_size=2),
                save_canonical_metadata(tmp_path, "vllm", vocab_size=2),
            ],
            expected_fixtures=expected_pair(),
            comparison_policy=same_prefix_policy(),
            tolerance=ToleranceCalibration(
                atol=0.0,
                observed_max_abs_diff=0.0,
                calibration_factor=2.0,
                method="pending",
            ),
        )

        with pytest.raises(ValueError, match="unsupported canonical fixture shape"):
            validate_calibration_coverage(tmp_path, manifest)

    def test_oracle_length_mismatch_fails_before_calibration_is_recorded(self, tmp_path):
        manifest = build_manifest(
            fixtures=[
                save_canonical_metadata(tmp_path, "transformers", num_tokens=1),
                save_canonical_metadata(tmp_path, "vllm", num_tokens=2),
            ],
            expected_fixtures=expected_pair(),
            comparison_policy=same_prefix_policy(),
            tolerance=ToleranceCalibration(
                atol=0.0,
                observed_max_abs_diff=0.0,
                calibration_factor=2.0,
                method="pending",
            ),
        )

        with pytest.raises(ValueError, match="oracle pair requires identical token lengths"):
            validate_calibration_coverage(tmp_path, manifest)
