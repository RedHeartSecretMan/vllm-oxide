"""Behavioral seam of the versioned pure model-comparison module."""

import numpy as np
import pytest


def test_identical_distributions_report_zero_but_pending_budgets_cannot_pass() -> None:
    from golden_gen.layered_accuracy import compare_case

    logits = np.array([[0.0, 1.0], [2.0, -2.0]], dtype=np.float32)
    result = compare_case(logits, logits, logits, [1, 0])
    assert result["protocol"] == "layered-accuracy-v1"
    assert result["verdict"] == "INVALID"
    assert result["accepting"] is False
    assert result["numerical_checks"]["k_mean"] == 0.0
    assert result["numerical_checks"]["e_mean"] == 0.0
    assert result["behavior_checks"]["g_peak"] == 0.0


def test_full_distribution_metrics_use_paired_rows_and_reference_choice_loss() -> None:
    from golden_gen.layered_accuracy import compare_case

    reference = np.log(np.array([[0.75, 0.25], [0.5, 0.5]]))
    candidate = np.log(np.array([[0.5, 0.5], [0.75, 0.25]]))
    baseline = reference.copy()
    result = compare_case(reference, candidate, baseline, [0, 0])
    numerical = result["numerical_checks"]
    assert numerical["k_mean"] == pytest.approx((0.13081203594113697 + 0.14384103622589042) / 2)
    assert numerical["e_mean"] == pytest.approx(numerical["k_mean"])
    assert numerical["tv_mean"] == pytest.approx(0.25)
    assert numerical["tv_p95"] == pytest.approx(0.25)
    assert numerical["kl_p95"] == pytest.approx(0.14318958621165274)
    shifted = compare_case(reference, candidate + 8, baseline - 4, [0, 0])
    assert shifted["numerical_checks"]["k_mean"] == pytest.approx(numerical["k_mean"])
    assert result["behavior_checks"]["g_peak"] == 0.0


def test_wrong_raw_greedy_is_behavior_failure_even_when_reference_choice_is_optimal() -> None:
    from golden_gen.layered_accuracy import compare_case

    result = compare_case(
        np.array([[1.0, 1.0]]), np.array([[2.0, 1.0]]), np.array([[1.0, 1.0]]), [1]
    )
    assert result["behavior_checks"]["g_peak"] == 0.0
    assert result["behavior_checks"]["prediction_valid"] is False


def test_each_frozen_budget_is_independent_and_raw_prediction_cannot_be_excused() -> None:
    from golden_gen.layered_accuracy import Budgets, compare_case

    # Synthetic unit-test budgets, never policy defaults or proposed release values.
    ref = np.array([[1.0, 0.0]])
    rust = np.array([[0.0, 1.0]])
    baseline = np.array([[-10.0, 10.0]])
    result = compare_case(ref, rust, baseline, [1], Budgets(0, 0, 0, 0))
    assert result["verdict"] == "FAIL"
    assert result["numerical_checks"]["conditions"] == {
        "absolute_mean": False,
        "absolute_peak": False,
        "paired_mean": True,
    }
    assert result["behavior_checks"]["choice_loss_passed"] is False
    invalid_prediction = compare_case(ref, ref, ref, [1], Budgets(2, 2, 0, 2))
    assert invalid_prediction["verdict"] == "FAIL"
    exact = compare_case(ref, ref, ref, [0], Budgets(0, 0, 0, 0))
    assert exact["verdict"] == "PASS"
    assert exact["accepting"] is False  # A pure case result is never a release authorization.
    with pytest.raises(ValueError):
        Budgets(1, 0, 0, 0)


def test_arithmetic_floor_underflow_and_invalid_rows_are_explicit() -> None:
    from golden_gen.layered_accuracy import checked_kl, compare_case

    assert checked_kl(-1e-12) == 0
    with pytest.raises(ValueError):
        checked_kl(-1.0001e-12)
    ref = np.array([[1000.0, -1000.0]])
    result = compare_case(ref, np.zeros((1, 2)), ref, [0])
    assert result["numerical_checks"]["k_mean"] == pytest.approx(np.log(2))
    assert result["rows"][0]["probability_underflow_counts"] == [1, 0, 1]
    for bad in (
        np.array([[np.nan, 0]]),
        np.array([[np.inf, 0]]),
        np.empty((0, 2)),
        np.empty((1, 0)),
    ):
        with pytest.raises(ValueError):
            compare_case(bad, bad, bad, [0])
