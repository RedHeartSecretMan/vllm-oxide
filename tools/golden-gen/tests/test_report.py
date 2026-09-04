from __future__ import annotations

import pytest
from pydantic import ValidationError

from golden_gen.report import ReleaseReportInput, render_release_report
from tests.support import pinned_kernel_paths, release_runtime


def _benchmark() -> dict:
    repetition = {
        "prefill_tokens_per_second": 100.0,
        "decode_tokens_per_second": 20.0,
        "time_to_first_token_ns": [10_000_000],
        "inter_token_latency_ns": {
            "samples": [2_000_000, 3_000_000],
            "mean": 2_500_000.0,
            "p50": 2_500_000.0,
            "p95": 2_950_000.0,
        },
        "memory": {
            "baseline_mib": 2000,
            "peak_mib": 3250,
            "delta_mib": 1250,
            "polling_interval_ms": 50,
            "sample_count": 3,
        },
    }
    return {
        "canonical_04": {"repetitions": [repetition, repetition, repetition]},
        "canonical_05": {"repetitions": [repetition, repetition, repetition]},
    }


def _input() -> ReleaseReportInput:
    return ReleaseReportInput(
        observation_commit="1" * 40,
        observation_tree="2" * 40,
        policy_checkpoint_commit="3" * 40,
        policy_checkpoint_tree="4" * 40,
        measurement_commit="5" * 40,
        measurement_tree="6" * 40,
        runtime=release_runtime(),
        kernel_paths=pinned_kernel_paths(),
        lifecycle={
            "expected": 56,
            "discovered": 56,
            "generated": 56,
            "calibration_compared": 56,
            "reference_compared": 28,
            "missing": 0,
            "unexpected": 0,
            "skipped": 0,
            "failed": 0,
            "duplicate": 0,
            "stale": 0,
            "unmatched": 0,
        },
        tolerance={
            "l1_near_tie_max_abs_logit_gap": 0.0,
            "l2_atol": 0.015625,
            "observation_sha256": "7" * 64,
            "raw_evidence_sha256": "8" * 64,
        },
        benchmark=_benchmark(),
        manifest_sha256="9" * 64,
        archive_sha256="a" * 64,
        limitations=["Single GPU Qwen3 offline generation only."],
    )


def test_report_records_three_commit_roles_all_metrics_and_no_self_reference():
    report = render_release_report(_input())

    for heading in (
        "Observation commit",
        "Policy checkpoint",
        "Measurement commit",
        "Prefill throughput",
        "Decode throughput",
        "Time to first token",
        "Per-token latency",
        "Peak memory",
    ):
        assert heading in report
    assert "final candidate commit" not in report.lower()
    assert "recorded externally" in report


def test_report_rejects_missing_metric_or_a_final_commit_self_reference():
    payload = _input().model_dump()
    del payload["benchmark"]["canonical_04"]["repetitions"][0]["memory"]
    with pytest.raises(ValidationError, match="benchmark"):
        ReleaseReportInput.model_validate(payload)

    payload = _input().model_dump()
    payload["final_commit"] = "f" * 40
    with pytest.raises(ValidationError, match="final_commit"):
        ReleaseReportInput.model_validate(payload)
