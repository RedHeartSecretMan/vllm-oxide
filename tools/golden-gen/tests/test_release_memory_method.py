"""Synthetic completed method records; never approved real R performance."""

import copy

import pytest

from golden_gen.release_memory_method import validate_memory_method


def method_record():
    samples = []
    for i, used in enumerate((2 << 20, 3 << 20, 2 << 20)):
        observed = (i + 1) * 6_000_000
        samples.append(
            dict(
                api_enter_ns=observed - 1_000_000,
                api_return_ns=observed - 100,
                api_wall_span_ns=999_900,
                observed_ns=observed,
                return_code=0,
                bytes=dict(total=4 << 20, free=(4 << 20) - used, used=used),
                error=None,
                final_sample=i == 2,
            )
        )
    library = dict(
        requested_soname="libnvidia-ml.so.1",
        loader_selection=(
            "fixed soname through platform dynamic linker; no application library override"
        ),
        loaded_library_path="/synthetic/libnvidia-ml.so.1",
        canonical_library_path="/synthetic/libnvidia-ml.so.1",
        library_sha256="e" * 64,
        library_size_bytes=100,
        memory_query_symbol_file=dict(
            path="/synthetic/libnvidia-ml.so.1",
            canonical_path="/synthetic/libnvidia-ml.so.1",
            sha256="e" * 64,
            size_bytes=100,
        ),
        actual_library_mappings=["0000-0100 r-xp 0000 00:00 0 /synthetic/libnvidia-ml.so.1"],
        device_index=0,
        memory_size=24,
        memory_offsets=[0, 8, 16],
        nvml_version="synthetic",
        driver_version="synthetic",
    )
    return dict(
        protocol="memory-method-record-v1",
        schema_version=1,
        method="direct-nvml-v1",
        record_only=True,
        accepting=False,
        call_id="canonical_04-repetition-1",
        source=dict(commit="c" * 40, tree="c" * 40),
        build_source_id="d" * 40,
        benchmark_pid=123,
        sampling_target_ms=5,
        sampling_schedule="previous validated observation + 5ms; no catch-up",
        completion_gap_limit_ns=50_000_000,
        additional_api_wall_span_limit_ns=50_000_000,
        worker_ack_timeout_seconds=5.0,
        raw_sample_limit=20000,
        primary_failure=None,
        worker_joined=True,
        clock="same benchmark-stage Rust Instant origin; elapsed nanoseconds",
        completion_observation="observed_ns after successful API return and memory validation",
        sampling_method="direct nvmlDeviceGetMemoryInfo, device0; no CLI/Python sampler",
        projection=(
            "used_mib=floor(used_bytes/2^20); raw bytes retained; not old native parser evidence"
        ),
        times=dict(
            post_initialization_monitor_started_ns=1_000_000,
            baseline_ready_ns=6_100_000,
            generate_invoked_ns=7_000_000,
            generate_returned_ns=12_100_000,
            generate_succeeded_after_internal_final_step_sync_ns=12_200_000,
            finish_requested_ns=13_000_000,
            worker_join_completed_ns=19_000_000,
        ),
        trace=dict(
            library=library,
            raw_samples=samples,
            projected_samples=[
                dict(elapsed_ms=0, used_mib=2),
                dict(elapsed_ms=6, used_mib=3),
                dict(elapsed_ms=12, used_mib=2),
            ],
            baseline_observed_ns=6_000_000,
            maximum_gap_ns=6_000_000,
            finish_requested_ns=13_000_000,
            final_sample_enter_ns=17_000_000,
            final_sample_return_ns=17_999_900,
            final_sample_observed_ns=18_000_000,
            worker_finished_ns=18_700_000,
            primary_error=None,
            shutdown_error=None,
            shutdown_completed=True,
            shutdown_enter_ns=18_100_000,
            shutdown_return_ns=18_600_000,
        ),
    )


def test_method_raw_bytes_reproduce_legacy_memory_projection():
    data = method_record()
    observed = validate_memory_method(
        data, "canonical_04-repetition-1", dict(commit="c" * 40, tree="c" * 40), "d" * 40, 123
    )
    assert observed == dict(
        polling_interval_ms=6,
        sample_count=3,
        baseline_mib=2,
        peak_mib=3,
        delta_mib=1,
        samples=[
            dict(elapsed_ms=0, used_mib=2),
            dict(elapsed_ms=6, used_mib=3),
            dict(elapsed_ms=12, used_mib=2),
        ],
        other_compute_processes=[],
    )


@pytest.mark.parametrize(
    "fault",
    [
        "pending",
        "gap",
        "api_span",
        "projection",
        "final_before_finish",
        "wrong_source",
        "shutdown",
        "library_symbol",
        "baseline_before_initialization",
    ],
)
def test_memory_method_rejects_incomplete_or_misattributed_raw(fault):
    data = copy.deepcopy(method_record())
    if fault == "pending":
        data["trace"]["raw_samples"][-1]["api_return_ns"] = None
    elif fault == "gap":
        data["trace"]["raw_samples"][-1]["observed_ns"] = 62_000_001
    elif fault == "api_span":
        data["trace"]["raw_samples"][0]["api_wall_span_ns"] = 50_000_001
    elif fault == "projection":
        data["trace"]["projected_samples"][1]["used_mib"] = 2
    elif fault == "final_before_finish":
        data["times"]["finish_requested_ns"] = 18_000_001
    elif fault == "wrong_source":
        data["source"]["commit"] = "a" * 40
    elif fault == "shutdown":
        data["trace"]["shutdown_completed"] = False
    elif fault == "library_symbol":
        data["trace"]["library"]["memory_query_symbol_file"]["sha256"] = "f" * 64
    elif fault == "baseline_before_initialization":
        data["times"]["post_initialization_monitor_started_ns"] = 5_500_000
    with pytest.raises(ValueError):
        validate_memory_method(
            data, "canonical_04-repetition-1", dict(commit="c" * 40, tree="c" * 40), "d" * 40, 123
        )
