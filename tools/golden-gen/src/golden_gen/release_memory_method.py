"""Recompute the explicit R direct-NVML record from raw bytes and real times.

Pure CPU verification. Records stay nonaccepting; only the full source-role
adapter can combine them with approved sources, actual owner guards and Y math.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

LIMIT = 50_000_000
CALL_IDS = tuple(f"{w}-repetition-{n}" for w in ("canonical_04", "canonical_05") for n in (1, 2, 3))


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _ns(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("memory method requires actual nonnegative integer nanoseconds/bytes")
    return value


def _library(data: dict[str, Any]) -> None:
    _require(
        data.get("requested_soname") == "libnvidia-ml.so.1"
        and data.get("loader_selection")
        == "fixed soname through platform dynamic linker; no application library override"
        and data.get("device_index") == 0
        and type(data.get("device_index")) is int
        and data.get("memory_size") == 24
        and data.get("memory_offsets") == [0, 8, 16],
        "memory method actual NVML library/device/layout differs",
    )
    for key in ("loaded_library_path", "canonical_library_path"):
        _require(
            isinstance(data.get(key), str) and PurePosixPath(data[key]).is_absolute(),
            "memory method lacks actual absolute library path",
        )
    digest = data.get("library_sha256")
    _require(
        isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and _ns(data.get("library_size_bytes")) > 0,
        "memory method lacks actual library hash/size",
    )
    symbol = data.get("memory_query_symbol_file", {})
    _require(
        symbol.get("canonical_path") == data["canonical_library_path"]
        and symbol.get("sha256") == digest
        and symbol.get("size_bytes") == data["library_size_bytes"]
        and isinstance(symbol.get("path"), str)
        and PurePosixPath(symbol["path"]).is_absolute(),
        "memory query symbol does not belong to the recorded loaded library",
    )
    mappings = data.get("actual_library_mappings")
    _require(
        isinstance(mappings, list)
        and bool(mappings)
        and all(isinstance(row, str) and "libnvidia-ml.so" in row for row in mappings)
        and any(
            "r-xp" in row
            and (data["canonical_library_path"] in row or data["loaded_library_path"] in row)
            for row in mappings
        ),
        "memory method lacks actual executable library mappings",
    )
    _require(
        all(
            isinstance(data.get(key), str) and bool(data[key])
            for key in ("nvml_version", "driver_version")
        ),
        "memory method library versions are missing",
    )


def validate_memory_method(
    data: dict[str, Any],
    call_id: str,
    source: dict[str, str],
    build_source_id: str,
    benchmark_pid: int,
) -> dict[str, Any]:
    _require(
        call_id in CALL_IDS
        and data.get("protocol") == "memory-method-record-v1"
        and type(data.get("schema_version")) is int
        and data["schema_version"] == 1
        and data.get("method") == "direct-nvml-v1"
        and data.get("record_only") is True
        and data.get("accepting") is False
        and "kind" not in data
        and data.get("call_id") == call_id
        and data.get("source") == source
        and data.get("build_source_id") == build_source_id
        and type(data.get("benchmark_pid")) is int
        and data["benchmark_pid"] == benchmark_pid,
        "memory method source/identity/version differs",
    )
    _require(
        data.get("sampling_target_ms") == 5
        and type(data.get("sampling_target_ms")) is int
        and data.get("sampling_schedule") == "previous validated observation + 5ms; no catch-up"
        and data.get("completion_gap_limit_ns") == LIMIT
        and data.get("additional_api_wall_span_limit_ns") == LIMIT
        and data.get("clock") == "same benchmark-stage Rust Instant origin; elapsed nanoseconds"
        and data.get("completion_observation")
        == "observed_ns after successful API return and memory validation"
        and data.get("worker_ack_timeout_seconds") == 5.0
        and data.get("sampling_method")
        == "direct nvmlDeviceGetMemoryInfo, device0; no CLI/Python sampler"
        and data.get("projection")
        == "used_mib=floor(used_bytes/2^20); raw bytes retained; not old native parser evidence"
        and data.get("raw_sample_limit") == 20_000,
        "memory method fixed clock/schedule/limits/projection differs",
    )
    trace = data["trace"]
    _require(
        data.get("worker_joined") is True
        and data.get("primary_failure") is None
        and trace.get("primary_error") is None
        and trace.get("shutdown_error") is None
        and trace.get("shutdown_completed") is True,
        "memory method worker/error/shutdown incomplete",
    )
    _library(trace["library"])
    times = data["times"]
    ordered = [
        times.get(key)
        for key in (
            "post_initialization_monitor_started_ns",
            "baseline_ready_ns",
            "generate_invoked_ns",
            "generate_returned_ns",
            "generate_succeeded_after_internal_final_step_sync_ns",
            "finish_requested_ns",
            "worker_join_completed_ns",
        )
    ]
    ordered = [_ns(value) for value in ordered]
    _require(
        all(a <= b for a, b in zip(ordered, ordered[1:], strict=False)),
        "memory method baseline/inference/final-sync lifecycle differs",
    )
    raw = trace["raw_samples"]
    _require(
        isinstance(raw, list) and 2 <= len(raw) <= 20_000,
        "memory method raw sample inventory incomplete",
    )
    _require(
        ordered[0] <= _ns(raw[0].get("api_enter_ns")),
        "memory method baseline sampling starts before post-initialization monitor",
    )
    observations, projected, previous = [], [], None
    for index, sample in enumerate(raw):
        entered, returned, span, observed = (
            _ns(sample.get(key))
            for key in ("api_enter_ns", "api_return_ns", "api_wall_span_ns", "observed_ns")
        )
        _require(
            entered <= returned <= observed
            and returned - entered == span
            and span <= LIMIT
            and type(sample.get("return_code")) is int
            and sample["return_code"] == 0
            and sample.get("error") is None
            and sample.get("final_sample") is (index == len(raw) - 1),
            "memory method API return/span/error/final sample invalid",
        )
        if previous is not None:
            _require(
                entered >= previous + 5_000_000 and 0 < observed - previous <= LIMIT,
                "memory method actual completion gap or no-catch-up schedule invalid",
            )
        memory = sample["bytes"]
        total, free, used = (_ns(memory.get(key)) for key in ("total", "free", "used"))
        _require(total > 0 and free + used == total, "memory method raw memory accounting differs")
        observations.append(observed)
        projected.append(
            dict(elapsed_ms=(observed - observations[0]) // 1_000_000, used_mib=used // 2**20)
        )
        previous = observed
    gap = max(b - a for a, b in zip(observations, observations[1:], strict=False))
    _require(
        trace.get("maximum_gap_ns") == gap
        and trace["projected_samples"] == projected
        and trace.get("baseline_observed_ns") == observations[0] <= times["baseline_ready_ns"]
        and trace.get("finish_requested_ns") == times["finish_requested_ns"],
        "memory method raw/projection/baseline/finish differs",
    )
    last = raw[-1]
    _require(
        trace.get("final_sample_enter_ns") == last["api_enter_ns"] >= times["finish_requested_ns"]
        and trace.get("final_sample_return_ns") == last["api_return_ns"]
        and trace.get("final_sample_observed_ns") == observations[-1],
        "memory method final sample does not cover final sync",
    )
    closed = [
        _ns(trace.get(key))
        for key in ("shutdown_enter_ns", "shutdown_return_ns", "worker_finished_ns")
    ]
    _require(
        observations[-1]
        <= closed[0]
        <= closed[1]
        <= closed[2]
        <= times["worker_join_completed_ns"],
        "memory method shutdown/join lifecycle differs",
    )
    used_values = [sample["used_mib"] for sample in projected]
    _require(
        all(
            a["elapsed_ms"] < b["elapsed_ms"]
            for a, b in zip(projected, projected[1:], strict=False)
        ),
        "memory method legacy floor-ms observation must increase",
    )
    return dict(
        polling_interval_ms=(gap + 999_999) // 1_000_000,
        sample_count=len(projected),
        baseline_mib=used_values[0],
        peak_mib=max(used_values),
        delta_mib=max(used_values) - used_values[0],
        samples=projected,
        other_compute_processes=[],
    )
