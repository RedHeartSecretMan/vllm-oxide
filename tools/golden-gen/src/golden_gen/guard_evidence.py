"""Reconstruct successful supervision from recorded observations, not PASS flags."""

from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid guard timestamp/resource number")
    return float(value)


def _integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("invalid guard integer")
    return value


def _gpu(value: Any) -> list[tuple[int, int, int]]:
    if not isinstance(value, str) or not value:
        raise ValueError("missing GPU resource data")
    rows = []
    for line in value.splitlines():
        parts = line.split(",")
        if len(parts) != 3:
            raise ValueError("malformed GPU resource data")
        row = tuple(int(part.strip()) for part in parts)
        if any(item < 0 for item in row):
            raise ValueError("negative GPU resource data")
        rows.append((row[0], row[1], row[2]))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("duplicate GPU index")
    return rows


def _compute(value: Any) -> set[int]:
    if not isinstance(value, str):
        raise ValueError("missing compute resource data")
    result = set()
    for line in value.splitlines():
        parts = line.split(",")
        if len(parts) != 3 or int(parts[0].strip()) <= 0:
            raise ValueError("malformed compute resource data")
        result.add(int(parts[0].strip()))
    return result


def validate_guard_timeline(record: dict[str, Any]) -> None:
    """Validate the fixed ADR-0019 policy; no caller-selected tolerance is accepted."""
    if (
        record.get("schema_version") != 2
        or type(record.get("child_returncode")) is not int
        or record["child_returncode"] != 0
        or record.get("failure") is not None
        or record.get("cleanup_failure") is not None
        or record.get("telemetry_cleanup_failure") is not None
        or record.get("remaining_owned_pids") != []
        or record.get("remaining_telemetry_pids") != []
        or record.get("ram_poll_interval_ms") != 100
        or record.get("telemetry_interval_ms") != 1000
    ):
        raise ValueError("unsuccessful or incompatible supervised guard")
    pid = _integer(record["child_pid"], 1)
    if record.get("owned_pgid") != pid:
        raise ValueError("guard owner PGID mismatch")
    duration = _number(record["elapsed_seconds"])
    launched = _number(record["child_started_seconds"])
    exited = _number(record["child_exit_observed_seconds"])
    cleaned = _number(record["owner_cleanup_completed_seconds"])
    if not launched <= exited <= cleaned <= duration:
        raise ValueError("invalid owner lifecycle order")
    fast = record["fast_ram_samples"]
    if (
        not fast
        or type(record["ram_sample_count"]) is not int
        or len(fast) != record["ram_sample_count"]
    ):
        raise ValueError("invalid independent RAM sample count")
    times = [_number(sample["elapsed_seconds"]) for sample in fast]
    if any(b <= a for a, b in zip(times, times[1:], strict=False)) or times[-1] > duration:
        raise ValueError("invalid independent RAM timeline")
    gap = max((b - a for a, b in zip(times, times[1:], strict=False)), default=0)
    if _number(record["maximum_fast_poll_gap_seconds"]) != gap:
        raise ValueError("false independent RAM polling gap")
    rams = [_integer(sample["available_ram_bytes"], 16 * 1024**3) for sample in fast]
    if not any(launched <= when <= exited for when in times):
        raise ValueError("missing RAM monitoring coverage during the worker lifecycle")
    known = {pid}
    previous = launched
    observed_owner = False
    for observation in record["owned_process_samples"]:
        when = _number(observation["elapsed_seconds"])
        if not previous <= when <= cleaned:
            raise ValueError("invalid owned-process timeline")
        previous = when
        members = observation["pids"]
        if members != sorted(set(members)):
            raise ValueError("invalid owned-process membership")
        known.update(_integer(member, 1) for member in members)
        observed_owner |= pid in members and when <= exited
    if not observed_owner:
        raise ValueError("missing owned-process monitoring coverage")
    samples = record["resource_samples"]
    events = record["telemetry_events"]
    if not samples or not events:
        raise ValueError("missing fresh resource evidence")
    if times[0] > samples[0]["started_seconds"] or times[-1] < samples[-1]["completed_seconds"]:
        raise ValueError("RAM monitoring coverage does not bracket resource supervision")
    fresh: list[tuple[str, dict[str, Any]]] = []
    timeout_at: int | None = None
    previous_end = 0.0
    phase_rank = {"before": 0, "active": 1, "after": 2}
    previous_rank = 0
    active_start: float | None = None
    for index, event in enumerate(events):
        start, end = _number(event["started_seconds"]), _number(event["ended_seconds"])
        phase = event["phase"]
        if (
            event["attempt"] != index
            or phase not in phase_rank
            or phase_rank[phase] < previous_rank
            or not previous_end <= start <= end <= duration
        ):
            raise ValueError("overlapping or unordered telemetry attempts")
        if (
            (phase == "before" and end > launched)
            or (phase == "active" and not launched <= start <= exited <= cleaned)
            or (phase == "active" and end > cleaned)
            or (phase == "after" and start < cleaned)
        ):
            raise ValueError("telemetry phase lies outside the owner lifecycle")
        if (
            phase == "active"
            and active_start is not None
            and events[index - 1]["outcome"] != "timeout"
            and start - active_start < 1
        ):
            raise ValueError("telemetry exceeds approved sampling rate")
        if phase == "active":
            active_start = start
        previous_rank, previous_end = phase_rank[phase], end
        queries = event["queries"]
        expected = ["gpu_memory", "compute_processes"]
        if not queries or len(queries) > 2:
            raise ValueError("missing or extra telemetry query")
        last_end = start
        for number, query in enumerate(queries):
            begin, finish = _number(query["started_seconds"]), _number(query["ended_seconds"])
            if (
                query["kind"] != expected[number]
                or type(query["timeout_seconds"]) is not int
                or query["timeout_seconds"] != 5
                or not last_end <= begin <= finish <= end
            ):
                raise ValueError("invalid telemetry query identity/deadline/order")
            _integer(query["pid"], 1)
            if query["outcome"] == "ok":
                if query["error"] is not None or finish - begin > 5:
                    raise ValueError("late or erroneous successful query")
            elif query["outcome"] == "timeout":
                if not query["error"] or number != len(queries) - 1 or finish - begin < 5:
                    raise ValueError("invalid timed-out query")
                if not any(begin < when < finish for when in times):
                    raise ValueError("missing independent RAM monitoring during timed-out query")
            else:
                raise ValueError("fatal telemetry query cannot be accepted")
            last_end = finish
        if event["outcome"] == "timeout":
            if (
                timeout_at is not None
                or event["snapshot_index"] is not None
                or not event["error"]
                or queries[-1]["outcome"] != "timeout"
            ):
                raise ValueError("invalid or repeated telemetry timeout")
            timeout_at = index
        elif event["outcome"] == "fresh":
            if (
                len(queries) != 2
                or any(query["outcome"] != "ok" for query in queries)
                or event["error"] is not None
            ):
                raise ValueError("partial snapshot cannot be fresh")
            if (
                type(event["snapshot_index"]) is not int
                or event["snapshot_index"] != len(fresh)
                or event["snapshot_index"] >= len(samples)
            ):
                raise ValueError("fresh snapshot index mismatch")
            sample = samples[event["snapshot_index"]]
            if (
                sample["started_seconds"] != start
                or sample["completed_seconds"] != end
                or sample["elapsed_seconds"] != end
                or end - start > 10
            ):
                raise ValueError("stale or late resource snapshot")
            if timeout_at == index - 1:
                prior = events[timeout_at]
                if phase != prior["phase"] or end - prior["started_seconds"] > 15:
                    raise ValueError("telemetry recovery window exceeded")
            fresh.append((phase, sample))
        else:
            raise ValueError("fatal telemetry attempt cannot be accepted")
    if timeout_at == len(events) - 1 or len(fresh) != len(samples):
        raise ValueError("unrecovered timeout or unbound resource sample")
    if exited - launched >= record["telemetry_interval_ms"] / 1000 and not any(
        phase == "active" for phase, _ in fresh
    ):
        raise ValueError("missing active telemetry monitoring coverage")
    if (
        fresh[0][0] != "before"
        or fresh[-1][0] != "after"
        or fresh[0][1] != record["before"]
        or fresh[-1][1] != record["after"]
        or fresh[0][1]["completed_seconds"] > launched
        or fresh[-1][1]["started_seconds"] < cleaned
    ):
        raise ValueError("missing fresh initial or post-cleanup snapshot")
    disks, gpu = [], []
    for phase, sample in fresh:
        rams.append(_integer(sample["available_ram_bytes"], 16 * 1024**3))
        disks.append(_integer(sample["disk_free_bytes"]))
        gpu.extend(_gpu(sample["gpu_memory"]))
        compute = _compute(sample["compute_processes"])
        if (phase != "active" and compute) or compute - known:
            raise ValueError("foreign compute in supervised resource evidence")
    if (
        record["minimum_available_ram_bytes"] != min(rams)
        or record["minimum_disk_free_bytes"] != min(disks)
        or record["peak_gpu_used_mib"] != max(row[1] for row in gpu)
        or record["minimum_gpu_free_mib"] != min(row[2] for row in gpu)
    ):
        raise ValueError("guard resource extrema differ from actual samples")
