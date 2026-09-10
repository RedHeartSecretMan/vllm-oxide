"""Supervise a GPU owner for its full lifetime, including model initialization."""

from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from golden_gen.telemetry import IsolatedTelemetry


@contextmanager
def _adopt_owned_descendants() -> Iterator[None]:
    """Linux subreaper permits reaping grandchildren after an owner exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    previous = ctypes.c_int()
    if prctl(37, ctypes.addressof(previous), 0, 0, 0) != 0 or prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot establish owner subreaper")
    try:
        yield
    finally:
        if prctl(36, previous.value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot restore owner subreaper")


def _group_members(pgid: int) -> set[int]:
    members = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) == pgid:
                members.add(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def _cleanup_owned_group(
    child: subprocess.Popen[Any], on_poll: Callable[[], None] | None = None
) -> list[str]:
    """Signal only the fresh owner's PGID, even after its direct child exits."""
    signals = []
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 1)):
        if not _group_members(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
            signals.append(sig.name)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            child.poll()
            if on_poll is not None:
                on_poll()
            for pid in _group_members(child.pid) - {child.pid}:
                with suppress(ChildProcessError):
                    os.waitpid(pid, os.WNOHANG)
            if not _group_members(child.pid):
                return signals
            time.sleep(0.01)
    if _group_members(child.pid):
        raise RuntimeError("owned process group could not be fully cleaned")
    return signals


def resource_snapshot(directory: Path) -> dict[str, Any]:
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    return {
        "available_ram_bytes": available,
        "disk_free_bytes": shutil.disk_usage(directory).free,
        "gpu_memory": gpu,
        "compute_processes": processes,
    }


def run_guarded(
    command: list[str],
    evidence: Path,
    *,
    probe: Callable[[], dict[str, Any]] | None = None,
    interval: float = 0.1,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    supervision: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> None:
    """Supervise one owner with independent RAM checks and bounded telemetry."""
    with _adopt_owned_descendants():
        _run_guarded(
            command,
            evidence,
            probe=probe,
            interval=interval,
            cwd=cwd,
            env=env,
            supervision=supervision,
            timeout_seconds=timeout_seconds,
        )


def _run_guarded(
    command: list[str],
    evidence: Path,
    *,
    probe: Callable[[], dict[str, Any]] | None,
    interval: float,
    cwd: Path | None,
    env: dict[str, str] | None,
    supervision: dict[str, Any] | None,
    timeout_seconds: float | None,
) -> None:
    from golden_gen.telemetry import read_ram

    if (
        not command
        or evidence.exists()
        or evidence.is_symlink()
        or type(interval) not in (int, float)
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("guard requires a command, positive interval and fresh evidence path")
    if timeout_seconds is not None and (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("invalid stage deadline")
    if supervision is not None:
        if supervision.get("role") == "measurement_owner":
            expected = {
                "role",
                "measurement_source",
                "supervision_source",
                "supervision_policy_sha256",
                "measurement_invocation",
            }
            invocation = supervision.get("measurement_invocation", {})
            if (
                cwd is None
                or str(cwd.resolve()) != invocation.get("cwd")
                or env is None
                or env.get("PYTHONPATH") != invocation.get("pythonpath")
                or env.get("PYTHONDONTWRITEBYTECODE") != invocation.get("pythondontwritebytecode")
            ):
                raise ValueError("actual worker invocation differs from its supervision binding")
        elif supervision.get("role") == "auxiliary_stage":
            expected = {"role", "stage_source", "stage_invocation"}
            if timeout_seconds is None:
                raise ValueError("auxiliary stage requires an independent deadline")
        else:
            raise ValueError("unknown supervision role")
        if set(supervision) != expected:
            raise ValueError("unexpected supervision metadata fields")
    origin = time.monotonic()
    child: subprocess.Popen[Any] | None = None
    telemetry: IsolatedTelemetry | None = None
    telemetry_processes: list[IsolatedTelemetry] = []
    failure: BaseException | None = None
    cleanup_error: BaseException | None = None
    telemetry_error: BaseException | None = None
    cleanup_signals: list[str] = []
    known_owned: set[int] = set()
    samples: list[dict[str, Any]] = []
    fast_samples: list[dict[str, Any]] = []
    owned_samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    child_started: float | None = None
    child_exited: float | None = None
    cleanup_completed: float | None = None
    recovered = 0
    last_fast = -float("inf")
    closing = False

    def elapsed() -> float:
        return time.monotonic() - origin

    def fast(force: bool = False) -> None:
        nonlocal last_fast, child_exited
        now = elapsed()
        if timeout_seconds is not None and now >= timeout_seconds:
            raise RuntimeError("independent stage deadline exceeded")
        if child is not None and cleanup_completed is None:
            members = _group_members(child.pid)
            known_owned.update(members)
            if not owned_samples or owned_samples[-1]["pids"] != sorted(members):
                owned_samples.append(dict(elapsed_seconds=now, pids=sorted(members)))
            status = child.poll()
            if status is not None:
                if child_exited is None:
                    child_exited = now
                if status != 0 and not closing:
                    raise RuntimeError(f"guarded command failed with exit status {status}")
        if force or now - last_fast >= interval:
            available = read_ram()
            fast_samples.append(dict(elapsed_seconds=now, available_ram_bytes=available))
            last_fast = now
            if available < 16 * 1024**3:
                raise RuntimeError("available host RAM fell below 16 GiB during stage")

    def checked(value: dict[str, Any], phase: str) -> str | None:
        ram = value.get("available_ram_bytes")
        if type(ram) is not int or ram < 0:
            raise ValueError("malformed available host RAM")
        disk = value.get("disk_free_bytes")
        if disk is not None and (type(disk) is not int or disk < 0):
            raise ValueError("malformed disk resource data")
        gpu = value.get("gpu_memory", "")
        if not isinstance(gpu, str):
            raise ValueError("malformed GPU resource data")
        for line in gpu.splitlines():
            parts = line.split(",")
            if len(parts) != 3 or any(int(part.strip()) < 0 for part in parts):
                raise ValueError("malformed GPU resource data")
        compute_text = value.get("compute_processes", "")
        if not isinstance(compute_text, str):
            raise ValueError("malformed compute resource data")
        compute = set()
        for line in compute_text.splitlines():
            parts = line.split(",")
            if len(parts) != 3 or int(parts[0].strip()) <= 0:
                raise ValueError("malformed compute resource data")
            compute.add(int(parts[0].strip()))
        if probe is None and (not gpu or disk is None):
            raise ValueError("incomplete resource snapshot")
        if ram < 16 * 1024**3:
            return "available host RAM is below 16 GiB"
        if (phase != "active" and compute) or compute - known_owned:
            return "unrelated CUDA compute processes are active"
        return None

    def snapshot(phase: str) -> dict[str, Any]:
        nonlocal recovered, telemetry, before, after
        if telemetry is None:
            telemetry = IsolatedTelemetry(probe, evidence.parent, origin)
            telemetry_processes.append(telemetry)
        window_start = time.monotonic()
        retry = False
        while True:
            started = elapsed()
            event: dict[str, Any] = dict(
                attempt=len(events),
                phase=phase,
                started_seconds=started,
                ended_seconds=None,
                outcome=None,
                snapshot_index=None,
                error=None,
                queries=[],
            )
            events.append(event)
            try:
                deadline = (
                    min(time.monotonic() + 10, window_start + 15)
                    if retry
                    else time.monotonic() + 10
                )
                value = telemetry.snapshot(fast, deadline)
                violation = checked(value, phase)
                completed = elapsed()
                if retry and time.monotonic() > window_start + 15:
                    raise RuntimeError("telemetry recovery exceeded 15 seconds")
                sample = dict(
                    value,
                    started_seconds=started,
                    completed_seconds=completed,
                    elapsed_seconds=completed,
                )
                event.update(
                    ended_seconds=completed,
                    outcome="fresh",
                    snapshot_index=len(samples),
                    queries=list(telemetry.queries),
                )
                samples.append(sample)
                if phase == "before":
                    before = sample
                elif phase == "after":
                    after = sample
                if violation is not None:
                    raise RuntimeError(violation)
                return sample
            except BaseException as error:
                if event["outcome"] != "fresh":
                    event.update(
                        ended_seconds=elapsed(),
                        error=str(error) or type(error).__name__,
                        outcome="timeout"
                        if isinstance(error, subprocess.TimeoutExpired)
                        else "fatal",
                        queries=list(telemetry.queries),
                    )
                if isinstance(error, subprocess.TimeoutExpired) and recovered == 0:
                    recovered += 1
                    retry = True
                    if time.monotonic() < window_start + 15:
                        continue
                raise

    try:
        fast()
        before = snapshot("before")
        child = subprocess.Popen(command, start_new_session=True, cwd=cwd, env=env)
        child_started = elapsed()
        known_owned.add(child.pid)
        fast(force=True)
        next_telemetry = time.monotonic()
        while child.poll() is None:
            fast()
            if child.returncode is not None:
                break
            if time.monotonic() >= next_telemetry:
                snapshot("active")
                next_telemetry = time.monotonic() + 1
            time.sleep(interval)
        if child_exited is None:
            child_exited = elapsed()
        if child.returncode != 0:
            raise RuntimeError(f"guarded command failed with exit status {child.returncode}")
    except BaseException as error:
        failure = error
    finally:
        closing = True
        try:
            fast(force=True)
        except BaseException as error:
            failure = failure or error
        if child is not None:
            try:

                def cleanup_tick() -> None:
                    nonlocal failure
                    try:
                        fast()
                    except BaseException as error:
                        failure = failure or error

                cleanup_signals = _cleanup_owned_group(child, cleanup_tick)
                cleanup_completed = elapsed()
                if child_exited is None:
                    child_exited = elapsed()
            except BaseException as error:
                cleanup_error = error
                failure = failure or error
            # A failed/pending query is terminated before the required final probe.
            if failure is not None and telemetry is not None and telemetry.pending:
                try:
                    telemetry.close()
                    telemetry = None
                except BaseException as error:
                    telemetry_error = error
                    failure = failure or error
            try:
                after = snapshot("after")
            except BaseException as error:
                failure = failure or error
        if telemetry is not None:
            try:
                telemetry.close()
            except BaseException as error:
                telemetry_error = error
                failure = failure or error
        try:
            fast(force=True)
        except BaseException as error:
            failure = failure or error
        ram_values = [sample["available_ram_bytes"] for sample in fast_samples + samples]
        disks = [sample["disk_free_bytes"] for sample in samples if "disk_free_bytes" in sample]
        gpu_rows = [
            tuple(int(part.strip()) for part in line.split(","))
            for sample in samples
            for line in sample.get("gpu_memory", "").splitlines()
        ]
        times = [sample["elapsed_seconds"] for sample in fast_samples]
        record = dict(
            schema_version=2,
            command=command,
            role="auxiliary_stage",
            stage_deadline_seconds=timeout_seconds,
            child_pid=child.pid if child is not None else None,
            child_returncode=child.returncode if child is not None else None,
            owned_pgid=child.pid if child is not None else None,
            cleanup_signals=cleanup_signals,
            remaining_owned_pids=sorted(_group_members(child.pid)) if child is not None else [],
            cleanup_failure=str(cleanup_error) if cleanup_error else None,
            remaining_telemetry_pids=sorted(
                {pid for process in telemetry_processes for pid in process.remaining()}
            ),
            telemetry_cleanup_failure=str(telemetry_error) if telemetry_error else None,
            owned_process_samples=owned_samples,
            before=before,
            after=after,
            minimum_available_ram_bytes=min(ram_values) if ram_values else None,
            minimum_disk_free_bytes=min(disks) if disks else None,
            peak_gpu_used_mib=max(row[1] for row in gpu_rows) if gpu_rows else None,
            minimum_gpu_free_mib=min(row[2] for row in gpu_rows) if gpu_rows else None,
            resource_samples=samples,
            fast_ram_samples=fast_samples,
            ram_poll_interval_ms=interval * 1000,
            ram_sample_count=len(fast_samples),
            maximum_fast_poll_gap_seconds=max(
                (b - a for a, b in zip(times, times[1:], strict=False)), default=0
            ),
            telemetry_interval_ms=1000,
            telemetry_events=events,
            child_started_seconds=child_started,
            child_exit_observed_seconds=child_exited,
            owner_cleanup_completed_seconds=cleanup_completed,
            elapsed_seconds=elapsed(),
            failure=str(failure) or type(failure).__name__ if failure is not None else None,
        )
        if supervision is not None:
            record.update(supervision)
        with evidence.open("x") as output:
            json.dump(record, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    if failure is not None:
        raise failure
