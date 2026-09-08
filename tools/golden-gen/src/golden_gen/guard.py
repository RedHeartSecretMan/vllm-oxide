"""Supervise a GPU owner for its full lifetime, including model initialization."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


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


def _cleanup_owned_group(child: subprocess.Popen[Any]) -> list[str]:
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
        if child.poll() is None:
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                while os.waitpid(-child.pid, os.WNOHANG)[0] > 0:
                    pass
            except ChildProcessError:
                pass
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
) -> None:
    """Supervise one fresh owner and reap its whole process group before return."""
    with _adopt_owned_descendants():
        _run_guarded(command, evidence, probe=probe, interval=interval)


def _run_guarded(
    command: list[str],
    evidence: Path,
    *,
    probe: Callable[[], dict[str, Any]] | None,
    interval: float,
) -> None:
    """Fail closed on child failure, interrupted monitoring, or the host RAM floor.

    ``probe`` is the host-resource seam used by CPU tests. The production probe
    records disk/GPU/process state before and after, and RAM throughout execution.
    """
    if not command or evidence.exists() or evidence.is_symlink():
        raise ValueError("guard requires a command and fresh evidence path")
    snapshot = probe or (lambda: resource_snapshot(evidence.parent))
    before = snapshot()
    if before.get("compute_processes"):
        raise RuntimeError("unrelated CUDA compute processes are active before stage")
    if before["available_ram_bytes"] < 16 * 1024**3:
        raise RuntimeError("available host RAM is below 16 GiB before stage")
    child = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    minimum = before["available_ram_bytes"]
    samples = 1
    failure: str | None = None
    cleanup_signals: list[str] = []
    cleanup_error: BaseException | None = None
    after: dict[str, Any] = {}
    observed: list[dict[str, Any]] = []
    known_owned = {child.pid}
    minimum_disk: int | None = None
    peak_gpu: int | None = None
    minimum_gpu: int | None = None

    def retain(sample: dict[str, Any]) -> None:
        nonlocal minimum, minimum_disk, peak_gpu, minimum_gpu
        minimum = min(minimum, sample["available_ram_bytes"])
        if "disk_free_bytes" in sample:
            free = sample["disk_free_bytes"]
            minimum_disk = free if minimum_disk is None else min(minimum_disk, free)
        for line in sample.get("gpu_memory", "").splitlines():
            _, used, free_gpu = (int(part.strip()) for part in line.split(","))
            peak_gpu = used if peak_gpu is None else max(peak_gpu, used)
            minimum_gpu = free_gpu if minimum_gpu is None else min(minimum_gpu, free_gpu)
        observed.append(dict(elapsed_seconds=time.monotonic() - started, **sample))

    try:
        retain(before)
        while child.poll() is None:
            known_owned.update(_group_members(child.pid))
            current = snapshot()
            known_owned.update(_group_members(child.pid))
            retain(current)
            available = current["available_ram_bytes"]
            samples += 1
            if available < 16 * 1024**3:
                raise RuntimeError("available host RAM fell below 16 GiB during stage")
            compute = {
                int(line.split(",", 1)[0].strip())
                for line in current.get("compute_processes", "").splitlines()
            }
            if compute - known_owned:
                raise RuntimeError("unrelated CUDA compute process appeared during stage")
            time.sleep(interval)
        if child.returncode != 0:
            raise RuntimeError(f"guarded command failed with exit status {child.returncode}")
    except BaseException as error:
        failure = str(error) or type(error).__name__
        raise
    finally:
        try:
            cleanup_signals = _cleanup_owned_group(child)
        except BaseException as error:
            failure = failure or str(error)
            cleanup_error = error
        try:
            after = snapshot()
            retain(after)
            if after["available_ram_bytes"] < 16 * 1024**3 and failure is None:
                raise RuntimeError("available host RAM is below 16 GiB after stage")
            if after.get("compute_processes") and failure is None:
                raise RuntimeError("CUDA compute processes remain after owner cleanup")
        except BaseException as error:
            failure = failure or str(error)
            raise
        finally:
            record = {
                "schema_version": 1,
                "command": command,
                "child_pid": child.pid,
                "child_returncode": child.returncode,
                "owned_pgid": child.pid,
                "cleanup_signals": cleanup_signals,
                "remaining_owned_pids": sorted(_group_members(child.pid)),
                "cleanup_failure": str(cleanup_error) if cleanup_error is not None else None,
                "before": before,
                "after": after,
                "minimum_available_ram_bytes": minimum,
                "minimum_disk_free_bytes": minimum_disk,
                "peak_gpu_used_mib": peak_gpu,
                "minimum_gpu_free_mib": minimum_gpu,
                "resource_samples": observed,
                "ram_poll_interval_ms": interval * 1000,
                "ram_sample_count": samples,
                "elapsed_seconds": time.monotonic() - started,
                "failure": failure,
            }
            with evidence.open("x") as output:
                json.dump(record, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
        if cleanup_error is not None:
            raise cleanup_error
