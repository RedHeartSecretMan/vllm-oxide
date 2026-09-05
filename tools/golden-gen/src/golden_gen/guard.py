"""Supervise a GPU owner for its full lifetime, including model initialization."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


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
    after: dict[str, Any] = {}
    try:
        while child.poll() is None:
            if probe is None:
                available = next(
                    int(line.split()[1]) * 1024
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            else:
                available = snapshot()["available_ram_bytes"]
            samples += 1
            minimum = min(minimum, available)
            if available < 16 * 1024**3:
                raise RuntimeError("available host RAM fell below 16 GiB during stage")
            time.sleep(interval)
        if child.returncode != 0:
            raise RuntimeError(f"guarded command failed with exit status {child.returncode}")
    except BaseException as error:
        failure = str(error) or type(error).__name__
        raise
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        try:
            after = snapshot()
            if after["available_ram_bytes"] < 16 * 1024**3 and failure is None:
                raise RuntimeError("available host RAM is below 16 GiB after stage")
        except BaseException as error:
            failure = failure or str(error)
            raise
        finally:
            record = {
                "schema_version": 1,
                "command": command,
                "child_pid": child.pid,
                "child_returncode": child.returncode,
                "before": before,
                "after": after,
                "minimum_available_ram_bytes": minimum,
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
