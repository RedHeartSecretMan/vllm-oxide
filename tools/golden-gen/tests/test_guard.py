from __future__ import annotations

import json
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from golden_gen.guard import run_guarded


def test_active_guard_stops_child_when_ram_drops_and_records_before_after(tmp_path):
    probes = iter([{"available_ram_bytes": 32 * 1024**3}, {"available_ram_bytes": 1024}])
    evidence = tmp_path / "guard.json"
    with pytest.raises(RuntimeError, match="RAM"):
        run_guarded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            evidence,
            probe=lambda: next(probes, {"available_ram_bytes": 1024}),
            interval=0.01,
        )
    record = json.loads(evidence.read_bytes())
    assert record["before"]["available_ram_bytes"] == 32 * 1024**3
    assert record["after"]["available_ram_bytes"] == 1024
    assert record["child_returncode"] < 0
    assert record["failure"]


def test_guard_preserves_child_failure_and_never_turns_it_into_success(tmp_path):
    with pytest.raises(RuntimeError, match="exit status 7"):
        run_guarded(
            [sys.executable, "-c", "raise SystemExit(7)"],
            tmp_path / "guard.json",
            probe=lambda: {"available_ram_bytes": 32 * 1024**3},
            interval=0.01,
        )


def test_guard_cleans_owned_descendants_after_direct_child_exits(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    evidence = tmp_path / "guard.json"
    command = [
        sys.executable,
        "-c",
        "import subprocess,sys,pathlib; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))",
    ]
    child_pid = None
    try:
        run_guarded(
            command, evidence, probe=lambda: {"available_ram_bytes": 32 * 1024**3}, interval=0.01
        )
        child_pid = int(pid_file.read_text())
        assert not Path(f"/proc/{child_pid}").exists(), (
            "owned descendant must be stopped and reaped"
        )
    finally:
        if child_pid is None and pid_file.exists():
            child_pid = int(pid_file.read_text())
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


def test_guard_detects_external_compute_and_retains_continuous_resource_samples(tmp_path):
    observations = iter(
        [
            dict(
                available_ram_bytes=32 * 1024**3,
                disk_free_bytes=9999,
                gpu_memory="0, 100, 900",
                compute_processes="",
            ),
            dict(
                available_ram_bytes=31 * 1024**3,
                disk_free_bytes=8888,
                gpu_memory="0, 200, 800",
                compute_processes="99999999, external, 100",
            ),
        ]
    )
    last = dict(
        available_ram_bytes=32 * 1024**3,
        disk_free_bytes=7777,
        gpu_memory="0, 100, 900",
        compute_processes="",
    )
    evidence = tmp_path / "guard.json"
    with pytest.raises(RuntimeError, match="unrelated"):
        run_guarded(
            [sys.executable, "-c", "import time; time.sleep(.1)"],
            evidence,
            probe=lambda: next(observations, last),
            interval=0.01,
        )
    raw = json.loads(evidence.read_text())
    assert raw["remaining_owned_pids"] == []
    assert raw["minimum_disk_free_bytes"] == 7777
    assert raw["peak_gpu_used_mib"] == 200
    assert raw["minimum_gpu_free_mib"] == 800
