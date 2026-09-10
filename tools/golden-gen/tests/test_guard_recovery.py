"""CPU monitoring-policy feedback; injected faults are not driver diagnosis."""

import json
import subprocess
import sys
import time

import pytest

from golden_gen.guard import run_guarded


def fake_query(tmp_path, monkeypatch, delays):
    import os

    executable = tmp_path / "nvidia-smi"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,sys,time\nfrom pathlib import Path\n"
        f"path=Path({str(tmp_path / 'query-counts.json')!r})\n"
        'counts=json.loads(path.read_text()) if path.exists() else {"gpu":0,"compute":0}\n'
        'key="gpu" if "--query-gpu=index,memory.used,memory.free" in sys.argv else "compute"\n'
        "counts[key]+=1\npath.write_text(json.dumps(counts))\n"
        f"delays={delays!r}\ntime.sleep(delays.get((key,counts[key]),0))\n"
        'print("0, 1000, 15000" if key=="gpu" else "")\n'
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_real_query_timeout_keeps_five_second_deadline_and_recovers(tmp_path, monkeypatch):
    from golden_gen.guard_evidence import validate_guard_timeline

    fake_query(tmp_path, monkeypatch, {("gpu", 1): 6})
    evidence = tmp_path / "real-timeout.json"
    run_guarded([sys.executable, "-c", "import time; time.sleep(.12)"], evidence)
    record = json.loads(evidence.read_text())
    validate_guard_timeline(record)
    timeouts = [event for event in record["telemetry_events"] if event["outcome"] == "timeout"]
    assert len(timeouts) == 1
    assert timeouts[0]["queries"][0]["timeout_seconds"] == 5
    assert record["failure"] is None


def test_recovery_window_stops_blocked_retry_without_waiting_another_five_seconds(
    tmp_path, monkeypatch
):
    import pytest

    fake_query(
        tmp_path,
        monkeypatch,
        {("gpu", 2): 4.2, ("compute", 2): 6, ("gpu", 3): 4.2, ("compute", 3): 6},
    )
    evidence = tmp_path / "window.json"
    with pytest.raises(RuntimeError, match="deadline"):
        run_guarded([sys.executable, "-c", "import time; time.sleep(60)"], evidence)
    record = json.loads(evidence.read_text())
    failed = [event for event in record["telemetry_events"] if event["outcome"] == "fatal"][0]
    timeout = [event for event in record["telemetry_events"] if event["outcome"] == "timeout"][0]
    assert 15 <= failed["ended_seconds"] - timeout["started_seconds"] < 16
    assert any(
        timeout["started_seconds"] < sample["elapsed_seconds"] < timeout["ended_seconds"]
        for sample in record["fast_ram_samples"]
    )
    assert record["remaining_owned_pids"] == record["remaining_telemetry_pids"] == []


def test_successful_owner_survives_one_transient_telemetry_timeout(tmp_path):
    ready = tmp_path / "worker-exited-soon"
    evidence = tmp_path / "guard.json"
    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        if calls == 2:
            deadline = time.monotonic() + 2
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            # The real child has returned zero while the external query was blocked.
            time.sleep(0.15)
            raise subprocess.TimeoutExpired(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.free"], 5
            )
        return dict(
            available_ram_bytes=48 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            gpu_memory="0, 1000, 15000",
            compute_processes="",
        )

    run_guarded(
        [
            sys.executable,
            "-c",
            "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(.05)",
            str(ready),
        ],
        evidence,
        probe=probe,
    )
    result = json.loads(evidence.read_text())
    assert result["failure"] is None
    assert result["child_returncode"] == 0
    assert result["remaining_owned_pids"] == []
    assert result["after"]["gpu_memory"] == "0, 1000, 15000"


def test_ram_floor_is_not_blocked_by_slow_gpu_telemetry(tmp_path, monkeypatch):
    from pathlib import Path

    import pytest

    original_read = Path.read_text
    started = time.monotonic()
    calls = 0

    def proc_memory(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            amount = 48 * 1024**2 if time.monotonic() - started < 0.10 else 1024
            return f"MemAvailable: {amount} kB\n"
        return original_read(path, *args, **kwargs)

    def slow_probe():
        nonlocal calls
        calls += 1
        if calls > 1:
            time.sleep(0.6)
        return dict(
            available_ram_bytes=48 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            gpu_memory="0, 1000, 15000",
            compute_processes="",
        )

    monkeypatch.setattr(Path, "read_text", proc_memory)
    with pytest.raises(RuntimeError, match="RAM"):
        run_guarded(
            [sys.executable, "-c", "import time; time.sleep(.3)"],
            tmp_path / "ram-guard.json",
            probe=slow_probe,
            interval=0.01,
        )


def test_guard_records_real_query_and_fast_ram_timeline(tmp_path, monkeypatch):
    import os

    executable = tmp_path / "nvidia-smi"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\n"
        'print("0, 1000, 15000" if '
        '"--query-gpu=index,memory.used,memory.free" in sys.argv else "")\n'
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    evidence = tmp_path / "timeline.json"
    run_guarded([sys.executable, "-c", "import time; time.sleep(.12)"], evidence)
    record = json.loads(evidence.read_text())
    assert record["schema_version"] == 2
    assert record["ram_sample_count"] == len(record["fast_ram_samples"])
    assert record["remaining_telemetry_pids"] == []
    assert record["telemetry_cleanup_failure"] is None
    after = [event for event in record["telemetry_events"] if event["phase"] == "after"][-1]
    assert after["started_seconds"] >= record["owner_cleanup_completed_seconds"]
    assert [query["kind"] for query in after["queries"]] == ["gpu_memory", "compute_processes"]
    assert all(query["pid"] > 0 and query["timeout_seconds"] == 5 for query in after["queries"])
    from golden_gen.guard_evidence import validate_guard_timeline

    validate_guard_timeline(record)
    record["telemetry_events"][-1]["queries"][0]["timeout_seconds"] = 6
    import pytest

    with pytest.raises(ValueError, match="query"):
        validate_guard_timeline(record)
    record["telemetry_events"][-1]["queries"][0]["timeout_seconds"] = 5
    import copy

    missing_owned = copy.deepcopy(record)
    missing_owned["owned_process_samples"] = []
    with pytest.raises(ValueError, match="monitoring"):
        validate_guard_timeline(missing_owned)
    wrong_phase = copy.deepcopy(record)
    active = next(event for event in wrong_phase["telemetry_events"] if event["phase"] == "active")
    active["phase"] = "after"
    with pytest.raises(ValueError, match="lifecycle"):
        validate_guard_timeline(wrong_phase)
    truncated = copy.deepcopy(record)
    truncated["resource_samples"].pop()
    with pytest.raises(ValueError, match="snapshot"):
        validate_guard_timeline(truncated)


def test_persistent_timeout_cannot_become_success(tmp_path):
    import pytest

    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        if calls in (2, 3):
            raise subprocess.TimeoutExpired(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.free"], 5
            )
        return dict(
            available_ram_bytes=48 * 1024**3, compute_processes="", gpu_memory="0, 1000, 15000"
        )

    evidence = tmp_path / "persistent.json"
    with pytest.raises(subprocess.TimeoutExpired):
        run_guarded([sys.executable, "-c", "import time; time.sleep(30)"], evidence, probe=probe)
    record = json.loads(evidence.read_text())
    assert record["failure"]
    assert record["remaining_owned_pids"] == []


def test_recovered_telemetry_does_not_hide_worker_exit_seven(tmp_path):
    import pytest

    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(0.15)
            raise subprocess.TimeoutExpired(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.free"], 5
            )
        return dict(
            available_ram_bytes=48 * 1024**3, compute_processes="", gpu_memory="0, 1000, 15000"
        )

    evidence = tmp_path / "bad-worker.json"
    with pytest.raises(RuntimeError, match="exit status 7"):
        run_guarded([sys.executable, "-c", "raise SystemExit(7)"], evidence, probe=probe)
    record = json.loads(evidence.read_text())
    assert record["child_returncode"] == 7 and record["failure"]
    assert record["remaining_owned_pids"] == []


def test_auxiliary_stage_has_an_independent_wall_deadline(tmp_path):
    import pytest

    evidence = tmp_path / "deadline.json"
    with pytest.raises(RuntimeError, match="stage deadline"):
        run_guarded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            evidence,
            probe=lambda: dict(available_ram_bytes=32 * 1024**3, compute_processes=""),
            timeout_seconds=0.15,
        )
    record = json.loads(evidence.read_text())
    assert record["failure"] and record["remaining_owned_pids"] == []


def test_final_snapshot_does_not_reset_recovery_allowance(tmp_path):
    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        if calls in (2, 4):
            time.sleep(0.15)
            raise subprocess.TimeoutExpired(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.free"], 5
            )
        return dict(available_ram_bytes=32 * 1024**3, compute_processes="", gpu_memory="0, 1, 2")

    evidence = tmp_path / "final-timeout.json"
    with pytest.raises(subprocess.TimeoutExpired):
        run_guarded([sys.executable, "-c", "import time; time.sleep(.05)"], evidence, probe=probe)
    record = json.loads(evidence.read_text())
    assert [
        event["phase"] for event in record["telemetry_events"] if event["outcome"] == "timeout"
    ] == ["active", "after"]
    assert record["failure"] and record["after"] == {}
    assert record["remaining_owned_pids"] == record["remaining_telemetry_pids"] == []


@pytest.mark.parametrize("fault", ["exit", "csv"])
def test_non_timeout_probe_errors_do_not_launch_or_retry_worker(tmp_path, monkeypatch, fault):
    import os

    executable = tmp_path / "nvidia-smi"
    action = "raise SystemExit(9)" if fault == "exit" else 'print("malformed")'
    executable.write_text(f"#!{sys.executable}\n{action}\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    evidence = tmp_path / "invalid-probe.json"
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        run_guarded([sys.executable, "-c", "raise SystemExit(0)"], evidence)
    record = json.loads(evidence.read_text())
    assert record["child_pid"] is None and record["failure"]
    assert len(record["telemetry_events"]) == 1
    assert record["remaining_telemetry_pids"] == []


@pytest.mark.parametrize("target", ["owner", "telemetry"])
def test_cleanup_permission_failure_is_retained_and_rejected(tmp_path, monkeypatch, target):
    import multiprocessing
    import os
    import signal
    from contextlib import suppress

    owner_pid = tmp_path / "owner.pid"
    telemetry_pid = tmp_path / "telemetry.pid"

    def probe():
        telemetry_pid.write_text(str(os.getpid()))
        return dict(available_ram_bytes=32 * 1024**3, compute_processes="", gpu_memory="0, 1, 2")

    killpg = os.killpg

    def denied(pgid, signum):
        path = owner_pid if target == "owner" else telemetry_pid
        if path.exists() and pgid == int(path.read_text()):
            raise PermissionError("synthetic cleanup denied")
        return killpg(pgid, signum)

    monkeypatch.setattr(os, "killpg", denied)
    command = [
        sys.executable,
        "-c",
        "import pathlib,os,subprocess,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        'subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])',
        str(owner_pid),
    ]
    evidence = tmp_path / "cleanup.json"
    try:
        with pytest.raises(PermissionError, match="cleanup denied"):
            run_guarded(command, evidence, probe=probe)
        record = json.loads(evidence.read_text())
        field = "cleanup_failure" if target == "owner" else "telemetry_cleanup_failure"
        assert record[field] and record["failure"]
    finally:
        monkeypatch.setattr(os, "killpg", killpg)
        for path in (owner_pid, telemetry_pid):
            if path.exists():
                pgid = int(path.read_text())
                assert pgid != os.getpgrp()
                with suppress(ProcessLookupError):
                    killpg(pgid, signal.SIGKILL)
                for child in multiprocessing.active_children():
                    if child.pid == pgid:
                        child.join(timeout=1)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    try:
                        reaped, _ = os.waitpid(-pgid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if reaped == 0:
                        time.sleep(0.01)


def test_guard_rejects_erased_active_monitoring_interval():
    from golden_gen.guard_evidence import validate_guard_timeline

    ram = 32 * 1024**3
    samples = [
        dict(
            started_seconds=start,
            completed_seconds=end,
            elapsed_seconds=end,
            available_ram_bytes=ram,
            disk_free_bytes=1000,
            gpu_memory="0, 1, 2",
            compute_processes="",
        )
        for start, end in ((0.0, 0.5), (101.1, 101.5))
    ]
    events = []
    for index, (phase, sample) in enumerate(zip(("before", "after"), samples, strict=True)):
        start = sample["started_seconds"]
        queries = [
            dict(
                kind=kind,
                pid=200 + index * 2 + offset,
                timeout_seconds=5,
                started_seconds=start + 0.01 + offset * 0.2,
                ended_seconds=start + 0.1 + offset * 0.2,
                outcome="ok",
                error=None,
            )
            for offset, kind in enumerate(("gpu_memory", "compute_processes"))
        ]
        events.append(
            dict(
                attempt=index,
                phase=phase,
                started_seconds=start,
                ended_seconds=sample["completed_seconds"],
                outcome="fresh",
                snapshot_index=index,
                error=None,
                queries=queries,
            )
        )
    record = dict(
        schema_version=2,
        child_pid=10,
        owned_pgid=10,
        child_returncode=0,
        failure=None,
        cleanup_failure=None,
        telemetry_cleanup_failure=None,
        remaining_owned_pids=[],
        remaining_telemetry_pids=[],
        ram_poll_interval_ms=100,
        telemetry_interval_ms=1000,
        child_started_seconds=1.0,
        child_exit_observed_seconds=100.0,
        owner_cleanup_completed_seconds=101.0,
        elapsed_seconds=102.0,
        fast_ram_samples=[dict(elapsed_seconds=102.0, available_ram_bytes=ram)],
        ram_sample_count=1,
        maximum_fast_poll_gap_seconds=0,
        owned_process_samples=[],
        resource_samples=samples,
        telemetry_events=events,
        before=samples[0],
        after=samples[1],
        minimum_available_ram_bytes=ram,
        minimum_disk_free_bytes=1000,
        peak_gpu_used_mib=1,
        minimum_gpu_free_mib=2,
    )
    with pytest.raises(ValueError, match="monitoring|coverage"):
        validate_guard_timeline(record)
