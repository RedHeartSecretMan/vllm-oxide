from __future__ import annotations

import json
import sys

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
