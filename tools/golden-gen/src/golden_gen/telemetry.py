"""Single-flight, killable resource sampling isolated from the owner watchdog."""

from __future__ import annotations

import multiprocessing
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


def read_ram() -> int:
    value = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if value < 0:
        raise ValueError("invalid available RAM")
    return value


def _query(pipe: Connection, kind: str, argument: str, origin: float) -> str:
    started = time.monotonic() - origin
    command = ["nvidia-smi", argument, "--format=csv,noheader,nounits"]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    record = dict(
        kind=kind,
        pid=child.pid,
        timeout_seconds=5,
        started_seconds=started,
        ended_seconds=None,
        outcome=None,
        error=None,
    )
    pipe.send(("query", record))
    try:
        out, err = child.communicate(timeout=5)
        if child.returncode != 0:
            raise subprocess.CalledProcessError(child.returncode, command, out, err)
    except BaseException as error:
        child.kill()
        child.communicate()
        record.update(
            ended_seconds=time.monotonic() - origin,
            outcome="timeout" if isinstance(error, subprocess.TimeoutExpired) else "fatal",
            error=str(error),
        )
        pipe.send(("query", record))
        raise
    record.update(ended_seconds=time.monotonic() - origin, outcome="ok")
    pipe.send(("query", record))
    return out.strip()


def _serve(
    pipe: Connection, probe: Callable[[], dict[str, Any]] | None, directory: Path, origin: float
) -> None:
    os.setsid()
    pipe.send(("ready", None))
    while pipe.recv() == "snapshot":
        try:
            if probe is None:
                value = dict(
                    available_ram_bytes=read_ram(),
                    disk_free_bytes=shutil.disk_usage(directory).free,
                    gpu_memory=_query(
                        pipe, "gpu_memory", "--query-gpu=index,memory.used,memory.free", origin
                    ),
                    compute_processes=_query(
                        pipe,
                        "compute_processes",
                        "--query-compute-apps=pid,process_name,used_memory",
                        origin,
                    ),
                )
            else:
                value = probe()
            pipe.send(("ok", value))
        except BaseException as error:
            pipe.send(("error", error))


class IsolatedTelemetry:
    """One persistent probe process; callers keep checking safety while it waits."""

    def __init__(
        self, probe: Callable[[], dict[str, Any]] | None, directory: Path, origin: float
    ) -> None:
        context = multiprocessing.get_context("fork")
        self.pipe, child_pipe = context.Pipe()
        self.process = context.Process(target=_serve, args=(child_pipe, probe, directory, origin))
        self.queries: list[dict[str, Any]] = []
        self.pending = False
        self.process.start()
        child_pipe.close()
        if not self.pipe.poll(1) or self.pipe.recv()[0] != "ready":
            self.close()
            raise RuntimeError("telemetry process did not become ready")

    def snapshot(self, on_poll: Callable[[], None], deadline: float) -> dict[str, Any]:
        on_poll()
        self.queries = []
        self.pipe.send("snapshot")
        self.pending = True
        while True:
            on_poll()
            if self.pipe.poll(0.01):
                status, value = self.pipe.recv()
                if status == "query":
                    self.queries = [item for item in self.queries if item["pid"] != value["pid"]]
                    self.queries.append(value)
                    continue
                if status == "error":
                    self.pending = False
                    raise value
                if status != "ok" or not isinstance(value, dict):
                    raise ValueError("invalid telemetry result")
                self.pending = False
                if time.monotonic() >= deadline:
                    raise RuntimeError("late telemetry snapshot")
                return value
            if time.monotonic() >= deadline:
                raise RuntimeError("telemetry snapshot deadline exceeded")

    def close(self) -> None:
        if self.process.pid is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if self.process.is_alive():
                    self.process.kill()
            self.process.join(timeout=1)
            if self.process.is_alive():
                raise RuntimeError("telemetry process could not be cleaned")
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                try:
                    if os.waitpid(-self.process.pid, os.WNOHANG)[0] == 0:
                        time.sleep(0.01)
                        continue
                except ChildProcessError:
                    break
            if self.remaining():
                raise RuntimeError("telemetry descendants could not be cleaned")
        self.pipe.close()

    def remaining(self) -> list[int]:
        members = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                if int(fields[2]) == self.process.pid:
                    members.append(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
        return sorted(members)
