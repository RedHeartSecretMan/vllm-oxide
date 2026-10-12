"""Kernel capability decisions at the default GPU guard boundary; no GPU jobs."""

from contextlib import nullcontext
from pathlib import Path

import pytest

from golden_gen import guard


def kernel(monkeypatch, value):
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/sys/kernel/osrelease"):
            if isinstance(value, Exception):
                raise value
            return value
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


@pytest.mark.parametrize(
    "release", ["6.6.87.2-microsoft-standard-WSL2\n", "4.4.0-19041-Microsoft\n"]
)
def test_wsl_is_rejected_before_any_gpu_owner_or_query(monkeypatch, tmp_path, release):
    kernel(monkeypatch, release)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported platform must fail before any owner/query starts")

    monkeypatch.setattr(guard, "_adopt_owned_descendants", forbidden)
    monkeypatch.setattr(guard, "_run_guarded", forbidden)
    monkeypatch.setattr(guard.subprocess, "Popen", forbidden)
    monkeypatch.setattr(guard.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="WSL.*process enumeration"):
        guard.run_guarded(["never-start-CUDA"], tmp_path / "guard.json")
    assert not (tmp_path / "guard.json").exists()


def test_native_linux_preserves_default_guard_delegation(monkeypatch, tmp_path):
    kernel(monkeypatch, "6.8.0-generic\n")
    calls = []
    monkeypatch.setattr(guard, "_adopt_owned_descendants", nullcontext)
    monkeypatch.setattr(guard, "_run_guarded", lambda *args, **kwargs: calls.append(kwargs))
    guard.run_guarded(["mock-default-GPU-owner"], tmp_path / "guard.json")
    assert len(calls) == 1
    assert calls[0]["probe"] is None


@pytest.mark.parametrize("release", ["6.6.87-microsoft-standard-WSL2", OSError("unreadable")])
def test_custom_cpu_probe_does_not_require_gpu_kernel_capability(monkeypatch, tmp_path, release):
    kernel(monkeypatch, release)
    calls = []

    def probe():
        return {"available_ram_bytes": 32 * 1024**3}

    monkeypatch.setattr(guard, "_adopt_owned_descendants", nullcontext)
    monkeypatch.setattr(guard, "_run_guarded", lambda *args, **kwargs: calls.append(kwargs))
    guard.run_guarded(["mock-CPU-owner"], tmp_path / "guard.json", probe=probe)
    assert len(calls) == 1
    assert calls[0]["probe"] is probe


@pytest.mark.parametrize("failure", [OSError("unreadable"), UnicodeError("invalid"), "\n"])
def test_default_gpu_probe_fails_closed_when_kernel_cannot_be_established(
    monkeypatch, tmp_path, failure
):
    kernel(monkeypatch, failure)

    def forbidden(*args, **kwargs):
        raise AssertionError("no owner/query on unknown kernel capability")

    monkeypatch.setattr(guard, "_adopt_owned_descendants", forbidden)
    monkeypatch.setattr(guard, "_run_guarded", forbidden)
    with pytest.raises(RuntimeError, match="Cannot establish GPU compute-process enumeration"):
        guard.run_guarded(["never-start-CUDA"], tmp_path / "guard.json")
