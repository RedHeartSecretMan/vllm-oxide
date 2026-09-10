"""Source-bound fixed CPU gate producer and offline raw-log validation."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from golden_gen.layered_accuracy import PROTOCOL
from golden_gen.layered_artifacts import atomic_json, bound_file, sha, source_identity

PRESSURE = "mixed_chunked_prefill_and_decode_survive_three_block_pressure"


def gate_commands(python: str) -> dict[str, list[str]]:
    cargo = ["cargo"]
    return {
        "default-core": cargo
        + ["test", "--offline", "-p", "vllm_oxide", "--no-default-features", "--lib"],
        "workspace": cargo + ["test", "--offline", "--workspace"],
        "internal-golden": cargo
        + ["test", "--offline", "--workspace", "--features", "internal-golden"],
        "fmt": cargo + ["fmt", "--all", "--check"],
        "clippy": cargo
        + [
            "clippy",
            "--offline",
            "--workspace",
            "--all-targets",
            "--features",
            "internal-golden",
            "--",
            "-D",
            "warnings",
        ],
        "dependencies": cargo + ["deny", "--frozen", "check"],
        "python": [python, "-m", "pytest", "tools/golden-gen/tests", "-q"],
        "ruff": [
            python,
            "-m",
            "ruff",
            "check",
            "tools/golden-gen/src",
            "tools/golden-gen/tests",
            "tools/golden-gen/scripts",
        ],
        "mypy": [
            python,
            "-m",
            "mypy",
            "--config-file",
            "tools/golden-gen/pyproject.toml",
            "tools/golden-gen/src",
            "tools/golden-gen/scripts",
        ],
    }


def _output_ok(name: str, stdout: str, stderr: str) -> None:
    if name in ("default-core", "workspace", "internal-golden"):
        if "test result: ok." not in stdout or re.search(
            r"\b[1-9][0-9]* (?:failed|ignored)", stdout
        ):
            raise ValueError("CPU Rust tests lack complete successful raw output")
        if name == "default-core" and not re.search(rf"test [^\n]*{PRESSURE} \.\.\. ok", stdout):
            raise ValueError("mandatory public pressure regression was not executed")
    if name == "python" and (
        not re.search(r"\b[1-9][0-9]* passed\b", stdout)
        or re.search(r"\b[1-9][0-9]* (?:failed|skipped|xfailed|xpassed|error)", stdout)
    ):
        raise ValueError("Python CPU gates are incomplete or skipped")
    if name == "ruff" and "All checks passed!" not in stdout:
        raise ValueError("Ruff raw success output missing")
    if name == "mypy" and "Success: no issues found" not in stdout:
        raise ValueError("mypy raw success output missing")
    if (
        name == "dependencies"
        and "advisories ok, bans ok, licenses ok, sources ok" not in stdout + stderr
    ):
        raise ValueError("dependency policy raw success output missing")


def collect_cpu(
    repo: Path, directory: Path, python: Path, worker: Path, binary: Path, target: Path
) -> Path:
    source = source_identity(repo)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    directory.mkdir(mode=0o700)
    environment = {
        "CARGO_BUILD_JOBS": "1",
        "CARGO_TARGET_DIR": str(target.resolve()),
        "CARGO_NET_OFFLINE": "true",
        "PYTHONPATH": str(repo / "tools/golden-gen/src"),
        "GOLDEN_WORKER_TEST_PYTHON": str(worker.absolute()),
        "GOLDEN_TRANSPORT_TEST_BINARY": str(binary.resolve()),
        "PYTEST_ADDOPTS": "",
        "PYTHONNOUSERSITE": "1",
        "RUSTFLAGS": "",
        "CARGO_ENCODED_RUSTFLAGS": "",
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_OXIDE_INTERNAL_")}
    env.update(environment)
    versions = {
        tool: subprocess.check_output(command, cwd=repo, env=env, text=True).strip()
        for tool, command in {
            "rustc": ["rustc", "--version", "--verbose"],
            "cargo": ["cargo", "--version"],
            "python": [str(python), "--version"],
        }.items()
    }
    records = []
    for name, command in gate_commands(str(python.absolute())).items():
        completed = subprocess.run(command, cwd=repo, env=env, capture_output=True, check=False)
        outputs = {}
        for channel in ("stdout", "stderr"):
            path = directory / f"{name}.{channel}.log"
            with path.open("xb") as stream:
                stream.write(getattr(completed, channel))
                stream.flush()
                os.fsync(stream.fileno())
            outputs[channel] = dict(path=path.name, sha256=sha(path))
        records.append(dict(gate=name, command=command, returncode=completed.returncode, **outputs))
        if completed.returncode:
            raise ValueError(f"CPU gate failed: {name}; original logs retained")
        _output_ok(name, completed.stdout.decode(), completed.stderr.decode())
    if source_identity(repo) != source:
        raise ValueError("source changed during CPU gates")
    path = directory / "evidence.json"
    atomic_json(
        path,
        dict(
            protocol=PROTOCOL,
            schema_version=1,
            kind="recorded_cpu_gates",
            producer="golden_gen.release_cpu-v1",
            source=source,
            python=str(python.absolute()),
            environment=environment,
            toolchain=versions,
            gates=records,
        ),
    )
    return path


def validate_cpu(path: Path, source: dict[str, str]) -> list[Path]:
    data = json.loads(path.read_text())
    if (
        data.get("protocol") != PROTOCOL
        or data.get("schema_version") != 1
        or data.get("kind") != "recorded_cpu_gates"
        or data.get("producer") != "golden_gen.release_cpu-v1"
        or data.get("source") != source
    ):
        raise ValueError("invalid CPU execution provenance")
    environment = data["environment"]
    if (
        environment.get("CARGO_BUILD_JOBS") != "1"
        or environment.get("CARGO_NET_OFFLINE") != "true"
        or not all(
            environment.get(k)
            for k in (
                "GOLDEN_WORKER_TEST_PYTHON",
                "GOLDEN_TRANSPORT_TEST_BINARY",
                "CARGO_TARGET_DIR",
                "PYTHONPATH",
            )
        )
        or set(data["toolchain"]) != {"rustc", "cargo", "python"}
        or not all(data["toolchain"].values())
    ):
        raise ValueError("CPU execution environment/toolchain missing")
    expected = gate_commands(data["python"])
    records = data["gates"]
    if [r["gate"] for r in records] != list(expected):
        raise ValueError("missing/duplicate CPU gate")
    closure = [path]
    for record in records:
        name = record["gate"]
        if (
            record["command"] != expected[name]
            or type(record["returncode"]) is not int
            or record["returncode"] != 0
        ):
            raise ValueError("CPU gate command/features or exit status mismatch")
        files = {c: bound_file(path.parent, record[c]) for c in ("stdout", "stderr")}
        _output_ok(name, files["stdout"].read_text(), files["stderr"].read_text())
        closure.extend(files.values())
    return closure


def validate_cpu_roles(
    path: Path, measurement: dict[str, str], supervision: dict[str, str]
) -> list[Path]:
    """Keep both original CPU executions separate; copying never changes their source."""
    data = json.loads(path.read_text())
    if (
        data.get("protocol") != PROTOCOL
        or data.get("schema_version") != 1
        or data.get("kind") != "role_cpu_gates"
        or data.get("measurement_source") != measurement
        or data.get("supervision_source") != supervision
        or set(data)
        != {
            "protocol",
            "schema_version",
            "kind",
            "measurement_source",
            "supervision_source",
            "measurement",
            "supervision",
        }
    ):
        raise ValueError("CPU role provenance mismatch")
    original = bound_file(path.parent, data["measurement"])
    current = bound_file(path.parent, data["supervision"])
    if original == current or measurement == supervision:
        raise ValueError("supervised CPU roles cannot collapse to one source")
    return [path, *validate_cpu(original, measurement), *validate_cpu(current, supervision)]
