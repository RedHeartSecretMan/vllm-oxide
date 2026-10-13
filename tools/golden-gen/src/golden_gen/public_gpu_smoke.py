"""Source-bound public CUDA smoke; never a numerical acceptance marker."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from golden_gen.environment import validate_release_model
from golden_gen.guard import run_guarded
from golden_gen.layered_artifacts import atomic_json, sha, source_identity

_LOG_WORKER = """import subprocess, sys
with open(sys.argv[1], 'xb') as out, open(sys.argv[2], 'xb') as err:
    raise SystemExit(subprocess.run(sys.argv[3:], stdout=out, stderr=err, check=False).returncode)
"""


def validate_api_output(stdout: str, source: dict[str, str]) -> dict[str, Any]:
    if not re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", stdout):
        raise ValueError("public GPU smoke did not execute its one required test")
    matches = re.findall(r"PUBLIC_GPU_SMOKE (\{[^\n]+\})", stdout)
    if len(matches) != 1:
        raise ValueError("public GPU smoke requires one complete behavior record")
    value = json.loads(matches[0])
    expected = {
        "source": source,
        "warmup": True,
        "repeated_calls": 100,
        "mixed_requests": 3,
        "unique_request_ids": 107,
        "passed": True,
    }
    if value != expected or value["passed"] is not True or value["warmup"] is not True:
        raise ValueError("public GPU smoke behavior or source identity is incomplete")
    if any(
        type(value[key]) is not int
        for key in ("repeated_calls", "mixed_requests", "unique_request_ids")
    ):
        raise ValueError("public GPU smoke counts must be integers")
    return dict(value)


def _run(
    repo: Path,
    output: Path,
    name: str,
    command: list[str],
    env: dict[str, str],
    deadline: int,
) -> dict[str, Any]:
    stdout, stderr, guard = (
        output / f"{name}.{suffix}" for suffix in ("stdout", "stderr", "guard.json")
    )
    run_guarded(
        [sys.executable, "-c", _LOG_WORKER, str(stdout), str(stderr), *command],
        guard,
        cwd=repo,
        env=env,
        timeout_seconds=deadline,
    )
    return {
        "command": command,
        "files": [{"path": path.name, "sha256": sha(path)} for path in (stdout, stderr, guard)],
    }


def collect(repo: Path, model: Path, output: Path) -> Path:
    repo, model, output = repo.resolve(strict=True), model.resolve(strict=True), output.absolute()
    if Path(__file__).resolve() != repo / "tools/golden-gen/src/golden_gen/public_gpu_smoke.py":
        raise ValueError("public smoke tool must execute from its declared checkout")
    source = source_identity(repo)
    validate_release_model(model)
    if any(name.startswith("VLLM_OXIDE_INTERNAL_") for name in os.environ):
        raise ValueError("public GPU smoke refuses private capture/replay configuration")
    output.mkdir(mode=0o700)  # A fresh directory is mandatory; never overwrite evidence.
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(repo / "tools/golden-gen/src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CARGO_NET_OFFLINE": "true",
        "CARGO_BUILD_JOBS": "1",
        "RUSTFLAGS": "",
        "CARGO_ENCODED_RUSTFLAGS": "",
        "QWEN3_MODEL_DIR": str(model),
        "VLLM_SMOKE_COMMIT": source["commit"],
        "VLLM_SMOKE_TREE": source["tree"],
    }
    versions = {
        name: subprocess.check_output(command, cwd=repo, env=env, text=True, timeout=30).strip()
        for name, command in {
            "rustc": ["rustc", "--version", "--verbose"],
            "cuda": ["nvcc", "--version"],
            "gpu": ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        }.items()
    }
    build = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "-p",
        "vllm_oxide_cli",
        "-p",
        "vllm_oxide_test",
        "--features",
        "cuda",
    ]
    test = [
        "cargo",
        "test",
        "--locked",
        "--release",
        "-p",
        "vllm_oxide_test",
        "--features",
        "cuda",
        "--test",
        "public_gpu",
    ]
    records = [
        _run(repo, output, "build", build, env, 1800),
        _run(repo, output, "build-public-test", [*test, "--no-run"], env, 1800),
        _run(
            repo,
            output,
            "public-api",
            [
                *test,
                "--",
                "--exact",
                "cuda_public_generation_contract",
                "--nocapture",
                "--test-threads=1",
            ],
            env,
            300,
        ),
    ]
    public = validate_api_output((output / "public-api.stdout").read_text(), source)
    cli = [
        "cargo",
        "run",
        "--locked",
        "--release",
        "-p",
        "vllm_oxide_cli",
        "--features",
        "cuda",
        "--",
        "--model",
        str(model),
        "--max-tokens",
        "2",
        "The capital of France is",
    ]
    records.append(_run(repo, output, "cli", cli, env, 300))
    text = (output / "cli.stdout").read_text()
    if not text.strip():
        raise ValueError("the exact Quick Start command returned empty CLI output")
    if source_identity(repo) != source:
        raise ValueError("source changed during public GPU smoke")
    marker = output / "public-gpu-smoke.json"
    atomic_json(
        marker,
        {
            "schema_version": 1,
            "kind": "public_gpu_smoke",
            "source": source,
            "passed": True,
            "numerical_acceptance": False,
            "environment": versions,
            "stages": records,
            "public_api": public,
            "cli_output": text,
        },
    )
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(collect(args.repo_root, args.model_dir, args.output))


if __name__ == "__main__":
    main()
