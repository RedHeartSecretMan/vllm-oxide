"""Release-host preflight for the pinned ADR-0012 environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from golden_gen.config import (
    CUDA_TOOLKIT_VERSION,
    MODEL_CONFIG_SHA256,
    MODEL_WEIGHTS_SHA256,
    PYTORCH_CUDA_VERSION,
    PYTORCH_VERSION,
    TOKENIZER_SHA256,
    TRANSFORMERS_VERSION,
    TRITON_VERSION,
    VLLM_VERSION,
    XGRAMMAR_VERSION,
)
from golden_gen.schema import RuntimeInfo, WheelIdentity


@dataclass(frozen=True)
class InstallEvidence:
    frozen: bool
    registry_no_build: bool
    source_builds: tuple[str, ...]


def validate_deterministic_environment(environment: Mapping[str, str]) -> None:
    expected = {
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, value in expected.items():
        if environment.get(name) != value:
            raise ValueError(f"{name} must equal {value} before any runtime import")


def validate_model_artifacts(model_dir: Path, expected: Mapping[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        path = Path(model_dir) / filename
        if not path.is_file():
            raise ValueError(f"required model artifact is missing: {filename}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected.get(filename):
            raise ValueError(f"{filename} SHA-256 does not match ADR-0012")
        observed[filename] = digest
    return observed


def validate_wheel_only_install(evidence: InstallEvidence) -> None:
    if not evidence.frozen or not evidence.registry_no_build:
        raise ValueError("environment install must be frozen and registry wheel-only")
    unexpected = sorted(set(evidence.source_builds) - {"golden-gen"})
    if unexpected:
        raise ValueError(f"third-party source build is forbidden: {', '.join(unexpected)}")


def require_available_ram() -> None:
    """Abort an active GPU-owning stage below the 16 GiB host-RAM floor."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            if available_kib < 16 * 1024 * 1024:
                raise RuntimeError("available host RAM fell below the 16 GiB release floor")
            return
    raise RuntimeError("MemAvailable is unavailable; refusing release work")


def _command(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _locked_core_wheels(lock_path: Path) -> list[WheelIdentity]:
    with Path(lock_path).open("rb") as source:
        lock = tomllib.load(source)
    expected = {
        "torch": PYTORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "vllm": VLLM_VERSION,
        "xgrammar": XGRAMMAR_VERSION,
        "triton": TRITON_VERSION,
    }
    identities: list[WheelIdentity] = []
    for name, version in expected.items():
        package = next(
            (
                item
                for item in lock.get("package", [])
                if item.get("name") == name and item.get("version") == version
            ),
            None,
        )
        if package is None or "registry" not in package.get("source", {}):
            raise ValueError(f"locked registry package is missing: {name}=={version}")
        compatible = []
        for wheel in package.get("wheels", []):
            filename = unquote(Path(urlparse(wheel["url"]).path).name)
            is_platform_wheel = any(
                marker in filename for marker in ("cp312", "cp38-abi3", "py3-none-any")
            )
            if is_platform_wheel and ("x86_64" in filename or filename.endswith("-any.whl")):
                compatible.append((filename, wheel["hash"].removeprefix("sha256:")))
        if not compatible:
            raise ValueError(f"no locked x86_64 Python 3.12 wheel for {name}=={version}")
        # Build-tagged wheels sort after untagged wheels, matching the resolver's preference.
        filename, sha256 = sorted(compatible)[-1]
        identities.append(
            WheelIdentity(name=name, version=version, filename=filename, sha256=sha256)
        )
    return identities


def collect_release_runtime(model_dir: Path, repo_root: Path) -> RuntimeInfo:
    """Probe the already-installed release host without resolving or downloading anything."""
    validate_deterministic_environment(os.environ)
    validate_wheel_only_install(
        InstallEvidence(frozen=True, registry_no_build=True, source_builds=("golden-gen",))
    )
    expected_hashes = {
        "config.json": MODEL_CONFIG_SHA256,
        "tokenizer.json": TOKENIZER_SHA256,
        "model.safetensors": MODEL_WEIGHTS_SHA256,
    }
    validate_model_artifacts(model_dir, expected_hashes)
    expected_versions = {
        "torch": PYTORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "vllm": VLLM_VERSION,
        "xgrammar": XGRAMMAR_VERSION,
        "triton": TRITON_VERSION,
    }
    observed_versions = {name: importlib.metadata.version(name) for name in expected_versions}
    if observed_versions != expected_versions:
        raise ValueError(
            f"installed oracle versions do not match the lock: {json.dumps(observed_versions)}"
        )
    try:
        importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("unexpected third-party flash-attn distribution is installed")

    import torch

    if torch.version.cuda != PYTORCH_CUDA_VERSION:
        raise ValueError(
            f"PyTorch bundled CUDA must be {PYTORCH_CUDA_VERSION}, got {torch.version.cuda}"
        )
    vllm_files = [str(path) for path in (importlib.metadata.files("vllm") or [])]
    if not any(path.endswith("vllm_flash_attn/_vllm_fa2_C.abi3.so") for path in vllm_files):
        raise ValueError("vLLM wheel does not contain the prebuilt FlashAttention-2 extension")

    nvcc = _command("nvcc", "--version")
    if f"V{CUDA_TOOLKIT_VERSION}" not in nvcc:
        raise ValueError(f"CUDA toolkit must be {CUDA_TOOLKIT_VERSION}")
    gpu_line = _command(
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    )
    gpu_parts = [part.strip() for part in gpu_line.split(",")]
    if len(gpu_parts) != 3 or gpu_parts[1] != "8.9":
        raise ValueError(f"release GPU must be one sm_89 device, got {gpu_line}")
    repo_root = Path(repo_root)
    lock_path = repo_root / "tools/golden-gen/uv.lock"
    with lock_path.open("rb") as source:
        uv_lock_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    python_version = platform.python_version()
    if sys.version_info[:2] != (3, 12):
        raise ValueError(f"release Python must be 3.12, got {python_version}")
    return RuntimeInfo(
        evidence_mode="release",
        registry_install_mode="locked-wheels-only",
        pythonhashseed="0",
        cublas_workspace_config=":4096:8",
        python_version=python_version,
        torch_version=PYTORCH_VERSION,
        torch_cuda_version=PYTORCH_CUDA_VERSION,
        transformers_version=TRANSFORMERS_VERSION,
        vllm_version=VLLM_VERSION,
        xgrammar_version=XGRAMMAR_VERSION,
        triton_version=TRITON_VERSION,
        cuda_toolkit_version=CUDA_TOOLKIT_VERSION,
        rustc_version=_command("rustc", "--version"),
        nvidia_driver_version=gpu_parts[2],
        gpu_name=gpu_parts[0],
        compute_capability="8.9",
        os_kernel=f"{platform.system()} {platform.release()}",
        generator_commit=_command("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        uv_lock_sha256=uv_lock_sha256,
        wheels=_locked_core_wheels(lock_path),
    )
