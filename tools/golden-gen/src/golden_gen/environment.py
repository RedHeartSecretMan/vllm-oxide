"""Release-host preflight for the pinned ADR-0012 environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from email.parser import Parser
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


def collect_installed_wheels(
    lock_path: Path,
    *,
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
) -> list[WheelIdentity]:
    """Match every live registry distribution's wheel tags/build to one locked archive.

    The archive digest comes from the frozen installer lock, not from installed
    metadata. The calling stage also retains the successful wheel-only installer
    transcript and checks that uv's complete resolved environment is synchronized.
    """
    with Path(lock_path).open("rb") as source:
        lock = tomllib.load(source)
    identities: list[WheelIdentity] = []
    seen: set[str] = set()
    live = importlib.metadata.distributions() if distributions is None else distributions
    for distribution in live:
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in seen:
            raise ValueError(f"duplicate installed distribution: {name}")
        seen.add(name)
        if name == "golden-gen":
            continue  # The sole explicitly permitted local project build.
        version = distribution.version
        if distribution.read_text("direct_url.json") is not None:
            raise ValueError(f"unexpected direct/source distribution: {name}")
        packages = [
            item
            for item in lock.get("package", [])
            if item.get("name") == name
            and item.get("version") == version
            and "registry" in item.get("source", {})
        ]
        if len(packages) != 1:
            raise ValueError(f"installed package is not a unique locked registry entry: {name}")
        wheel_metadata = distribution.read_text("WHEEL")
        installer = distribution.read_text("INSTALLER")
        if not wheel_metadata or not installer or installer.strip() != "uv":
            raise ValueError(f"installed wheel metadata or uv installer identity missing: {name}")
        metadata = Parser().parsestr(wheel_metadata)
        tags = set(metadata.get_all("Tag", []))
        build = metadata.get("Build", "")
        compatible = []
        for wheel in packages[0].get("wheels", []):
            filename = unquote(Path(urlparse(wheel["url"]).path).name)
            parts = filename.removesuffix(".whl").split("-")
            if len(parts) not in (5, 6):
                continue
            wheel_tags = {
                f"{python}-{abi}-{platform_tag}"
                for python in parts[-3].split(".")
                for abi in parts[-2].split(".")
                for platform_tag in parts[-1].split(".")
            }
            if tags == wheel_tags and build == (parts[2] if len(parts) == 6 else ""):
                compatible.append((filename, wheel["hash"].removeprefix("sha256:")))
        if len(compatible) != 1:
            raise ValueError(f"live wheel tags/build do not identify one locked archive: {name}")
        filename, sha256 = compatible[0]
        identities.append(
            WheelIdentity(name=name, version=version, filename=filename, sha256=sha256)
        )
    return sorted(identities, key=lambda wheel: wheel.name.encode())


def collect_release_runtime(model_dir: Path, repo_root: Path) -> RuntimeInfo:
    """Probe the already-installed release host without resolving or downloading anything."""
    validate_deterministic_environment(os.environ)
    if sys.prefix == sys.base_prefix:
        raise ValueError("release environment must be an isolated virtual environment")
    _command(
        "uv",
        "sync",
        "--project",
        str(Path(repo_root) / "tools/golden-gen"),
        "--extra",
        "gpu",
        "--frozen",
        "--check",
        "--offline",
        "--no-build",
        "--python",
        sys.executable,
        "--no-python-downloads",
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
        wheels=collect_installed_wheels(lock_path),
    )
