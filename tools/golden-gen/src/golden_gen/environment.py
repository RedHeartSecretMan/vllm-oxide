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


def validate_release_model(model_dir: Path) -> Path:
    """Validate and return the actual local source used by an oracle GPU owner."""
    model_dir = model_dir.resolve(strict=True)
    validate_model_artifacts(
        model_dir,
        {
            "config.json": MODEL_CONFIG_SHA256,
            "tokenizer.json": TOKENIZER_SHA256,
            "model.safetensors": MODEL_WEIGHTS_SHA256,
        },
    )
    if any(path.name != "model.safetensors" for path in model_dir.glob("*.safetensors")) or any(
        model_dir.glob("*.safetensors.index.json")
    ):
        raise ValueError("unexpected alternative release model weights")
    return model_dir


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
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"{args[0]} preflight failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _expanded_wheel_tags(tag: str) -> set[str]:
    python, abi, platform_tag = tag.split("-")
    return {
        f"{python_tag}-{abi_tag}-{platform}"
        for python_tag in python.split(".")
        for abi_tag in abi.split(".")
        for platform in platform_tag.split(".")
    }


def _cached_wheel_matches(
    distribution: importlib.metadata.Distribution,
    name: str,
    filename: str,
    digest: str,
    cache_root: Path,
) -> bool:
    """Prove a vendor-retagged wheel through uv's actual hash-bound cache entry.

    uv wheels-v6 retains a MessagePack archive identity before its HTTP policy.
    Accept only the observed four-field format and a WHEEL hardlink to that exact
    archive. Unsupported cache formats and copied/unidentifiable installs fail.
    This does not guess equivalence between linux and manylinux tags.
    """

    def string(value: str) -> bytes:
        encoded = value.encode()
        if len(encoded) < 32:
            return bytes([0xA0 + len(encoded)]) + encoded
        if len(encoded) < 256:
            return bytes([0xD9, len(encoded)]) + encoded
        raise ValueError("unsupported uv cache identity string length")

    installed = [
        Path(str(distribution.locate_file(path)))
        for path in (distribution.files or [])
        if str(path).endswith(".dist-info/WHEEL")
    ]
    if len(installed) != 1:
        return False
    suffix = filename.removeprefix(name.replace("-", "_") + "-").removesuffix(".whl")
    entries = [
        *cache_root.glob(f"wheels-v6/index/*/{name}/{suffix}"),
        *cache_root.glob(f"wheels-v6/pypi/{name}/{suffix}"),
    ]
    for entry in entries:
        if not entry.is_symlink():
            continue
        archive = entry.resolve(strict=True)
        origin = entry.with_name(entry.name + ".http")
        expected = (
            b"\x94"
            + string(archive.name)
            + b"\x91\x92"
            + string("Sha256")
            + string(digest)
            + string(filename)
            + b"\x00"
        )
        cached_wheel = archive / installed[0].parent.name / "WHEEL"
        if (
            origin.is_file()
            and origin.read_bytes().startswith(expected)
            and cached_wheel.is_file()
            and installed[0].samefile(cached_wheel)
        ):
            return True
    return False


def collect_installed_wheels(
    lock_path: Path,
    *,
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
    cache_root: Path | None = None,
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
        if name == "golden-gen":
            # An editable project exposes both its installed dist-info and source
            # egg-info. uv --check validates this sole permitted local project.
            continue
        if name in seen:
            raise ValueError(f"duplicate installed distribution: {name}")
        seen.add(name)
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
        tags = {
            expanded
            for tag in metadata.get_all("Tag", [])
            for expanded in _expanded_wheel_tags(tag)
        }
        build = metadata.get("Build", "")
        compatible = []
        for wheel in packages[0].get("wheels", []):
            filename = unquote(Path(urlparse(wheel["url"]).path).name)
            parts = filename.removesuffix(".whl").split("-")
            if len(parts) not in (5, 6):
                continue
            wheel_tags = _expanded_wheel_tags("-".join(parts[-3:]))
            if tags == wheel_tags and build == (parts[2] if len(parts) == 6 else ""):
                compatible.append((filename, wheel["hash"].removeprefix("sha256:")))
        if len(compatible) != 1:
            cache_root = cache_root or Path(_command("uv", "cache", "dir"))
            compatible = []
            for wheel in packages[0].get("wheels", []):
                filename = unquote(Path(urlparse(wheel["url"]).path).name)
                sha256 = wheel["hash"].removeprefix("sha256:")
                if _cached_wheel_matches(distribution, name, filename, sha256, cache_root):
                    compatible.append((filename, sha256))
            if len(compatible) != 1:
                raise ValueError(
                    f"live wheel does not identify one hash-bound locked archive: {name}"
                )
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
        # --check is read-only. A global --no-build here rejects the permitted
        # editable local project even when the complete environment is current.
        # Registry installation itself remains --no-build in validate-release.sh.
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
