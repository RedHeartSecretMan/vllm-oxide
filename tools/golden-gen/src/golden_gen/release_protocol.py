"""Fail-closed, CPU-testable contracts for the goldens-v0.2 release stages."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from golden_gen.config import (
    BASELINE_KERNEL_PATH,
    CANDIDATE_KERNEL_PATH,
    CUDA_TOOLKIT_VERSION,
    PYTORCH_CUDA_VERSION,
    PYTORCH_VERSION,
    REFERENCE_KERNEL_PATH,
    TRANSFORMERS_VERSION,
    TRITON_VERSION,
    VLLM_VERSION,
    XGRAMMAR_VERSION,
)
from golden_gen.schema import DiscoveredFixture, KernelPaths, Manifest, RuntimeInfo, WheelIdentity

_APPROVED_CASES = {
    *(f"canonical_{index:02d}" for index in range(1, 5)),
    *(f"canonical_05{suffix}" for suffix in "abcd"),
    *(f"regression_{index:02d}" for index in range(1, 21)),
}


def _approved_case_families() -> dict[str, str]:
    return {
        **{f"canonical_{index:02d}": "canonical" for index in range(1, 5)},
        **{f"canonical_05{suffix}": "batch" for suffix in "abcd"},
        **{f"regression_{index:02d}": "regression" for index in range(1, 21)},
    }


_ARTIFACT_PARENT = Path("/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts")
_STAGES = (
    "env",
    "generate",
    "observe",
    "authoritative",
    "benchmark",
    "report",
    "bundle",
    "publish",
    "verify",
)


class StageLedger:
    """Resolve fresh, ticket-owned stage directories without hidden resume state."""

    def __init__(self, run_root: Path) -> None:
        resolved = Path(run_root).resolve(strict=False)
        try:
            relative = resolved.relative_to(_ARTIFACT_PARENT)
        except ValueError as error:
            raise ValueError(
                f"run path must be below ticket artifact root {_ARTIFACT_PARENT}"
            ) from error
        if not relative.parts:
            raise ValueError("run path must name a fresh run below the ticket artifact root")
        self.run_root = resolved

    def output_path(self, stage: str) -> Path:
        if stage not in _STAGES:
            raise ValueError(f"unknown goldens-v0.2 stage: {stage}")
        return self.run_root / stage

    def require_fresh_output(self, stage: str) -> Path:
        output = self.output_path(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"stage requires a fresh non-existing output path: {output}")
        return output


@dataclass(frozen=True)
class CorpusContract:
    """Exact concrete-case and oracle-asset accounting from ADR-0012."""

    case_count: int
    asset_count: int
    assets_by_family: dict[str, int]
    reference_count: int
    reference_l1_count: int
    reference_l2_count: int
    baseline_calibration_count: int

    @classmethod
    def from_discovered(cls, discovered: list[DiscoveredFixture]) -> CorpusContract:
        case_ids = [case.prompt_id for case in discovered]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("release corpus contains duplicate concrete case identifiers")
        if set(case_ids) != _APPROVED_CASES:
            missing = sorted(_APPROVED_CASES - set(case_ids))
            unexpected = sorted(set(case_ids) - _APPROVED_CASES)
            raise ValueError(
                f"release corpus identity mismatch: missing={missing}, unexpected={unexpected}"
            )

        cases_by_family = {
            family: sum(case.family == family for case in discovered)
            for family in ("canonical", "batch", "regression")
        }
        l2_cases = cases_by_family["canonical"] + cases_by_family["batch"]
        case_count = len(discovered)
        return cls(
            case_count=case_count,
            asset_count=case_count * 2,
            assets_by_family={family: count * 2 for family, count in cases_by_family.items()},
            reference_count=case_count,
            reference_l1_count=case_count,
            reference_l2_count=l2_cases,
            baseline_calibration_count=case_count,
        )


def pinned_kernel_paths() -> KernelPaths:
    """Return the only three kernel identities accepted by ADR-0012."""
    return KernelPaths(
        reference=REFERENCE_KERNEL_PATH,
        baseline=BASELINE_KERNEL_PATH,
        candidate=CANDIDATE_KERNEL_PATH,
    )


def dry_run_runtime_info() -> RuntimeInfo:
    """Build schema-valid metadata that remains explicitly non-publishable."""
    versions = {
        "torch": PYTORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "vllm": VLLM_VERSION,
        "xgrammar": XGRAMMAR_VERSION,
        "triton": TRITON_VERSION,
    }
    wheels = [
        WheelIdentity(
            name=name,
            version=version,
            filename=f"{name}-{version}-dry-run.whl",
            sha256="0" * 64,
        )
        for name, version in versions.items()
    ]
    return RuntimeInfo(
        evidence_mode="dry-run",
        registry_install_mode="locked-wheels-only",
        pythonhashseed="0",
        cublas_workspace_config=":4096:8",
        python_version=platform.python_version(),
        torch_version=PYTORCH_VERSION,
        torch_cuda_version=PYTORCH_CUDA_VERSION,
        transformers_version=TRANSFORMERS_VERSION,
        vllm_version=VLLM_VERSION,
        xgrammar_version=XGRAMMAR_VERSION,
        triton_version=TRITON_VERSION,
        cuda_toolkit_version=CUDA_TOOLKIT_VERSION,
        rustc_version="dry-run",
        nvidia_driver_version="dry-run",
        gpu_name="dry-run",
        compute_capability="8.9",
        os_kernel="dry-run",
        generator_commit="0" * 40,
        uv_lock_sha256="0" * 64,
        wheels=wheels,
    )


def validate_release_manifest_coverage(manifest: Manifest) -> CorpusContract:
    """Require the exact approved corpus and complete baseline calibration."""
    if manifest.runtime.evidence_mode != "release":
        raise ValueError("release bundle rejects dry-run runtime evidence")
    families = _approved_case_families()
    expected_contract = {
        (prompt_id, oracle): family
        for prompt_id, family in families.items()
        for oracle in ("transformers", "vllm")
    }
    declared = {(item.prompt_id, item.oracle): item.family for item in manifest.expected_fixtures}
    if declared != expected_contract:
        raise ValueError("release manifest does not declare the exact 28-case/56-asset corpus")
    generated = {(item.prompt_id, item.oracle) for item in manifest.fixtures}
    if generated != set(expected_contract):
        raise ValueError("release bundle requires every and only the 56 expected fixtures")
    calibrated = {f"{prompt_id}.vllm" for prompt_id in families}
    if set(manifest.calibrated_fixtures) != calibrated:
        raise ValueError(
            "release bundle requires complete baseline calibration for all 28 fixtures"
        )
    discovered = [
        DiscoveredFixture(prompt_id=prompt_id, family=family, prompt="contract-only")
        for prompt_id, family in families.items()
    ]
    return CorpusContract.from_discovered(discovered)
