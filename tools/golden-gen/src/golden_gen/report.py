"""Generate the evidence-only goldens-v0.2 release report."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from golden_gen.schema import KernelPaths, Manifest, RuntimeInfo


class LifecycleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected: int
    discovered: int
    generated: int
    calibration_compared: int
    reference_compared: int
    missing: int
    unexpected: int
    skipped: int
    failed: int
    duplicate: int
    stale: int
    unmatched: int

    @model_validator(mode="after")
    def exact_release_totals(self) -> LifecycleEvidence:
        required = {
            "expected": 56,
            "discovered": 56,
            "generated": 56,
            "calibration_compared": 56,
            "reference_compared": 28,
            "missing": 0,
            "unexpected": 0,
            "skipped": 0,
            "failed": 0,
            "duplicate": 0,
            "stale": 0,
            "unmatched": 0,
        }
        if self.model_dump() != required:
            raise ValueError("release lifecycle totals are not the exact ADR-0012 values")
        return self


class InterTokenLatency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    samples: list[int] = Field(min_length=1)
    mean: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)


class MemoryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_mib: int = Field(ge=0)
    peak_mib: int = Field(ge=0)
    delta_mib: int = Field(ge=0)
    polling_interval_ms: int = Field(ge=1, le=50)
    sample_count: int = Field(ge=2)
    samples: list[dict[str, int]] = Field(default_factory=list)
    other_compute_processes: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def peak_and_delta_are_consistent(self) -> MemoryEvidence:
        if self.peak_mib < self.baseline_mib or self.peak_mib - self.baseline_mib != self.delta_mib:
            raise ValueError("benchmark memory baseline, peak, and delta are inconsistent")
        return self


class BenchmarkRepetition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prefill_tokens_per_second: float = Field(gt=0.0)
    decode_tokens_per_second: float = Field(gt=0.0)
    time_to_first_token_ns: list[int] = Field(min_length=1)
    inter_token_latency_ns: InterTokenLatency
    memory: MemoryEvidence
    telemetry_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class WorkloadEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repetitions: list[BenchmarkRepetition] = Field(min_length=3, max_length=3)
    headline_prefill_tokens_per_second: float | None = None
    headline_decode_tokens_per_second: float | None = None
    headline_time_to_first_token_ns: float | None = None
    headline_peak_memory_mib: float | None = None


class BenchmarkEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_04: WorkloadEvidence
    canonical_05: WorkloadEvidence


class ToleranceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    l1_near_tie_max_abs_logit_gap: float = Field(ge=0.0, le=0.0625)
    l2_atol: float = Field(ge=0.0, le=0.25)
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rationale: str = Field(min_length=1)


class ReleaseReportInput(BaseModel):
    """Inputs available before the evidence-only final candidate exists."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    observation_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    policy_checkpoint_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    policy_checkpoint_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    measurement_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    measurement_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime: RuntimeInfo
    kernel_paths: KernelPaths
    lifecycle: LifecycleEvidence
    tolerance: ToleranceEvidence
    benchmark: BenchmarkEvidence
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def commit_roles_are_distinct_and_release_bound(self) -> ReleaseReportInput:
        commits = {
            self.observation_commit,
            self.policy_checkpoint_commit,
            self.measurement_commit,
        }
        if len(commits) != 3:
            raise ValueError("observation, policy checkpoint, and measurement commits must differ")
        if self.runtime.evidence_mode != "release":
            raise ValueError("release report rejects dry-run runtime evidence")
        return self


def _headline(workload: WorkloadEvidence, field: str) -> float:
    return float(median(getattr(repetition, field) for repetition in workload.repetitions))


def validate_report_sources(
    evidence: ReleaseReportInput,
    repo: Path,
    manifest: Manifest,
    observation: dict[str, Any],
    benchmark: dict[str, Any],
) -> None:
    """Bind caller-supplied report labels to machine evidence and committed Git objects."""
    if (
        benchmark.get("measurement_commit") != evidence.measurement_commit
        or benchmark.get("measurement_tree") != evidence.measurement_tree
    ):
        raise ValueError("report measurement identity differs from benchmark evidence")
    identity = observation["identity"]
    if (
        identity["measurement_commit"] != evidence.observation_commit
        or identity["measurement_tree"] != evidence.observation_tree
    ):
        raise ValueError("report observation identity differs from calibration evidence")
    required = {
        f"measurement-commit:{evidence.measurement_commit}",
        f"measurement-tree:{evidence.measurement_tree}",
        f"observation-measurement-commit:{evidence.observation_commit}",
        f"observation-measurement-tree:{evidence.observation_tree}",
        f"definition-observation-sha256:{evidence.tolerance.observation_sha256}",
    }
    if not required.issubset(manifest.tolerance_policy.evidence):
        raise ValueError("report identities differ from the approved manifest policy")
    if evidence.tolerance.rationale != manifest.tolerance_policy.rationale:
        raise ValueError("report rationale differs from the approved policy")

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True
        ).stdout

    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("report source worktree is dirty")
    if git("rev-parse", "HEAD").decode().strip() != evidence.measurement_commit:
        raise ValueError("report must be created from the frozen measurement commit")
    for role in ("observation", "policy_checkpoint", "measurement"):
        commit = getattr(evidence, f"{role}_commit")
        tree = getattr(evidence, f"{role}_tree")
        if git("rev-parse", f"{commit}^{{tree}}").decode().strip() != tree:
            raise ValueError(f"report {role} commit/tree identity is invalid")
    git(
        "merge-base",
        "--is-ancestor",
        evidence.observation_commit,
        evidence.policy_checkpoint_commit,
    )
    git(
        "merge-base",
        "--is-ancestor",
        evidence.policy_checkpoint_commit,
        evidence.measurement_commit,
    )
    path = "docs/releases/goldens-v0.2-calibration-observation.json"
    checkpoint_bytes = git("show", f"{evidence.policy_checkpoint_commit}:{path}")
    if hashlib.sha256(checkpoint_bytes).hexdigest() != evidence.tolerance.observation_sha256:
        raise ValueError("report policy checkpoint contains a different observation")
    checkpoint_blob = (
        git("rev-parse", f"{evidence.policy_checkpoint_commit}:{path}").decode().strip()
    )
    index = json.loads(
        git("show", f"{evidence.policy_checkpoint_commit}:.dag/definition-index.json")
    )
    selected = [item for item in index["inputs"] if item.get("path") == path]
    if selected != [{"path": path, "blob_oid": checkpoint_blob}]:
        raise ValueError("report observation was not bound by the policy checkpoint")


def render_release_report(evidence: ReleaseReportInput) -> str:
    """Render durable Markdown without embedding the future candidate identity."""
    lines = [
        "# goldens-v0.2 release evidence",
        "",
        (
            f"- Observation commit: `{evidence.observation_commit}` "
            f"(tree `{evidence.observation_tree}`)"
        ),
        (
            f"- Policy checkpoint: `{evidence.policy_checkpoint_commit}` "
            f"(tree `{evidence.policy_checkpoint_tree}`)"
        ),
        (
            f"- Measurement commit: `{evidence.measurement_commit}` "
            f"(tree `{evidence.measurement_tree}`)"
        ),
        (
            "- The evidence-only candidate and tag identity are recorded externally "
            "by the DAG and GitHub Release metadata."
        ),
        "",
        "## Environment and kernels",
        "",
        (
            f"- GPU: {evidence.runtime.gpu_name}, "
            f"compute capability {evidence.runtime.compute_capability}"
        ),
        (
            f"- Driver/toolkit: {evidence.runtime.nvidia_driver_version} / "
            f"{evidence.runtime.cuda_toolkit_version}"
        ),
        f"- Reference: `{evidence.kernel_paths.reference}`",
        f"- Baseline: `{evidence.kernel_paths.baseline}`",
        f"- Candidate: `{evidence.kernel_paths.candidate}`",
        "",
        "```json",
        evidence.runtime.model_dump_json(indent=2),
        "```",
        "",
        "## Fixture lifecycle",
        "",
        (
            "Expected/discovered/generated/calibration-compared: 56/56/56/56; "
            "reference-compared: 28; missing/unexpected/skipped/failed/duplicate/"
            "stale/unmatched: all zero."
        ),
        "",
        "## Tolerance policy",
        "",
        f"- L1 near-tie candidate gap: {evidence.tolerance.l1_near_tie_max_abs_logit_gap}",
        f"- L2 absolute tolerance: {evidence.tolerance.l2_atol}",
        f"- Observation SHA-256: `{evidence.tolerance.observation_sha256}`",
        f"- Raw evidence SHA-256: `{evidence.tolerance.raw_evidence_sha256}`",
        f"- Approved error classes and rationale: {evidence.tolerance.rationale}",
        "",
        "## Performance",
    ]
    for name in ("canonical_04", "canonical_05"):
        workload = getattr(evidence.benchmark, name)
        ttft = [
            value
            for repetition in workload.repetitions
            for value in repetition.time_to_first_token_ns
        ]
        itl = [
            value
            for repetition in workload.repetitions
            for value in repetition.inter_token_latency_ns.samples
        ]
        peak = [repetition.memory.peak_mib for repetition in workload.repetitions]
        lines.extend(
            [
                "",
                f"### {name}",
                "",
                (
                    "- Prefill throughput median: "
                    f"{_headline(workload, 'prefill_tokens_per_second'):.3f} tok/s"
                ),
                (
                    "- Decode throughput median: "
                    f"{_headline(workload, 'decode_tokens_per_second'):.3f} tok/s"
                ),
                f"- Time to first token median: {median(ttft):.3f} ns",
                (
                    f"- Per-token latency median: {median(itl):.3f} ns "
                    "(all raw samples retained below)"
                ),
                f"- Peak memory median: {median(peak):.3f} MiB",
                "",
                "Complete repetitions (TTFT and ITL in ns, memory in MiB):",
                "",
                "```json",
                workload.model_dump_json(indent=2, exclude_none=True),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Assets",
            "",
            f"- `manifest.json`: `{evidence.manifest_sha256}`",
            f"- `goldens-v0.2.tar.gz`: `{evidence.archive_sha256}`",
            "",
            "## Known limitations",
            "",
            *(f"- {limitation}" for limitation in evidence.limitations),
            "",
        ]
    )
    return "\n".join(lines)
