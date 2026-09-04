"""Fresh-process oracle generation, replay verification, and fixture assembly."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from golden_gen.assets import build_fixture_archive
from golden_gen.config import COMPARISON_KERNEL_SCOPE, MODEL_DTYPE
from golden_gen.environment import require_available_ram
from golden_gen.generate import run_all
from golden_gen.manifest import build_expected_fixtures, build_manifest, write_manifest
from golden_gen.oracles.base import Oracle
from golden_gen.prompts import discover_fixtures, load_prompts
from golden_gen.release_protocol import pinned_kernel_paths
from golden_gen.replay import ReplayEvidence, verify_oracle_replay
from golden_gen.schema import (
    BaselineCalibration,
    FixtureMetadata,
    RuntimeInfo,
    TolerancePolicy,
)


class OracleRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    oracle: Literal["transformers", "vllm"]
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixtures: list[FixtureMetadata]

    @model_validator(mode="after")
    def exact_oracle_corpus(self) -> OracleRun:
        expected_ids = {
            *(f"canonical_{index:02d}" for index in range(1, 5)),
            *(f"canonical_05{suffix}" for suffix in "abcd"),
            *(f"regression_{index:02d}" for index in range(1, 21)),
        }
        observed = {fixture.prompt_id for fixture in self.fixtures}
        if len(self.fixtures) != 28 or observed != expected_ids:
            raise ValueError("oracle run must contain the exact 28 concrete cases")
        if any(fixture.oracle != self.oracle for fixture in self.fixtures):
            raise ValueError("oracle run contains a fixture from a different oracle")
        return self


class ReplayRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    oracle: Literal["transformers", "vllm"]
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_count: Literal[28]
    verified_filenames: list[str] = Field(min_length=28, max_length=28)


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _require_fresh_ticket_path(path: Path) -> Path:
    resolved = Path(path).resolve(strict=False)
    parent = Path("/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts")
    try:
        relative = resolved.relative_to(parent)
    except ValueError as error:
        raise ValueError(f"release output must stay below {parent}") from error
    if not relative.parts or resolved.exists() or resolved.is_symlink():
        raise ValueError("release output must be a fresh non-existing ticket artifact path")
    return resolved


def generate_oracle_run(
    oracle: Literal["transformers", "vllm"],
    runtime_record: Path,
    prompts_dir: Path,
    output_dir: Path,
) -> OracleRun:
    runtime_bytes = Path(runtime_record).read_bytes()
    runtime = RuntimeInfo.model_validate_json(runtime_bytes)
    if runtime.evidence_mode != "release":
        raise ValueError("oracle generation requires release runtime evidence")
    output_dir = _require_fresh_ticket_path(output_dir)
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    fixtures_dir = output_dir / "fixtures"
    fixtures_dir.mkdir(mode=0o700)
    prompts = load_prompts(prompts_dir)
    discovered = discover_fixtures(prompts)
    if len(discovered) != 28:
        raise ValueError("oracle generation did not discover the exact 28-case corpus")
    adapter: Oracle
    if oracle == "transformers":
        from golden_gen.oracles.transformers_oracle import TransformersOracle

        adapter = TransformersOracle()
    else:
        from golden_gen.oracles.vllm_oracle import VllmOracle

        adapter = VllmOracle()
    try:
        fixtures = run_all(
            [adapter],
            prompts,
            fixtures_dir,
            resource_guard=require_available_ram,
        )
    finally:
        adapter.close()
    run = OracleRun(
        schema_version=1,
        oracle=oracle,
        runtime_sha256=hashlib.sha256(runtime_bytes).hexdigest(),
        fixtures=fixtures,
    )
    with (output_dir / "oracle-run.json").open("xb") as destination:
        destination.write(run.model_dump_json(indent=2).encode())
        destination.write(b"\n")
    return run


def verify_fresh_replay(
    oracle: Literal["transformers", "vllm"],
    primary_dir: Path,
    replay_dir: Path,
    output: Path,
) -> ReplayRecord:
    primary = OracleRun.model_validate_json((Path(primary_dir) / "oracle-run.json").read_bytes())
    replay = OracleRun.model_validate_json((Path(replay_dir) / "oracle-run.json").read_bytes())
    if primary.oracle != oracle or replay.oracle != oracle:
        raise ValueError("oracle replay identity does not match the requested engine")
    if primary.runtime_sha256 != replay.runtime_sha256:
        raise ValueError("oracle replay runtime identity changed")
    expected = tuple(sorted((fixture.filename for fixture in primary.fixtures), key=str.encode))
    evidence: ReplayEvidence = verify_oracle_replay(
        Path(primary_dir) / "fixtures",
        Path(replay_dir) / "fixtures",
        expected,
    )
    record = ReplayRecord(
        schema_version=1,
        oracle=oracle,
        runtime_sha256=primary.runtime_sha256,
        fixture_count=evidence.fixture_count,
        verified_filenames=list(evidence.verified_filenames),
    )
    with Path(output).open("xb") as destination:
        destination.write(record.model_dump_json(indent=2).encode())
        destination.write(b"\n")
    return record


def assemble_release_fixtures(
    runtime_record: Path,
    reference_dir: Path,
    baseline_dir: Path,
    prompts_dir: Path,
    output_dir: Path,
) -> Path:
    runtime_bytes = Path(runtime_record).read_bytes()
    runtime = RuntimeInfo.model_validate_json(runtime_bytes)
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    reference = OracleRun.model_validate_json(
        (Path(reference_dir) / "oracle-run.json").read_bytes()
    )
    baseline = OracleRun.model_validate_json((Path(baseline_dir) / "oracle-run.json").read_bytes())
    if (
        reference.oracle != "transformers"
        or baseline.oracle != "vllm"
        or reference.runtime_sha256 != runtime_sha256
        or baseline.runtime_sha256 != runtime_sha256
    ):
        raise ValueError("oracle runs do not share the selected release runtime")
    output_dir = _require_fresh_ticket_path(output_dir)
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    fixtures = [*reference.fixtures, *baseline.fixtures]
    for run_dir, run in ((reference_dir, reference), (baseline_dir, baseline)):
        for fixture in run.fixtures:
            source = Path(run_dir) / "fixtures" / fixture.filename
            destination = output_dir / fixture.filename
            try:
                destination.hardlink_to(source)
            except OSError:
                shutil.copyfile(source, destination)
            if _sha256(destination) != fixture.sha256:
                raise ValueError(f"assembled fixture checksum mismatch: {fixture.filename}")
    archive = build_fixture_archive(output_dir, fixtures, output_dir / "goldens-v0.2.tar.gz")
    expected = build_expected_fixtures(discover_fixtures(load_prompts(prompts_dir)))
    manifest = build_manifest(
        fixtures,
        BaselineCalibration(
            candidate_atol=0.0,
            observed_max_abs_diff=0.0,
            calibration_factor=2.0,
            method="pending baseline calibration",
        ),
        archive=archive,
        tolerance_policy=TolerancePolicy(
            version="same-prefix-v1",
            dtype=MODEL_DTYPE,
            kernel=COMPARISON_KERNEL_SCOPE,
            l1_near_tie_max_abs_logit_gap=0.0,
            l2_atol=0.0,
            rationale="pending empirical Definition Revision",
            evidence=[],
        ),
        expected_fixtures=expected,
        runtime=runtime,
        kernel_paths=pinned_kernel_paths(),
    )
    write_manifest(manifest, output_dir / "manifest.json")
    return output_dir
