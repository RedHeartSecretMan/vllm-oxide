"""Explicit source-role preparation, separate from the accepted X/Y protocol.

The proposed adapter requires independently approved external freezing and a
selected role Definition. No caller summary or archive code supplies authority.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from golden_gen.layered_artifacts import sha
from golden_gen.release_transport import Entrypoints, Source

POLICY_PATH = "docs/validation/release-source-role-policy.json"
MONITOR_PATHS = {
    "crates/vllm-oxide-test/src/benchmark/runner.rs",
    "crates/vllm-oxide-test/src/benchmark/nvml_ffi.rs",
    "crates/vllm-oxide-test/src/benchmark/nvml_monitor.rs",
}
HASH = r"^[0-9a-f]{64}$"
OID = r"^[0-9a-f]{40}$"
ADAPTER_PATHS = {
    ".github/workflows/ci.yml",
    "tools/golden-gen/src/golden_gen/release_roles.py",
    "tools/golden-gen/src/golden_gen/release_memory_method.py",
    "tools/golden-gen/src/golden_gen/release_performance.py",
    "tools/golden-gen/src/golden_gen/release_cpu.py",
    "tools/golden-gen/src/golden_gen/release_adapter.py",
    "tools/golden-gen/src/golden_gen/release_cli.py",
    "tools/golden-gen/src/golden_gen/supervision.py",
    ".dag/definition-index.json",
    POLICY_PATH,
    "docs/validation/release-source-role-policy.proposed.json",
    "docs/adr/0029-proposed-release-source-roles.md",
    "tools/golden-gen/tests/test_release_roles.py",
    "tools/golden-gen/tests/test_release_role_cpu.py",
    "tools/golden-gen/tests/test_release_memory_method.py",
    "tools/golden-gen/tests/test_role_performance.py",
    "tools/golden-gen/tests/test_release_role_adapter.py",
}


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("schema_version", check_fields=False, mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("record schema_version must be an integer, not a coercion")
        return value


class GitObject(StrictRecord):
    mode: Literal["100644", "100755", "040000"]
    kind: Literal["blob", "tree"]
    oid: str = Field(pattern=OID)


class ObjectChange(StrictRecord):
    path: str
    before: GitObject | None
    after: GitObject

    @model_validator(mode="after")
    def safe_path(self) -> ObjectChange:
        from golden_gen.release_transport import logical_path

        logical_path(self.path)
        if self.before is not None and self.before.mode != self.after.mode:
            raise ValueError("source-role changes cannot change a tracked mode")
        return self


class BinaryBinding(StrictRecord):
    sha256: str = Field(pattern=HASH)
    build_source_id: str = Field(pattern=OID)


class NumericalBinding(StrictRecord):
    measurement: Source
    evaluator: Source
    calibration: Source
    policy_sha256: str = Field(pattern=HASH)
    registry_sha256: str = Field(pattern=HASH)
    manifest_sha256: str = Field(pattern=HASH)
    result_sha256: str = Field(pattern=HASH)
    marker_sha256: str = Field(pattern=HASH)
    original_cpu_roles_sha256: str = Field(pattern=HASH)


class RolePolicy(StrictRecord):
    protocol: Literal["layered-release-source-roles-v1"]
    schema_version: Literal[1]
    status: Literal["approved"]
    numerical: NumericalBinding
    performance_producer: Source
    benchmark_binary: BinaryBinding
    producer_changes: list[ObjectChange] = Field(min_length=1)
    monitor_method: Literal["direct-nvml-v1"]
    monitor_schema_version: Literal[1]

    @model_validator(mode="after")
    def narrow_scope(self) -> RolePolicy:
        names = [change.path for change in self.producer_changes]
        if len(set(names)) != len(names) or any(name not in MONITOR_PATHS for name in names):
            raise ValueError(
                "producer scope is limited to exact reviewed benchmark monitor objects"
            )
        if self.performance_producer.commit in (
            self.numerical.measurement.commit,
            self.numerical.evaluator.commit,
        ):
            raise ValueError("performance source cannot relabel numerical source")
        return self


class ReviewBinding(StrictRecord):
    source: Source
    base: Source
    diff_sha256: str = Field(pattern=HASH)
    standards_findings: int = Field(ge=0, le=0)
    spec_findings: int = Field(ge=0, le=0)


class ExternalFreeze(StrictRecord):
    protocol: Literal["layered-source-role-freeze-v1"]
    schema_version: Literal[1]
    status: Literal["approved"]
    policy_sha256: str = Field(pattern=HASH)
    release_source: Source
    performance_repo: str
    benchmark_path: str
    transport_consumer_path: str
    transport_consumer_sha256: str = Field(pattern=HASH)
    cpu_envelope_sha256: str = Field(pattern=HASH)
    release_changes: list[ObjectChange] = Field(min_length=1)
    performance_review: ReviewBinding
    release_review: ReviewBinding
    user_scope_approval_sha256: str = Field(pattern=HASH)


class SourceRoles(BaseModel):
    """Actual identities remain distinct across the four declared roles."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    numerical_measurement: Source
    numerical_evaluator: Source
    performance_producer: Source
    release_evaluator: Source

    @model_validator(mode="after")
    def distinct(self) -> SourceRoles:
        values = [
            self.numerical_measurement,
            self.numerical_evaluator,
            self.performance_producer,
            self.release_evaluator,
        ]
        if len({source.commit for source in values}) != 4:
            raise ValueError("release source roles must remain distinct")
        return self


@dataclass(frozen=True)
class FreezeBinding:
    """Trusted external input: the expected hash comes from explicit scope approval."""

    path: Path
    sha256: str


def _approved_freeze(freeze: FreezeBinding | None) -> ExternalFreeze:
    if freeze is None:
        raise ValueError("approved external source-role freeze is required")
    if freeze.path.is_symlink() or not freeze.path.is_file() or sha(freeze.path) != freeze.sha256:
        raise ValueError("external source-role freeze hash/path differs")
    value = json.loads(freeze.path.read_text())
    if not isinstance(value, dict) or value.get("status") != "approved":
        raise ValueError("approved external source-role freeze is required")
    return ExternalFreeze.model_validate(value)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def _source_object(repo: Path, source: Source) -> None:
    if _git(repo, "rev-parse", source.commit + "^{tree}").decode().strip() != source.tree:
        raise ValueError("source-role commit/tree does not exist in trusted Git")


def _objects(repo: Path, source: Source) -> dict[str, GitObject]:
    result = {}
    for item in _git(repo, "ls-tree", "-r", "-z", source.commit).split(b"\0"):
        if item:
            header, path = item.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            result[path.decode()] = GitObject.model_validate(dict(mode=mode, kind=kind, oid=oid))
    return result


def _exact_changes(repo: Path, old: Source, new: Source, changes: list[ObjectChange]) -> None:
    _source_object(repo, old)
    _source_object(repo, new)
    _git(repo, "merge-base", "--is-ancestor", old.commit, new.commit)
    before, after = _objects(repo, old), _objects(repo, new)
    actual = {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}
    if actual != {change.path for change in changes}:
        raise ValueError("source-role diff differs from exact reviewed object changes")
    for change in changes:
        if before.get(change.path) != change.before or after.get(change.path) != change.after:
            raise ValueError("source-role Git object/mode differs from reviewed change")


def _role_policy(repo: Path) -> tuple[dict[str, Any], str]:
    """Select only this new Definition; the old NUM/registry/SUP loader is unchanged."""
    index_raw = (repo / ".dag/definition-index.json").read_bytes()
    if index_raw != _git(repo, "show", "HEAD:.dag/definition-index.json"):
        raise ValueError("source-role Definition index is not the tracked checkpoint")
    inputs = json.loads(index_raw)["inputs"]
    selected = [entry for entry in inputs if entry["path"] == POLICY_PATH]
    if len(selected) != 1:
        raise ValueError("source-role Definition is pending or duplicated")
    path = repo / POLICY_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("source-role Definition must be a tracked regular file")
    raw = path.read_bytes()
    oid = _git(repo, "rev-parse", "HEAD:" + POLICY_PATH).decode().strip()
    if oid != selected[0]["blob_oid"] or raw != _git(repo, "show", "HEAD:" + POLICY_PATH):
        raise ValueError("source-role Definition differs from selected tracked bytes")
    return json.loads(raw), sha(path)


def _preserve_definitions(repo: Path, old: Source, new: Source) -> None:
    before, after = (_objects(repo, source) for source in (old, new))
    index = json.loads(_git(repo, "show", old.commit + ":.dag/definition-index.json"))
    for entry in index["inputs"]:
        path = entry["path"]
        if before.get(path) != after.get(path) or before[path].oid != entry["blob_oid"]:
            raise ValueError("original selected Definition Git object/mode changed")
    new_index = json.loads(_git(repo, "show", new.commit + ":.dag/definition-index.json"))
    if {key: value for key, value in index.items() if key != "inputs"} != {
        key: value for key, value in new_index.items() if key != "inputs"
    }:
        raise ValueError("source-role selection changed original Definition index metadata")
    original = {entry["path"]: entry for entry in index["inputs"]}
    current = {entry["path"]: entry for entry in new_index["inputs"]}
    if (
        len(current) != len(new_index["inputs"])
        or any(current.get(path) != entry for path, entry in original.items())
        or current.keys() - original.keys() - {POLICY_PATH}
    ):
        raise ValueError("source-role index changed original selected Definition inputs")


def _load_binding(
    repo: Path, binding: FreezeBinding | None
) -> tuple[ExternalFreeze, RolePolicy, SourceRoles]:
    from golden_gen.layered_artifacts import source_identity

    freeze = _approved_freeze(binding)
    if source_identity(repo) != freeze.release_source.model_dump():
        raise ValueError("actual release evaluator differs from approved frozen W")
    document, digest = _role_policy(repo)
    if digest != freeze.policy_sha256:
        raise ValueError("selected source-role Definition differs from actual approval")
    policy = RolePolicy.model_validate(document)
    roles = SourceRoles(
        numerical_measurement=policy.numerical.measurement,
        numerical_evaluator=policy.numerical.evaluator,
        performance_producer=policy.performance_producer,
        release_evaluator=freeze.release_source,
    )
    _exact_changes(
        repo, roles.numerical_measurement, roles.performance_producer, policy.producer_changes
    )
    _exact_changes(repo, roles.numerical_evaluator, roles.release_evaluator, freeze.release_changes)
    _preserve_definitions(repo, roles.numerical_measurement, roles.performance_producer)
    _preserve_definitions(repo, roles.numerical_evaluator, roles.release_evaluator)
    monitor_names = {change.path for change in policy.producer_changes}
    for change in freeze.release_changes:
        path = change.path
        if path not in ADAPTER_PATHS | monitor_names:
            raise ValueError(
                "release adapter change exceeds the narrowly reviewed source-role scope"
            )
    producer_objects = _objects(repo, roles.performance_producer)
    release_objects = _objects(repo, roles.release_evaluator)
    if any(producer_objects[path] != release_objects.get(path) for path in monitor_names):
        raise ValueError("R/W monitor objects differ")
    for review, source, base in (
        (freeze.performance_review, roles.performance_producer, roles.numerical_measurement),
        (freeze.release_review, roles.release_evaluator, roles.numerical_evaluator),
    ):
        import hashlib

        if (
            review.source != source
            or review.base != base
            or review.diff_sha256
            != hashlib.sha256(
                _git(repo, "diff", "--no-renames", "--binary", base.commit, source.commit)
            ).hexdigest()
        ):
            raise ValueError("actual fixed-range source review differs")
    return freeze, policy, roles


def _numerical_result(
    repo: Path, root: Path, entries: Entrypoints, policy: RolePolicy, roles: SourceRoles
) -> tuple[dict[str, Any], list[Path]]:
    from golden_gen.layered_artifacts import (
        bound_file,
        manifest_artifact_closure,
        source_identity,
        verify_marker,
    )
    from golden_gen.release_transport import logical_path

    manifest = root / logical_path(entries.authoritative_manifest)
    marker = root / logical_path(entries.authoritative_marker)
    if (
        manifest.is_symlink()
        or marker.is_symlink()
        or sha(manifest) != policy.numerical.manifest_sha256
        or sha(marker) != policy.numerical.marker_sha256
    ):
        raise ValueError("original numerical manifest/marker changed")
    preserved = verify_marker(marker, roles.numerical_evaluator.model_dump())
    marker_root = marker.parent.parent
    result_path = bound_file(marker_root, preserved["outputs"][0])
    if (
        sha(result_path) != policy.numerical.result_sha256
        or bound_file(marker_root, preserved["outputs"][1]).resolve() != manifest.resolve()
    ):
        raise ValueError("original numerical result/marker edge changed")
    expected = json.loads(result_path.read_text())
    if expected.get("accepting") is not True or expected.get("verdict") != "PASS":
        raise ValueError("source-role route requires original accepted numerical evidence")
    with tempfile.TemporaryDirectory(prefix="trusted-role-evaluator-") as directory:
        checkout = Path(directory) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(checkout)],
            check=True,
            capture_output=True,
        )
        _git(checkout, "checkout", "--quiet", "--detach", roles.numerical_evaluator.commit)
        if source_identity(checkout) != roles.numerical_evaluator.model_dump():
            raise ValueError("trusted Y checkout identity differs")
        instruction = (
            "import json,sys;from pathlib import Path;"
            "from golden_gen.layered_artifacts import source_identity;"
            "from golden_gen.layered_manifest import evaluate_manifest;"
            "repo=Path(sys.argv[1]);expected=json.loads(sys.argv[3]);"
            "assert source_identity(repo)==expected;"
            "result=evaluate_manifest(repo,Path(sys.argv[2]),authoritative=True);"
            "Path(sys.argv[4]).write_text(json.dumps(result,sort_keys=True,allow_nan=False))"
        )
        output = Path(directory) / "result.json"
        guard_path = Path(directory) / "guard.json"
        command = [
            sys.executable,
            "-c",
            instruction,
            str(checkout),
            str(manifest.absolute()),
            json.dumps(roles.numerical_evaluator.model_dump()),
            str(output),
        ]
        from golden_gen.guard import run_guarded
        from golden_gen.telemetry import read_ram

        def offline_probe() -> dict[str, Any]:
            return dict(
                available_ram_bytes=read_ram(),
                disk_free_bytes=shutil.disk_usage(directory).free,
                gpu_memory="",
                compute_processes="",
            )

        # guard.py and telemetry.py are preserved old-Y Git objects. This CPU
        # probe never starts NVML/nvidia-smi or imports code from the archive.
        run_guarded(
            command,
            guard_path,
            probe=offline_probe,
            interval=0.1,
            cwd=checkout,
            env=dict(
                os.environ,
                PYTHONPATH=str(checkout / "tools/golden-gen/src"),
                PYTHONDONTWRITEBYTECODE="1",
            ),
            supervision=dict(
                role="auxiliary_stage",
                stage_source=roles.numerical_evaluator.model_dump(),
                stage_invocation=dict(cwd=str(checkout), command=command),
            ),
            timeout_seconds=7200,
        )
        guard = json.loads(guard_path.read_text())
        if (
            guard["child_returncode"] != 0
            or guard["failure"] is not None
            or guard["cleanup_failure"] is not None
            or guard["telemetry_cleanup_failure"] is not None
            or guard["remaining_owned_pids"] != []
            or guard["remaining_telemetry_pids"] != []
            or guard["command"] != command
            or guard["ram_poll_interval_ms"] != 100
            or guard["stage_deadline_seconds"] != 7200
        ):
            raise ValueError("independent trusted Y CPU owner did not complete guarded cleanup")
        reproduced = json.loads(output.read_text())
        if (
            reproduced != expected
            or source_identity(checkout) != roles.numerical_evaluator.model_dump()
        ):
            raise ValueError("actual independent trusted Y full re-evaluation differs")
    closure = [marker, result_path, *manifest_artifact_closure(manifest)]
    if (
        verify_marker(marker, roles.numerical_evaluator.model_dump()) != preserved
        or source_identity(repo) != roles.release_evaluator.model_dump()
    ):
        raise ValueError("numerical originals or W source changed during independent evaluation")
    return reproduced, list(dict.fromkeys(closure))


def _precondition(
    repo: Path, root: Path, entries: Entrypoints, binding: FreezeBinding | None
) -> dict[str, Any]:
    from golden_gen.layered_release import REGISTRY_PATH, Registry, definition_document
    from golden_gen.release_cpu import validate_release_cpu_roles
    from golden_gen.release_transport import logical_path

    freeze, policy, roles = _load_binding(repo, binding)
    cpu = root / logical_path(entries.cpu_gates)
    if sha(cpu) != freeze.cpu_envelope_sha256:
        raise ValueError("actual four-role CPU envelope differs from external freeze")
    cpu_closure = validate_release_cpu_roles(cpu, roles, freeze.policy_sha256)
    original = json.loads(cpu.read_text())["original_cpu_roles"]
    if original["sha256"] != policy.numerical.original_cpu_roles_sha256:
        raise ValueError("original X/Y CPU role bytes changed")
    registry_data, digest = definition_document(repo, REGISTRY_PATH)
    if digest != policy.numerical.registry_sha256:
        raise ValueError("source-role registry differs from original numerical registry")
    result, numerical_closure = _numerical_result(repo, root, entries, policy, roles)
    if (
        result["source"] != roles.numerical_measurement.model_dump()
        or result["evaluator_source"] != roles.numerical_evaluator.model_dump()
        or result["calibration_source"] != policy.numerical.calibration.model_dump()
        or result["policy_sha256"] != policy.numerical.policy_sha256
        or result["registry_sha256"] != policy.numerical.registry_sha256
    ):
        raise ValueError("numerical source roles were relabeled")
    return dict(
        freeze=freeze,
        policy=policy,
        roles=roles,
        numerical=result,
        registry=Registry.model_validate(registry_data),
        closure=[*cpu_closure, *numerical_closure],
    )


def prepare_performance_binding(
    trusted_repo: Path,
    evidence_root: Path,
    entrypoints: Entrypoints,
    *,
    freeze: FreezeBinding | None = None,
) -> dict[str, Any]:
    prepared = _precondition(trusted_repo, evidence_root, entrypoints, freeze)
    return dict(accepting=False, **prepared)


def validate_release_roles(
    trusted_repo: Path,
    evidence_root: Path,
    entrypoints: Entrypoints,
    *,
    freeze: FreezeBinding | None = None,
) -> dict[str, Any]:
    from golden_gen.release_performance import validate_role_performance
    from golden_gen.release_transport import logical_path

    prepared = _precondition(trusted_repo, evidence_root, entrypoints, freeze)
    raw, performance_closure = validate_role_performance(
        evidence_root / logical_path(entrypoints.performance),
        prepared["roles"],
        prepared["policy"],
        prepared["numerical"]["runtime_profile"],
        prepared["registry"],
        prepared["freeze"].policy_sha256,
    )
    return {
        **prepared,
        "accepting": True,
        "verdict": "PASS",
        "performance": raw,
        "closure": [*prepared["closure"], *performance_closure],
    }
