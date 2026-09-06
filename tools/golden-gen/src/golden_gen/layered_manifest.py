"""Independent IO/identity/coverage validation before the pure numerical seam."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from golden_gen import config
from golden_gen.fixed_prefix import (
    require_collection_equivalence,
    validate_capture,
    validate_control_capture,
    validate_execution_events,
)
from golden_gen.layered_accuracy import PROTOCOL, Budgets, compare_case
from golden_gen.layered_artifacts import bound_file, sha, source_identity
from golden_gen.layered_release import (
    POLICY_PATH,
    REGISTRY_PATH,
    BudgetPolicy,
    Registry,
    definition_document,
    release_verdict,
)
from golden_gen.replay import tensor_bits_equal
from golden_gen.schema import RuntimeInfo


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaptureEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    execution_group_id: str
    engine: Literal["reference", "baseline", "candidate"]
    primary: Artifact
    replay: Artifact
    control: Artifact
    primary_receipt: Artifact
    replay_receipt: Artifact
    control_receipt: Artifact


class AuxiliaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    verification_id: str
    primary: Artifact
    replay: Artifact
    primary_receipt: Artifact
    replay_receipt: Artifact


class LayeredManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["layered-accuracy-v1"]
    schema_version: Literal[1]
    source: dict[str, str]
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str | None = None
    purpose: Literal["observation", "authoritative"]
    captures: list[CaptureEntry] = Field(min_length=1)
    operator_checks: list[AuxiliaryEntry] = Field(default_factory=list)
    behavior_checks: list[AuxiliaryEntry] = Field(default_factory=list)


def _read(root: Path, artifact: Artifact) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(bound_file(root, artifact.model_dump()).read_text())
    if not isinstance(value, dict):
        raise ValueError("layered artifact must be a JSON object")
    return value


def _receipt(
    root: Path,
    artifact: Artifact,
    capture: Artifact,
    engine: str,
    source: dict[str, str],
    registry_sha: str,
    mode: str,
) -> dict[str, Any]:
    value = _read(root, artifact)
    kernels = {
        "reference": config.REFERENCE_KERNEL_PATH,
        "baseline": config.BASELINE_KERNEL_PATH,
        "candidate": config.CANDIDATE_KERNEL_PATH,
    }
    expected_model = dict(
        revision=config.MODEL_REVISION,
        config_sha256=config.MODEL_CONFIG_SHA256,
        tokenizer_sha256=config.TOKENIZER_SHA256,
        weights_sha256=config.MODEL_WEIGHTS_SHA256,
        dtype="bfloat16",
        vocab_size=config.VOCAB_SIZE,
    )
    if (
        value.get("protocol") != PROTOCOL
        or value.get("schema_version") != 1
        or value.get("source") != source
        or value.get("registry_sha256") != registry_sha
        or value.get("engine") != engine
        or value.get("mode") != mode
        or value.get("capture_sha256") != capture.sha256
        or value.get("kernel") != kernels[engine]
        or value.get("model") != expected_model
    ):
        raise ValueError("capture receipt source/model/kernel/mode mismatch")
    runtime = RuntimeInfo.model_validate(value["runtime"])
    if runtime.evidence_mode != "release" or runtime.generator_commit != source["commit"]:
        raise ValueError("capture runtime is synthetic or belongs to another source")
    guard = json.loads(bound_file(root, value["guard"]).read_text())
    if (
        guard.get("child_returncode") != 0
        or guard.get("failure") is not None
        or guard.get("minimum_available_ram_bytes", 0) < 16 * 1024**3
        or guard.get("before", {}).get("compute_processes") != ""
        or guard.get("after", {}).get("compute_processes") != ""
        or value.get("driver_pid") != guard.get("child_pid")
    ):
        raise ValueError("capture lacks a successful fresh guarded process")
    if engine == "candidate" and (
        not isinstance(value.get("binary_sha256"), str)
        or len(value["binary_sha256"]) != 64
        or not isinstance(value.get("build_source_id"), str)
        or len(value["build_source_id"]) != 40
    ):
        raise ValueError("candidate receipt lacks binary/build identity")
    return value


def evaluate_manifest(repo: Path, manifest_path: Path, *, authoritative: bool) -> dict[str, Any]:
    """Invalid evidence returns INVALID and never opens sealed captures before approval."""
    try:
        return _evaluate(repo, manifest_path, authoritative=authoritative)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        return dict(
            protocol=PROTOCOL,
            schema_version=1,
            accepting=False,
            verdict="INVALID",
            observation_complete=False,
            reasons=[str(error)],
            operator_checks=[],
            numerical_checks=[],
            behavior_checks=[],
        )


def _evaluate(repo: Path, manifest_path: Path, *, authoritative: bool) -> dict[str, Any]:
    source = source_identity(repo)
    registry_data, registry_sha = definition_document(repo, REGISTRY_PATH)
    registry = Registry.model_validate(registry_data)
    policy: BudgetPolicy | None = None
    policy_sha = None
    if authoritative:
        policy_data, policy_sha = definition_document(repo, POLICY_PATH)
        policy = BudgetPolicy.model_validate(policy_data)
        if policy.values is None or any(
            policy.operator_budgets.get(p.profile_id) is None for p in registry.operator_profiles
        ):
            raise ValueError("budgets_pending")
        if policy.registry_sha256 != registry_sha or not all(
            (
                policy.calibration_source,
                policy.calibration_evidence_sha256,
                policy.fault_evidence_sha256,
                policy.rationale,
            )
        ):
            raise ValueError("budget approval provenance is incomplete or targets another registry")
    # Do not read even the manifest's capture paths before the budget gate above.
    manifest = LayeredManifest.model_validate_json(manifest_path.read_text())
    if (
        manifest.source != source
        or manifest.registry_sha256 != registry_sha
        or manifest.policy_sha256 != policy_sha
        or manifest.purpose != ("authoritative" if authoritative else "observation")
    ):
        raise ValueError("layered manifest has stale source/registry/policy/purpose")
    selected = (
        registry.numerical_cases
        if authoritative
        else [c for c in registry.numerical_cases if c.split == "calibration"]
    )
    if any(p.vocab_size != config.VOCAB_SIZE for c in selected for p in [c.plan, *c.setup_calls]):
        raise ValueError("release numerical evidence must contain the entire pinned vocabulary")
    expected = {
        (case.plan.execution_group_id, engine)
        for case in selected
        for engine in ("reference", "baseline", "candidate")
    }
    found = [(entry.execution_group_id, entry.engine) for entry in manifest.captures]
    if not expected or len(found) != len(expected) or set(found) != expected:
        raise ValueError("missing/duplicate/unexpected capture group, engine, or sealed subset")
    root = manifest_path.parent
    numeric = []
    coverage = []
    budgets = Budgets(**policy.values) if policy is not None and policy.values is not None else None
    for case in selected:
        by_engine = {}
        for engine in ("reference", "baseline", "candidate"):
            entry = next(
                e
                for e in manifest.captures
                if (e.execution_group_id, e.engine) == (case.plan.execution_group_id, engine)
            )
            receipts = [
                _receipt(root, receipt, capture, engine, source, registry_sha, mode)
                for receipt, capture, mode in (
                    (entry.primary_receipt, entry.primary, "fixed_prefix"),
                    (entry.replay_receipt, entry.replay, "fixed_prefix"),
                    (entry.control_receipt, entry.control, "collection_control"),
                )
            ]
            if len({r["driver_pid"] for r in receipts}) != 3:
                raise ValueError("primary/replay/control must run in separate fresh processes")
            if (
                engine == "candidate"
                and len({(r["binary_sha256"], r["build_source_id"]) for r in receipts}) != 1
            ):
                raise ValueError("candidate binary changed across capture replays")
            setup_rows = []
            for receipt in receipts:
                refs = receipt.get("setup_captures", [])
                if len(refs) != len(case.setup_calls):
                    raise ValueError("setup capture count differs from the frozen execution group")
                setup_rows.append(
                    [
                        validate_capture(plan, _read(root, Artifact.model_validate(ref)))
                        for plan, ref in zip(case.setup_calls, refs, strict=True)
                    ]
                )
            for index, setup in enumerate(case.setup_calls):
                for member in setup.members:
                    values = [
                        np.asarray(
                            [r["logits"] for r in variant[index][member.member_id]],
                            dtype=np.float32,
                        )
                        for variant in setup_rows
                    ]
                    if any(not tensor_bits_equal(values[0], value) for value in values[1:]):
                        raise ValueError("setup calls changed across primary/replay/control owners")
            first, second, control = (
                _read(root, ref) for ref in (entry.primary, entry.replay, entry.control)
            )
            primary = validate_capture(case.plan, first)
            replay = validate_capture(case.plan, second)
            shared = require_collection_equivalence(case.plan, first, control)
            from golden_gen.execution_checks import verify_prefix_reuse

            for raw, validated in (
                (first, primary),
                (second, replay),
                (control, validate_control_capture(case.plan, control)),
            ):
                verify_prefix_reuse(case, engine, raw, validated)
            mechanisms = validate_execution_events(case.plan, first, primary)
            # Scheduling mechanisms currently describe the Rust execution target;
            # reference records its own actual incremental path without pretending to page/preempt.
            if not set(case.required_mechanisms.get(engine, [])).issubset(mechanisms):
                raise ValueError(f"declared {engine} mechanism was not actually executed")
            for member in case.plan.members:
                left, right = primary[member.member_id], replay[member.member_id]
                a, b = (
                    np.asarray([r["logits"] for r in rows], dtype=np.float32)
                    for rows in (left, right)
                )
                if not tensor_bits_equal(a, b) or any(
                    x["predicted_token_id"] != y["predicted_token_id"]
                    for x, y in zip(left, right, strict=True)
                ):
                    raise ValueError("same-engine fresh replay is not bit-identical")
            by_engine[engine] = primary
            coverage.append(
                dict(
                    execution_group_id=case.plan.execution_group_id,
                    engine=engine,
                    observed_mechanisms=sorted(mechanisms),
                    control_shared_rows=shared,
                )
            )
        for member in case.plan.members:
            rows = [by_engine[e][member.member_id] for e in ("reference", "candidate", "baseline")]
            values = [
                np.asarray([r["logits"] for r in engine_rows], dtype=np.float32)
                for engine_rows in rows
            ]
            comparison = compare_case(
                values[0], values[1], values[2], [r["predicted_token_id"] for r in rows[1]], budgets
            )
            comparison.update(
                case_id=member.case_id,
                member_id=member.member_id,
                execution_group_id=case.plan.execution_group_id,
                expected_rows=len(member.continuation),
            )
            numeric.append(comparison)
    operators = []
    if len(manifest.operator_checks) > 1:
        raise ValueError("duplicate operator suite")
    for auxiliary_entry in manifest.operator_checks:
        from golden_gen.operator_verification import verify_operator_capture

        if auxiliary_entry.verification_id != "operators":
            raise ValueError("unknown operator suite")
        receipts = [
            _receipt(
                root, receipt, capture, "candidate", source, registry_sha, "operator_verification"
            )
            for receipt, capture in (
                (auxiliary_entry.primary_receipt, auxiliary_entry.primary),
                (auxiliary_entry.replay_receipt, auxiliary_entry.replay),
            )
        ]
        if (
            receipts[0]["driver_pid"] == receipts[1]["driver_pid"]
            or len({(r["binary_sha256"], r["build_source_id"]) for r in receipts}) != 1
        ):
            raise ValueError("operator replay must use fresh processes and the same binary")
        first, second = _read(root, auxiliary_entry.primary), _read(root, auxiliary_entry.replay)
        operators = verify_operator_capture(registry.operator_profiles, first, require_cuda=True)
        verify_operator_capture(registry.operator_profiles, second, require_cuda=True)
        left, right = first["operator_checks"], second["operator_checks"]
        for profile in registry.operator_profiles:
            a, b = (
                next(r for r in rows if r["profile_id"] == profile.profile_id)
                for rows in (left, right)
            )
            if not tensor_bits_equal(
                np.asarray(a["values"], dtype=np.float32), np.asarray(b["values"], dtype=np.float32)
            ):
                raise ValueError("operator replay is not bit-identical")
    behaviors = []
    seen_behavior_ids: set[str] = set()
    for auxiliary_entry in manifest.behavior_checks:
        from golden_gen.behavior_verification import compare_behavior

        behavior_case = next(
            (c for c in registry.behavior_cases if c.case_id == auxiliary_entry.verification_id),
            None,
        )
        if behavior_case is None or auxiliary_entry.verification_id in seen_behavior_ids:
            raise ValueError("unknown/duplicate public behavior case")
        seen_behavior_ids.add(auxiliary_entry.verification_id)
        receipts = [
            _receipt(root, receipt, capture, "candidate", source, registry_sha, "free_generation")
            for receipt, capture in (
                (auxiliary_entry.primary_receipt, auxiliary_entry.primary),
                (auxiliary_entry.replay_receipt, auxiliary_entry.replay),
            )
        ]
        if (
            receipts[0]["driver_pid"] == receipts[1]["driver_pid"]
            or len({(r["binary_sha256"], r["build_source_id"]) for r in receipts}) != 1
        ):
            raise ValueError("behavior replay must use fresh processes and the same binary")
        first, second = _read(root, auxiliary_entry.primary), _read(root, auxiliary_entry.replay)
        for raw in (first, second):
            if any(
                call.get("binding") is not None and call["binding"].get("device") != "cuda:0"
                for call in raw.get("calls", [])
            ):
                raise ValueError("release public behavior must execute on CUDA")
            compare_behavior(behavior_case, raw)
        if first != second:
            raise ValueError("same-engine free generation replay differs")
        behaviors.append(compare_behavior(behavior_case, first))
    result = release_verdict(
        registry if authoritative else registry.model_copy(update={"numerical_cases": selected}),
        policy,
        numeric,
        operators,
        behaviors,
    )
    result.update(
        source=source,
        registry_sha256=registry_sha,
        policy_sha256=policy_sha,
        manifest_sha256=sha(manifest_path),
        coverage=coverage,
        observation_complete=not authoritative,
        expected_capture_groups=len(expected),
        compared_capture_groups=len(found),
    )
    # Accurate numerical acceptance is not publication authority or a benchmark result.
    result["accepting"] = authoritative and result["verdict"] == "PASS"
    return result
