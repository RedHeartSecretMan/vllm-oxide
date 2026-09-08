"""Independent IO/identity/coverage validation before the pure numerical seam."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from golden_gen import config
from golden_gen.fixed_prefix import (
    require_collection_equivalence,
    validate_baseline_owner_bindings,
    validate_capture,
    validate_control_capture,
    validate_execution_events,
)
from golden_gen.layered_accuracy import (
    PROTOCOL,
    Budgets,
    compare_case,
    free_generation_diagnostics,
    summarize_cases,
)
from golden_gen.layered_artifacts import bound_file, sha, source_identity, verify_marker
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
    control_replay: Artifact | None = None
    control_replay_receipt: Artifact | None = None


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
    calibration_evidence: Artifact | None = None
    calibration_manifest: Artifact | None = None
    calibration_marker: Artifact | None = None
    fault_evidence: Artifact | None = None


def _read(root: Path, artifact: Artifact) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(bound_file(root, artifact.model_dump()).read_text())
    if not isinstance(value, dict):
        raise ValueError("layered artifact must be a JSON object")
    return value


def _common_runtime(value: dict[str, Any]) -> dict[str, Any]:
    profile = RuntimeInfo.model_validate(value).model_dump(exclude={"generator_commit"})
    profile["wheels"] = sorted(profile["wheels"], key=lambda w: w["name"].lower().replace("_", "-"))
    return profile


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
    evidence = value.get("engine_evidence", {})
    if engine == "reference":
        if (
            evidence.get("deterministic_algorithms") is not True
            or evidence.get("warn_only") is not False
            or evidence.get("attention_backend") != "SDPBackend.MATH"
        ):
            raise ValueError("reference lacks actual deterministic MATH backend evidence")
    elif engine == "baseline":
        from golden_gen.worker_determinism import BaselineWorkerEvidence

        states = [
            BaselineWorkerEvidence.model_validate(record)
            for record in evidence.get("worker_states", [])
        ]
        if (
            len(states) != 2
            or [s.phase for s in states] != ["ready", "complete"]
            or states[0].pid != states[1].pid
            or any(s.expected_attention_layers != 28 for s in states)
        ):
            raise ValueError("baseline worker lifecycle/backend evidence is incomplete")
    elif (
        evidence.get("source") != source
        or evidence.get("cuda_feature_enabled") is not True
        or evidence.get("build_source_id") != value.get("build_source_id")
        or type(evidence.get("producer_pid")) is not int
        or evidence["producer_pid"] <= 0
    ):
        raise ValueError("candidate lacks actual CUDA build/process evidence")
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


def _calibration_provenance(
    repo: Path, root: Path, manifest: LayeredManifest, registry: Registry, policy: BudgetPolicy
) -> tuple[dict[str, dict[str, bool | None]], dict[str, Any], dict[str, Any]]:
    from golden_gen.layered_faults import verify_global_fault_evidence
    from golden_gen.operator_faults import evaluate_distribution_fault, evaluate_fault_models

    if (
        manifest.calibration_evidence is None
        or manifest.calibration_manifest is None
        or manifest.fault_evidence is None
        or manifest.calibration_marker is None
    ):
        raise ValueError("approved calibration/fault artifacts are missing")
    if (
        manifest.calibration_evidence.sha256 != policy.calibration_evidence_sha256
        or manifest.fault_evidence.sha256 != policy.fault_evidence_sha256
    ):
        raise ValueError("calibration/fault artifacts differ from the approved policy")
    old = policy.calibration_source
    if (
        old is None
        or set(old) != {"commit", "tree"}
        or any(re.fullmatch(r"[0-9a-f]{40}", v) is None for v in old.values())
    ):
        raise ValueError("invalid calibration source identity")
    tree = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", old["commit"] + "^{tree}"], text=True
    ).strip()
    if tree != old["tree"]:
        raise ValueError("calibration source tree does not match its commit")
    marker = verify_marker(bound_file(root, manifest.calibration_marker.model_dump()), old)
    if (
        marker["stage"] != "observation"
        or marker["outputs"][0]["sha256"] != manifest.calibration_evidence.sha256
        or not any(
            ref["sha256"] == manifest.calibration_manifest.sha256 for ref in marker["outputs"]
        )
    ):
        raise ValueError("approved calibration is not covered by its original observation marker")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            old["commit"],
            manifest.source["commit"],
        ],
        check=True,
        capture_output=True,
    )
    changed = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--name-only",
                "-z",
                old["commit"],
                manifest.source["commit"],
                "--",
            ]
        )
        .decode()
        .split("\0")
    )
    allowed = {
        REGISTRY_PATH,
        POLICY_PATH,
        "CONTEXT.md",
        ".dag/definition-index.json",
        ".dag/definitions/v0.2.0-github.json",
    }
    if any(
        path and path not in allowed and not (path.startswith("docs/adr/") and path.endswith(".md"))
        for path in changed
    ):
        raise ValueError(
            "execution source changed since approved calibration; fresh calibration required"
        )
    recorded = _read(root, manifest.calibration_evidence)
    if (
        recorded.get("source") != old
        or recorded.get("registry_sha256") != manifest.registry_sha256
        or recorded.get("manifest_sha256") != manifest.calibration_manifest.sha256
    ):
        raise ValueError("approved calibration report identity mismatch")
    calibration_path = bound_file(root, manifest.calibration_manifest.model_dump())
    recomputed = _evaluate(repo, calibration_path, authoritative=False, source_override=old)
    if (
        recomputed != recorded
        or not recomputed.get("observation_complete")
        or recomputed.get("accepting") is not False
    ):
        raise ValueError("approved calibration report cannot be reproduced from bound raw evidence")
    faults = _read(root, manifest.fault_evidence)
    if (
        faults.get("source") != old
        or faults.get("registry_sha256") != manifest.registry_sha256
        or faults.get("numerical_fault_rule") != "obvious_distribution_swap_v1"
    ):
        raise ValueError("fault evidence is not bound to the approved calibration source/registry")
    if not verify_global_fault_evidence(repo, calibration_path, registry, old, faults):
        raise ValueError("a required structural/identity/argmax fault was not detected")
    return (
        evaluate_fault_models(registry.operator_profiles, faults, policy.operator_budgets),
        evaluate_distribution_fault(
            Budgets(**policy.values) if policy.values is not None else None,
            faults["numerical_fault_definition"],
        ),
        recomputed["runtime_profile"],
    )


def _evaluate(
    repo: Path,
    manifest_path: Path,
    *,
    authoritative: bool,
    source_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    source = source_identity(repo) if source_override is None else source_override
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
    from golden_gen.layered_inventory import frozen_owner_inventory, owner_key

    inventory = frozen_owner_inventory(registry, authoritative=authoritative)
    declared_keys = {owner_key(owner) for owner in inventory}
    found_owners = []
    for capture_entry in manifest.captures:
        for variant in ("primary", "replay", "control"):
            found_owners.append(
                ("execution_group", capture_entry.execution_group_id, capture_entry.engine, variant)
            )
        if capture_entry.control_replay is not None:
            found_owners.append(
                (
                    "execution_group",
                    capture_entry.execution_group_id,
                    capture_entry.engine,
                    "control-replay",
                )
            )
    for auxiliary_entry in manifest.operator_checks:
        found_owners.extend(
            ("operator_suite", auxiliary_entry.verification_id, "candidate", v)
            for v in ("primary", "replay")
        )
    for auxiliary_entry in manifest.behavior_checks:
        found_owners.extend(
            ("standalone_behavior", auxiliary_entry.verification_id, "candidate", v)
            for v in ("primary", "replay")
        )
    if len(found_owners) != len(declared_keys) or set(found_owners) != declared_keys:
        raise ValueError("manifest does not cover the exact frozen unique owner inventory")
    if (
        manifest.source != source
        or manifest.registry_sha256 != registry_sha
        or manifest.policy_sha256 != policy_sha
        or manifest.purpose != ("authoritative" if authoritative else "observation")
    ):
        raise ValueError("layered manifest has stale source/registry/policy/purpose")
    root = manifest_path.parent
    fault_checks = None
    distribution_fault = None
    runtime_profile: dict[str, Any] | None = None
    if authoritative:
        if policy is None:
            raise ValueError("approved policy missing")
        fault_checks, distribution_fault, runtime_profile = _calibration_provenance(
            repo, root, manifest, registry, policy
        )

    def bind_runtime(records: list[dict[str, Any]]) -> None:
        nonlocal runtime_profile
        for record in records:
            actual = _common_runtime(record["runtime"])
            if runtime_profile is None:
                runtime_profile = actual
            elif actual != runtime_profile:
                raise ValueError(
                    "measurement runtime differs across engines/replays "
                    "or from approved calibration"
                )

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
    numeric = []
    free_generation = []
    coverage = []
    execution_captures = {}
    unforced_captures = {}
    budgets = Budgets(**policy.values) if policy is not None and policy.values is not None else None
    for case in selected:
        by_engine = {}
        own_tokens = {}
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
            public_control = engine == "candidate" and any(
                b.mode == "unforced_control"
                and b.scenario.get("execution_groups") == [case.plan.execution_group_id]
                for b in registry.behavior_cases
            )
            if public_control:
                if entry.control_replay is None or entry.control_replay_receipt is None:
                    raise ValueError("required unforced public control replay is missing")
                receipts.append(
                    _receipt(
                        root,
                        entry.control_replay_receipt,
                        entry.control_replay,
                        engine,
                        source,
                        registry_sha,
                        "collection_control",
                    )
                )
            elif entry.control_replay is not None or entry.control_replay_receipt is not None:
                raise ValueError("unexpected unforced control replay owner")
            bind_runtime(receipts)
            if len({r["driver_pid"] for r in receipts}) != len(receipts):
                raise ValueError("primary/replay/control must run in separate fresh processes")
            if (
                engine == "candidate"
                and len({(r["binary_sha256"], r["build_source_id"]) for r in receipts}) != 1
            ):
                raise ValueError("candidate binary changed across capture replays")
            setup_rows = []
            setup_raws = []
            for variant_index, receipt in enumerate(receipts):
                refs = receipt.get("setup_captures", [])
                if len(refs) != len(case.setup_calls):
                    raise ValueError("setup capture count differs from the frozen execution group")
                setup_values = [_read(root, Artifact.model_validate(ref)) for ref in refs]
                setup_raws.append(setup_values)
                setup_rows.append(
                    [
                        (validate_capture if variant_index < 2 else validate_control_capture)(
                            plan, raw
                        )
                        for plan, raw in zip(case.setup_calls, setup_values, strict=True)
                    ]
                )
            for index, setup in enumerate(case.setup_calls):
                for controls in setup_raws[2:]:
                    require_collection_equivalence(setup, setup_raws[0][index], controls[index])
                for member in setup.members:
                    values = [
                        np.asarray(
                            [r["logits"] for r in variant[index][member.member_id]],
                            dtype=np.float32,
                        )
                        for variant in setup_rows
                    ]
                    if not tensor_bits_equal(values[0], values[1]) or (
                        len(values) == 4 and not tensor_bits_equal(values[2], values[3])
                    ):
                        raise ValueError("setup calls changed across same-mode fresh replay owners")
            first, second, control = (
                _read(root, ref) for ref in (entry.primary, entry.replay, entry.control)
            )
            if engine == "baseline":
                for setups, raw in zip(setup_raws, (first, second, control), strict=True):
                    validate_baseline_owner_bindings(
                        zip([*case.setup_calls, case.plan], [*setups, raw], strict=True)
                    )
            primary = validate_capture(case.plan, first)
            if engine == "candidate":
                execution_captures[case.plan.execution_group_id] = (case, first)
            replay = validate_capture(case.plan, second)
            shared = require_collection_equivalence(case.plan, first, control)
            control_rows = validate_control_capture(case.plan, control)
            own_tokens[engine] = {
                m.member_id: [r["predicted_token_id"] for r in control_rows[m.member_id]]
                for m in case.plan.members
            }
            if public_control:
                if entry.control_replay is None:
                    raise ValueError("unforced replay missing")
                control_replay = _read(root, entry.control_replay)
                unforced_left = validate_control_capture(case.plan, control)
                unforced_right = validate_control_capture(case.plan, control_replay)
                for member in case.plan.members:
                    a, b = (
                        np.asarray([r["logits"] for r in rows[member.member_id]], dtype=np.float32)
                        for rows in (unforced_left, unforced_right)
                    )
                    if not tensor_bits_equal(a, b):
                        raise ValueError("unforced public fresh replay logits differ")
                if control.get("public_call") != control_replay.get("public_call") or any(
                    a.get("public_call") != b.get("public_call")
                    for a, b in zip(setup_raws[2], setup_raws[3], strict=True)
                ):
                    raise ValueError("unforced LLM::generate return values changed on fresh replay")
                unforced_captures[case.plan.execution_group_id] = (
                    case,
                    setup_raws[2],
                    control,
                    setup_raws[3],
                    control_replay,
                )
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
            free = free_generation_diagnostics(
                own_tokens["reference"][member.member_id],
                own_tokens["candidate"][member.member_id],
                own_tokens["baseline"][member.member_id],
            )
            free.update(
                case_id=member.case_id,
                member_id=member.member_id,
                execution_group_id=case.plan.execution_group_id,
            )
            free_generation.append(free)
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
        bind_runtime(receipts)
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
        if (
            behavior_case is None
            or behavior_case.mode != "free_generation"
            or auxiliary_entry.verification_id in seen_behavior_ids
        ):
            raise ValueError("unknown/duplicate public behavior case")
        seen_behavior_ids.add(auxiliary_entry.verification_id)
        receipts = [
            _receipt(root, receipt, capture, "candidate", source, registry_sha, "free_generation")
            for receipt, capture in (
                (auxiliary_entry.primary_receipt, auxiliary_entry.primary),
                (auxiliary_entry.replay_receipt, auxiliary_entry.replay),
            )
        ]
        bind_runtime(receipts)
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
    for behavior_case in registry.behavior_cases:
        if behavior_case.mode == "fixed_prefix_execution" and (
            authoritative or behavior_case.split == "calibration"
        ):
            from golden_gen.execution_checks import compare_execution_behavior

            behaviors.append(compare_execution_behavior(behavior_case, execution_captures))
        if behavior_case.mode == "unforced_control" and (
            authoritative or behavior_case.split == "calibration"
        ):
            from golden_gen.execution_checks import compare_unforced_control_behavior

            groups = behavior_case.scenario.get("execution_groups", [])
            if len(groups) != 1 or groups[0] not in unforced_captures:
                raise ValueError("unforced public behavior capture group missing")
            group, left_setups, left_control, right_setups, right_control = unforced_captures[
                groups[0]
            ]
            evaluations = []
            for setups, raw in ((left_setups, left_control), (right_setups, right_control)):
                if any(
                    c.get("public_call", {}).get("binding", {}).get("device") != "cuda:0"
                    for c in [*setups, raw]
                ):
                    raise ValueError("unforced public release behavior must execute on CUDA")
                evaluated = compare_unforced_control_behavior(behavior_case, group, setups, raw)
                evaluations.append(evaluated)
            combined = dict(evaluations[0])
            combined["checks"] = {
                key: all(e["checks"][key] for e in evaluations)
                for key in behavior_case.required_checks
            }
            combined["evidence_complete"] = all(e["evidence_complete"] for e in evaluations)
            combined["missing_mechanisms"] = sorted(
                {m for e in evaluations for m in e["missing_mechanisms"]}
            )
            combined["verdict"] = (
                "INVALID"
                if not combined["evidence_complete"]
                else "PASS"
                if all(combined["checks"].values())
                else "FAIL"
            )
            behaviors.append(combined)
    if fault_checks is not None:
        for operator in operators:
            operator["fault_checks"] = fault_checks[operator["profile_id"]]
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
        observation_complete=not authoritative
        and all(b.get("evidence_complete", True) for b in behaviors),
        expected_capture_groups=len(expected),
        compared_capture_groups=len(found),
        expected_unique_owners=len(inventory),
        validated_unique_owners=len(found_owners),
        case_equal_summary=summarize_cases(numeric),
        runtime_profile=runtime_profile,
        free_generation_diagnostics=free_generation,
        calibration_source=policy.calibration_source if policy is not None else None,
        distribution_fault=distribution_fault,
    )
    if (
        any(b.get("evidence_complete") is False for b in behaviors)
        and "incomplete_behavior_coverage" not in result["reasons"]
    ):
        result["reasons"].append("incomplete_behavior_coverage")
    if distribution_fault is not None and distribution_fault["verdict"] != "FAIL":
        if result["verdict"] != "INVALID":
            result["verdict"] = "FAIL"
        result["reasons"].append("approved_budget_did_not_reject_wrong_distribution_fault")
    # Accurate numerical acceptance is not publication authority or a benchmark result.
    result["accepting"] = authoritative and result["verdict"] == "PASS"
    return result
