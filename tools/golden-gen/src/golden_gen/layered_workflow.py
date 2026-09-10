"""Prepare versioned artifacts from the frozen inventory, without claiming acceptance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from golden_gen.layered_artifacts import bound_file, sha, source_identity
from golden_gen.layered_inventory import frozen_owner_inventory, owner_key
from golden_gen.layered_release import (
    POLICY_PATH,
    REGISTRY_PATH,
    BudgetPolicy,
    Registry,
    definition_document,
)


def artifact_reference(root: Path, path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("owner artifact is not a confined regular file")
    reference = dict(path=path.relative_to(root).as_posix(), sha256=sha(path))
    bound_file(root, reference)
    return reference


def assemble_entries(root: Path, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen = set()
    for owner in inventory:
        kind, name, engine, variant = owner_key(owner)
        if (kind, name, engine, variant) in seen:
            raise ValueError("duplicate assembly owner")
        seen.add((kind, name, engine, variant))
        directory = root / f"{'aux-' if kind != 'execution_group' else ''}{name}-{engine}-{variant}"
        entry = grouped.setdefault(
            (kind, name, engine),
            dict(execution_group_id=name, engine=engine)
            if kind == "execution_group"
            else dict(verification_id=name),
        )
        field = variant.replace("-", "_")
        entry[field] = artifact_reference(root, directory / "capture.json")
        entry[field + "_receipt"] = artifact_reference(root, directory / "receipt.json")
    return dict(
        captures=[v for k, v in grouped.items() if k[0] == "execution_group"],
        operator_checks=[v for k, v in grouped.items() if k[0] == "operator_suite"],
        behavior_checks=[v for k, v in grouped.items() if k[0] == "standalone_behavior"],
    )


def assemble_manifest(
    repo: Path,
    root: Path,
    *,
    authoritative: bool,
    calibration: dict[str, Path] | None = None,
    supervision_policy: Path | None = None,
    retained_owner_ledger: Path | None = None,
) -> dict[str, Any]:
    from golden_gen.layered_manifest import LayeredManifest

    source = source_identity(repo)
    from golden_gen.supervision import POLICY_PATH as SUPERVISION_POLICY_PATH
    from golden_gen.supervision import load_policy, validate_equivalence

    supervised = authoritative and (repo / SUPERVISION_POLICY_PATH).exists()
    roles: dict[str, Any] = {}
    if supervised:
        if supervision_policy is None or retained_owner_ledger is None:
            raise ValueError(
                "supervision assembly requires policy and retained ledger dependencies"
            )
        supervision, digest = load_policy(repo)
        validate_equivalence(repo, supervision, source)
        policy_ref = artifact_reference(root, supervision_policy)
        ledger_ref = artifact_reference(root, retained_owner_ledger)
        if (
            policy_ref["sha256"] != digest
            or ledger_ref["sha256"] != supervision["retained_ledger_sha256"]
        ):
            raise ValueError("supervision assembly dependency identity mismatch")
        roles = dict(
            evaluator_source=source,
            supervision_source=source,
            supervision_policy=policy_ref,
            retained_owner_ledger=ledger_ref,
        )
        source = supervision["measurement_source"]
    elif supervision_policy is not None or retained_owner_ledger is not None:
        raise ValueError("unexpected supervision dependencies for legacy assembly")
    data, registry_sha = definition_document(repo, REGISTRY_PATH)
    registry = Registry.model_validate(data)
    policy_sha = None
    if authoritative:
        data, policy_sha = definition_document(repo, POLICY_PATH)
        policy = BudgetPolicy.model_validate(data)
        if (
            policy.values is None
            or policy.registry_sha256 != registry_sha
            or any(
                policy.operator_budgets.get(p.profile_id) is None
                for p in registry.operator_profiles
            )
        ):
            raise ValueError("budgets_pending: cannot assemble sealed authoritative evidence")
        if calibration is None or set(calibration) != {
            "calibration_evidence",
            "calibration_manifest",
            "calibration_marker",
            "fault_evidence",
        }:
            raise ValueError(
                "authoritative assembly requires all approved calibration/fault predecessors"
            )
    elif calibration:
        raise ValueError("observation cannot borrow calibration approvals")
    inventory = frozen_owner_inventory(registry, authoritative=authoritative)
    records = assemble_entries(root, inventory)
    records.update(
        protocol="layered-accuracy-v1",
        schema_version=2 if supervised else 1,
        source=source,
        registry_sha256=registry_sha,
        policy_sha256=policy_sha,
        purpose="authoritative" if authoritative else "observation",
        **roles,
    )
    if calibration:
        records.update({key: artifact_reference(root, path) for key, path in calibration.items()})
    return LayeredManifest.model_validate(records).model_dump(exclude_none=True)
