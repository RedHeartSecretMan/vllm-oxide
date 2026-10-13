"""CPU-only source-role seams; no fixture represents an approved real R/W."""

import hashlib
import json
from pathlib import Path

import pytest

from golden_gen.release_roles import (
    FreezeBinding,
    SourceRoles,
    prepare_performance_binding,
    validate_release_roles,
)
from golden_gen.release_transport import Entrypoints


def test_new_performance_context_cannot_use_the_legacy_x_y_route_without_approval(tmp_path):
    from golden_gen.supervision import performance_role_context

    with pytest.raises(ValueError, match="approved external source-role freeze"):
        performance_role_context(
            tmp_path / "W", tmp_path / "R", "/synthetic/python", tmp_path / "R-binary"
        )


def test_source_roles_cannot_collapse_performance_into_numerical_identity():
    # Type-only synthetic fixture; no repository, approval or measured source.
    sources = {
        name: dict(commit=value * 40, tree=value * 40)
        for name, value in [
            ("numerical_measurement", "a"),
            ("numerical_evaluator", "b"),
            ("performance_producer", "c"),
            ("release_evaluator", "d"),
        ]
    }
    assert SourceRoles.model_validate(sources).performance_producer.commit == "c" * 40
    sources["performance_producer"] = sources["numerical_measurement"]
    with pytest.raises(ValueError, match="distinct"):
        SourceRoles.model_validate(sources)


def test_source_roles_require_external_approval_before_accessing_evidence(tmp_path: Path):
    root = tmp_path / "unobserved-evidence"
    entries = Entrypoints(
        authoritative_manifest="manifest.json",
        authoritative_marker="layered-markers/authoritative.complete.json",
        performance="performance/evidence.json",
        cpu_gates="cpu-roles.json",
    )
    for method in (prepare_performance_binding, validate_release_roles):
        with pytest.raises(ValueError, match="approved external source-role freeze"):
            method(tmp_path / "repo", root, entries)
    assert not root.exists()


def test_approved_flag_alone_cannot_supply_source_role_authority(tmp_path: Path):
    path = tmp_path / "forged-summary.json"
    path.write_text(json.dumps(dict(status="approved", supplied_verdict="PASS")))
    freeze = FreezeBinding(path, hashlib.sha256(path.read_bytes()).hexdigest())
    entries = Entrypoints(
        authoritative_manifest="manifest.json",
        authoritative_marker="marker.json",
        performance="performance.json",
        cpu_gates="cpu.json",
    )
    with pytest.raises(ValueError):
        prepare_performance_binding(
            tmp_path / "repo", tmp_path / "evidence", entries, freeze=freeze
        )
    assert not (tmp_path / "evidence").exists()


def test_proposed_freeze_cannot_authorize_the_role_route(tmp_path: Path):
    root = tmp_path / "unobserved-evidence"
    path = tmp_path / "proposed-freeze.json"
    path.write_text(json.dumps(dict(status="proposed", real_R_W_not_frozen=True)))
    freeze = FreezeBinding(path, hashlib.sha256(path.read_bytes()).hexdigest())
    entries = Entrypoints(
        authoritative_manifest="manifest.json",
        authoritative_marker="layered-markers/authoritative.complete.json",
        performance="performance/evidence.json",
        cpu_gates="cpu-roles.json",
    )
    for method in (prepare_performance_binding, validate_release_roles):
        with pytest.raises(ValueError, match="approved external source-role freeze"):
            method(tmp_path / "repo", root, entries, freeze=freeze)
    assert not root.exists()


def test_unselected_proposed_policy_and_wrong_freeze_hash_fail_before_writes(tmp_path):
    from golden_gen.release_adapter import prepare_bundle, validate_evidence, verify_bundle

    entries = Entrypoints(
        authoritative_manifest="manifest.json",
        authoritative_marker="marker.json",
        performance="performance.json",
        cpu_gates="cpu.json",
    )
    path = tmp_path / "external.json"
    path.write_text('{"status":"approved"}')
    freeze = FreezeBinding(path, "0" * 64)
    with pytest.raises(ValueError, match="hash/path"):
        validate_evidence(tmp_path / "repo", tmp_path / "evidence", entries, freeze=freeze)
    with pytest.raises(ValueError, match="hash/path"):
        prepare_bundle(
            tmp_path / "repo",
            tmp_path / "evidence",
            tmp_path / "bundle",
            entries.model_dump(),
            freeze=freeze,
        )
    with pytest.raises(ValueError, match="hash/path"):
        verify_bundle(
            tmp_path / "repo",
            tmp_path / "bundle",
            tmp_path / "cache",
            tmp_path / "consumer",
            freeze=freeze,
        )
    assert not (tmp_path / "bundle").exists() and not (tmp_path / "cache").exists()
