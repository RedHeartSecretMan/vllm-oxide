"""CPU-only source-role seams; no fixture represents an approved real R/W."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from golden_gen import release_roles as _roles
from golden_gen.layered_artifacts import sha, source_identity
from golden_gen.layered_release import POLICY_PATH as NUM_PATH
from golden_gen.layered_release import REGISTRY_PATH
from golden_gen.release_roles import (
    FreezeBinding,
    SourceRoles,
    prepare_performance_binding,
    validate_release_roles,
)
from golden_gen.release_transport import Entrypoints
from golden_gen.supervision import POLICY_PATH as SUP_PATH
from tests.test_release_role_adapter import synthetic_role_inputs

CAPABILITY_PATHS = (
    "tools/golden-gen/src/golden_gen/guard.py",
    "tools/golden-gen/tests/conftest.py",
    "tools/golden-gen/tests/test_guard_capability.py",
    "tools/golden-gen/tests/test_guard_recovery.py",
)


def synthetic_reviewed_candidate(repo, freeze, changes):
    """Rebind only temporary Git fixtures; never approve a real source."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

    for name, raw in changes.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    git("add", "--", *changes)
    git("commit", "-qm", "synthetic candidate object changes")
    source = source_identity(repo)
    document = json.loads(freeze.path.read_text())
    base = document["release_review"]["base"]["commit"]

    def objects(commit):
        values = {}
        for row in git("ls-tree", "-r", commit).splitlines():
            header, name = row.split("\t")
            mode, kind, oid = header.split()
            values[name] = dict(mode=mode, kind=kind, oid=oid)
        return values

    before, after = objects(base), objects(source["commit"])
    document["release_source"] = source
    document["release_changes"] = [
        dict(path=name, before=before.get(name), after=after[name])
        for name in after
        if before.get(name) != after[name]
    ]
    document["release_review"].update(
        source=source,
        diff_sha256=hashlib.sha256(
            _roles._git(repo, "diff", "--no-renames", "--binary", base, source["commit"])
        ).hexdigest(),
    )
    path = freeze.path.with_name(f"synthetic-candidate-{source['commit']}.json")
    path.write_text(json.dumps(document))
    return FreezeBinding(path, sha(path))


def synthetic_capability_candidate(tmp_path):
    repo, _, _, old_freeze = synthetic_role_inputs(tmp_path)
    original = Path(__file__).resolve().parents[3]
    candidate = synthetic_reviewed_candidate(
        repo, old_freeze, {name: (original / name).read_bytes() for name in CAPABILITY_PATHS}
    )
    return repo, old_freeze, candidate


def test_guard_capability_candidate_requires_exact_new_external_freeze(tmp_path):
    repo, old_freeze, candidate = synthetic_capability_candidate(tmp_path)
    with pytest.raises(ValueError, match="approved external source-role freeze"):
        _roles._load_binding(repo, None)
    with pytest.raises(ValueError, match="actual release evaluator differs"):
        _roles._load_binding(repo, old_freeze)
    with pytest.raises(ValueError, match="hash/path"):
        _roles._load_binding(repo, FreezeBinding(candidate.path, "0" * 64))
    frozen, policy, roles = _roles._load_binding(repo, candidate)
    old = json.loads(old_freeze.path.read_text())
    assert roles.numerical_evaluator.model_dump() == old["release_review"]["base"]
    assert roles.numerical_measurement.model_dump() == old["performance_review"]["base"]
    assert roles.performance_producer.model_dump() == old["performance_review"]["source"]
    assert policy.numerical.evaluator == roles.numerical_evaluator
    assert {change.path for change in frozen.release_changes} - {
        change["path"] for change in old["release_changes"]
    } == set(CAPABILITY_PATHS)
    telemetry = "tools/golden-gen/src/golden_gen/telemetry.py"
    assert _roles._git(repo, "show", "HEAD:" + telemetry) == _roles._git(
        repo, "show", roles.numerical_evaluator.commit + ":" + telemetry
    )
    document = json.loads(candidate.path.read_text())
    document["release_review"]["diff_sha256"] = "0" * 64
    candidate.path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="actual fixed-range source review differs"):
        _roles._load_binding(repo, FreezeBinding(candidate.path, sha(candidate.path)))


@pytest.mark.parametrize(
    "path,reason",
    [
        ("tools/golden-gen/src/golden_gen/telemetry.py", "narrowly reviewed source-role scope"),
        (
            "tools/golden-gen/src/golden_gen/layered_manifest.py",
            "narrowly reviewed source-role scope",
        ),
        (NUM_PATH, "original selected Definition"),
        (REGISTRY_PATH, "original selected Definition"),
        (SUP_PATH, "original selected Definition"),
    ],
)
def test_capability_paths_do_not_authorize_other_protected_objects(tmp_path, path, reason):
    repo, _, candidate = synthetic_capability_candidate(tmp_path)
    extra = synthetic_reviewed_candidate(
        repo, candidate, {path: (repo / path).read_bytes() + b"\n"}
    )
    with pytest.raises(ValueError, match=reason):
        _roles._load_binding(repo, extra)


def test_capability_allowlist_does_not_authorize_unreviewed_guard_edits(tmp_path):
    repo, _, candidate = synthetic_capability_candidate(tmp_path)
    guard = repo / CAPABILITY_PATHS[0]
    guard.write_bytes(guard.read_bytes() + b"\n# synthetic unreviewed guard edit\n")
    _roles._git(repo, "add", "--", CAPABILITY_PATHS[0])
    _roles._git(repo, "commit", "-qm", "synthetic unreviewed guard edit")
    document = json.loads(candidate.path.read_text())
    document["release_source"] = source_identity(repo)
    document["release_review"]["source"] = document["release_source"]
    candidate.path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Git object/mode differs from reviewed change"):
        _roles._load_binding(repo, FreezeBinding(candidate.path, sha(candidate.path)))


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
