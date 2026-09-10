"""Truthful measurement/supervision identities and exact retained-owner authority."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from golden_gen.layered_artifacts import bound_file, sha, source_identity, worker_metadata
from golden_gen.layered_inventory import owner_key
from golden_gen.layered_release import definition_document

POLICY_PATH = "docs/validation/layered-supervision-policy.json"


def load_policy(repo: Path) -> tuple[dict[str, Any], str]:
    data, digest = definition_document(repo, POLICY_PATH)
    fixed = dict(
        protocol="layered-supervision-policy-v1",
        schema_version=1,
        policy_id="bounded-telemetry-recovery-v1",
        ram_floor_bytes=16 * 1024**3,
        fast_poll_interval_ms=100,
        telemetry_interval_ms=1000,
        query_timeout_seconds=5,
        recovery_window_seconds=15,
        max_recovered_timeouts_per_owner=1,
        retryable_error="subprocess.TimeoutExpired",
        retry_scope="complete_fresh_snapshot",
        require_fresh_before=True,
        require_fresh_after_cleanup=True,
        require_single_flight=True,
        require_owned_and_telemetry_cleanup=True,
        retained_index_start=0,
    )
    for key, expected in fixed.items():
        if type(data.get(key)) is not type(expected) or data[key] != expected:
            raise ValueError(f"unsupported supervision policy: {key}")
    for key in ("registry_sha256", "numerical_policy_sha256", "retained_ledger_sha256"):
        if not isinstance(data.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", data[key]) is None:
            raise ValueError("invalid supervision policy digest")
    count = data.get("retained_owner_count")
    if (
        type(count) is not int
        or count <= 0
        or data.get("retained_index_end_inclusive") != count - 1
    ):
        raise ValueError("invalid retained owner policy range")
    from golden_gen.layered_release import POLICY_PATH as NUMERICAL_POLICY_PATH
    from golden_gen.layered_release import REGISTRY_PATH

    if (
        definition_document(repo, REGISTRY_PATH)[1] != data["registry_sha256"]
        or definition_document(repo, NUMERICAL_POLICY_PATH)[1] != data["numerical_policy_sha256"]
    ):
        raise ValueError("supervision policy changed numerical scope")
    return data, digest


def measurement_context(
    repo: Path,
    measurement_repo: Path,
    python: str,
    binary: Path | None,
) -> dict[str, Any]:
    """Bind the actual immutable checkout and binary before constructing a worker."""
    policy, digest = load_policy(repo)
    supervisor = source_identity(repo)
    validate_equivalence(repo, policy, supervisor)
    measured = source_identity(measurement_repo)
    if measured != policy["measurement_source"]:
        raise ValueError("measurement checkout identity mismatch")
    root = str(measurement_repo.resolve())
    candidate = None
    if binary is not None:
        if binary.is_symlink() or sha(binary) != policy["measurement_binary"]["sha256"]:
            raise ValueError("measurement binary identity mismatch")
        candidate = dict(path=str(binary.resolve()), **policy["measurement_binary"])
    return dict(
        role="measurement_owner",
        measurement_source=measured,
        supervision_source=supervisor,
        supervision_policy_sha256=digest,
        measurement_invocation=dict(
            cwd=root,
            repo_root=root,
            pythonpath=str(measurement_repo.resolve() / "tools/golden-gen/src"),
            pythondontwritebytecode="1",
            python_executable=python,
            candidate_binary=candidate,
        ),
    )


def validate_invocation(
    guard: dict[str, Any],
    policy: dict[str, Any],
    key: tuple[str, str, str, str],
    receipt: dict[str, Any],
) -> None:
    invocation = guard.get("measurement_invocation", {})
    root = invocation.get("repo_root")
    if (
        guard.get("role") != "measurement_owner"
        or guard.get("measurement_source") != policy["measurement_source"]
        or receipt.get("source") != policy["measurement_source"]
        or not isinstance(root, str)
        or not Path(root).is_absolute()
        or invocation.get("cwd") != root
        or invocation.get("pythonpath") != str(Path(root) / "tools/golden-gen/src")
        or invocation.get("pythondontwritebytecode") != "1"
    ):
        raise ValueError("false measurement invocation/source")
    command = guard.get("command", [])
    if (
        not isinstance(command, list)
        or not all(isinstance(value, str) for value in command)
        or len(command) < 4
        or command[0] != invocation.get("python_executable")
        or command[1:3] != ["-m", "golden_gen.layered_cli"]
        or command[3] != ("worker" if key[0] == "execution_group" else "worker-aux")
        or len(command[4:]) % 2
    ):
        raise ValueError("invalid measurement worker invocation")
    pairs = list(zip(command[4::2], command[5::2], strict=True))
    options = dict(pairs)
    required = {"--repo-root", "--run-dir", "--model-dir", "--group", "--engine", "--variant"}
    if key[2] == "candidate":
        required.add("--candidate-binary")
    if (
        len(options) != len(pairs)
        or set(options) != required
        or options["--repo-root"] != root
        or options["--group"] != key[1]
        or options["--engine"] != key[2]
        or options["--variant"] != key[3]
        or receipt.get("engine") != key[2]
        or receipt.get("driver_pid") != guard.get("child_pid")
        or guard.get("owned_pgid") != guard.get("child_pid")
    ):
        raise ValueError("measurement invocation owner mismatch")
    binary = invocation.get("candidate_binary")
    if key[2] == "candidate":
        expected = policy["measurement_binary"]
        if (
            not isinstance(binary, dict)
            or binary.get("path") != options["--candidate-binary"]
            or binary.get("sha256") != expected["sha256"]
            or binary.get("build_source_id") != expected["build_source_id"]
            or receipt.get("binary_sha256") != expected["sha256"]
            or receipt.get("build_source_id") != expected["build_source_id"]
        ):
            raise ValueError("measurement invocation binary mismatch")
    elif binary is not None:
        raise ValueError("unexpected measurement binary invocation")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def _source(repo: Path, value: dict[str, str]) -> None:
    if set(value) != {"commit", "tree"} or any(
        not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{40}", item) is None
        for item in value.values()
    ):
        raise ValueError("invalid supervision source identity")
    if _git(repo, "rev-parse", f"{value['commit']}^{{tree}}").decode().strip() != value["tree"]:
        raise ValueError("supervision source tree mismatch")


def validate_equivalence(repo: Path, policy: dict[str, Any], source: dict[str, str]) -> None:
    """Check actual Git object/mode equality, never waive measurement source checks."""
    measured = policy["measurement_source"]
    _source(repo, measured)
    _source(repo, source)
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            measured["commit"],
            source["commit"],
        ],
        check=False,
        capture_output=True,
    )
    if status.returncode != 0:
        raise ValueError("supervision source is not a measurement descendant")
    paths = policy["unchanged_measurement_paths"]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("invalid protected measurement paths")
    for path in paths:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError("unsafe protected measurement path")
        original = _git(repo, "ls-tree", "-z", measured["commit"], "--", path)
        current = _git(repo, "ls-tree", "-z", source["commit"], "--", path)
        if not original or original != current:
            raise ValueError(f"protected measurement path changed: {path}")


def validate_retained_ledger(
    root: Path,
    artifact: dict[str, Any],
    policy: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Authorize only the frozen byte identities, not arbitrary old source claims."""
    if artifact.get("sha256") != policy["retained_ledger_sha256"]:
        raise ValueError("retained ledger differs from approved policy")
    ledger = json.loads(bound_file(root, artifact).read_text())
    count = policy["retained_owner_count"]
    if (
        ledger.get("protocol") != "layered-retained-owner-ledger-v1"
        or ledger.get("schema_version") != 1
        or ledger.get("measurement_source") != policy["measurement_source"]
        or ledger.get("registry_sha256") != policy["registry_sha256"]
        or ledger.get("policy_sha256") != policy["numerical_policy_sha256"]
        or ledger.get("owner_count") != count
        or len(ledger.get("owners", [])) != count
        or policy["retained_index_start"] != 0
        or policy["retained_index_end_inclusive"] != count - 1
        or count > len(inventory)
    ):
        raise ValueError("invalid retained ledger identity/count")
    result = {}
    for index, entry in enumerate(ledger["owners"]):
        key = owner_key(inventory[index])
        if entry.get("index") != index or entry.get("owner_key") != list(key) or key in result:
            raise ValueError("retained ledger owner inventory mismatch")
        capture_path = bound_file(root, entry["capture"])
        receipt_path = bound_file(root, entry["receipt"])
        guard_path = bound_file(root, entry["guard"])
        receipt = json.loads(receipt_path.read_text())
        guard = json.loads(guard_path.read_text())
        if (
            receipt.get("source") != policy["measurement_source"]
            or receipt.get("guard") != entry["guard"]
        ):
            raise ValueError("retained receipt identity mismatch")
        if (
            guard.get("schema_version") != 1
            or guard.get("child_returncode") != 0
            or guard.get("failure") is not None
            or guard.get("cleanup_failure") is not None
            or guard.get("remaining_owned_pids") != []
            or guard.get("minimum_available_ram_bytes", 0) < 16 * 1024**3
            or guard.get("before", {}).get("compute_processes") != ""
            or guard.get("after", {}).get("compute_processes") != ""
            or receipt.get("driver_pid") != guard.get("child_pid")
        ):
            raise ValueError("retained owner lacks successful strict legacy guard")
        samples = guard.get("resource_samples", [])
        if not samples or any(
            type(s.get("available_ram_bytes")) is not int or s["available_ram_bytes"] < 16 * 1024**3
            for s in samples
        ):
            raise ValueError("invalid retained guard samples")
        files = entry.get("files", [])
        names = [value["path"] for value in files]
        required = {entry[field]["path"] for field in ("capture", "receipt", "guard")}
        required.update(value["path"] for value in receipt.get("setup_captures", []))
        if len(set(names)) != len(names) or not required <= set(names):
            raise ValueError("retained ledger omits or duplicates dependencies")
        for value in files:
            path = bound_file(root, value)
            if path.parent != capture_path.parent:
                raise ValueError("retained dependency belongs to another owner")
        if receipt_path.parent != capture_path.parent or guard_path.parent != capture_path.parent:
            raise ValueError("retained owner directories differ")
        result[key] = entry
    return result


def manifest_roles(
    repo: Path,
    root: Path,
    manifest: dict[str, Any],
    inventory: list[dict[str, Any]],
    evaluator: dict[str, str],
) -> dict[str, Any]:
    """Validate both source roles and every owner before numerical files are parsed."""
    from golden_gen.guard_evidence import validate_guard_timeline

    policy, digest = load_policy(repo)
    validate_equivalence(repo, policy, evaluator)
    if (
        manifest.get("evaluator_source") != evaluator
        or manifest.get("supervision_source") != evaluator
        or manifest.get("source") != policy["measurement_source"]
        or manifest.get("supervision_policy", {}).get("sha256") != digest
    ):
        raise ValueError("supervised manifest source/policy mismatch")
    if json.loads(bound_file(root, manifest["supervision_policy"]).read_text()) != policy:
        raise ValueError("supervision policy artifact differs from selected Definition")
    retained = validate_retained_ledger(root, manifest["retained_owner_ledger"], policy, inventory)
    records = {}
    for category, kind in (
        ("captures", "execution_group"),
        ("operator_checks", "operator_suite"),
        ("behavior_checks", "standalone_behavior"),
    ):
        for entry in manifest.get(category, []):
            name = (
                entry["execution_group_id"]
                if kind == "execution_group"
                else entry["verification_id"]
            )
            engine = entry["engine"] if kind == "execution_group" else "candidate"
            for variant in ("primary", "replay", "control", "control_replay"):
                if entry.get(variant) is None:
                    continue
                key = (kind, name, engine, variant.replace("_", "-"))
                if key in records:
                    raise ValueError("duplicate supervised owner")
                records[key] = entry[variant], entry[variant + "_receipt"]
    if set(records) != {owner_key(owner) for owner in inventory}:
        raise ValueError("supervised owner inventory mismatch")
    for key, (capture, receipt_ref) in records.items():
        receipt = json.loads(bound_file(root, receipt_ref).read_text())
        guard = json.loads(bound_file(root, receipt["guard"]).read_text())
        if key in retained:
            expected = retained[key]
            if (
                capture != expected["capture"]
                or receipt_ref != expected["receipt"]
                or receipt["guard"] != expected["guard"]
            ):
                raise ValueError("retained owner differs from exact approved ledger")
        else:
            if (
                guard.get("schema_version") != 2
                or guard.get("supervision_source") != evaluator
                or guard.get("supervision_policy_sha256") != digest
            ):
                raise ValueError("new owner lacks supervised guard identity")
            worker_metadata(root, bound_file(root, receipt_ref), receipt)
            validate_guard_timeline(guard)
            validate_invocation(guard, policy, key, receipt)
    return {
        name: manifest[name]
        for name in (
            "evaluator_source",
            "supervision_source",
            "supervision_policy",
            "retained_owner_ledger",
        )
    }
