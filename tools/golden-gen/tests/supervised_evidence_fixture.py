"""Small synthetic role-separated closure for CPU tests, never GPU measurements."""

import json
import subprocess
from pathlib import Path

from golden_gen.layered_artifacts import sha
from golden_gen.layered_inventory import frozen_owner_inventory, owner_key
from golden_gen.layered_release import POLICY_PATH, REGISTRY_PATH, Registry
from golden_gen.supervision import POLICY_PATH as SUPERVISION_POLICY_PATH
from tests.release_evidence_fixture import complete_release_inputs


def supervised_release_inputs(tmp_path):
    repo, run, measured, entries = complete_release_inputs(tmp_path)
    manifest_path = run / entries["authoritative_manifest"]
    manifest = json.loads(manifest_path.read_text())
    registry = Registry.model_validate_json((repo / REGISTRY_PATH).read_text())
    inventory = frozen_owner_inventory(registry, authoritative=True)

    def artifact(path):
        return dict(path=str(path.relative_to(run)), sha256=sha(path))

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
                if entry.get(variant) is not None:
                    records[(kind, name, engine, variant.replace("_", "-"))] = (entry, variant)
    owners = []
    for index, definition in enumerate(inventory):
        key = owner_key(definition)
        entry, variant = records[key]
        receipt_path = run / entry[variant + "_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text())
        guard_path = run / receipt["guard"]["path"]
        guard = json.loads(guard_path.read_text())
        sample = dict(
            available_ram_bytes=32 * 1024**3,
            compute_processes="",
            gpu_memory="0, 1, 2",
            disk_free_bytes=1000,
        )
        guard.update(
            schema_version=1,
            child_pid=receipt["driver_pid"],
            before=sample,
            after=sample,
            cleanup_failure=None,
            remaining_owned_pids=[],
            resource_samples=[sample, sample],
            minimum_available_ram_bytes=32 * 1024**3,
        )
        guard_path.write_text(json.dumps(guard))
        receipt["guard"] = artifact(guard_path)
        receipt_path.write_text(json.dumps(receipt))
        entry[variant + "_receipt"] = artifact(receipt_path)
        files = [
            entry[variant],
            entry[variant + "_receipt"],
            receipt["guard"],
            *receipt.get("setup_captures", []),
        ]
        owners.append(
            dict(
                index=index,
                owner_key=list(key),
                capture=entry[variant],
                receipt=entry[variant + "_receipt"],
                guard=receipt["guard"],
                files=files,
                original_execution_root=str(run),
                retained_from_root=str(run),
                origin_role="synthetic-test",
            )
        )
    ledger = dict(
        protocol="layered-retained-owner-ledger-v1",
        schema_version=1,
        measurement_source=measured,
        registry_sha256=sha(repo / REGISTRY_PATH),
        policy_sha256=sha(repo / POLICY_PATH),
        owner_count=len(owners),
        owners=owners,
    )
    ledger_path = run / "retained-ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    policy = json.loads((Path(__file__).resolve().parents[3] / SUPERVISION_POLICY_PATH).read_text())
    policy.update(
        measurement_source=measured,
        retained_ledger_sha256=sha(ledger_path),
        retained_owner_count=len(owners),
        retained_index_end_inclusive=len(owners) - 1,
        registry_sha256=sha(repo / REGISTRY_PATH),
        numerical_policy_sha256=sha(repo / POLICY_PATH),
        unchanged_measurement_paths=[REGISTRY_PATH, POLICY_PATH],
    )
    (repo / SUPERVISION_POLICY_PATH).write_text(json.dumps(policy))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    index_path = repo / ".dag/definition-index.json"
    definition_index = json.loads(index_path.read_text())
    definition_index["inputs"].append(
        dict(path=SUPERVISION_POLICY_PATH, blob_oid=git("hash-object", SUPERVISION_POLICY_PATH))
    )
    index_path.write_text(json.dumps(definition_index))
    git("add", ".")
    git("commit", "-qm", "synthetic reviewed supervision Definition")
    evaluator = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
    policy_path = run / "supervision-policy.json"
    policy_path.write_bytes((repo / SUPERVISION_POLICY_PATH).read_bytes())
    manifest.update(
        schema_version=2,
        evaluator_source=evaluator,
        supervision_source=evaluator,
        supervision_policy=artifact(policy_path),
        retained_owner_ledger=artifact(ledger_path),
    )
    manifest_path.write_text(json.dumps(manifest))
    return repo, run, measured, evaluator, entries
