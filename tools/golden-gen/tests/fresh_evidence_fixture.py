"""Complete synthetic fresh-owner closure for CPU tests, never release evidence."""

import json
import shutil
import subprocess

from golden_gen.layered_artifacts import sha
from golden_gen.supervision import POLICY_PATH
from tests.supervised_evidence_fixture import supervised_release_inputs
from tests.supervision_fixture_helpers import bind_synthetic_performance, synthetic_guard


def fresh_release_inputs(tmp_path):
    repo, run, measured, _, entries = supervised_release_inputs(tmp_path)
    manifest_path = run / entries["authoritative_manifest"]
    manifest = json.loads(manifest_path.read_text())
    ledger = json.loads((run / manifest.pop("retained_owner_ledger")["path"]).read_text())
    policy = json.loads((repo / POLICY_PATH).read_text())
    for key in ("retained_ledger_sha256", "retained_index_start", "retained_index_end_inclusive"):
        del policy[key]
    candidate = next(o for o in ledger["owners"] if o["owner_key"][2] == "candidate")
    receipt = json.loads((run / candidate["receipt"]["path"]).read_text())
    policy.update(
        protocol="layered-supervision-policy-v2",
        schema_version=2,
        policy_id="bounded-telemetry-fresh-v1",
        retained_owner_count=0,
        measurement_binary=dict(
            sha256=receipt["binary_sha256"], build_source_id=receipt["build_source_id"]
        ),
    )
    (repo / POLICY_PATH).write_text(json.dumps(policy))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    index_path = repo / ".dag/definition-index.json"
    index = json.loads(index_path.read_text())
    for item in index["inputs"]:
        if item["path"] == POLICY_PATH:
            item["blob_oid"] = git("hash-object", POLICY_PATH)
    index_path.write_text(json.dumps(index))
    git("add", ".")
    git("commit", "-qm", "synthetic fresh supervision definition")
    evaluator = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))

    def artifact(path):
        return dict(path=str(path.relative_to(run)), sha256=sha(path))

    policy_path = run / manifest["supervision_policy"]["path"]
    policy_path.write_bytes((repo / POLICY_PATH).read_bytes())
    manifest.update(
        schema_version=3,
        evaluator_source=evaluator,
        supervision_source=evaluator,
        supervision_policy=artifact(policy_path),
    )
    by_key = {tuple(owner["owner_key"]): owner for owner in ledger["owners"]}
    guards = []
    for category, kind in (
        ("captures", "execution_group"),
        ("operator_checks", "operator_suite"),
        ("behavior_checks", "standalone_behavior"),
    ):
        for entry in manifest.get(category, []):
            name = entry["execution_group_id" if kind == "execution_group" else "verification_id"]
            engine = entry.get("engine", "candidate")
            for variant in ("primary", "replay", "control", "control_replay"):
                if entry.get(variant) is None:
                    continue
                key = (kind, name, engine, variant.replace("_", "-"))
                owner = by_key[key]
                receipt = json.loads((run / owner["receipt"]["path"]).read_text())
                directory = run / (
                    f"{'aux-' if kind != 'execution_group' else ''}{name}-{engine}-{key[3]}"
                )
                # The base synthetic fixture also exercises assembly, so its
                # disposable conventional owner directories may already exist.
                directory.mkdir(exist_ok=True)
                shutil.copyfile(run / entry[variant]["path"], directory / "capture.json")
                entry[variant] = artifact(directory / "capture.json")
                receipt_path = directory / "receipt.json"
                guard = synthetic_guard(receipt["driver_pid"])
                binary = (
                    dict(path="/measurement/candidate", **policy["measurement_binary"])
                    if engine == "candidate"
                    else None
                )
                command = [
                    "/python",
                    "-m",
                    "golden_gen.layered_cli",
                    "worker" if kind == "execution_group" else "worker-aux",
                    "--repo-root",
                    "/measurement",
                    "--run-dir",
                    str(run),
                    "--model-dir",
                    "/model",
                    "--group",
                    name,
                    "--engine",
                    engine,
                    "--variant",
                    key[3],
                ]
                if binary:
                    command += ["--candidate-binary", binary["path"]]
                guard.update(
                    role="measurement_owner",
                    measurement_source=measured,
                    supervision_source=evaluator,
                    supervision_policy_sha256=sha(policy_path),
                    command=command,
                    measurement_invocation=dict(
                        cwd="/measurement",
                        repo_root="/measurement",
                        pythonpath="/measurement/tools/golden-gen/src",
                        pythondontwritebytecode="1",
                        python_executable="/python",
                        candidate_binary=binary,
                    ),
                )
                guard_path = directory / "guard.json"
                guard_path.write_text(json.dumps(guard))
                receipt["guard"] = artifact(guard_path)
                receipt_path.write_text(json.dumps(receipt))
                receipt_path.with_name("worker.json").write_text(
                    json.dumps({k: v for k, v in receipt.items() if k != "guard"})
                )
                entry[variant + "_receipt"] = artifact(receipt_path)
                guards.append((guard_path, receipt_path, entry, variant))
    manifest_path.write_text(json.dumps(manifest))
    bind_synthetic_performance(run, entries, policy, measured, evaluator, policy_path)
    return repo, run, measured, evaluator, entries, guards
