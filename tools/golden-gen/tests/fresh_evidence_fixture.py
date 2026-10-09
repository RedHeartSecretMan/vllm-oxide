"""Complete synthetic fresh-owner closure for CPU tests, never release evidence."""

import json
import shutil
import subprocess

from golden_gen.layered_artifacts import sha
from golden_gen.supervision import POLICY_PATH
from tests.supervised_evidence_fixture import supervised_release_inputs


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
    return repo, run, measured, evaluator, entries, guards


def synthetic_guard(pid):
    """A short owner with complete before/after queries and active RAM/process samples."""
    ram = 32 * 1024**3
    samples = []
    events = []
    for index, (phase, start, end) in enumerate((("before", 0.0, 0.05), ("after", 0.17, 0.22))):
        samples.append(
            dict(
                started_seconds=start,
                completed_seconds=end,
                elapsed_seconds=end,
                available_ram_bytes=ram,
                disk_free_bytes=1000,
                gpu_memory="0, 1, 2",
                compute_processes="",
            )
        )
        events.append(
            dict(
                attempt=index,
                phase=phase,
                started_seconds=start,
                ended_seconds=end,
                outcome="fresh",
                snapshot_index=index,
                error=None,
                queries=[
                    dict(
                        kind=kind,
                        pid=200 + index * 2 + offset,
                        timeout_seconds=5,
                        started_seconds=start + 0.01 + offset * 0.02,
                        ended_seconds=start + 0.02 + offset * 0.02,
                        outcome="ok",
                        error=None,
                    )
                    for offset, kind in enumerate(("gpu_memory", "compute_processes"))
                ],
            )
        )
    return dict(
        schema_version=2,
        child_pid=pid,
        owned_pgid=pid,
        child_returncode=0,
        failure=None,
        cleanup_failure=None,
        telemetry_cleanup_failure=None,
        remaining_owned_pids=[],
        remaining_telemetry_pids=[],
        ram_poll_interval_ms=100,
        telemetry_interval_ms=1000,
        child_started_seconds=0.06,
        child_exit_observed_seconds=0.15,
        owner_cleanup_completed_seconds=0.16,
        elapsed_seconds=0.25,
        fast_ram_samples=[
            dict(elapsed_seconds=t, available_ram_bytes=ram) for t in (0.0, 0.1, 0.25)
        ],
        ram_sample_count=3,
        maximum_fast_poll_gap_seconds=0.25 - 0.1,
        owned_process_samples=[dict(elapsed_seconds=0.1, pids=[pid])],
        resource_samples=samples,
        telemetry_events=events,
        before=samples[0],
        after=samples[1],
        minimum_available_ram_bytes=ram,
        minimum_disk_free_bytes=1000,
        peak_gpu_used_mib=1,
        minimum_gpu_free_mib=2,
    )
