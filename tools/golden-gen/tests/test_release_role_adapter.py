"""Synthetic Git/CPU evidence; no fixture is a real source-role approval."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from golden_gen.layered_artifacts import sha, source_identity, write_marker
from golden_gen.layered_manifest import evaluate_manifest
from golden_gen.layered_release import POLICY_PATH as NUM_PATH
from golden_gen.layered_release import REGISTRY_PATH
from golden_gen.release_roles import (
    FreezeBinding,
    prepare_performance_binding,
    validate_release_roles,
)
from golden_gen.release_transport import Entrypoints
from tests.supervised_evidence_fixture import supervised_release_inputs
from tests.test_release_role_cpu import cpu_document
from tests.test_role_performance import role_performance


def synthetic_role_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo, run, x, _, entries = supervised_release_inputs(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

    # The child executes real, immutable old-Y numerical Python from trusted Git,
    # against only the six synthetic prediction cases and synthetic raw records.
    original = Path(__file__).resolve().parents[3]
    y_commit = "ba5eb799acbf156fcab588d71aad09b6f1b15615"
    paths = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(original),
                "ls-tree",
                "-r",
                "--name-only",
                y_commit,
                "tools/golden-gen/src",
            ]
        )
        .decode()
        .splitlines()
    )
    for name in paths:
        output = repo / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(
            subprocess.check_output(["git", "-C", str(original), "show", y_commit + ":" + name])
        )
    git("add", ".")
    git("commit", "-qm", "synthetic trusted Y code and Definitions")
    y = source_identity(repo)
    manifest_path = run / entries["authoritative_manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest.update(evaluator_source=y, supervision_source=y)
    manifest_path.write_text(json.dumps(manifest))
    result = evaluate_manifest(repo, manifest_path, authoritative=True)
    assert result["accepting"] is True, result
    marker_path = run / entries["authoritative_marker"]
    prior = json.loads(marker_path.read_text())["predecessor"]["path"]
    marker_path.unlink()  # This isolated fixture regenerates its own synthetic marker.
    result_path = run / "authoritative-result.json"
    result_path.write_text(json.dumps(result))
    write_marker(run, "authoritative", y, [result_path, manifest_path], run / prior)

    def ref(path):
        return dict(path=path.relative_to(run).as_posix(), sha256=sha(path))

    cpu_x, cpu_y = (cpu_document(run, label, source) for label, source in (("X", x), ("Y", y)))
    original_cpu = run / "original-cpu-roles.json"
    original_cpu.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                kind="role_cpu_gates",
                measurement_source=x,
                supervision_source=y,
                measurement=ref(cpu_x),
                supervision=ref(cpu_y),
            )
        )
    )
    git("checkout", "--quiet", "--detach", x["commit"])
    monitor = "crates/vllm-oxide-test/src/benchmark/nvml_monitor.rs"
    (repo / monitor).parent.mkdir(parents=True)
    (repo / monitor).write_text("// synthetic monitor fixture, never built\n")
    git("add", monitor)
    git("commit", "-qm", "synthetic monitor-only R")
    r = source_identity(repo)
    r_object = dict(mode="100644", kind="blob", oid=git("rev-parse", "HEAD:" + monitor))
    git("checkout", "--quiet", "--detach", y["commit"])
    (repo / monitor).parent.mkdir(parents=True, exist_ok=True)
    (repo / monitor).write_text("// synthetic monitor fixture, never built\n")
    policy = dict(
        protocol="layered-release-source-roles-v1",
        schema_version=1,
        status="approved",
        numerical=dict(
            measurement=x,
            evaluator=y,
            calibration=result["calibration_source"],
            policy_sha256=sha(repo / NUM_PATH),
            registry_sha256=sha(repo / REGISTRY_PATH),
            manifest_sha256=sha(manifest_path),
            result_sha256=sha(result_path),
            marker_sha256=sha(marker_path),
            original_cpu_roles_sha256=sha(original_cpu),
        ),
        performance_producer=r,
        benchmark_binary=dict(sha256="a" * 64, build_source_id="e" * 40),
        producer_changes=[dict(path=monitor, before=None, after=r_object)],
        monitor_method="direct-nvml-v1",
        monitor_schema_version=1,
    )
    role_policy = "docs/validation/release-source-role-policy.json"
    (repo / role_policy).write_text(json.dumps(policy))
    index_path = repo / ".dag/definition-index.json"
    index = json.loads(index_path.read_text())
    index["inputs"].append(dict(path=role_policy, blob_oid=git("hash-object", role_policy)))
    index_path.write_text(json.dumps(index))
    git("add", ".")
    git("commit", "-qm", "synthetic selected role Definition W")
    w = source_identity(repo)
    cpu_r, cpu_w = (cpu_document(run, label, source) for label, source in (("R", r), ("W", w)))
    cpu = run / "four-role-cpu.json"
    scope = run / "selected-role-policy.json"
    scope.write_bytes((repo / role_policy).read_bytes())
    cpu.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=2,
                kind="release_role_cpu_gates",
                sources=dict(
                    numerical_measurement=x,
                    numerical_evaluator=y,
                    performance_producer=r,
                    release_evaluator=w,
                ),
                original_cpu_roles=ref(original_cpu),
                performance_cpu=ref(cpu_r),
                release_cpu=ref(cpu_w),
                source_role_policy=ref(scope),
            )
        )
    )

    def objects(commit):
        values = {}
        for row in git("ls-tree", "-r", commit).splitlines():
            header, name = row.split("\t")
            mode, kind, oid = header.split()
            values[name] = dict(mode=mode, kind=kind, oid=oid)
        return values

    before, after = objects(y["commit"]), objects(w["commit"])
    changes = [
        dict(path=p, before=before.get(p), after=after[p])
        for p in after
        if before.get(p) != after[p]
    ]

    def review(source, base):
        raw = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--no-renames",
                "--binary",
                base["commit"],
                source["commit"],
            ]
        )
        return dict(
            source=source,
            base=base,
            diff_sha256=hashlib.sha256(raw).hexdigest(),
            standards_findings=0,
            spec_findings=0,
        )

    freeze_path = tmp_path / "synthetic-freeze.json"
    freeze_path.write_text(
        json.dumps(
            dict(
                protocol="layered-source-role-freeze-v1",
                schema_version=1,
                status="approved",
                policy_sha256=sha(scope),
                release_source=w,
                performance_repo="/synthetic/R",
                benchmark_path="/synthetic/binary",
                transport_consumer_path="/synthetic/consumer",
                transport_consumer_sha256="a" * 64,
                cpu_envelope_sha256=sha(cpu),
                release_changes=changes,
                performance_review=review(r, x),
                release_review=review(w, y),
                user_scope_approval_sha256="a" * 64,
            )
        )
    )
    entries["cpu_gates"] = cpu.name
    return (
        repo,
        run,
        Entrypoints.model_validate(entries),
        FreezeBinding(freeze_path, sha(freeze_path)),
    )


@pytest.mark.usefixtures("controlled_host_ram")
def test_role_preparation_executes_trusted_y_full_numerical_code_in_a_separate_process(
    tmp_path, monkeypatch
):
    from golden_gen import guard

    repo, run, entries, freeze = synthetic_role_inputs(tmp_path)
    before = {p: sha(p) for p in run.rglob("*") if p.is_file()}
    owners = []
    original_guard = guard.run_guarded

    def retain_actual_cpu_owner(command, evidence, **kwargs):
        original_guard(command, evidence, **kwargs)
        owner = json.loads(evidence.read_text())
        assert owner["child_pid"] != os.getpid()
        assert kwargs["cwd"] != repo
        assert kwargs["env"]["PYTHONPATH"] == str(kwargs["cwd"] / "tools/golden-gen/src")
        assert source_identity(kwargs["cwd"]) == json.loads(command[5])
        assert owner["remaining_owned_pids"] == []
        owners.append(owner)

    monkeypatch.setattr(guard, "run_guarded", retain_actual_cpu_owner)
    result = prepare_performance_binding(repo, run, entries, freeze=freeze)
    assert len(owners) == 1
    assert result["accepting"] is False
    assert result["numerical"]["accepting"] is True
    assert len(result["numerical"]["numerical_checks"]) == 6
    assert (
        result["numerical"]["evaluator_source"] == result["roles"].numerical_evaluator.model_dump()
    )
    assert {p: sha(p) for p in before} == before


@pytest.mark.usefixtures("controlled_host_ram")
def test_rehashed_pass_summary_cannot_replace_independent_trusted_y_math(tmp_path):
    repo, run, entries, freeze = synthetic_role_inputs(tmp_path)
    preserved_result = run / "authoritative-result.json"
    value = json.loads(preserved_result.read_text())
    value["numerical_checks"][0]["rows"][0]["d_r"] = 999
    preserved_result.write_text(json.dumps(value))
    marker_path = run / entries.authoritative_marker
    marker = json.loads(marker_path.read_text())
    marker["outputs"][0]["sha256"] = sha(preserved_result)
    marker_path.write_text(json.dumps(marker))
    # Rebind the entire synthetic policy/freeze/CPU graph, so only independent
    # numerical reproduction can reject the coherent forged PASS summary.
    policy_path = repo / "docs/validation/release-source-role-policy.json"
    policy = json.loads(policy_path.read_text())
    policy["numerical"].update(result_sha256=sha(preserved_result), marker_sha256=sha(marker_path))
    policy_path.write_text(json.dumps(policy))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

    index_path = repo / ".dag/definition-index.json"
    index = json.loads(index_path.read_text())
    index["inputs"][-1]["blob_oid"] = git("hash-object", str(policy_path))
    index_path.write_text(json.dumps(index))
    git("add", ".")
    git("commit", "-qm", "synthetic rehashed forged summary")
    w = source_identity(repo)
    scope = run / "selected-role-policy.json"
    scope.write_bytes(policy_path.read_bytes())
    cpu_w = run / "W/evidence.json"
    cpu_data = json.loads(cpu_w.read_text())
    cpu_data["source"] = w
    cpu_w.write_text(json.dumps(cpu_data))
    cpu_path = run / entries.cpu_gates
    cpu = json.loads(cpu_path.read_text())
    cpu["sources"]["release_evaluator"] = w
    cpu["release_cpu"]["sha256"] = sha(cpu_w)
    cpu["source_role_policy"]["sha256"] = sha(scope)
    cpu_path.write_text(json.dumps(cpu))
    external = json.loads(freeze.path.read_text())
    external.update(policy_sha256=sha(scope), release_source=w, cpu_envelope_sha256=sha(cpu_path))
    for change in external["release_changes"]:
        change["after"]["oid"] = git("rev-parse", "HEAD:" + change["path"])
    review = external["release_review"]
    review["source"] = w
    review["diff_sha256"] = hashlib.sha256(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--no-renames",
                "--binary",
                review["base"]["commit"],
                w["commit"],
            ]
        )
    ).hexdigest()
    freeze.path.write_text(json.dumps(external))
    with pytest.raises(ValueError, match="independent trusted Y full re-evaluation differs"):
        prepare_performance_binding(
            repo, run, entries, freeze=FreezeBinding(freeze.path, sha(freeze.path))
        )


def complete_role_inputs(tmp_path):
    repo, run, entries, freeze = synthetic_role_inputs(tmp_path / "release")
    performance, _, _, _, _, _, wrapper = role_performance(tmp_path / "performance")
    target = run / entries.performance
    shutil.copytree(performance.parent, target.parent, dirs_exist_ok=True)
    external = json.loads(freeze.path.read_text())
    policy = json.loads((repo / "docs/validation/release-source-role-policy.json").read_text())
    roles = dict(
        numerical_measurement=policy["numerical"]["measurement"],
        numerical_evaluator=policy["numerical"]["evaluator"],
        performance_producer=policy["performance_producer"],
        release_evaluator=external["release_source"],
    )
    raw_path, guard_path = target.parent / "benchmark.json", target.parent / "guard.json"
    raw, guard = (json.loads(path.read_text()) for path in (raw_path, guard_path))
    raw.update(
        measurement_commit=roles["performance_producer"]["commit"],
        measurement_tree=roles["performance_producer"]["tree"],
    )
    guard.update(
        measurement_source=roles["performance_producer"],
        supervision_source=roles["release_evaluator"],
        supervision_policy_sha256=external["policy_sha256"],
    )
    for name, key in (("--measurement-commit", "commit"), ("--measurement-tree", "tree")):
        guard["command"][guard["command"].index(name) + 1] = roles["performance_producer"][key]
    raw_path.write_text(json.dumps(raw))
    guard_path.write_text(json.dumps(guard))
    scope = target.parent / wrapper["source_role_policy"]["path"]
    scope.write_text(json.dumps(policy))
    wrapper.update(
        sources=roles, authoritative_manifest_sha256=policy["numerical"]["manifest_sha256"]
    )
    wrapper["runtime"]["generator_commit"] = roles["performance_producer"]["commit"]
    for name, path in (("raw", raw_path), ("guard", guard_path), ("source_role_policy", scope)):
        wrapper[name]["sha256"] = sha(path)
    for artifact in wrapper["memory_methods"]:
        path = target.parent / artifact["path"]
        record = json.loads(path.read_text())
        record["source"] = roles["performance_producer"]
        path.write_text(json.dumps(record))
        artifact["sha256"] = sha(path)
    target.write_text(json.dumps(wrapper))
    return repo, run, entries, freeze


@pytest.mark.usefixtures("controlled_host_ram")
def test_complete_role_graph_revalidates_original_numerical_and_actual_performance_roles(tmp_path):
    repo, run, entries, freeze = complete_role_inputs(tmp_path)
    result = validate_release_roles(repo, run, entries, freeze=freeze)
    assert result["accepting"] is True and result["verdict"] == "PASS"
    assert result["numerical"]["source"] == result["roles"].numerical_measurement.model_dump()
    assert (
        result["performance"]["measurement_commit"] == result["roles"].performance_producer.commit
    )


@pytest.mark.usefixtures("controlled_host_ram")
def test_role_assets_survive_actual_rust_python_consumers_and_exact_report_candidate(tmp_path):
    from golden_gen.release_adapter import REPORT, git, prepare_bundle, render_report, verify_bundle

    binary = os.environ.get("GOLDEN_TRANSPORT_TEST_BINARY")
    if not binary:
        pytest.skip("CPU transport binary required")
    repo, run, entries, freeze = complete_role_inputs(tmp_path)
    consumer = Path(binary).resolve()
    external = json.loads(freeze.path.read_text())
    external.update(transport_consumer_path=str(consumer), transport_consumer_sha256=sha(consumer))
    freeze.path.write_text(json.dumps(external))
    freeze = FreezeBinding(freeze.path, sha(freeze.path))
    originals = {path: sha(path) for path in run.rglob("*") if path.is_file()}
    bundle = prepare_bundle(repo, run, tmp_path / "bundle", entries.model_dump(), freeze=freeze)
    verified = verify_bundle(repo, bundle, tmp_path / "cache", consumer, freeze=freeze)
    assert verified["accepting"] is True
    assert verified["source_roles"]["performance_producer"] != verified["manifest"]["source"]
    assert {path: sha(path) for path in originals} == originals
    report = render_report(verified)
    assert "Separate numerical and performance sources" in report
    for source in verified["source_roles"].values():
        assert source["commit"] in report

    report_path = repo / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    git(repo, "add", REPORT)
    git(repo, "commit", "-qm", "synthetic exact role report descendant")
    final = verify_bundle(repo, bundle, tmp_path / "final-cache", consumer, freeze=freeze)
    assert final["source_roles"] == verified["source_roles"]
    report_path.write_text(report + "unapproved summary\n")
    git(repo, "add", REPORT)
    git(repo, "commit", "-qm", "synthetic wrong report descendant")
    with pytest.raises(ValueError, match="exact production rendering"):
        verify_bundle(repo, bundle, tmp_path / "wrong-report-cache", consumer, freeze=freeze)
    assert not (tmp_path / "wrong-report-cache").exists()


def test_role_cli_requires_independent_freeze_hash_and_refuses_proposed_before_assets(
    tmp_path, monkeypatch, capsys
):
    from golden_gen.release_cli import main

    freeze = tmp_path / "proposed.json"
    freeze.write_text('{"status":"proposed"}')
    marker = tmp_path / "complete.json"
    arguments = [
        "release",
        "verify",
        "--repo-root",
        str(tmp_path / "repo"),
        "--bundle-dir",
        str(tmp_path / "unopened-bundle"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--rust-binary",
        str(tmp_path / "unopened-consumer"),
        "--marker",
        str(marker),
        "--source-role-freeze",
        str(freeze),
    ]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit) as failed:
        main()
    assert failed.value.code == 2
    assert "independently approved expected hash" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", arguments + ["--source-role-freeze-sha256", sha(freeze)])
    with pytest.raises(SystemExit) as failed:
        main()
    assert failed.value.code == 2
    assert "approved external source-role freeze" in capsys.readouterr().out
    assert not marker.exists() and not (tmp_path / "cache").exists()
