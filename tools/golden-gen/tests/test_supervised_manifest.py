from golden_gen.layered_manifest import evaluate_manifest
from tests.supervised_evidence_fixture import supervised_release_inputs


def test_supervised_manifest_recomputes_complete_legacy_closure(tmp_path, monkeypatch):
    repo, run, measured, evaluator, entries = supervised_release_inputs(tmp_path)
    result = evaluate_manifest(repo, run / entries["authoritative_manifest"], authoritative=True)
    assert result["verdict"] == "PASS", result.get("reasons")
    assert result["accepting"] is True
    assert result["source"] == measured and result["evaluator_source"] == evaluator
    import json

    from golden_gen.layered_artifacts import verify_marker, write_marker

    manifest_path = run / entries["authoritative_manifest"]
    manifest = json.loads(manifest_path.read_text())
    (run / entries["authoritative_marker"]).rename(run / "original-authoritative-marker.json")
    output = run / "supervised-result.json"
    output.write_text(json.dumps(result))
    marker = write_marker(
        run,
        "authoritative",
        evaluator,
        [output, manifest_path],
        run / manifest["calibration_marker"]["path"],
    )
    verified = verify_marker(marker, evaluator)
    assert verified["schema_version"] == 2
    assert verified["measurement_source"] == measured
    assert verified["source"] == evaluator
    import copy

    import pytest

    manifest_bytes = manifest_path.read_bytes()
    wrong = copy.deepcopy(manifest)
    wrong["evaluator_source"] = wrong["supervision_source"] = dict(commit="a" * 40, tree="b" * 40)
    manifest_path.write_text(json.dumps(wrong))
    assert evaluate_manifest(repo, manifest_path, authoritative=True)["verdict"] == "INVALID"
    manifest_path.write_bytes(manifest_bytes)
    first = manifest["captures"][0]["primary"]
    alias = run / "unapproved-diagnostic-capture.json"
    alias.write_bytes((run / first["path"]).read_bytes())
    wrong = copy.deepcopy(manifest)
    wrong["captures"][0]["primary"]["path"] = alias.name
    manifest_path.write_text(json.dumps(wrong))
    assert evaluate_manifest(repo, manifest_path, authoritative=True)["verdict"] == "INVALID"
    manifest_path.write_bytes(manifest_bytes)
    marker_bytes = marker.read_bytes()
    bad_marker = json.loads(marker_bytes)
    bad_marker["measurement_source"] = evaluator
    marker.write_text(json.dumps(bad_marker))
    with pytest.raises(ValueError, match="source"):
        verify_marker(marker, evaluator)
    marker.write_bytes(marker_bytes)
    import pytest

    from golden_gen.layered_workflow import assemble_manifest

    calibration = {
        name: run / manifest[name]["path"]
        for name in (
            "calibration_evidence",
            "calibration_manifest",
            "calibration_marker",
            "fault_evidence",
        )
    }
    with pytest.raises(ValueError, match="supervision"):
        assemble_manifest(repo, run, authoritative=True, calibration=calibration)
    import shutil

    from golden_gen.layered_artifacts import sha
    from golden_gen.release_adapter import prepare_bundle

    shutil.copytree(run / "cpu", run / "supervision-cpu")
    cpu_path = run / "supervision-cpu/evidence.json"
    cpu = json.loads(cpu_path.read_text())
    cpu["source"] = evaluator  # This is a newly constructed synthetic CPU fixture, not real logs.
    cpu_path.write_text(json.dumps(cpu))

    def ref(path):
        return dict(path=str(path.relative_to(run)), sha256=sha(path))

    roles = run / "cpu-roles.json"
    roles.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                kind="role_cpu_gates",
                measurement_source=measured,
                supervision_source=evaluator,
                measurement=ref(run / "cpu/evidence.json"),
                supervision=ref(cpu_path),
            )
        )
    )
    entries["cpu_gates"] = "cpu-roles.json"
    performance = run / "benchmark/evidence.json"
    wrapper = json.loads(performance.read_text())
    wrapper["authoritative_manifest_sha256"] = sha(manifest_path)
    performance.write_text(json.dumps(wrapper))
    bundle = prepare_bundle(repo, run, tmp_path / "bundle", entries)
    assert (bundle / "manifest.json").exists()
    import os
    from pathlib import Path

    from golden_gen.release_adapter import verify_bundle
    from golden_gen.release_cpu import validate_cpu_roles

    binary = os.environ.get("GOLDEN_TRANSPORT_TEST_BINARY")
    if not binary:
        pytest.skip("CPU transport binary required")
    verified_bundle = verify_bundle(repo, bundle, tmp_path / "cache", Path(binary))
    assert verified_bundle["verdict"] == "PASS"
    assert verified_bundle["manifest"]["source"] == measured
    assert verified_bundle["result"]["evaluator_source"] == evaluator
    original = roles.read_bytes()
    broken = json.loads(original)
    del broken["supervision"]
    roles.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="CPU role"):
        validate_cpu_roles(roles, measured, evaluator)
    broken = json.loads(original)
    broken["measurement"], broken["supervision"] = broken["supervision"], broken["measurement"]
    roles.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="CPU execution provenance"):
        validate_cpu_roles(roles, measured, evaluator)
    roles.write_bytes(original)
    import hashlib

    from golden_gen.layered_artifacts import source_identity
    from golden_gen.layered_publication import publish
    from golden_gen.release_adapter import REPORT, git, render_report

    class StopBeforeWrites:
        calls = 0

        def ensure_absent(self):
            self.calls += 1
            raise RuntimeError("FAKE_STOP_BEFORE_WRITES")

    fake = StopBeforeWrites()
    monkeypatch.delenv("VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH", raising=False)
    with pytest.raises(ValueError, match="independent user authority"):
        publish(
            repo,
            bundle,
            tmp_path / "denied-cache",
            Path(binary),
            tmp_path / "no-review",
            evaluator["commit"],
            fake,
        )
    assert fake.calls == 0
    report = repo / REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(verified_bundle))
    git(repo, "add", REPORT)
    git(repo, "commit", "-qm", "synthetic supervised evidence-only report")
    candidate = source_identity(repo)
    review_root = tmp_path / "reviews"
    review_root.mkdir()
    axes = {}
    for axis in ("standards", "spec"):
        path = review_root / (axis + ".md")
        path.write_text("Synthetic CPU review fixture, not a real review.\n")
        axes[axis] = dict(unresolved_findings=[], report=dict(path=path.name, sha256=sha(path)))
    review = review_root / "review.json"
    review.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                kind="full_candidate_review",
                base=evaluator["commit"],
                candidate=candidate,
                diff_sha256=hashlib.sha256(
                    git(repo, "diff", "--binary", f"{evaluator['commit']}...{candidate['commit']}")
                ).hexdigest(),
                axes=axes,
            )
        )
    )
    monkeypatch.setenv("VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH", "goldens-v0.2")
    with pytest.raises(RuntimeError, match="FAKE_STOP_BEFORE_WRITES"):
        publish(
            repo, bundle, tmp_path / "fake-cache", Path(binary), review, evaluator["commit"], fake
        )
    assert fake.calls == 1
