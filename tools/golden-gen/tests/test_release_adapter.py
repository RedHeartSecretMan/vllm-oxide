import os
from pathlib import Path

import pytest

from golden_gen.release_adapter import prepare_bundle


def test_pending_release_refuses_before_creating_bundle_or_reading_holdout(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="authoritative"):
        prepare_bundle(
            repo,
            tmp_path,
            tmp_path / "bundle",
            {
                "authoritative_manifest": "sealed.json",
                "authoritative_marker": "layered-markers/authoritative.complete.json",
                "performance": "benchmark/evidence.json",
                "cpu_gates": "cpu/evidence.json",
            },
        )
    assert not (tmp_path / "bundle").exists()


def test_complete_raw_evidence_survives_python_bundle_rust_clean_consumer(tmp_path, monkeypatch):
    from golden_gen.release_adapter import render_report, verify_bundle
    from tests.release_evidence_fixture import complete_release_inputs

    binary = os.environ.get("GOLDEN_TRANSPORT_TEST_BINARY")
    if not binary:
        pytest.skip("CPU transport binary required")
    repo, run, source, entries = complete_release_inputs(tmp_path)
    bundle = prepare_bundle(repo, run, tmp_path / "bundle", entries)
    result = verify_bundle(repo, bundle, tmp_path / "cache", Path(binary))
    assert result["verdict"] == "PASS" and result["accepting"] is True
    assert result["manifest"]["source"] == source
    assert result["manifest"]["counts"]["numerical_cases"] == 6
    report = render_report(result)
    assert result["archive_sha256"] in report
    assert result["manifest_sha256"] in report
    assert "Performance raw samples" in report
    import hashlib
    import json
    import shutil

    from golden_gen.layered_artifacts import sha, source_identity
    from golden_gen.layered_publication import publish
    from golden_gen.release_adapter import REPORT, git

    # Rehashed summary mutations must still fail raw semantic recomputation.
    raw = run / "benchmark/benchmark.json"
    wrapper = run / "benchmark/evidence.json"
    original_raw, original_wrapper = raw.read_bytes(), wrapper.read_bytes()
    changed = json.loads(original_raw)
    changed["workloads"]["canonical_04"]["repetitions"][0]["time_to_first_token_ns"] = [999]
    raw.write_text(json.dumps(changed))
    changed = json.loads(original_wrapper)
    changed["raw"]["sha256"] = sha(raw)
    wrapper.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="latency"):
        prepare_bundle(repo, run, tmp_path / "wrong-performance", entries)
    raw.write_bytes(original_raw)
    wrapper.write_bytes(original_wrapper)
    telemetry = run / "benchmark/canonical_04-repetition-1.telemetry.json"
    original_telemetry = telemetry.read_bytes()
    for key in ("prompt_token_ids", "sampling_params", "cold_prefix_state", "sampled_token_ids"):
        data = json.loads(original_telemetry)
        if key == "prompt_token_ids":
            data["binding"][key] = [[3]]  # Same length, different actual input token.
        elif key == "sampling_params":
            data["binding"][key][0]["temperature"] = 1
        elif key == "cold_prefix_state":
            data["binding"][key] = False
        else:
            data[key][0][0] = 1
        telemetry.write_text(json.dumps(data))
        summary = json.loads(original_raw)
        summary["workloads"]["canonical_04"]["repetitions"][0]["telemetry_artifact_sha256"] = sha(
            telemetry
        )
        raw.write_text(json.dumps(summary))
        wrapper_data = json.loads(original_wrapper)
        wrapper_data["raw"]["sha256"] = sha(raw)
        wrapper_data["telemetry"][0]["sha256"] = sha(telemetry)
        wrapper.write_text(json.dumps(wrapper_data))
        with pytest.raises(ValueError, match="binding|token stream"):
            prepare_bundle(repo, run, tmp_path / ("wrong-" + key), entries)
        telemetry.write_bytes(original_telemetry)
        raw.write_bytes(original_raw)
        wrapper.write_bytes(original_wrapper)
    cpu = run / "cpu/evidence.json"
    original_cpu = cpu.read_bytes()
    changed = json.loads(original_cpu)
    changed["gates"][0]["command"].append("--features=cuda")
    cpu.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="CPU gate command"):
        prepare_bundle(repo, run, tmp_path / "wrong-cpu", entries)
    cpu.write_bytes(original_cpu)

    class FakeTransport:
        def __init__(self, corrupt=False, existing=False):
            self.calls = []
            self.corrupt = corrupt
            self.existing = existing

        def ensure_absent(self):
            self.calls.append("absent")
            if self.existing:
                raise ValueError("remote exists")

        def create_tag(self, candidate):
            self.calls.append("tag")
            self.candidate = candidate

        def create_release(self, notes):
            self.calls.append("release")
            self.notes = notes.read_text()

        def upload(self, bundle):
            self.calls.append("upload")
            self.bundle = bundle

        def readback(self, directory):
            self.calls.append("readback")
            assets = []
            for p in self.bundle.iterdir():
                shutil.copyfile(p, directory / p.name)
                assets.append(dict(name=p.name, size=p.stat().st_size))
            if self.corrupt:
                (directory / "manifest.json").write_text("{}")
            return dict(
                tag=dict(
                    ref="refs/tags/goldens-v0.2", object=dict(type="commit", sha=self.candidate)
                ),
                release=dict(tagName="goldens-v0.2", isDraft=False, body=self.notes, assets=assets),
            )

    report_path = repo / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    git(repo, "add", REPORT)
    git(repo, "commit", "-qm", "synthetic evidence-only report")
    candidate = source_identity(repo)
    base = git(repo, "rev-parse", source["commit"] + "^").decode().strip()
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    axes = {}
    for axis in ("standards", "spec"):
        p = reviews / (axis + ".md")
        p.write_text("Synthetic review fixture; not a real release review.\n")
        axes[axis] = dict(unresolved_findings=[], report=dict(path=p.name, sha256=sha(p)))
    review = reviews / "review.json"
    review.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                kind="full_candidate_review",
                base=base,
                candidate=candidate,
                diff_sha256=hashlib.sha256(
                    git(repo, "diff", "--binary", f"{base}...{candidate['commit']}")
                ).hexdigest(),
                axes=axes,
            )
        )
    )
    monkeypatch.setenv("VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH", "goldens-v0.2")
    for corruption in ("report", "source"):
        changed_path = report_path if corruption == "report" else repo / "unapproved_execution.py"
        changed_path.write_text("unapproved synthetic mutation\n")
        git(repo, "add", str(changed_path.relative_to(repo)))
        git(repo, "commit", "-qm", f"synthetic wrong {corruption}")
        refused = FakeTransport()
        with pytest.raises(ValueError, match="committed report|post-measurement"):
            publish(
                repo,
                bundle,
                tmp_path / f"wrong-{corruption}-cache",
                Path(binary),
                review,
                base,
                refused,
            )
        assert refused.calls == []
        git(repo, "checkout", "--quiet", "--detach", candidate["commit"])
    transport = FakeTransport()
    published = publish(
        repo, bundle, tmp_path / "publication-cache", Path(binary), review, base, transport
    )
    assert published["remote_verified"] is True
    assert transport.calls == ["absent", "tag", "release", "upload", "readback"]
    transport = FakeTransport(corrupt=True)
    with pytest.raises(ValueError, match="downloaded asset bytes"):
        publish(
            repo, bundle, tmp_path / "bad-readback-cache", Path(binary), review, base, transport
        )
    assert transport.calls == ["absent", "tag", "release", "upload", "readback"]
    transport = FakeTransport(existing=True)
    with pytest.raises(ValueError, match="remote exists"):
        publish(
            repo, bundle, tmp_path / "existing-remote-cache", Path(binary), review, base, transport
        )
    assert transport.calls == ["absent"]
