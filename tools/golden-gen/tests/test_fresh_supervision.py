"""Fresh evidence must prove every owner; it cannot reinterpret retained receipts."""

import json

import pytest

from golden_gen.layered_artifacts import manifest_artifact_closure, sha, verify_marker, write_marker
from golden_gen.layered_manifest import evaluate_manifest
from tests.fresh_evidence_fixture import fresh_release_inputs


def test_fresh_manifest_recomputes_every_owner_without_legacy_ledger(tmp_path):
    repo, run, measured, evaluator, entries, _ = fresh_release_inputs(tmp_path)
    path = run / entries["authoritative_manifest"]
    result = evaluate_manifest(repo, path, authoritative=True)
    assert result["verdict"] == "PASS", result.get("reasons")
    assert result["accepting"] is True
    assert result["source"] == measured and result["evaluator_source"] == evaluator
    assert "retained_owner_ledger" not in result
    closure = manifest_artifact_closure(path)
    assert not any(p.name == "retained-ledger.json" for p in closure)
    assert any(p.name == "worker.json" for p in closure)
    manifest = json.loads(path.read_text())
    (run / entries["authoritative_marker"]).rename(run / "original-marker.json")
    output = run / "fresh-result.json"
    output.write_text(json.dumps(result))
    marker = write_marker(
        run,
        "authoritative",
        evaluator,
        [output, path],
        run / manifest["calibration_marker"]["path"],
    )
    assert verify_marker(marker, evaluator)["measurement_source"] == measured
    from golden_gen.layered_workflow import assemble_manifest

    assembled = assemble_manifest(
        repo,
        run,
        authoritative=True,
        supervision_policy=run / manifest["supervision_policy"]["path"],
        calibration={
            name: run / manifest[name]["path"]
            for name in (
                "calibration_evidence",
                "calibration_manifest",
                "calibration_marker",
                "fault_evidence",
            )
        },
    )
    assert assembled["schema_version"] == 3
    assert "retained_owner_ledger" not in assembled
    rebuilt = run / "fresh-assembled.json"
    rebuilt.write_text(json.dumps(assembled))
    assert evaluate_manifest(repo, rebuilt, authoritative=True)["verdict"] == "PASS"
    from golden_gen.layered_manifest import LayeredManifest

    with pytest.raises(ValueError, match="cannot retain"):
        LayeredManifest.model_validate(
            dict(manifest, retained_owner_ledger=manifest["supervision_policy"])
        )
    with pytest.raises(ValueError, match="requires its original ledger"):
        LayeredManifest.model_validate(dict(manifest, schema_version=2))


@pytest.mark.parametrize("fault", ["legacy_guard", "missing_worker", "wrong_source", "missing_ram"])
def test_fresh_owner_cannot_bypass_supervision(tmp_path, fault):
    repo, run, _, _, entries, owners = fresh_release_inputs(tmp_path)
    path = run / entries["authoritative_manifest"]
    guard_path, receipt_path, entry, variant = owners[0]
    if fault == "missing_worker":
        receipt_path.with_name("worker.json").unlink()
    else:
        guard = json.loads(guard_path.read_text())
        if fault == "legacy_guard":
            guard["schema_version"] = 1
        elif fault == "wrong_source":
            guard["supervision_source"]["commit"] = "a" * 40
        else:
            guard["fast_ram_samples"] = []
        guard_path.write_text(json.dumps(guard))
        receipt = json.loads(receipt_path.read_text())
        receipt["guard"]["sha256"] = sha(guard_path)
        receipt_path.write_text(json.dumps(receipt))
        manifest = json.loads(path.read_text())
        # Update the hash to ensure rejection comes from supervision validation.
        for category in ("captures", "operator_checks", "behavior_checks"):
            for actual in manifest.get(category, []):
                if actual == entry:
                    actual[variant + "_receipt"]["sha256"] = sha(receipt_path)
        path.write_text(json.dumps(manifest))
    result = evaluate_manifest(repo, path, authoritative=True)
    assert result["verdict"] == "INVALID" and result["accepting"] is False
    expected = {
        "legacy_guard": "new owner lacks supervised guard identity",
        "missing_worker": "missing or unsafe original worker metadata",
        "wrong_source": "new owner lacks supervised guard identity",
        "missing_ram": "invalid independent RAM sample count",
    }
    assert expected[fault] in result["reasons"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("retained_owner_count", 1),
        ("retained_owner_count", False),
        ("retained_ledger_sha256", "a" * 64),
        ("retained_index_end_inclusive", -1),
    ],
)
def test_fresh_policy_cannot_smuggle_retained_authority(tmp_path, monkeypatch, field, value):
    from pathlib import Path

    import golden_gen.supervision as supervision
    from golden_gen.layered_release import POLICY_PATH, REGISTRY_PATH

    repo = Path(__file__).resolve().parents[3]
    policy, _ = supervision.load_policy(repo)
    policy = dict(
        policy,
        protocol="layered-supervision-policy-v2",
        schema_version=2,
        policy_id="bounded-telemetry-fresh-v1",
        retained_owner_count=0,
    )
    for key in ("retained_ledger_sha256", "retained_index_start", "retained_index_end_inclusive"):
        policy.pop(key, None)

    def definition(_repo, path):
        if path == supervision.POLICY_PATH:
            return policy, "c" * 64
        return {}, policy["registry_sha256" if path == REGISTRY_PATH else "numerical_policy_sha256"]

    assert POLICY_PATH != supervision.POLICY_PATH
    monkeypatch.setattr(supervision, "definition_document", definition)
    assert supervision.load_policy(tmp_path)[0]["retained_owner_count"] == 0
    policy[field] = value
    with pytest.raises(ValueError, match="cannot retain"):
        supervision.load_policy(tmp_path)
