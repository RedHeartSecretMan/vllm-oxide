import json
from pathlib import Path

import pytest


def test_new_marker_cannot_accept_pending_results_or_changed_outputs(tmp_path: Path) -> None:
    from golden_gen.layered_artifacts import verify_marker, write_marker

    source = dict(commit="a" * 40, tree="b" * 40)
    output = tmp_path / "observation.json"
    output.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                source=source,
                accepting=False,
                verdict="INVALID",
                observation_complete=True,
            )
        )
    )
    with pytest.raises(ValueError):
        write_marker(tmp_path, "authoritative", source, [output])
    marker = write_marker(tmp_path, "observation", source, [output])
    assert json.loads(marker.read_text())["protocol"] == "layered-accuracy-v1"
    assert verify_marker(marker, source)["accepting"] is False
    output.write_text("{}")
    with pytest.raises(ValueError):
        verify_marker(marker, source)


def test_authoritative_marker_keeps_the_approved_calibration_source_not_a_relabel(
    tmp_path: Path,
) -> None:
    from golden_gen.layered_artifacts import verify_marker, write_marker

    old = dict(commit="a" * 40, tree="b" * 40)
    new = dict(commit="c" * 40, tree="d" * 40)
    observation = tmp_path / "observation.json"
    observation.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                source=old,
                observation_complete=True,
                accepting=False,
                verdict="INVALID",
            )
        )
    )
    prior = write_marker(tmp_path, "observation", old, [observation])
    result = tmp_path / "authoritative.json"
    result.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                source=new,
                calibration_source=old,
                accepting=True,
                verdict="PASS",
            )
        )
    )
    final = write_marker(tmp_path, "authoritative", new, [result], prior)
    assert verify_marker(final, new)["predecessor"]["source"] == old
    assert json.loads(prior.read_text())["source"] == old
    corrupted = json.loads(final.read_text())
    corrupted["stage"] = "observation"
    final.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError):
        verify_marker(final, new)


def test_marker_binds_transitive_capture_receipt_and_guard_bytes(tmp_path: Path) -> None:
    from golden_gen.layered_artifacts import sha, verify_marker, write_marker

    def save(name: str, data: dict) -> dict:
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return dict(path=name, sha256=sha(path))

    source = dict(commit="a" * 40, tree="b" * 40)
    raw = save("raw.json", dict(logits=[1]))
    guard = save("guard.json", dict(child_returncode=0))
    receipt = save("receipt.json", dict(guard=guard, setup_captures=[]))
    manifest = save(
        "manifest.json",
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            source=source,
            purpose="observation",
            captures=[dict(primary=raw, primary_receipt=receipt)],
        ),
    )
    report = save(
        "result.json",
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            source=source,
            manifest_sha256=manifest["sha256"],
            accepting=False,
            observation_complete=True,
            verdict="INVALID",
        ),
    )
    marker = write_marker(
        tmp_path, "observation", source, [tmp_path / report["path"], tmp_path / manifest["path"]]
    )
    assert {x["path"] for x in verify_marker(marker, source)["outputs"]} == {
        "result.json",
        "manifest.json",
        "raw.json",
        "receipt.json",
        "guard.json",
    }
    (tmp_path / "raw.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_marker(marker, source)
