import json
from pathlib import Path

import pytest


def test_new_marker_cannot_accept_pending_results_or_changed_outputs(tmp_path: Path) -> None:
    from golden_gen.layered_artifacts import verify_marker, write_marker

    output = tmp_path / "observation.json"
    output.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                accepting=False,
                verdict="INVALID",
                observation_complete=True,
            )
        )
    )
    source = dict(commit="a" * 40, tree="b" * 40)
    with pytest.raises(ValueError):
        write_marker(tmp_path, "authoritative", source, [output])
    marker = write_marker(tmp_path, "observation", source, [output])
    assert json.loads(marker.read_text())["protocol"] == "layered-accuracy-v1"
    assert verify_marker(marker, source)["accepting"] is False
    output.write_text("{}")
    with pytest.raises(ValueError):
        verify_marker(marker, source)
