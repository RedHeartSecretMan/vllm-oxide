from __future__ import annotations

import json

import pytest

from golden_gen.publication import verify_publication


def test_publication_asserts_tag_exact_assets_notes_and_downloaded_bytes(tmp_path):
    publish = tmp_path / "publish"
    publish.mkdir()
    bundle = tmp_path / "bundle/goldens-v0.2"
    bundle.mkdir(parents=True)
    downloads = tmp_path / "verify/downloads"
    downloads.mkdir(parents=True)
    candidate = "a" * 40
    (publish / "remote-tag.txt").write_text(f"{candidate}\trefs/tags/goldens-v0.2\n")
    (publish / "release-notes.md").write_text("measured and final identities\n")
    assets = [{"name": name, "size": 4} for name in ("manifest.json", "goldens-v0.2.tar.gz")]
    for asset in assets:
        (bundle / asset["name"]).write_bytes(b"data")
        (downloads / asset["name"]).write_bytes(b"data")
    release = {
        "tagName": "goldens-v0.2",
        "isDraft": False,
        "assets": assets,
        "body": "measured and final identities\n",
    }
    (publish / "release.json").write_text(json.dumps(release))
    verify_publication(tmp_path, candidate, downloaded=True)
    (downloads / "manifest.json").write_bytes(b"fake")
    with pytest.raises(ValueError, match="downloaded asset checksum"):
        verify_publication(tmp_path, candidate, downloaded=True)
    with pytest.raises(ValueError, match="remote tag"):
        verify_publication(tmp_path, "b" * 40)
    release["assets"].append({"name": "unexpected", "size": 4})
    (publish / "release.json").write_text(json.dumps(release))
    with pytest.raises(ValueError, match="exact asset names"):
        verify_publication(tmp_path, candidate)
