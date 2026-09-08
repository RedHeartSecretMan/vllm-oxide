"""Candidate-bound publication preparation and remote readback assertions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ASSETS = ("goldens-v0.2.tar.gz", "manifest.json")


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare_publication(repo: Path, run_root: Path, candidate: str) -> Path:
    """Reject the superseded publisher before reading evidence or writing notes."""
    raise ValueError(
        "legacy publication is disabled: layered-accuracy-v1 requires a reviewed "
        "L0/L1/L2 publication adapter; historical manifests cannot authorize a new release"
    )


def verify_publication(run_root: Path, candidate: str, *, downloaded: bool = False) -> None:
    """Assert remote identities and, after download, the exact frozen asset bytes."""
    publication = run_root / "publish"
    tag = (publication / "remote-tag.txt").read_text().splitlines()
    if tag != [f"{candidate}\trefs/tags/goldens-v0.2"]:
        raise ValueError("remote tag does not point directly to the frozen candidate")
    release = json.loads((publication / "release.json").read_bytes())
    assets = release.get("assets", [])
    if (
        release.get("tagName") != "goldens-v0.2"
        or release.get("isDraft") is not False
        or sorted(asset.get("name", "") for asset in assets) != list(ASSETS)
    ):
        raise ValueError("remote release tag, draft status, or exact asset names differ")
    notes = (publication / "release-notes.md").read_text()
    if release.get("body") != notes:
        raise ValueError("remote release notes do not name the frozen measured and tagged commits")
    bundle = run_root / "bundle/goldens-v0.2"
    for asset in assets:
        name = asset["name"]
        if asset.get("size") != (bundle / name).stat().st_size:
            raise ValueError(f"remote asset size differs from frozen bundle: {name}")
        if downloaded and _sha(run_root / "verify/downloads" / name) != _sha(bundle / name):
            raise ValueError(f"downloaded asset checksum differs from frozen bundle: {name}")
    if downloaded:
        names = sorted(path.name for path in (run_root / "verify/downloads").iterdir())
        if names != list(ASSETS):
            raise ValueError("downloaded release does not contain exactly the two frozen assets")
