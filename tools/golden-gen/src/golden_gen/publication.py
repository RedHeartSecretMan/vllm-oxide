"""Candidate-bound publication preparation and remote readback assertions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from golden_gen.schema import Manifest

ASSETS = ("goldens-v0.2.tar.gz", "manifest.json")
REPORT = "docs/releases/goldens-v0.2.md"


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare_publication(repo: Path, run_root: Path, candidate: str) -> Path:
    """Require the exact clean evidence-only candidate before any tag/release write."""
    if _git(repo, "rev-parse", "HEAD").decode().strip() != candidate:
        raise ValueError("publication HEAD differs from the frozen candidate")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("publication candidate worktree is dirty")
    bundle = run_root / "bundle/goldens-v0.2"
    manifest = Manifest.from_json(bundle / "manifest.json")
    measurements = [
        value.split(":", 1)[1]
        for value in manifest.tolerance_policy.evidence
        if value.startswith("measurement-commit:")
    ]
    trees = [
        value.split(":", 1)[1]
        for value in manifest.tolerance_policy.evidence
        if value.startswith("measurement-tree:")
    ]
    if len(measurements) != 1 or len(trees) != 1:
        raise ValueError("manifest does not bind one measurement commit and tree")
    measured = measurements[0]
    if _git(repo, "rev-parse", f"{measured}^{{tree}}").decode().strip() != trees[0]:
        raise ValueError("measurement tree differs from its commit")
    _git(repo, "merge-base", "--is-ancestor", measured, candidate)
    changed = set(
        _git(repo, "diff", "--name-only", "-z", f"{measured}..{candidate}").split(b"\0")
    ) - {b""}
    if changed != {REPORT.encode()}:
        raise ValueError("final candidate must change only the committed release evidence report")
    report = (run_root / "report/goldens-v0.2.md").read_bytes()
    if _git(repo, "show", f"{candidate}:{REPORT}") != report:
        raise ValueError("committed report differs from the frozen stage report")
    for name in ASSETS:
        digest = _sha(bundle / name)
        if f"- `{name}`: `{digest}`".encode() not in report:
            raise ValueError(f"report does not bind the final asset bytes: {name}")
    if _sha(bundle / ASSETS[0]) != manifest.archive.sha256:
        raise ValueError("bundle archive does not match the manifest")
    benchmark = json.loads((run_root / "benchmark/benchmark.json").read_bytes())
    if benchmark["measurement_commit"] != measured or benchmark["measurement_tree"] != trees[0]:
        raise ValueError("benchmark and approved manifest measure different source identities")
    notes = run_root / "publish/release-notes.md"
    repository = "https://github.com/RedHeartSecretMan/vllm-oxide"
    with notes.open("xb") as output:
        output.write(
            (
                f"Measured commit: [{measured}]({repository}/commit/{measured})\n\n"
                f"Final tagged commit: [{candidate}]({repository}/tree/{candidate})\n\n"
            ).encode()
            + report
        )
    return notes


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
