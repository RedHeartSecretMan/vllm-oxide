"""Candidate-bound schema5 publication with an explicit external transport seam."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

from golden_gen.layered_artifacts import bound_file, sha, source_identity
from golden_gen.release_adapter import REPORT, git, render_report, verify_bundle
from golden_gen.release_transport import ASSETS, check_upload_size

REPOSITORY = "RedHeartSecretMan/vllm-oxide"
PUSH_URL = f"https://github.com/{REPOSITORY}.git"
TAG = "goldens-v0.2"


class PublicationTransport(Protocol):
    def ensure_absent(self) -> None: ...
    def create_tag(self, candidate: str) -> None: ...
    def create_release(self, notes: Path) -> None: ...
    def upload(self, bundle: Path) -> None: ...
    def readback(self, directory: Path) -> dict[str, Any]: ...


class GitHubTransport:
    """Real writes occur only after publish() has verified every local prerequisite."""

    def __init__(self, repo: Path | None = None):
        self.repo = repo

    @staticmethod
    def _gh(*args: str) -> bytes:
        return subprocess.check_output(["gh", *args], stderr=subprocess.PIPE)

    def ensure_absent(self) -> None:
        for endpoint in (
            f"repos/{REPOSITORY}/git/ref/tags/{TAG}",
            f"repos/{REPOSITORY}/releases/tags/{TAG}",
        ):
            result = subprocess.run(
                ["gh", "api", "--include", endpoint], capture_output=True, check=False
            )
            if result.returncode == 0 or not result.stdout.startswith(
                (b"HTTP/2.0 404", b"HTTP/2 404", b"HTTP/1.1 404")
            ):
                raise ValueError(
                    "remote tag/release is existing, partial, or absence cannot be proved"
                )

    def create_tag(self, candidate: str) -> None:
        if self.repo is None or re.fullmatch(r"[0-9a-f]{40}", candidate) is None:
            raise ValueError("tag push requires the reviewed repository and full candidate OID")
        # Upload the local evidence-only descendant's objects and only this tag.
        # A refs API call cannot create a ref to an object absent on GitHub.
        subprocess.run(
            ["git", "-C", str(self.repo), "push", "--", PUSH_URL, f"{candidate}:refs/tags/{TAG}"],
            check=True,
            capture_output=True,
        )

    def create_release(self, notes: Path) -> None:
        self._gh(
            "release",
            "create",
            TAG,
            "--repo",
            REPOSITORY,
            "--verify-tag",
            "--title",
            TAG,
            "--notes-file",
            str(notes),
        )

    def upload(self, bundle: Path) -> None:
        self._gh(
            "release",
            "upload",
            TAG,
            "--repo",
            REPOSITORY,
            *(str(bundle / name) for name in sorted(ASSETS)),
        )

    def readback(self, directory: Path) -> dict[str, Any]:
        tag = json.loads(self._gh("api", f"repos/{REPOSITORY}/git/ref/tags/{TAG}"))
        release = json.loads(
            self._gh(
                "release",
                "view",
                TAG,
                "--repo",
                REPOSITORY,
                "--json",
                "tagName,isDraft,assets,body",
            )
        )
        if {a["name"] for a in release["assets"]} != ASSETS or len(release["assets"]) != 2:
            raise ValueError("remote release asset set differs")
        self._gh("release", "download", TAG, "--repo", REPOSITORY, "--dir", str(directory))
        return dict(tag=tag, release=release)


def verify_review(repo: Path, path: Path, base: str, candidate: dict[str, str]) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", base) is None:
        raise ValueError("review Base must be a full commit OID")
    review = json.loads(path.read_text())
    diff = git(repo, "diff", "--binary", f"{base}...{candidate['commit']}")
    if (
        review.get("protocol") != "layered-accuracy-v1"
        or review.get("schema_version") != 1
        or review.get("kind") != "full_candidate_review"
        or review.get("base") != base
        or review.get("candidate") != candidate
        or base == candidate["commit"]
        or not diff
        or review.get("diff_sha256") != hashlib.sha256(diff).hexdigest()
        or set(review.get("axes", {})) != {"standards", "spec"}
    ):
        raise ValueError("fresh full-range review candidate/Base/diff binding missing")
    for axis in review["axes"].values():
        if (
            axis.get("unresolved_findings") != []
            or not bound_file(path.parent, axis["report"]).read_text().strip()
        ):
            raise ValueError("full review has unresolved findings or lacks its original report")


def publish(
    repo: Path,
    bundle: Path,
    cache: Path,
    rust_binary: Path,
    review: Path,
    base: str,
    transport: PublicationTransport,
) -> dict[str, Any]:
    if os.environ.get("VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH") != TAG:
        raise ValueError("publication requires independent user authority for goldens-v0.2")
    candidate = source_identity(repo)
    verified = verify_bundle(repo, bundle, cache, rust_binary)
    measured = verified["manifest"]["source"]
    if candidate == measured:
        raise ValueError("publication requires a reviewed evidence-only report descendant")
    git(repo, "merge-base", "--is-ancestor", base, measured["commit"])
    report = render_report(verified)
    if git(repo, "show", f"{candidate['commit']}:{REPORT}") != report.encode():
        raise ValueError("committed report differs from revalidated evidence and asset hashes")
    verify_review(repo, review, base, candidate)
    for name in ASSETS:
        check_upload_size((bundle / name).stat().st_size)
    if source_identity(repo) != candidate:
        raise ValueError("candidate changed before publication")
    notes = (
        f"Measured commit: https://github.com/{REPOSITORY}/commit/{measured['commit']}\n\n"
        f"Final tagged commit: https://github.com/{REPOSITORY}/tree/{candidate['commit']}\n\n"
        + report
    )
    with tempfile.TemporaryDirectory(prefix="layered-publication-") as directory:
        root = Path(directory)
        notes_path = root / "notes.md"
        notes_path.write_text(notes)
        transport.ensure_absent()
        transport.create_tag(candidate["commit"])
        transport.create_release(notes_path)
        transport.upload(bundle)
        downloads = root / "download"
        downloads.mkdir()
        remote = transport.readback(downloads)
        tag, release = remote["tag"], remote["release"]
        if (
            tag.get("ref") != f"refs/tags/{TAG}"
            or tag.get("object", {}).get("type") != "commit"
            or tag["object"].get("sha") != candidate["commit"]
            or release.get("tagName") != TAG
            or release.get("isDraft") is not False
            or release.get("body") != notes
            or len(release.get("assets", [])) != 2
            or {a["name"] for a in release["assets"]} != ASSETS
        ):
            raise ValueError("remote tag/release/notes/asset identity differs; no automatic repair")
        for asset in release["assets"]:
            name = asset["name"]
            if asset["size"] != (bundle / name).stat().st_size or sha(downloads / name) != sha(
                bundle / name
            ):
                raise ValueError("remote downloaded asset bytes differ; no automatic repair")
        readback = verify_bundle(repo, downloads, root / "verified-readback", rust_binary)
        if (
            readback["manifest_sha256"] != verified["manifest_sha256"]
            or readback["archive_sha256"] != verified["archive_sha256"]
        ):
            raise ValueError("remote clean consumer does not verify frozen assets")
    return dict(
        protocol="layered-accuracy-v1",
        schema_version=5,
        stage="publication",
        candidate=candidate,
        measurement=measured,
        manifest_sha256=verified["manifest_sha256"],
        archive_sha256=verified["archive_sha256"],
        remote_verified=True,
    )
