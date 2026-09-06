"""Atomic, protocol-bound IO for the new workflow; legacy markers are never inputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from golden_gen.layered_accuracy import PROTOCOL


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(repo: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repo.resolve() or git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("layered measurements require the exact clean repository root")
    return dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"layered artifact already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".layered-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bound_file(root: Path, record: dict[str, Any]) -> Path:
    relative = PurePosixPath(record["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("unsafe layered artifact path")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("layered artifact is not a confined regular file")
    if sha(path) != record["sha256"]:
        raise ValueError("layered artifact checksum mismatch")
    return path


def write_marker(
    root: Path,
    stage: str,
    source: dict[str, str],
    outputs: list[Path],
    predecessor: Path | None = None,
) -> Path:
    if stage not in ("observation", "authoritative") or not outputs:
        raise ValueError("unsupported or empty layered stage")
    result = json.loads(outputs[0].read_text())
    if result.get("protocol") != PROTOCOL or result.get("schema_version") != 1:
        raise ValueError("legacy/unknown result cannot create a layered marker")
    accepting = stage == "authoritative"
    if accepting:
        if result.get("accepting") is not True or result.get("verdict") != "PASS":
            raise ValueError("nonaccepting result cannot create an authoritative success marker")
    elif result.get("accepting") is not False or result.get("observation_complete") is not True:
        raise ValueError("incomplete observation cannot create a completion marker")
    prior = None
    if predecessor is not None:
        verify_marker(predecessor, source)
        prior = dict(path=predecessor.relative_to(root).as_posix(), sha256=sha(predecessor))
    elif accepting:
        raise ValueError("authoritative marker requires a verified predecessor")
    records = [dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in outputs]
    for record in records:
        bound_file(root, record)
    directory = root / "layered-markers"
    directory.mkdir(exist_ok=True, mode=0o700)
    destination = directory / f"{stage}.complete.json"
    atomic_json(
        destination,
        dict(
            protocol=PROTOCOL,
            schema_version=1,
            stage=stage,
            source=source,
            accepting=accepting,
            predecessor=prior,
            outputs=records,
        ),
    )
    return destination


def verify_marker(path: Path, source: dict[str, str]) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    if (
        value.get("protocol") != PROTOCOL
        or value.get("schema_version") != 1
        or value.get("source") != source
        or not value.get("outputs")
    ):
        raise ValueError("invalid or stale layered marker identity")
    root = path.parent.parent
    for record in value["outputs"]:
        bound_file(root, record)
    if value.get("predecessor") is not None:
        verify_marker(bound_file(root, value["predecessor"]), source)
    return value
