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


def manifest_artifact_closure(manifest_path: Path, seen: set[Path] | None = None) -> list[Path]:
    """Hash the complete declared dependency graph without parsing full logit tensors."""
    visited = set() if seen is None else seen
    identity = manifest_path.resolve()
    if identity in visited:
        raise ValueError("cyclic layered calibration manifest dependency")
    visited.add(identity)
    data = json.loads(manifest_path.read_text())
    if (
        data.get("protocol") != PROTOCOL
        or data.get("schema_version") != 1
        or data.get("purpose") not in ("observation", "authoritative")
    ):
        raise ValueError("invalid manifest in layered artifact closure")
    root = manifest_path.parent
    paths = [manifest_path]
    for entry in [
        *data.get("captures", []),
        *data.get("operator_checks", []),
        *data.get("behavior_checks", []),
    ]:
        for variant in ("primary", "replay", "control", "control_replay"):
            if entry.get(variant) is None:
                continue
            paths.append(bound_file(root, entry[variant]))
            receipt_path = bound_file(root, entry[variant + "_receipt"])
            paths.append(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            paths.append(bound_file(root, receipt["guard"]))
            paths.extend(bound_file(root, ref) for ref in receipt.get("setup_captures", []))
    for name in ("calibration_evidence", "fault_evidence", "calibration_marker"):
        if data.get(name) is not None:
            paths.append(bound_file(root, data[name]))
    if data.get("calibration_manifest") is not None:
        paths.extend(
            manifest_artifact_closure(bound_file(root, data["calibration_manifest"]), visited)
        )
    visited.remove(identity)
    return paths


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
    if (
        result.get("protocol") != PROTOCOL
        or result.get("schema_version") != 1
        or result.get("source") != source
    ):
        raise ValueError("legacy/unknown result cannot create a layered marker")
    accepting = stage == "authoritative"
    if accepting:
        if result.get("accepting") is not True or result.get("verdict") != "PASS":
            raise ValueError("nonaccepting result cannot create an authoritative success marker")
    elif result.get("accepting") is not False or result.get("observation_complete") is not True:
        raise ValueError("incomplete observation cannot create a completion marker")
    prior = None
    if predecessor is not None:
        if not accepting:
            raise ValueError("observation cannot borrow another stage predecessor")
        prior_source = result.get("calibration_source")
        if not isinstance(prior_source, dict):
            raise ValueError("authoritative result lacks approved calibration source")
        previous = verify_marker(predecessor, prior_source)
        if previous["stage"] != "observation":
            raise ValueError("authoritative predecessor must be the approved observation")
        prior = dict(
            path=predecessor.relative_to(root).as_posix(),
            sha256=sha(predecessor),
            source=prior_source,
        )
    elif accepting:
        raise ValueError("authoritative marker requires a verified predecessor")
    complete_outputs = list(outputs)
    if len(outputs) > 1:
        if result.get("manifest_sha256") != sha(outputs[1]):
            raise ValueError("stage result is not bound to its manifest bytes")
        complete_outputs.extend(manifest_artifact_closure(outputs[1]))
    complete_outputs = list(dict.fromkeys(complete_outputs))
    records = [dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in complete_outputs]
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
        or value.get("stage") not in ("observation", "authoritative")
        or type(value.get("accepting")) is not bool
        or value["accepting"] != (value["stage"] == "authoritative")
        or path.is_symlink()
        or path.name != f"{value.get('stage')}.complete.json"
        or path.parent.name != "layered-markers"
    ):
        raise ValueError("invalid or stale layered marker identity")
    root = path.parent.parent
    for record in value["outputs"]:
        bound_file(root, record)
    result = json.loads(bound_file(root, value["outputs"][0]).read_text())
    if (
        result.get("protocol") != PROTOCOL
        or result.get("schema_version") != 1
        or result.get("source") != source
    ):
        raise ValueError("marker result protocol/source differs")
    if len(value["outputs"]) > 1:
        manifest_path = bound_file(root, value["outputs"][1])
        if result.get("manifest_sha256") != sha(manifest_path):
            raise ValueError("marker manifest/result identity mismatch")
        closure = {p.relative_to(root).as_posix() for p in manifest_artifact_closure(manifest_path)}
        closure.add(value["outputs"][0]["path"])
        if (
            len(value["outputs"]) != len(closure)
            or {r["path"] for r in value["outputs"]} != closure
        ):
            raise ValueError("marker omits or adds declared artifact dependencies")
    predecessor = value.get("predecessor")
    if value["stage"] == "observation":
        if (
            predecessor is not None
            or result.get("accepting") is not False
            or result.get("observation_complete") is not True
        ):
            raise ValueError("invalid observation marker/result state")
    else:
        if (
            result.get("accepting") is not True
            or result.get("verdict") != "PASS"
            or not isinstance(predecessor, dict)
        ):
            raise ValueError("invalid authoritative marker/result state")
        prior_source = result.get("calibration_source")
        if not isinstance(prior_source, dict) or predecessor.get("source") != prior_source:
            raise ValueError("predecessor calibration source was relabeled")
        prior_path = bound_file(root, predecessor)
        if json.loads(prior_path.read_text()).get("stage") != "observation":
            raise ValueError("authoritative predecessor is not an observation")
        verify_marker(prior_path, prior_source)
    return value
