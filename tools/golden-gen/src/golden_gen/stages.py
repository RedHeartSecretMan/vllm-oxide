"""Content-bound completion markers for independently resumable release stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

_PREDECESSOR = {
    "env": None,
    "generate": "env",
    "calibrate": "generate",
    "observe": "calibrate",
    "authoritative": "observe",
    "benchmark": "authoritative",
    "report": "benchmark",
    "bundle": "report",
    "verify-local": "bundle",
    "publish": "verify-local",
    "verify": "publish",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _stage_outputs(run_root: Path, stage: str) -> list[Path]:
    root = run_root / stage
    if not root.is_dir():
        raise ValueError(f"stage output directory is missing: {root}")
    outputs = [
        path
        for path in root.rglob("*")
        if path.is_file() and ".venv" not in path.relative_to(root).parts
    ]
    return sorted(
        outputs,
        key=lambda path: path.relative_to(run_root).as_posix().encode(),
    )


def write_stage_marker(run_root: Path, stage: str, repo_root: Path) -> Path:
    run_root = Path(run_root).resolve()
    approved = Path("/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts")
    try:
        relative = run_root.relative_to(approved)
    except ValueError as error:
        raise ValueError("stage marker run root is outside the ticket artifact root") from error
    if not relative.parts or stage not in _PREDECESSOR:
        raise ValueError("unknown stage or non-specific ticket run root")
    markers = run_root / "markers"
    markers.mkdir(mode=0o700, exist_ok=True)
    destination = markers / f"{stage}.complete.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"stage marker already exists: {destination}")
    predecessor = _PREDECESSOR[stage]
    predecessor_sha256 = None
    if predecessor is not None:
        verify_stage_marker(run_root, predecessor, repo_root)
        predecessor_path = markers / f"{predecessor}.complete.json"
        if not predecessor_path.is_file():
            raise ValueError(f"predecessor marker is missing: {predecessor}")
        predecessor_sha256 = _sha256(predecessor_path)
    outputs = _stage_outputs(run_root, stage)
    if not outputs:
        raise ValueError(f"stage {stage} produced no auditable files")
    record = {
        "schema_version": 1,
        "stage": stage,
        "completed_at": datetime.now(UTC).isoformat(),
        "generator_commit": subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "generator_tree": subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "predecessor_marker_sha256": predecessor_sha256,
        "outputs": [
            {
                "path": path.relative_to(run_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{stage}.", suffix=".staging", dir=markers, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
        temporary = None
        directory = os.open(markers, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def verify_stage_marker(run_root: Path, stage: str, repo_root: Path) -> Path:
    """Rehash a stage and its predecessor marker before successor execution."""
    run_root = Path(run_root).resolve(strict=True)
    approved = Path("/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts")
    try:
        relative = run_root.relative_to(approved)
    except ValueError as error:
        raise ValueError("stage marker run root is outside the ticket artifact root") from error
    if not relative.parts or stage not in _PREDECESSOR:
        raise ValueError("unknown stage or non-specific ticket run root")
    marker = run_root / "markers" / f"{stage}.complete.json"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"stage marker is missing or unsafe: {stage}")
    record = json.loads(marker.read_bytes())
    if record.get("schema_version") != 1 or record.get("stage") != stage:
        raise ValueError(f"stage marker schema or identity is invalid: {stage}")
    commit = record.get("generator_commit")
    tree = record.get("generator_tree")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in (commit, tree)
    ):
        raise ValueError(f"stage marker has malformed Git identities: {stage}")
    resolved_tree = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", f"{commit}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved_tree != tree:
        raise ValueError(f"stage marker commit/tree identity is invalid: {stage}")
    subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=True,
        capture_output=True,
    )
    changes = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", "-z", f"{commit}..HEAD"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    allowed = {
        b"",
        b"CONTEXT.md",
        b".dag/definition-index.json",
        b".dag/definitions/v0.2.0-github.json",
        b"docs/releases/goldens-v0.2-calibration-observation.json",
        b"docs/releases/goldens-v0.2.md",
    }
    if any(
        path not in allowed and not (path.startswith(b"docs/adr/") and path.endswith(b".md"))
        for path in changes
    ):
        raise ValueError(f"executable inputs changed since completed stage: {stage}")
    predecessor = _PREDECESSOR[stage]
    expected_predecessor_sha = None
    if predecessor is not None:
        verify_stage_marker(run_root, predecessor, repo_root)
        predecessor_path = run_root / "markers" / f"{predecessor}.complete.json"
        if not predecessor_path.is_file() or predecessor_path.is_symlink():
            raise ValueError(f"predecessor marker is missing or unsafe: {predecessor}")
        expected_predecessor_sha = _sha256(predecessor_path)
    if record.get("predecessor_marker_sha256") != expected_predecessor_sha:
        raise ValueError(f"stage marker predecessor identity is invalid: {stage}")
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"stage marker has no output identities: {stage}")
    seen: set[str] = set()
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(output.get("path"), str):
            raise ValueError(f"stage marker output record is malformed: {stage}")
        relative_path = output["path"]
        if relative_path in seen:
            raise ValueError(f"stage marker output path is duplicated: {relative_path}")
        seen.add(relative_path)
        raw_path = run_root / relative_path
        if raw_path.is_symlink():
            raise ValueError(f"stage marker output is an unsafe symlink: {relative_path}")
        path = raw_path.resolve(strict=True)
        try:
            path.relative_to(run_root)
        except ValueError as error:
            raise ValueError(f"stage marker output escapes run root: {relative_path}") from error
        if not path.is_file():
            raise ValueError(f"stage marker output is missing: {relative_path}")
        if path.stat().st_size != output.get("size") or _sha256(path) != output.get("sha256"):
            raise ValueError(f"stage marker output identity changed: {relative_path}")
    actual = {path.relative_to(run_root).as_posix() for path in _stage_outputs(run_root, stage)}
    if seen != actual:
        raise ValueError(f"stage marker output set changed: {stage}")
    return marker
