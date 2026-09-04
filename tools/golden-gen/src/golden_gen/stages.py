"""Content-bound completion markers for independently resumable release stages."""

from __future__ import annotations

import hashlib
import json
import subprocess
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
    "publish": "bundle",
    "verify": "publish",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _stage_outputs(run_root: Path, stage: str) -> list[Path]:
    if stage == "calibrate":
        return [run_root / "generate/fixtures/manifest.json"]
    root = run_root / stage
    if not root.is_dir():
        raise ValueError(f"stage output directory is missing: {root}")
    outputs = [
        path
        for path in root.rglob("*")
        if path.is_file() and ".venv" not in path.relative_to(root).parts
    ]
    if stage == "authoritative":
        outputs.append(run_root / "generate/fixtures/manifest.json")
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
    with destination.open("xb") as output:
        output.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    return destination
