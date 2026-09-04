"""Build the deterministic archive in a golden asset bundle."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from golden_gen.config import ARCHIVE_FILENAME
from golden_gen.schema import ArchiveInfo, FixtureMetadata, Manifest


def _validate_fixture_filename(filename: str) -> None:
    if (
        not filename.isascii()
        or not filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
        or not filename.endswith(".safetensors")
    ):
        raise ValueError(f"unsupported fixture archive path: {filename}")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def build_fixture_archive(
    fixture_dir: Path,
    fixtures: list[FixtureMetadata],
    destination: Path,
) -> ArchiveInfo:
    """Write every declared fixture to one reproducible gzip-compressed USTAR archive."""
    fixture_dir = Path(fixture_dir)
    destination = Path(destination)
    if destination.name != ARCHIVE_FILENAME:
        raise ValueError(f"fixture archive must be named {ARCHIVE_FILENAME}")
    names = [fixture.filename for fixture in fixtures]
    if not names:
        raise ValueError("cannot build an empty fixture archive")
    if len(names) != len(set(names)):
        raise ValueError("duplicate fixture archive entry")
    for name in names:
        _validate_fixture_filename(name)

    declared = set(names)
    discovered = {path.name for path in fixture_dir.glob("*.safetensors") if path.is_file()}
    if discovered != declared:
        missing = sorted(declared - discovered)
        unexpected = sorted(discovered - declared)
        raise ValueError(
            f"fixture archive input mismatch: missing={missing}, unexpected={unexpected}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", dir=destination.parent, delete=False
        ) as raw:
            temporary = Path(raw.name)
            with (
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
                tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive,
            ):
                metadata_by_name = {fixture.filename: fixture for fixture in fixtures}
                for name in sorted(names, key=str.encode):
                    source_path = fixture_dir / name
                    if source_path.is_symlink():
                        raise ValueError(f"fixture source is not a regular file: {name}")
                    actual_sha256 = _sha256_file(source_path)
                    expected_sha256 = metadata_by_name[name].sha256
                    if actual_sha256 != expected_sha256:
                        raise ValueError(
                            f"fixture checksum mismatch for {name}: "
                            f"expected {expected_sha256}, got {actual_sha256}"
                        )
                    info = tarfile.TarInfo(name=name)
                    info.size = source_path.stat().st_size
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source_path.open("rb") as fixture_file:
                        archive.addfile(info, fixture_file)
            raw.flush()
            os.fsync(raw.fileno())
        assert temporary is not None
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return ArchiveInfo(filename=ARCHIVE_FILENAME, sha256=_sha256_file(destination))


def _fixture_id(fixture: FixtureMetadata) -> str:
    return f"{fixture.prompt_id}.{fixture.oracle}"


def _validate_release_coverage(manifest: Manifest) -> None:
    expected_ids = {fixture.fixture_id for fixture in manifest.expected_fixtures}
    generated_ids = {_fixture_id(fixture) for fixture in manifest.fixtures}
    if generated_ids != expected_ids:
        raise ValueError("release bundle requires every and only the expected fixtures")
    expected_calibration = {
        fixture.fixture_id
        for fixture in manifest.expected_fixtures
        if fixture.oracle_role == "baseline"
    }
    if set(manifest.calibrated_fixtures) != expected_calibration:
        raise ValueError("release bundle requires complete baseline calibration")


def publish_release_bundle(fixture_dir: Path, release_dir: Path) -> Path:
    """Create the exact two-file release bundle in a new, independent directory."""
    fixture_dir = Path(fixture_dir)
    release_dir = Path(release_dir)
    manifest_path = fixture_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = Manifest.model_validate_json(manifest_bytes)
    _validate_release_coverage(manifest)
    if os.path.lexists(release_dir):
        raise FileExistsError(f"release bundle destination already exists: {release_dir}")

    release_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_dir.name}.staging-", dir=release_dir.parent))
    try:
        archive_path = staging / ARCHIVE_FILENAME
        archive = build_fixture_archive(fixture_dir, manifest.fixtures, archive_path)
        if archive != manifest.archive:
            raise ValueError(
                "manifest archive identity does not match the deterministic fixture archive"
            )
        staged_manifest = staging / "manifest.json"
        with staged_manifest.open("xb") as output:
            output.write(manifest_bytes)
            output.flush()
            os.fsync(output.fileno())
        if {path.name for path in staging.iterdir()} != {"manifest.json", ARCHIVE_FILENAME}:
            raise ValueError("release bundle must contain exactly two assets")
        staging.rename(release_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return release_dir
