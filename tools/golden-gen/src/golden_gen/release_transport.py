"""Schema5 byte transport; installation alone never asserts release acceptance."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zlib
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from golden_gen.assets import _rename_no_replace
from golden_gen.layered_artifacts import sha

MANIFEST_LIMIT = 16 * 1024**2
ARTIFACT_LIMIT = 50_000
FILE_LIMIT = 8 * 1024**3 - 1
TOTAL_LIMIT = 1024**4
# GitHub's documented per-asset limit, checked 2026-09-08: strictly under 2 GiB.
UPLOAD_LIMIT = 2 * 1024**3
ARCHIVE = "goldens-v0.2.tar.gz"
ASSETS = {"manifest.json", ARCHIVE}
HEX64 = r"^[0-9a-f]{64}$"
HEX40 = r"^[0-9a-f]{40}$"


def logical_path(value: str) -> str:
    if (
        not value
        or not value.isascii()
        or len(value) > 512
        or "\\" in value
        or "\0" in value
        or any(p in ("", ".", "..", ".git") for p in value.split("/"))
    ):
        raise ValueError("unsafe logical artifact path")
    return value


def regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact is not a regular file")


def no_symlinks(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink in artifact or cache path")


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    commit: str = Field(pattern=HEX40)
    tree: str = Field(pattern=HEX40)


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    logical_path: str
    filename: str = Field(pattern=r"^artifact-[0-9]{6}\.bin$")
    sha256: str = Field(pattern=HEX64)
    size_bytes: int = Field(ge=0, le=FILE_LIMIT)

    @model_validator(mode="after")
    def safe(self) -> Artifact:
        logical_path(self.logical_path)
        return self

    @classmethod
    def inventory(cls, root: Path, paths: list[Path]) -> list[Artifact]:
        relative = sorted({p.relative_to(root).as_posix() for p in paths})
        result = []
        for index, name in enumerate(relative):
            logical_path(name)
            path = root / name
            regular(path)
            if any(p.is_symlink() for p in path.parents if p != root.parent):
                raise ValueError("symlink in artifact path")
            result.append(
                cls(
                    logical_path=name,
                    filename=f"artifact-{index:06}.bin",
                    sha256=sha(path),
                    size_bytes=path.stat().st_size,
                )
            )
        return result


class ArchiveIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    filename: Literal["goldens-v0.2.tar.gz"]
    sha256: str = Field(pattern=HEX64)


class Entrypoints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    authoritative_manifest: str
    authoritative_marker: str
    performance: str
    cpu_gates: str


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[5] = 5
    protocol: Literal["layered-accuracy-v1"] = "layered-accuracy-v1"
    product_version: Literal["v0.2.0"] = "v0.2.0"
    golden_version: Literal["goldens-v0.2"] = "goldens-v0.2"
    source: Source
    registry_sha256: str = Field(pattern=HEX64)
    policy_sha256: str = Field(pattern=HEX64)
    definition_index_blob: str = Field(pattern=HEX40)
    entrypoints: Entrypoints
    counts: dict[str, int]
    artifacts: list[Artifact] = Field(min_length=1, max_length=ARTIFACT_LIMIT)
    archive: ArchiveIdentity | None = None

    @model_validator(mode="after")
    def exact_inventory(self) -> ReleaseManifest:
        names = [a.logical_path for a in self.artifacts]
        if names != sorted(set(names)):
            raise ValueError("logical inventory must be sorted and unique")
        files = set(names)
        for index, item in enumerate(self.artifacts):
            if item.filename != f"artifact-{index:06}.bin":
                raise ValueError("noncanonical flat inventory filename")
            if any(
                "/".join(item.logical_path.split("/")[:n]) in files
                for n in range(1, len(item.logical_path.split("/")))
            ):
                raise ValueError("file/directory logical path conflict")
        if sum(a.size_bytes for a in self.artifacts) > TOTAL_LIMIT:
            raise ValueError("extracted size limit exceeded")
        if any(p not in files for p in self.entrypoints.model_dump().values()):
            raise ValueError("missing release entrypoint")
        if any(type(n) is not int or not 0 <= n <= 2**64 - 1 for n in self.counts.values()):
            raise ValueError("invalid release counts")
        return self


def read_manifest(path: Path) -> ReleaseManifest:
    regular(path)
    if path.stat().st_size > MANIFEST_LIMIT:
        raise ValueError("release manifest exceeds size limit")
    # Required wire keys cannot be supplied by producer-side model defaults.
    with path.open("rb") as stream:
        raw = stream.read(MANIFEST_LIMIT + 1)
    if len(raw) > MANIFEST_LIMIT:
        raise ValueError("release manifest exceeds size limit")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate release manifest key")
            result[key] = value
        return result

    data = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(data, dict) or set(data) != set(ReleaseManifest.model_fields):
        raise ValueError("release manifest missing or unknown wire fields")
    if type(data["schema_version"]) is not int:
        raise ValueError("release schema_version must be a wire integer")
    value = ReleaseManifest.model_validate(data)
    if value.archive is None:
        raise ValueError("release manifest has no archive identity")
    return value


def check_upload_size(size: int) -> None:
    if size >= UPLOAD_LIMIT:
        raise ValueError("capacity decision required: GitHub asset must be under 2 GiB")


def _capacity(parent: Path, size: int) -> None:
    if shutil.disk_usage(parent).free < size + MANIFEST_LIMIT:
        raise ValueError("capacity decision required: insufficient disk space")


def _header(item: Artifact) -> bytes:
    info = tarfile.TarInfo(item.filename)
    info.size = item.size_bytes
    info.mode = 0o644
    return info.tobuf(format=tarfile.USTAR_FORMAT)


def _verify_file(path: Path, item: Artifact) -> None:
    regular(path)
    if path.stat().st_size != item.size_bytes or sha(path) != item.sha256:
        raise ValueError("artifact size/checksum mismatch")


def build_bundle(root: Path, manifest: ReleaseManifest, destination: Path) -> Path:
    """Produce transport bytes only; the release adapter must first validate semantics."""
    no_symlinks(root)
    no_symlinks(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    _capacity(destination.parent, 2 * sum(a.size_bytes + 1024 for a in manifest.artifacts))
    staging = Path(tempfile.mkdtemp(prefix=".schema5-", dir=destination.parent))
    try:
        archive = staging / ARCHIVE
        with archive.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                for item in manifest.artifacts:
                    path = root / item.logical_path
                    no_symlinks(path)
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError("artifact escapes evidence root")
                    _verify_file(path, item)
                    compressed.write(_header(item))
                    copied = 0
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        while chunk := source.read(64 * 1024):
                            copied += len(chunk)
                            if copied > item.size_bytes:
                                raise ValueError("artifact grew during archive streaming")
                            digest.update(chunk)
                            compressed.write(chunk)
                    if copied != item.size_bytes or digest.hexdigest() != item.sha256:
                        raise ValueError("artifact changed during archive streaming")
                    compressed.write(bytes((-item.size_bytes) % 512))
                    _verify_file(path, item)
                compressed.write(bytes(1024))
            raw.flush()
            os.fsync(raw.fileno())
        check_upload_size(archive.stat().st_size)
        bound = manifest.model_copy(
            update={"archive": ArchiveIdentity(filename="goldens-v0.2.tar.gz", sha256=sha(archive))}
        )
        payload = (bound.model_dump_json(indent=2) + "\n").encode()
        if len(payload) > MANIFEST_LIMIT:
            raise ValueError("release manifest exceeds size limit")
        (staging / "manifest.json").write_bytes(payload)
        _rename_no_replace(staging, destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _inflate(path: Path) -> Generator[bytes, None, None]:
    decoder = zlib.decompressobj(31)
    with path.open("rb") as stream:
        if stream.read(10) != bytes.fromhex("1f8b08000000000002ff"):
            raise ValueError("noncanonical gzip header")
        stream.seek(0)
        pending = b""
        while not decoder.eof:
            pending = pending or stream.read(64 * 1024)
            if not pending:
                raise ValueError("truncated gzip stream")
            output = decoder.decompress(pending, 64 * 1024)
            pending = decoder.unconsumed_tail
            yield output
        if decoder.unused_data or pending or stream.read(1):
            raise ValueError("trailing compressed content")


class _Reader:
    def __init__(self, path: Path):
        self.chunks = _inflate(path)
        self.buffer = b""

    def read(self, size: int) -> bytes:
        while len(self.buffer) < size:
            chunk = next(self.chunks, None)
            if chunk is None:
                break
            self.buffer += chunk
        result, self.buffer = self.buffer[:size], self.buffer[size:]
        return result


def verify_install(installed: Path, manifest_bytes: bytes, manifest: ReleaseManifest) -> None:
    no_symlinks(installed)
    regular(installed / "manifest.json")
    if (installed / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("installed manifest differs")
    expected = {"manifest.json", *("evidence/" + a.logical_path for a in manifest.artifacts)}
    found = set()
    directories = {"evidence"}
    for artifact in manifest.artifacts:
        directories.update(
            p.as_posix()
            for p in Path("evidence", artifact.logical_path).parents
            if p.as_posix() != "."
        )
    for path in installed.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in immutable install")
        if path.is_file():
            found.add(path.relative_to(installed).as_posix())
        elif path.is_dir() and path.relative_to(installed).as_posix() not in directories:
            raise ValueError("unexpected directory in immutable install")
        elif not path.is_dir():
            raise ValueError("special entry in immutable install")
    if found != expected:
        raise ValueError("immutable inventory mismatch")
    for item in manifest.artifacts:
        _verify_file(installed / "evidence" / item.logical_path, item)


def install_transport(bundle: Path, cache: Path) -> Path:
    """Strict transport verification; caller must separately evaluate release semantics."""
    no_symlinks(bundle)
    no_symlinks(cache)
    if {p.name for p in bundle.iterdir()} != ASSETS:
        raise ValueError("release must contain exactly two assets")
    manifest = read_manifest(bundle / "manifest.json")
    assert manifest.archive is not None
    archive = bundle / ARCHIVE
    regular(archive)
    check_upload_size(archive.stat().st_size)
    if sha(archive) != manifest.archive.sha256:
        raise ValueError("archive checksum mismatch")
    parent = cache / "goldens-v0.2"
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or cache.is_symlink():
        raise ValueError("symlink cache root")
    final = parent / manifest.archive.sha256
    payload = (bundle / "manifest.json").read_bytes()
    if final.exists() or final.is_symlink():
        if final.is_symlink():
            raise ValueError("symlink immutable install")
        verify_install(final, payload, manifest)
        return final
    _capacity(parent, sum(a.size_bytes for a in manifest.artifacts))
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=parent))
    reader = _Reader(archive)
    try:
        for item in manifest.artifacts:
            if reader.read(512) != _header(item):
                raise ValueError("noncanonical or unexpected USTAR header")
            target = staging / "evidence" / item.logical_path
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            remaining = item.size_bytes
            with target.open("xb") as stream:
                while remaining:
                    chunk = reader.read(min(remaining, 64 * 1024))
                    if not chunk:
                        raise ValueError("truncated archive payload")
                    remaining -= len(chunk)
                    digest.update(chunk)
                    stream.write(chunk)
            if digest.hexdigest() != item.sha256:
                raise ValueError("extracted artifact checksum mismatch")
            padding = (-item.size_bytes) % 512
            if reader.read(padding) != bytes(padding):
                raise ValueError("nonzero USTAR payload padding")
        tail = reader.read(1024 * 1024 + 1)
        if len(tail) < 1024 or len(tail) > 1024 * 1024 or len(tail) % 512 or any(tail):
            raise ValueError("invalid USTAR end markers or trailing content")
        (staging / "manifest.json").write_bytes(payload)
        verify_install(staging, payload, manifest)
        try:
            _rename_no_replace(staging, final)
        except FileExistsError:
            verify_install(final, payload, manifest)
        return final
    finally:
        reader.chunks.close()
        if staging.exists():
            shutil.rmtree(staging)
