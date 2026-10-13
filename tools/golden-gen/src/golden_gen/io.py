"""Save and load fixture data using safetensors."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file, save


def _open_fixture_directory(path: Path, *, create: bool = False) -> int:
    """Traverse directory descriptors without resolving links or '..'."""
    path = Path(path)
    if ".." in path.parts or any("\\" in part or ":" in part for part in path.parts):
        raise ValueError("fixture paths must not contain traversal or platform-specific separators")
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise RuntimeError("safe fixture output requires no-follow directory descriptor support")
    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def prepare_fixture_directory(path: Path) -> None:
    """Create an output directory without following any linked ancestor."""
    descriptor = _open_fixture_directory(path, create=True)
    os.close(descriptor)


def require_regular_fixture_target(path: Path) -> None:
    """Permit fresh targets or an ordinary, singly linked idempotent retry target."""
    path = Path(path)
    descriptor = _open_fixture_directory(path.parent)
    try:
        try:
            status = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError(f"fixture target must be a regular non-linked file: {path}")
    finally:
        os.close(descriptor)


def _publish_fixture_no_replace(descriptor: int, source: str, destination: str) -> None:
    """Move a complete fixture atomically without replacing any destination entry."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS, "renameat2 is required for atomic fixture publication"
        ) from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(descriptor, os.fsencode(source), descriptor, os.fsencode(destination), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _save_fixture_bytes(path: Path, contents: bytes) -> str:
    """Publish a complete fixture exclusively; identical retries never overwrite."""
    descriptor = _open_fixture_directory(path.parent)
    temporary_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
    status: os.stat_result | None = None
    destination: int | None = None
    owns_destination = False
    try:
        destination = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode=0o600,
            dir_fd=descriptor,
        )
        # Establish close protection immediately after acquiring the raw FD.
        owns_destination = True
        status = os.fstat(destination)
        output = os.fdopen(destination, "wb")
        owns_destination = False
        with output:
            output.write(contents)
        # RENAME_NOREPLACE atomically moves the complete inode to its final name.
        # Even interruption immediately afterwards leaves one link and permits retry.
        try:
            _publish_fixture_no_replace(descriptor, temporary_name, path.name)
        except FileExistsError:
            source = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
            )
            try:
                existing = os.fdopen(source, "rb")
            except BaseException:
                with suppress(OSError):
                    os.close(source)
                raise
            with existing:
                existing_status = os.fstat(existing.fileno())
                if not stat.S_ISREG(existing_status.st_mode) or existing_status.st_nlink != 1:
                    raise ValueError(
                        f"fixture target must be a regular non-linked file: {path}"
                    ) from None
                if existing.read() != contents:
                    raise FileExistsError(
                        f"refusing to overwrite different fixture data: {path}"
                    ) from None
        return hashlib.sha256(contents).hexdigest()
    finally:
        # Identity recovery also covers failures in the first fstat. Cleanup
        # cannot replace the primary exception or remove another process's inode.
        if status is None and destination is not None and owns_destination:
            with suppress(OSError):
                status = os.stat(destination)
        if owns_destination and destination is not None:
            with suppress(OSError):
                os.close(destination)
        if status is not None:
            with suppress(OSError):
                current = os.stat(temporary_name, dir_fd=descriptor, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (status.st_dev, status.st_ino):
                    os.unlink(temporary_name, dir_fd=descriptor)
        os.close(descriptor)


def save_fixture(
    path: str | Path,
    token_ids: NDArray[np.int64],
    logits: NDArray[np.float32] | None,
    top5_indices: NDArray[np.int64] | None,
    top5_logits: NDArray[np.float32] | None,
    n_prompt_tokens: int,
) -> str:
    """Save fixture tensors to a safetensors file.

    Args:
        path: Output file path; links and different existing contents are rejected.
        token_ids: Shape [n_generated] int64 array.
        logits: Shape [n_generated, VOCAB_SIZE] float32, or None for regression.
        top5_indices: Shape [n_generated, 5] int64, or None for canonical.
        top5_logits: Shape [n_generated, 5] float32, or None for canonical.
        n_prompt_tokens: Number of prompt tokens.

    Returns:
        SHA-256 hex digest of the file contents.
    """
    tensors: dict[str, NDArray[Any]] = {
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "n_prompt_tokens": np.array(n_prompt_tokens, dtype=np.int64),
    }

    if logits is not None:
        tensors["logits"] = np.asarray(logits, dtype=np.float32)
    if top5_indices is not None:
        tensors["top5_indices"] = np.asarray(top5_indices, dtype=np.int64)
    if top5_logits is not None:
        tensors["top5_logits"] = np.asarray(top5_logits, dtype=np.float32)

    return _save_fixture_bytes(Path(path), save(tensors))


def load_fixture(path: str | Path) -> dict[str, NDArray[Any]]:
    """Load fixture tensors from a safetensors file.

    Returns:
        Dict with keys: token_ids, n_prompt_tokens, and optionally logits,
        top5_indices, top5_logits.
    """
    return load_file(str(path))
