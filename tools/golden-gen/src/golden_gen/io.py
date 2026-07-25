"""Save and load fixture data using safetensors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file


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
        path: Output file path.
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

    save_file(tensors, str(path))

    # Compute SHA-256 of the file after writing
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_fixture(path: str | Path) -> dict[str, NDArray[Any]]:
    """Load fixture tensors from a safetensors file.

    Returns:
        Dict with keys: token_ids, n_prompt_tokens, and optionally logits,
        top5_indices, top5_logits.
    """
    return load_file(str(path))
