from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from golden_gen.config import VOCAB_SIZE


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for fixture files."""
    d = tmp_path / "output"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def fake_logits() -> np.ndarray:
    """Return a small set of fake logits for testing."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((8, VOCAB_SIZE), dtype=np.float32)


@pytest.fixture
def fake_token_ids() -> np.ndarray:
    """Return fake token IDs for testing."""
    return np.array([123, 456, 789, 101, 202, 303, 404, 505], dtype=np.int64)


@pytest.fixture
def ticket_artifact_root(tmp_path: Path):
    root = (
        Path("/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts") / f"{tmp_path.parent.name}-{tmp_path.name}"
    )
    yield root
    shutil.rmtree(root, ignore_errors=True)
