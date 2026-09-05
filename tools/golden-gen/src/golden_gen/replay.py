"""Bit-identical fresh-process replay verification for oracle artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file


@dataclass(frozen=True)
class ReplayEvidence:
    fixture_count: int
    verified_filenames: tuple[str, ...]


def tensor_bits_equal(left: NDArray[np.generic], right: NDArray[np.generic]) -> bool:
    """Exclude container metadata, preserving signed zero and every tensor bit."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def verify_oracle_replay(
    primary_dir: Path,
    replay_dir: Path,
    expected_filenames: tuple[str, ...],
) -> ReplayEvidence:
    """Require identical tensor content and an exact, clean file set."""
    if not expected_filenames or len(expected_filenames) != len(set(expected_filenames)):
        raise ValueError("oracle replay contract must name a non-empty unique fixture set")
    expected = set(expected_filenames)
    for label, directory in (("primary", Path(primary_dir)), ("replay", Path(replay_dir))):
        discovered = {entry.name for entry in directory.iterdir() if entry.is_file()}
        if discovered != expected:
            missing = sorted(expected - discovered)
            unexpected = sorted(discovered - expected)
            raise ValueError(
                f"{label} replay file set mismatch: missing={missing}, unexpected={unexpected}"
            )

    for filename in sorted(expected_filenames, key=str.encode):
        primary = load_file(Path(primary_dir) / filename)
        replay = load_file(Path(replay_dir) / filename)
        if primary.keys() != replay.keys():
            raise ValueError(f"replay tensors are not bit-identical for {filename}: keys")
        for name in primary:
            left = primary[name]
            right = replay[name]
            if not tensor_bits_equal(left, right):
                raise ValueError(f"replay tensors are not bit-identical for {filename}:{name}")
    return ReplayEvidence(
        fixture_count=len(expected_filenames),
        verified_filenames=tuple(sorted(expected_filenames, key=str.encode)),
    )
