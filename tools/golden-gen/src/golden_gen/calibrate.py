"""Calibrate tolerance from canonical fixture pairs and compute regression skip map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from golden_gen.config import TOLERANCE_CALIBRATION_FACTOR
from golden_gen.io import load_fixture
from golden_gen.manifest import read_manifest, write_manifest
from golden_gen.schema import FixtureMetadata, Manifest, ToleranceCalibration


def _group_fixtures_by_oracle(
    manifest: Manifest, category_filter: str
) -> dict[str, dict[str, FixtureMetadata]]:
    """Group manifest fixtures by prompt_id then oracle name.

    Args:
        manifest: Loaded Manifest object.
        category_filter: ``"canonical"`` or ``"regression"``.

    Returns:
        Dict keyed by prompt_id, then by oracle name, yielding ``FixtureMetadata``.
    """
    grouped: dict[str, dict[str, FixtureMetadata]] = {}
    for f in manifest.fixtures:
        if f.category != category_filter:
            continue
        grouped.setdefault(f.prompt_id, {})[f.oracle] = f
    return grouped


def pairwise_max_abs_diff(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute per-element max absolute difference between a and b.

    Args:
        a: Logits array of shape [n, vocab_size].
        b: Logits array of shape [m, vocab_size].

    Returns:
        Maximum |a[i,j] - b[i,j]| across all shared positions.
    """
    min_len = min(a.shape[0], b.shape[0])
    a = a[:min_len]
    b = b[:min_len]
    return float(np.abs(a - b).max())


def count_argmax_mismatches(
    token_ids_a: NDArray[np.int64],
    token_ids_b: NDArray[np.int64],
) -> int:
    """Count how many positions disagree in token ID sequences.

    Args:
        token_ids_a: Token ID sequence from oracle A.
        token_ids_b: Token ID sequence from oracle B.

    Returns:
        Number of positions where token IDs differ (on shared prefix).
    """
    min_len = min(len(token_ids_a), len(token_ids_b))
    return int((token_ids_a[:min_len] != token_ids_b[:min_len]).sum())


def compute_skip_positions(
    token_ids_a: NDArray[np.int64],
    token_ids_b: NDArray[np.int64],
) -> list[int]:
    """Return list of positions where token IDs disagree.

    Only considers the shared prefix of the two sequences.
    """
    min_len = min(len(token_ids_a), len(token_ids_b))
    mismatches = token_ids_a[:min_len] != token_ids_b[:min_len]
    return [int(i) for i in range(min_len) if mismatches[i]]


def calibrate_from_fixtures(manifest_dir: Path) -> ToleranceCalibration:
    """Calibrate atol from transformers vs vllm canonical fixture pairs.

    Loads canonical fixtures from manifest_dir, computes per-element
    max absolute difference between transformers and vllm for each
    canonical prompt, then sets atol = TOLERANCE_CALIBRATION_FACTOR *
    max across all prompts.

    Args:
        manifest_dir: Directory containing manifest.json and .safetensors fixtures.

    Returns:
        ToleranceCalibration with atol, observed_max_abs_diff, etc.
    """
    manifest = read_manifest(manifest_dir / "manifest.json")
    grouped = _group_fixtures_by_oracle(manifest, "canonical")

    per_prompt_max_abs: list[float] = []
    for pid in sorted(grouped):
        oracles = grouped[pid]
        if "transformers" not in oracles or "vllm" not in oracles:
            continue

        transformers_data = load_fixture(manifest_dir / oracles["transformers"].filename)
        vllm_data = load_fixture(manifest_dir / oracles["vllm"].filename)

        if "logits" not in transformers_data or "logits" not in vllm_data:
            continue

        max_abs = pairwise_max_abs_diff(
            transformers_data["logits"].astype(np.float32),
            vllm_data["logits"].astype(np.float32),
        )
        per_prompt_max_abs.append(max_abs)

    observed_max_abs_diff = max(per_prompt_max_abs) if per_prompt_max_abs else 0.0
    atol = TOLERANCE_CALIBRATION_FACTOR * observed_max_abs_diff
    method = (
        f"{TOLERANCE_CALIBRATION_FACTOR}x max pairwise per-element |diff| "
        f"between transformers and vllm on canonical prompts"
    )

    return ToleranceCalibration(
        atol=atol,
        observed_max_abs_diff=observed_max_abs_diff,
        calibration_factor=TOLERANCE_CALIBRATION_FACTOR,
        method=method,
    )


def compute_regression_skip_map(manifest_dir: Path) -> dict[str, list[int]]:
    """Compute skip positions for L1 regression where vllm disagrees with transformers.

    For each regression prompt, compares token_ids between transformers and vllm
    fixtures. Positions where they disagree should be skipped during L1 comparison,
    since vllm itself disagrees with the reference oracle.

    Args:
        manifest_dir: Directory containing manifest.json and .safetensors fixtures.

    Returns:
        Dict mapping prompt_id -> list of position indices to skip.
    """
    manifest = read_manifest(manifest_dir / "manifest.json")
    grouped = _group_fixtures_by_oracle(manifest, "regression")

    skip_map: dict[str, list[int]] = {}
    for pid in sorted(grouped):
        oracles = grouped[pid]
        if "transformers" not in oracles or "vllm" not in oracles:
            continue

        transformers_data = load_fixture(manifest_dir / oracles["transformers"].filename)
        vllm_data = load_fixture(manifest_dir / oracles["vllm"].filename)

        skip_positions = compute_skip_positions(
            transformers_data["token_ids"].astype(np.int64),
            vllm_data["token_ids"].astype(np.int64),
        )
        if skip_positions:
            skip_map[pid] = skip_positions

    return skip_map
