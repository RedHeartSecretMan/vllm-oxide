"""Calibrate tolerance from canonical fixture pairs and compute regression skip map."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from golden_gen.config import TOLERANCE_CALIBRATION_FACTOR
from golden_gen.io import load_fixture
from golden_gen.manifest import read_manifest
from golden_gen.schema import ExpectedFixture, FixtureMetadata, Manifest, ToleranceCalibration


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


def validate_calibration_coverage(manifest_dir: Path, manifest: Manifest) -> list[str]:
    """Validate every baseline/reference pair and return calibrated baseline IDs."""
    generated = {(fixture.prompt_id, fixture.oracle): fixture for fixture in manifest.fixtures}
    calibrated: list[str] = []
    for expected in manifest.expected_fixtures:
        if expected.oracle_role != "baseline":
            continue
        reference = generated.get((expected.prompt_id, "transformers"))
        baseline = generated.get((expected.prompt_id, "vllm"))
        if reference is None or baseline is None:
            raise ValueError(f"missing oracle pair for calibration fixture {expected.fixture_id}")
        _validate_fixture_shape(manifest_dir, manifest, expected, reference)
        _validate_fixture_shape(manifest_dir, manifest, expected, baseline)
        calibrated.append(expected.fixture_id)
    if not calibrated:
        raise ValueError("empty calibration comparison set")
    return sorted(calibrated)


def _validate_fixture_shape(
    manifest_dir: Path,
    manifest: Manifest,
    expected: ExpectedFixture,
    metadata: FixtureMetadata,
) -> None:
    path = manifest_dir / metadata.filename
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != metadata.sha256:
        raise ValueError(f"fixture checksum mismatch for {metadata.filename}")
    data = load_fixture(path)
    token_ids = data.get("token_ids")
    prompt_tokens = data.get("n_prompt_tokens")
    if (
        token_ids is None
        or token_ids.dtype != np.int64
        or token_ids.shape != (metadata.num_tokens,)
    ):
        raise ValueError(f"unsupported token_ids shape for {metadata.filename}")
    if prompt_tokens is None or prompt_tokens.dtype != np.int64 or prompt_tokens.shape != ():
        raise ValueError(f"unsupported n_prompt_tokens shape for {metadata.filename}")
    if expected.family in ("canonical", "batch"):
        logits = data.get("logits")
        required_shape = (metadata.num_tokens, manifest.model.vocab_size)
        if (
            set(data) != {"token_ids", "n_prompt_tokens", "logits"}
            or logits is None
            or logits.dtype != np.float32
            or logits.shape != required_shape
            or metadata.logits_shape != required_shape
        ):
            raise ValueError(f"unsupported canonical fixture shape for {metadata.filename}")
    else:
        top5_indices = data.get("top5_indices")
        top5_logits = data.get("top5_logits")
        required_shape = (metadata.num_tokens, 5)
        if (
            set(data) != {"token_ids", "n_prompt_tokens", "top5_indices", "top5_logits"}
            or top5_indices is None
            or top5_indices.dtype != np.int64
            or top5_indices.shape != required_shape
            or top5_logits is None
            or top5_logits.dtype != np.float32
            or top5_logits.shape != required_shape
            or metadata.logits_shape != (0, 0)
        ):
            raise ValueError(f"unsupported regression fixture shape for {metadata.filename}")


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

    if not per_prompt_max_abs:
        raise ValueError("empty calibration comparison set")
    observed_max_abs_diff = max(per_prompt_max_abs)
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
