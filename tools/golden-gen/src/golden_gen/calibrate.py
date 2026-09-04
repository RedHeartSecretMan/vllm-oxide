"""Record calibration observations from canonical reference/baseline pairs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

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
        if reference.num_tokens != baseline.num_tokens:
            raise ValueError(
                f"oracle pair requires identical token lengths for {expected.prompt_id}: "
                f"{reference.num_tokens} != {baseline.num_tokens}"
            )
        reference_data = _validate_fixture_shape(manifest_dir, manifest, expected, reference)
        baseline_data = _validate_fixture_shape(manifest_dir, manifest, expected, baseline)
        if expected.family == "regression":
            count_argmax_mismatches(
                reference_data["token_ids"].astype(np.int64),
                baseline_data["token_ids"].astype(np.int64),
            )
        calibrated.append(expected.fixture_id)
    if not calibrated:
        raise ValueError("empty calibration comparison set")
    return sorted(calibrated)


def _validate_fixture_shape(
    manifest_dir: Path,
    manifest: Manifest,
    expected: ExpectedFixture,
    metadata: FixtureMetadata,
) -> dict[str, NDArray[Any]]:
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
    return data


def pairwise_max_abs_diff(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute per-element max absolute difference between a and b.

    Args:
        a: Logits array of shape [n, vocab_size].
        b: Logits array of shape [m, vocab_size].

    Returns:
        Maximum |a[i,j] - b[i,j]| across all shared positions.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"calibration logits require identical shapes, got {a.shape} and {b.shape}"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("calibration logits must contain only finite values")
    return float(np.abs(a - b).max())


def same_prefix_max_abs_diff(
    reference_logits: NDArray[np.float32],
    baseline_logits: NDArray[np.float32],
    reference_tokens: NDArray[np.int64],
    baseline_tokens: NDArray[np.int64],
) -> float:
    """Return the maximum logit difference through the first token divergence."""
    if reference_logits.shape != baseline_logits.shape:
        raise ValueError(
            "calibration logits require identical shapes, "
            f"got {reference_logits.shape} and {baseline_logits.shape}"
        )
    if reference_tokens.shape != baseline_tokens.shape:
        raise ValueError(
            "calibration token sequences require identical shapes, "
            f"got {reference_tokens.shape} and {baseline_tokens.shape}"
        )
    if reference_logits.ndim != 2 or reference_tokens.ndim != 1:
        raise ValueError("same-prefix calibration requires rank-2 logits and rank-1 tokens")
    if reference_logits.shape[0] != reference_tokens.shape[0]:
        raise ValueError("calibration logits and token sequences require identical step counts")
    if reference_tokens.size == 0:
        raise ValueError("empty same-prefix calibration comparison set")

    divergent_positions = np.flatnonzero(reference_tokens != baseline_tokens)
    compared_steps = (
        int(divergent_positions[0]) + 1 if divergent_positions.size > 0 else len(reference_tokens)
    )
    return pairwise_max_abs_diff(
        reference_logits[:compared_steps],
        baseline_logits[:compared_steps],
    )


def count_argmax_mismatches(
    token_ids_a: NDArray[np.int64],
    token_ids_b: NDArray[np.int64],
) -> int:
    """Count how many positions disagree in token ID sequences.

    Args:
        token_ids_a: Token ID sequence from oracle A.
        token_ids_b: Token ID sequence from oracle B.

    Returns:
        Number of positions where token IDs differ.
    """
    if len(token_ids_a) != len(token_ids_b):
        raise ValueError(
            "calibration token sequences require identical lengths, "
            f"got {len(token_ids_a)} and {len(token_ids_b)}"
        )
    return int((token_ids_a != token_ids_b).sum())


def calibrate_from_fixtures(manifest_dir: Path) -> ToleranceCalibration:
    """Calibrate atol from transformers vs vllm canonical fixture pairs.

    Loads canonical fixtures from manifest_dir, computes the per-element
    maximum only through each pair's first token divergence, then records a
    candidate atol as TOLERANCE_CALIBRATION_FACTOR times the observed maximum.
    This observation does not select the reference acceptance policy.

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

        max_abs = same_prefix_max_abs_diff(
            transformers_data["logits"].astype(np.float32),
            vllm_data["logits"].astype(np.float32),
            transformers_data["token_ids"].astype(np.int64),
            vllm_data["token_ids"].astype(np.int64),
        )
        per_prompt_max_abs.append(max_abs)

    if not per_prompt_max_abs:
        raise ValueError("empty calibration comparison set")
    observed_max_abs_diff = max(per_prompt_max_abs)
    atol = TOLERANCE_CALIBRATION_FACTOR * observed_max_abs_diff
    method = (
        f"{TOLERANCE_CALIBRATION_FACTOR}x max same-prefix per-element |diff| "
        f"through first divergence between transformers and vllm on canonical prompts"
    )

    return ToleranceCalibration(
        atol=atol,
        observed_max_abs_diff=observed_max_abs_diff,
        calibration_factor=TOLERANCE_CALIBRATION_FACTOR,
        method=method,
    )
