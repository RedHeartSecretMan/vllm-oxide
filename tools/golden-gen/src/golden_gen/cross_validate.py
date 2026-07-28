"""Cross-validate oracle outputs and calibrate tolerance."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from golden_gen.config import ORACLE_INVESTIGATION_THRESHOLD, TOLERANCE_CALIBRATION_FACTOR
from golden_gen.schema import KnownDeviation, OracleName, ToleranceCalibration


def pairwise_l2(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute max over steps of L2 norm of (a[t] - b[t]).

    Args:
        a: Logits array of shape [n, vocab_size].
        b: Logits array of shape [m, vocab_size].

    Returns:
        Maximum per-step L2 distance between a and b on their shared prefix.
    """
    min_len = min(a.shape[0], b.shape[0])
    a = a[:min_len]
    b = b[:min_len]
    per_step_l2 = np.linalg.norm(a - b, axis=1)
    return float(per_step_l2.max())


def pairwise_max_abs_diff(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute per-element max absolute difference between a and b.

    Unlike pairwise_l2 which aggregates over vocab_size dimensions,
    this returns the single largest element-wise difference across all
    steps and vocabulary positions.

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


def calibrate_tolerance(
    per_prompt_max_l2: dict[str, float],
    per_prompt_max_abs: dict[str, float] | None = None,
) -> ToleranceCalibration:
    """Calibrate atol and rtol from oracle-vs-oracle divergence.

    Uses per-element max absolute difference (infinity norm) for atol,
    and per-step L2 for observed_max_l2 (informational only, T8 Q8.2).

    Args:
        per_prompt_max_l2: Mapping from prompt_id -> max per-step L2 distance.
        per_prompt_max_abs: Mapping from prompt_id -> max per-element
            absolute difference. If None, falls back to L2-based calibration
            (legacy, less accurate).

    Returns:
        ToleranceCalibration with atol = rtol = factor × max pairwise abs diff.
    """
    observed_l2 = 0.0 if not per_prompt_max_l2 else max(per_prompt_max_l2.values())

    if per_prompt_max_abs:
        observed_max_abs = max(per_prompt_max_abs.values())
        tol = TOLERANCE_CALIBRATION_FACTOR * observed_max_abs
        method = (
            f"{TOLERANCE_CALIBRATION_FACTOR}x max pairwise per-element |diff| "
            f"across oracle triangle on canonical prompts"
        )
    else:
        tol = TOLERANCE_CALIBRATION_FACTOR * observed_l2
        method = (
            f"{TOLERANCE_CALIBRATION_FACTOR}x max pairwise L2 across "
            f"oracle triangle on canonical prompts (legacy L2 calibration)"
        )

    return ToleranceCalibration(
        atol=tol,
        rtol=tol,
        observed_max_l2=observed_l2,
        calibration_factor=TOLERANCE_CALIBRATION_FACTOR,
        method=method,
    )


def cross_validate_all(
    results: dict[tuple[OracleName, str], tuple[NDArray[Any], NDArray[Any]]],
) -> tuple[list[KnownDeviation], dict[str, float]]:
    """Run all pairwise cross-validations for canonical prompts.

    Args:
        results: Dict mapping (oracle_name, prompt_id) -> (token_ids, logits).
            Both arrays from canonical (full logits) prompts.

    Returns:
        Tuple of (known_deviations, per_prompt_max_l2).
    """
    pairs: list[tuple[OracleName, OracleName]] = [
        ("transformers", "nanovllm"),
        ("transformers", "vllm_v1"),
        ("nanovllm", "vllm_v1"),
    ]

    # Collect unique prompt IDs
    prompt_ids: set[str] = set()
    for _, pid in results:
        prompt_ids.add(pid)

    known_deviations: list[KnownDeviation] = []
    per_prompt_max_l2: dict[str, float] = {}
    per_prompt_max_abs: dict[str, float] = {}

    for pid in sorted(prompt_ids):
        prompt_max_l2 = 0.0
        prompt_max_abs = 0.0
        for oa, ob in pairs:
            key_a = (oa, pid)
            key_b = (ob, pid)
            if key_a not in results or key_b not in results:
                continue

            tok_a, logits_a = results[key_a]
            tok_b, logits_b = results[key_b]

            max_l2 = pairwise_l2(logits_a, logits_b)
            max_abs = pairwise_max_abs_diff(logits_a, logits_b)
            mismatches = count_argmax_mismatches(tok_a, tok_b)

            prompt_max_l2 = max(prompt_max_l2, max_l2)
            prompt_max_abs = max(prompt_max_abs, max_abs)

            if mismatches > 0 or max_l2 > 1e-6:
                note = f"L2={max_l2:.6f}, argmax_mismatches={mismatches} between {oa} and {ob}"
                known_deviations.append(
                    KnownDeviation(
                        pair=(oa, ob),
                        prompt_id=pid,
                        max_l2=max_l2,
                        argmax_mismatches=mismatches,
                        note=note,
                    )
                )

        per_prompt_max_l2[pid] = prompt_max_l2
        per_prompt_max_abs[pid] = prompt_max_abs

    return known_deviations, per_prompt_max_l2, per_prompt_max_abs


def flag_suspicious_divergence(
    per_prompt_max_l2: dict[str, float],
    threshold: float = ORACLE_INVESTIGATION_THRESHOLD,
) -> list[str]:
    """Return prompt_ids whose max pairwise L2 exceeds the investigation threshold.

    Spec T8 Q8.2: if oracle-vs-oracle divergence > 1e-1, the goldens from that
    prompt are NOT trustworthy — the operator must investigate the oracle before
    relying on those fixtures.

    Args:
        per_prompt_max_l2: Mapping from prompt_id -> max L2 distance
            observed across all oracle pairs for that prompt.
        threshold: L2 threshold above which to flag as suspicious.

    Returns:
        List of prompt_ids whose max L2 exceeds the threshold.
    """
    return [pid for pid, l2 in per_prompt_max_l2.items() if l2 > threshold]
