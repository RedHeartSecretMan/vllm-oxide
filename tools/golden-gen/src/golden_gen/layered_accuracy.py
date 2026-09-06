"""Pure T=1, full-vocabulary comparison for layered-accuracy-v1.

FP64 log-softmax and math.fsum accumulation. Underflow of exp(log_p) is
IEEE FP64 round-to-zero probability mass, recorded explicitly; log_q stays
in log space, with no epsilon or top-k approximation. No IO or GPU runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROTOCOL = "layered-accuracy-v1"
ALGORITHM = "fp64-logsoftmax-fsum-underflow-recorded-p95-linear-v1"


@dataclass(frozen=True)
class Budgets:
    """Values only; approval provenance is checked by the artifact consumer."""

    a_mean: float
    a_peak: float
    delta_mean: float
    g_limit: float

    def __post_init__(self) -> None:
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in (self.a_mean, self.a_peak, self.delta_mean, self.g_limit)
            )
            or self.a_mean > self.a_peak
        ):
            raise ValueError("budgets must be finite/nonnegative with A_mean <= A_peak")


def _log_probabilities(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = logits - logits.max()
    if not np.isfinite(centered).all():
        raise ValueError("FP64 log-softmax centering overflow")
    return np.asarray(centered - math.log(math.fsum(np.exp(centered).tolist())), dtype=np.float64)


def checked_kl(raw: float) -> float:
    if not math.isfinite(raw) or raw < -1e-12:
        raise ValueError("invalid KL arithmetic")
    return max(0.0, raw)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _top5(logits: NDArray[np.float64]) -> list[int]:
    count = min(5, len(logits))
    cutoff = np.partition(logits, len(logits) - count)[len(logits) - count]
    above = np.flatnonzero(logits > cutoff).tolist()
    tied = np.flatnonzero(logits == cutoff)[: count - len(above)].tolist()
    return sorted(above + tied, key=lambda token: (-logits[token], token))


def compare_case(
    reference: NDArray[Any],
    candidate: NDArray[Any],
    baseline: NDArray[Any],
    predicted_tokens: list[int],
    budgets: Budgets | None = None,
) -> dict[str, Any]:
    arrays = [np.asarray(x, dtype=np.float64) for x in (reference, candidate, baseline)]
    shape = arrays[0].shape
    if (
        len(shape) != 2
        or min(shape) == 0
        or any(x.shape != shape or not np.isfinite(x).all() for x in arrays)
    ):
        raise ValueError("complete finite aligned logits are required")
    if len(predicted_tokens) != shape[0]:
        raise ValueError("prediction row count mismatch")
    rows: list[dict[str, Any]] = []
    for step, (ref, rust, vllm) in enumerate(zip(*arrays, strict=True)):
        log_p, log_r, log_v = (_log_probabilities(x) for x in (ref, rust, vllm))
        p = np.exp(log_p)
        raw_dr = math.fsum((p * (log_p - log_r)).tolist())
        raw_dv = math.fsum((p * (log_p - log_v)).tolist())
        dr, dv = checked_kl(raw_dr), checked_kl(raw_dv)
        predicted = predicted_tokens[step]
        if type(predicted) is not int or not 0 <= predicted < shape[1]:
            raise ValueError("prediction token ID is outside the vocabulary")
        error = np.abs(ref - rust)
        reference_top5, candidate_top5 = _top5(ref), _top5(rust)
        rows.append(
            dict(
                step=step,
                d_r=dr,
                d_v=dv,
                raw_d_r=raw_dr,
                raw_d_v=raw_dv,
                paired_difference=dr - dv,
                g=float(ref.max() - ref[predicted]),
                tv=0.5 * math.fsum(np.abs(p - np.exp(log_r)).tolist()),
                probability_underflow_counts=[
                    int(np.count_nonzero(np.exp(lp) == 0)) for lp in (log_p, log_r, log_v)
                ],
                predicted_token_id=predicted,
                raw_greedy_token_id=int(np.argmax(rust)),
                reference_greedy_token_id=int(np.argmax(ref)),
                max_logit_error=float(error.max()),
                worst_token_id=int(np.argmax(error)),
                reference_top5_ids=reference_top5,
                candidate_top5_ids=candidate_top5,
                top5_overlap_count=len(set(reference_top5) & set(candidate_top5)),
            )
        )
    drs = [r["d_r"] for r in rows]
    tvs = [r["tv"] for r in rows]
    paired = [r["paired_difference"] for r in rows]
    agreement = [r["predicted_token_id"] == r["reference_greedy_token_id"] for r in rows]
    result: dict[str, Any] = dict(
        protocol=PROTOCOL,
        schema_version=1,
        algorithm=ALGORITHM,
        verdict="INVALID",
        accepting=False,
        reasons=["budgets_pending"],
        numerical_checks=dict(
            k_mean=_mean(drs),
            k_peak=max(drs),
            e_mean=_mean(paired),
            tv_mean=_mean(tvs),
            tv_max=max(tvs),
            tv_p95=_p95(tvs),
            kl_p95=_p95(drs),
            paired_difference_peak=max(paired),
            paired_difference_p95=_p95(paired),
            max_logit_error=max(r["max_logit_error"] for r in rows),
            kl_clamped_count=sum(r["raw_d_r"] < 0 for r in rows)
            + sum(r["raw_d_v"] < 0 for r in rows),
        ),
        behavior_checks=dict(
            g_peak=max(r["g"] for r in rows),
            prediction_valid=all(r["predicted_token_id"] == r["raw_greedy_token_id"] for r in rows),
            first_divergence=next((i for i, equal in enumerate(agreement) if not equal), None),
            token_agreement=sum(agreement) / len(agreement),
        ),
        rows=rows,
    )
    if budgets is not None:
        numerical = result["numerical_checks"]
        behavior = result["behavior_checks"]
        conditions = dict(
            absolute_mean=numerical["k_mean"] <= budgets.a_mean,
            absolute_peak=numerical["k_peak"] <= budgets.a_peak,
            paired_mean=numerical["e_mean"] <= budgets.delta_mean,
        )
        numerical["conditions"] = conditions
        behavior["choice_loss_passed"] = behavior["g_peak"] <= budgets.g_limit
        result["reasons"] = []
        result["verdict"] = (
            "PASS"
            if all(conditions.values())
            and behavior["choice_loss_passed"]
            and behavior["prediction_valid"]
            else "FAIL"
        )
    return result


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("case-equal summary requires unique nonempty cases")
    means = {
        key: _mean([c["numerical_checks"][key] for c in cases])
        for key in ("k_mean", "k_peak", "e_mean", "tv_mean")
    }
    means["g_peak"] = _mean([c["behavior_checks"]["g_peak"] for c in cases])
    if any(not math.isfinite(v) for v in means.values()):
        raise ValueError("nonfinite case summary")
    return dict(
        averaging="case_equal",
        accepting=False,
        case_count=len(cases),
        total_steps=sum(len(c["rows"]) for c in cases),
        case_equal_means=means,
        case_steps=[dict(case_id=c["case_id"], steps=len(c["rows"])) for c in cases],
    )


def free_generation_diagnostics(
    reference: list[int], candidate: list[int], baseline: list[int]
) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate) or len(reference) != len(baseline):
        raise ValueError("unforced control token streams must have their complete common budget")

    def compare(other: list[int]) -> dict[str, Any]:
        same = [a == b for a, b in zip(reference, other, strict=True)]
        return dict(
            first_divergence=next((i for i, equal in enumerate(same) if not equal), None),
            token_agreement=sum(same) / len(same),
            compared_output_positions=len(same),
        )

    return dict(
        protocol=PROTOCOL,
        mode="unforced_control",
        history_mode="each_engine_own_generated_history",
        ignore_eos=True,
        accepting=False,
        generated_token_ids=dict(reference=reference, candidate=candidate, baseline=baseline),
        reference_candidate=compare(candidate),
        reference_baseline=compare(baseline),
    )
