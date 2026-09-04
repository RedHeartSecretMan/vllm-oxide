"""Non-accepting candidate calibration observation from ADR-0012."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from golden_gen.io import load_fixture
from golden_gen.release_protocol import validate_release_manifest_coverage
from golden_gen.schema import Manifest, TolerancePolicy

CALIBRATION_IDS = (
    "canonical_01",
    "canonical_02",
    "canonical_03",
    "canonical_05a",
)
HOLDOUT_IDS = (
    "canonical_04",
    "canonical_05b",
    "canonical_05c",
    "canonical_05d",
)
L2_LADDER = (0.0, *(2.0**-exponent for exponent in range(12, 1, -1)))
L1_LADDER = (0.0, *(2.0**-exponent for exponent in range(12, 3, -1)))


@dataclass
class CalibrationAccessPlan:
    """Record the exact candidate files opened while keeping holdout sealed."""

    candidate_ids: tuple[str, ...] = CALIBRATION_IDS
    holdout_ids: tuple[str, ...] = HOLDOUT_IDS
    _opened: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def opened_ids(self) -> tuple[str, ...]:
        return tuple(self._opened)

    def record_open(self, fixture_id: str) -> None:
        if fixture_id in self.holdout_ids:
            raise ValueError(f"calibration observation cannot open sealed holdout {fixture_id}")
        next_index = len(self._opened)
        if next_index >= len(self.candidate_ids) or fixture_id != self.candidate_ids[next_index]:
            raise ValueError(
                "calibration observation must open each approved candidate exactly once in order"
            )
        self._opened.append(fixture_id)


@dataclass(frozen=True)
class CaseObservation:
    prompt_id: str
    compared_rows: int
    compared_elements: int
    first_divergence: int | None
    excluded_rows: int
    candidate_token_gap: float | None
    mean_abs_error: float
    rms_abs_error: float
    p50_abs_error: float
    p95_abs_error: float
    p99_abs_error: float
    p999_abs_error: float
    maximum_abs_error: float
    non_finite_count: int


@dataclass(frozen=True)
class ThresholdProposal:
    l1_near_tie_max_abs_logit_gap: float
    l2_atol: float


@dataclass(frozen=True)
class ObservationInput:
    reference_logits: NDArray[np.float32]
    candidate_logits: NDArray[np.float32]
    reference_tokens: NDArray[np.int64]
    candidate_tokens: NDArray[np.int64]


@dataclass(frozen=True)
class ObservationIdentity:
    measurement_commit: str
    measurement_tree: str
    manifest_sha256: str
    candidate_binary_sha256: str
    runtime_sha256: str
    kernel_scope: str
    raw_evidence_sha256: str


@dataclass(frozen=True)
class AggregateObservation:
    compared_rows: int
    compared_elements: int
    divergence_count: int
    mean_abs_error: float
    rms_abs_error: float
    p50_abs_error: float
    p95_abs_error: float
    p99_abs_error: float
    p999_abs_error: float
    maximum_abs_error: float
    non_finite_count: int


@dataclass(frozen=True)
class CalibrationObservationRecord:
    schema_version: int
    status: str
    accepting: bool
    input_l1_threshold: float
    input_l2_threshold: float
    opened_fixture_ids: tuple[str, ...]
    sealed_holdout_ids: tuple[str, ...]
    identity: ObservationIdentity
    cases: tuple[CaseObservation, ...]
    aggregate: AggregateObservation
    proposal: ThresholdProposal


def observe_case(
    prompt_id: str,
    reference_logits: NDArray[np.float32],
    candidate_logits: NDArray[np.float32],
    reference_tokens: NDArray[np.int64],
    candidate_tokens: NDArray[np.int64],
) -> CaseObservation:
    """Measure all finite elements through the first token divergence."""
    if (
        reference_logits.dtype != np.float32
        or candidate_logits.dtype != np.float32
        or reference_tokens.dtype != np.int64
        or candidate_tokens.dtype != np.int64
        or reference_logits.ndim != 2
        or candidate_logits.shape != reference_logits.shape
        or reference_tokens.ndim != 1
        or candidate_tokens.shape != reference_tokens.shape
        or reference_logits.shape[0] != reference_tokens.shape[0]
        or reference_logits.shape[0] == 0
        or reference_logits.shape[1] == 0
    ):
        raise ValueError(f"candidate observation shape or dtype mismatch for {prompt_id}")
    non_finite_count = int(
        np.count_nonzero(~np.isfinite(reference_logits))
        + np.count_nonzero(~np.isfinite(candidate_logits))
    )
    if non_finite_count:
        raise ValueError(f"candidate observation contains {non_finite_count} non-finite values")

    divergent = np.flatnonzero(reference_tokens != candidate_tokens)
    first_divergence = int(divergent[0]) if divergent.size else None
    compared_rows = first_divergence + 1 if first_divergence is not None else len(reference_tokens)
    errors = np.abs(
        reference_logits[:compared_rows].astype(np.float64)
        - candidate_logits[:compared_rows].astype(np.float64)
    ).reshape(-1)
    if errors.size == 0:
        raise ValueError(f"candidate observation has an empty comparison set for {prompt_id}")

    candidate_token_gap: float | None = None
    if first_divergence is not None:
        expected = int(reference_tokens[first_divergence])
        selected = int(candidate_tokens[first_divergence])
        vocab_size = candidate_logits.shape[1]
        if not (0 <= expected < vocab_size and 0 <= selected < vocab_size):
            raise ValueError(f"candidate observation token is outside logits shape for {prompt_id}")
        row = candidate_logits[first_divergence]
        candidate_token_gap = abs(float(row[expected]) - float(row[selected]))

    percentiles = np.percentile(errors, [50.0, 95.0, 99.0, 99.9], method="linear")
    return CaseObservation(
        prompt_id=prompt_id,
        compared_rows=compared_rows,
        compared_elements=int(errors.size),
        first_divergence=first_divergence,
        excluded_rows=len(reference_tokens) - compared_rows,
        candidate_token_gap=candidate_token_gap,
        mean_abs_error=float(np.mean(errors)),
        rms_abs_error=float(np.sqrt(np.mean(np.square(errors)))),
        p50_abs_error=float(percentiles[0]),
        p95_abs_error=float(percentiles[1]),
        p99_abs_error=float(percentiles[2]),
        p999_abs_error=float(percentiles[3]),
        maximum_abs_error=float(np.max(errors)),
        non_finite_count=0,
    )


def _smallest_covering_ladder(value: float, ladder: tuple[float, ...], label: str) -> float:
    for threshold in ladder:
        if value <= threshold:
            return threshold
    raise ValueError(f"{label} observation {value} exceeds approved ceiling {ladder[-1]}")


def propose_thresholds(cases: list[CaseObservation]) -> ThresholdProposal:
    if not cases:
        raise ValueError("candidate observation requires all four calibration cases")
    maximum_l2 = max(case.maximum_abs_error for case in cases)
    divergence_gaps = [
        case.candidate_token_gap for case in cases if case.candidate_token_gap is not None
    ]
    maximum_l1 = max(divergence_gaps, default=0.0)
    return ThresholdProposal(
        l1_near_tie_max_abs_logit_gap=_smallest_covering_ladder(
            maximum_l1, L1_LADDER, "L1 candidate gap"
        ),
        l2_atol=_smallest_covering_ladder(maximum_l2, L2_LADDER, "L2 absolute error"),
    )


def _same_prefix_errors(
    case: ObservationInput, first_divergence: int | None
) -> NDArray[np.float64]:
    compared_rows = (
        first_divergence + 1 if first_divergence is not None else len(case.reference_tokens)
    )
    return np.abs(
        case.reference_logits[:compared_rows].astype(np.float64)
        - case.candidate_logits[:compared_rows].astype(np.float64)
    ).reshape(-1)


def observe_calibration(
    loader: Callable[[str], ObservationInput],
    identity: ObservationIdentity,
) -> CalibrationObservationRecord:
    """Read only the calibration subset and return a normalized non-accepting record."""
    access = CalibrationAccessPlan()
    observations: list[CaseObservation] = []
    all_errors: list[NDArray[np.float64]] = []
    for prompt_id in access.candidate_ids:
        access.record_open(prompt_id)
        inputs = loader(prompt_id)
        observed = observe_case(
            prompt_id,
            inputs.reference_logits,
            inputs.candidate_logits,
            inputs.reference_tokens,
            inputs.candidate_tokens,
        )
        observations.append(observed)
        all_errors.append(_same_prefix_errors(inputs, observed.first_divergence))
    if access.opened_ids != CALIBRATION_IDS:
        raise ValueError("candidate observation did not open the exact calibration subset")

    errors = np.concatenate(all_errors)
    percentiles = np.percentile(errors, [50.0, 95.0, 99.0, 99.9], method="linear")
    aggregate = AggregateObservation(
        compared_rows=sum(case.compared_rows for case in observations),
        compared_elements=int(errors.size),
        divergence_count=sum(case.first_divergence is not None for case in observations),
        mean_abs_error=float(np.mean(errors)),
        rms_abs_error=float(np.sqrt(np.mean(np.square(errors)))),
        p50_abs_error=float(percentiles[0]),
        p95_abs_error=float(percentiles[1]),
        p99_abs_error=float(percentiles[2]),
        p999_abs_error=float(percentiles[3]),
        maximum_abs_error=float(np.max(errors)),
        non_finite_count=0,
    )
    return CalibrationObservationRecord(
        schema_version=1,
        status="non_accepting_calibration_observation",
        accepting=False,
        input_l1_threshold=0.0,
        input_l2_threshold=0.0,
        opened_fixture_ids=access.opened_ids,
        sealed_holdout_ids=HOLDOUT_IDS,
        identity=identity,
        cases=tuple(observations),
        aggregate=aggregate,
        proposal=propose_thresholds(observations),
    )


def canonical_observation_json(record: CalibrationObservationRecord) -> bytes:
    """Return stable bytes suitable for the second Definition checkpoint."""
    return (
        json.dumps(asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def observation_exit_code(record: CalibrationObservationRecord) -> int:
    """Observation is successful evidence but can never be an acceptance exit."""
    if record.accepting or record.input_l1_threshold != 0.0 or record.input_l2_threshold != 0.0:
        raise ValueError("calibration observation must remain non-accepting with zero thresholds")
    return 3


def _read_candidate_capture(path: Path) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"candidate capture is incomplete: {path}")
    parsed = [json.loads(line) for line in lines]
    header = parsed[0]
    trailer = parsed[-1]
    if (
        header.get("kind") != "header"
        or header.get("format") != "vllm-oxide-internal-golden-jsonl-v1"
        or header.get("schema_version") != 1
        or len(header.get("requests", [])) != 1
        or trailer.get("kind") != "trailer"
        or trailer.get("complete") is not True
    ):
        raise ValueError(f"candidate capture header or trailer is invalid: {path}")
    rows = parsed[1:-1]
    logits: list[list[float]] = []
    tokens: list[int] = []
    request_id = header["requests"][0]["request_id"]
    for index, row in enumerate(rows):
        values = row.get("logits")
        if (
            row.get("kind") != "row"
            or row.get("row_index") != index
            or row.get("input_position") != 0
            or row.get("request_id") != request_id
            or row.get("completion_step") != index
            or row.get("dtype") != "F32"
            or not isinstance(values, list)
            or row.get("row_shape") != [len(values)]
            or not values
        ):
            raise ValueError(f"candidate capture row is invalid: {path}:{index}")
        logits.append(values)
        tokens.append(int(row["selected_token"]))
    array = np.asarray(logits, dtype=np.float32)
    token_array = np.asarray(tokens, dtype=np.int64)
    if (
        trailer.get("row_count") != len(rows)
        or trailer.get("dtype") != "F32"
        or trailer.get("tensor_shape") != [len(rows), array.shape[1]]
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"candidate capture shape or values are invalid: {path}")
    return array, token_array


class CaptureEntry(TypedDict):
    prompt_id: str
    filename: str
    sha256: str


class CaptureIndex(TypedDict):
    schema_version: int
    measurement_commit: str
    measurement_tree: str
    manifest_sha256: str
    candidate_binary_sha256: str
    runtime_sha256: str
    kernel_scope: str
    opened_fixture_ids: list[str]
    sealed_holdout_ids: list[str]
    captures: list[CaptureEntry]


def _load_capture_index(path: Path) -> CaptureIndex:
    index = cast(CaptureIndex, json.loads((Path(path) / "capture-index.json").read_text()))
    if (
        index.get("schema_version") != 1
        or tuple(index.get("opened_fixture_ids", [])) != CALIBRATION_IDS
        or tuple(index.get("sealed_holdout_ids", [])) != HOLDOUT_IDS
        or len(index.get("captures", [])) != 4
    ):
        raise ValueError("candidate capture index violates the calibration/holdout contract")
    return index


def observe_capture_replays(
    manifest_path: Path,
    primary_dir: Path,
    replay_dir: Path,
) -> CalibrationObservationRecord:
    """Verify candidate replay, open four references, and build non-accepting evidence."""
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = Manifest.model_validate_json(manifest_bytes)
    validate_release_manifest_coverage(manifest)
    policy = manifest.tolerance_policy
    if (
        policy.l1_near_tie_max_abs_logit_gap != 0.0
        or policy.l2_atol != 0.0
        or policy.evidence
        or "pending" not in policy.rationale.lower()
    ):
        raise ValueError("candidate observation requires a pending zero-threshold manifest")
    primary = _load_capture_index(primary_dir)
    replay = _load_capture_index(replay_dir)
    identity_fields = (
        "measurement_commit",
        "measurement_tree",
        "manifest_sha256",
        "candidate_binary_sha256",
        "runtime_sha256",
        "kernel_scope",
    )
    if any(primary.get(field) != replay.get(field) for field in identity_fields):
        raise ValueError("candidate replay identity changed between fresh processes")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if primary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("candidate capture manifest identity is stale")
    if primary.get("kernel_scope") != manifest.kernel_paths.comparison_scope:
        raise ValueError("candidate capture kernel scope is stale")

    primary_entries = {item["prompt_id"]: item for item in primary["captures"]}
    replay_entries = {item["prompt_id"]: item for item in replay["captures"]}
    raw_digest = hashlib.sha256()

    def load_case(prompt_id: str) -> ObservationInput:
        primary_entry = primary_entries.get(prompt_id)
        replay_entry = replay_entries.get(prompt_id)
        if primary_entry is None or replay_entry is None:
            raise ValueError(f"candidate capture is missing calibration case {prompt_id}")
        primary_path = Path(primary_dir) / primary_entry["filename"]
        replay_path = Path(replay_dir) / replay_entry["filename"]
        primary_bytes = primary_path.read_bytes()
        replay_bytes = replay_path.read_bytes()
        if hashlib.sha256(primary_bytes).hexdigest() != primary_entry["sha256"]:
            raise ValueError(f"candidate primary capture checksum mismatch: {prompt_id}")
        if hashlib.sha256(replay_bytes).hexdigest() != replay_entry["sha256"]:
            raise ValueError(f"candidate replay capture checksum mismatch: {prompt_id}")
        primary_logits, primary_tokens = _read_candidate_capture(primary_path)
        replay_logits, replay_tokens = _read_candidate_capture(replay_path)
        if not np.array_equal(primary_logits, replay_logits) or not np.array_equal(
            primary_tokens, replay_tokens
        ):
            raise ValueError(f"candidate replay is not bit-identical: {prompt_id}")
        reference_meta = next(
            (
                fixture
                for fixture in manifest.fixtures
                if fixture.prompt_id == prompt_id and fixture.oracle == "transformers"
            ),
            None,
        )
        if reference_meta is None:
            raise ValueError(f"reference fixture metadata is missing: {prompt_id}")
        reference_path = Path(manifest_path).parent / reference_meta.filename
        reference_bytes = reference_path.read_bytes()
        if hashlib.sha256(reference_bytes).hexdigest() != reference_meta.sha256:
            raise ValueError(f"reference fixture checksum mismatch: {prompt_id}")
        reference = load_fixture(reference_path)
        raw_digest.update(prompt_id.encode())
        raw_digest.update(hashlib.sha256(reference_bytes).digest())
        raw_digest.update(hashlib.sha256(primary_bytes).digest())
        return ObservationInput(
            reference_logits=reference["logits"].astype(np.float32),
            candidate_logits=primary_logits,
            reference_tokens=reference["token_ids"].astype(np.int64),
            candidate_tokens=primary_tokens,
        )

    identity = ObservationIdentity(
        measurement_commit=str(primary["measurement_commit"]),
        measurement_tree=str(primary["measurement_tree"]),
        manifest_sha256=manifest_sha256,
        candidate_binary_sha256=str(primary["candidate_binary_sha256"]),
        runtime_sha256=str(primary["runtime_sha256"]),
        kernel_scope=str(primary["kernel_scope"]),
        raw_evidence_sha256="pending",
    )
    record = observe_calibration(load_case, identity)
    return CalibrationObservationRecord(
        **{
            **record.__dict__,
            "identity": ObservationIdentity(
                **{**record.identity.__dict__, "raw_evidence_sha256": raw_digest.hexdigest()}
            ),
        }
    )


def verify_candidate_capture_replay(primary_dir: Path, replay_dir: Path) -> dict[str, object]:
    """Verify all 28 candidate captures without sharing producer decoding code."""
    expected_ids = {
        *(f"canonical_{index:02d}" for index in range(1, 5)),
        *(f"canonical_05{suffix}" for suffix in "abcd"),
        *(f"regression_{index:02d}" for index in range(1, 21)),
    }
    expected_files = {f"{prompt_id}.candidate.jsonl" for prompt_id in expected_ids}
    for label, directory in (("primary", Path(primary_dir)), ("replay", Path(replay_dir))):
        discovered = {path.name for path in directory.iterdir() if path.is_file()}
        if discovered != expected_files:
            raise ValueError(
                f"candidate {label} replay file set mismatch: "
                f"missing={sorted(expected_files - discovered)}, "
                f"unexpected={sorted(discovered - expected_files)}"
            )
    verified: list[str] = []
    digest = hashlib.sha256()
    for filename in sorted(expected_files, key=str.encode):
        primary_path = Path(primary_dir) / filename
        replay_path = Path(replay_dir) / filename
        primary_logits, primary_tokens = _read_candidate_capture(primary_path)
        replay_logits, replay_tokens = _read_candidate_capture(replay_path)
        if not np.array_equal(primary_logits, replay_logits) or not np.array_equal(
            primary_tokens, replay_tokens
        ):
            raise ValueError(f"candidate correctness replay is not bit-identical: {filename}")
        digest.update(filename.encode())
        digest.update(primary_logits.tobytes(order="C"))
        digest.update(primary_tokens.tobytes(order="C"))
        verified.append(filename)
    return {
        "schema_version": 1,
        "fixture_count": 28,
        "verified_filenames": verified,
        "tensor_evidence_sha256": digest.hexdigest(),
    }


def approve_manifest_policy(
    manifest_path: Path,
    observation_path: Path,
    repo_root: Path,
    rationale: str,
) -> Manifest:
    """Apply only the Definition-tracked mechanical proposal to the pending manifest."""
    if not rationale.strip():
        raise ValueError("approved tolerance policy requires root-cause rationale")
    repo_root = Path(repo_root).resolve()
    observation_path = Path(observation_path).resolve()
    expected_path = repo_root / "docs/releases/goldens-v0.2-calibration-observation.json"
    if observation_path != expected_path:
        raise ValueError("policy approval requires the fixed tracked Definition observation path")
    relative = observation_path.relative_to(repo_root).as_posix()
    subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative],
        check=True,
        capture_output=True,
    )
    blob = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", f"HEAD:{relative}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = Manifest.model_validate_json(manifest_bytes)
    expected_baseline = {
        fixture.fixture_id
        for fixture in manifest.expected_fixtures
        if fixture.oracle_role == "baseline"
    }
    if len(expected_baseline) != 28 or set(manifest.calibrated_fixtures) != expected_baseline:
        raise ValueError("policy approval requires complete 28-case baseline calibration")
    if manifest.tolerance_policy.evidence or (
        manifest.tolerance_policy.l1_near_tie_max_abs_logit_gap != 0.0
        or manifest.tolerance_policy.l2_atol != 0.0
    ):
        raise ValueError("only a pending zero-threshold manifest can receive approval")
    observation_bytes = observation_path.read_bytes()
    observation = json.loads(observation_bytes)
    if (
        observation.get("schema_version") != 1
        or observation.get("status") != "non_accepting_calibration_observation"
        or observation.get("accepting") is not False
        or tuple(observation.get("opened_fixture_ids", [])) != CALIBRATION_IDS
        or tuple(observation.get("sealed_holdout_ids", [])) != HOLDOUT_IDS
        or observation["identity"]["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
        or observation["identity"]["kernel_scope"] != manifest.kernel_paths.comparison_scope
    ):
        raise ValueError(
            "tracked observation does not bind the pending manifest and sealed holdout"
        )
    proposal = observation["proposal"]
    proposed_l1 = float(proposal["l1_near_tie_max_abs_logit_gap"])
    proposed_l2 = float(proposal["l2_atol"])
    if proposed_l1 not in L1_LADDER or proposed_l2 not in L2_LADDER:
        raise ValueError("tracked observation proposal is outside the mechanical ladders")
    observation_sha256 = hashlib.sha256(observation_bytes).hexdigest()
    manifest.tolerance_policy = TolerancePolicy(
        version="same-prefix-v1",
        dtype=manifest.model.dtype,
        kernel=manifest.kernel_paths.comparison_scope,
        l1_near_tie_max_abs_logit_gap=proposed_l1,
        l2_atol=proposed_l2,
        rationale=rationale.strip(),
        evidence=[
            f"definition-observation-sha256:{observation_sha256}",
            f"definition-observation-blob:{blob}",
        ],
    )
    manifest = Manifest.model_validate(manifest.model_dump())
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".manifest-approved-", dir=Path(manifest_path).parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(manifest.model_dump_json(indent=2).encode())
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, manifest_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest
