"""CPU evidence checks for one canonical_03 two-step localization, never acceptance."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from golden_gen.replay import tensor_bits_equal

CHECKPOINTS = [
    "embedding",
    "layer0_input_norm",
    "layer0_q",
    "layer0_k",
    "layer0_v",
    *(f"layer_{i}" for i in range(28)),
    "final_norm",
]


def layer0_cause(
    reference: dict[str, NDArray[np.float32]], candidate: dict[str, NDArray[np.float32]]
) -> dict[str, Any]:
    if not tensor_bits_equal(reference["embedding"], candidate["embedding"]):
        raise ValueError("layer0 attribution requires bitwise identical embedding inputs")
    if not tensor_bits_equal(reference["layer0_input_norm"], candidate["layer0_input_norm"]):
        return {"first_path": "input_rmsnorm", "qkv_compared": False}
    different = [
        name
        for name in ("layer0_q", "layer0_k", "layer0_v")
        if not tensor_bits_equal(reference[name], candidate[name])
    ]
    return {
        "first_path": "qkv_projection" if different else "after_qkv",
        "qkv_compared": True,
        "different_projections": different,
    }


PREVIOUS_COMMIT = "0e09fb5fb9096f700d675f66b8c0ff81e56c4f81"
PREVIOUS_INDEX_SHA = "87e4fdbbaf563edf37e06ee49f984f0f8ba77d79c89df5bbae859485fecb5aea"
MANIFEST_SHA = "83ca9488d5e01643106005ba63c1016aaaae8994063f685f4ae0b43fd14e29bb"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(repo: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("layer localization requires a clean reviewed source")
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}")}


def decode_trace(lines: list[dict[str, Any]]) -> dict[str, NDArray[np.float32]]:
    if len(lines) != 36 or lines[-1] != {"kind": "trailer", "complete": True, "checkpoints": 34}:
        raise ValueError("incomplete layer diagnostic trace")
    header = lines[0]
    if (
        header.get("kind") != "header"
        or header.get("diagnostic_only") is not True
        or header.get("accepting") is not False
        or header.get("prompt_id") != "canonical_03"
        or header.get("step") not in (0, 1)
    ):
        raise ValueError("invalid layer diagnostic header")
    tokens, positions = header.get("token_ids"), header.get("positions")
    if (
        not isinstance(tokens, list)
        or not tokens
        or len(tokens) > 1024
        or any(type(v) is not int or v < 0 for v in tokens)
        or not isinstance(positions, list)
        or len(positions) != len(tokens)
    ):
        raise ValueError("invalid trace input identity")
    arrays: dict[str, NDArray[np.float32]] = {}
    for name, row in zip(CHECKPOINTS, lines[1:-1], strict=True):
        width = 2048 if name == "layer0_q" else 1024
        bits = row.get("bf16_bits")
        if (
            row.get("kind") != "checkpoint"
            or row.get("name") != name
            or row.get("dtype") != "BF16"
            or row.get("shape") != [len(tokens), width]
            or not isinstance(bits, list)
            or len(bits) != len(tokens) * width
            or any(type(v) is not int or not 0 <= v <= 65535 for v in bits)
        ):
            raise ValueError("invalid layer diagnostic checkpoint order/shape/bits")
        values = (np.array(bits, dtype=np.uint32) << 16).view(np.float32).reshape(-1, width)
        if not np.isfinite(values).all():
            raise ValueError("non-finite layer diagnostic checkpoint")
        arrays[name] = values
    return arrays


def require_prefix_equivalence(
    logits: NDArray[np.float32],
    tokens: NDArray[np.int64],
    previous_logits: NDArray[np.float32],
    previous_tokens: NDArray[np.int64],
) -> None:
    if (
        logits.dtype != np.float32
        or tokens.dtype != np.int64
        or logits.ndim != 2
        or logits.shape[0] != 2
        or tokens.shape != (2,)
        or int(tokens[0]) != 151667
        or not tensor_bits_equal(logits, previous_logits[:2])
        or not tensor_bits_equal(tokens, previous_tokens[:2])
    ):
        raise ValueError("instrumentation changed the exact original two-row prefix")


def compare(root: Path, repo: Path, previous: Path, manifest_path: Path) -> dict[str, Any]:
    """Interpret layers only after both engines reproduce the original raw logits."""
    from golden_gen.io import load_fixture
    from golden_gen.observation import _read_candidate_capture

    source = identity(repo)
    metadata = json.loads((root / "reference.json").read_text())
    if (
        metadata["source"] != source
        or metadata["equivalent_to_bc_reference"] is not True
        or metadata["diagnostic_only"] is not True
        or metadata["accepting"] is not False
        or metadata["script_sha256"]
        != sha(repo / "tools/golden-gen/scripts/canonical03_layer_probe.py")
        or metadata["deterministic_algorithms"] is not True
        or metadata["warn_only"] is not False
        or metadata["device"] != "cuda:0"
        or metadata["attention_backend"] != "SDPBackend.MATH"
    ):
        raise ValueError("stale reference localization identity")
    if set(metadata["artifacts"]) != {
        "request.json",
        "torch/step-0.jsonl",
        "torch/step-1.jsonl",
        "torch/logits.npz",
    }:
        raise ValueError("unexpected reference artifact set")
    for filename, expected in metadata["artifacts"].items():
        if sha(root / filename) != expected:
            raise ValueError("reference localization artifact changed")
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("wrong BC reference manifest")
    fixture = next(
        x
        for x in json.loads(manifest_path.read_text())["fixtures"]
        if x["prompt_id"] == "canonical_03" and x["oracle"] == "transformers"
    )
    fixture_path = manifest_path.parent / fixture["filename"]
    if sha(fixture_path) != fixture["sha256"]:
        raise ValueError("BC reference fixture changed")
    reference = load_fixture(fixture_path)
    with np.load(root / "torch/logits.npz", allow_pickle=False) as captured:
        require_prefix_equivalence(
            captured["logits"],
            captured["tokens"],
            reference["logits"].astype(np.float32),
            reference["token_ids"].astype(np.int64),
        )
    if sha(previous / "capture-index.json") != PREVIOUS_INDEX_SHA:
        raise ValueError("wrong uninstrumented observer index")
    old_index = json.loads((previous / "capture-index.json").read_text())
    index = json.loads((root / "rust/capture-index.json").read_text())
    if (
        old_index["measurement_commit"] != PREVIOUS_COMMIT
        or index["measurement_commit"] != source["commit"]
        or index["measurement_tree"] != source["tree"]
        or index["opened_fixture_ids"] != ["canonical_03"]
        or len(index["captures"]) != 1
        or index.get("diagnostic_only") is not True
        or index.get("accepting") is not False
        or index["manifest_sha256"] != MANIFEST_SHA
    ):
        raise ValueError("invalid diagnostic observer identity or subset")
    filename = "canonical_03.candidate.jsonl"
    old_entry = next(x for x in old_index["captures"] if x["prompt_id"] == "canonical_03")
    if (
        sha(previous / filename) != old_entry["sha256"]
        or sha(root / "rust" / filename) != index["captures"][0]["sha256"]
    ):
        raise ValueError("raw logits evidence checksum mismatch")
    logits, tokens = _read_candidate_capture(root / "rust" / filename)
    old_logits, old_tokens = _read_candidate_capture(previous / filename)
    require_prefix_equivalence(logits, tokens, old_logits, old_tokens)
    request = json.loads((root / "request.json").read_text())
    comparisons: list[dict[str, Any]] = []
    causes = []
    trace_hashes = {}
    for step in (0, 1):
        paths = [root / side / f"step-{step}.jsonl" for side in ("torch", "rust")]
        lines = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
        decoded = [decode_trace(value) for value in lines]
        expected_tokens = request["token_ids"] if step == 0 else [151667]
        expected_positions = (
            list(range(len(expected_tokens))) if step == 0 else [len(request["token_ids"])]
        )
        for side in lines:
            if (
                side[0]["step"] != step
                or side[0]["token_ids"] != expected_tokens
                or side[0]["positions"] != expected_positions
            ):
                raise ValueError("trace does not describe the shared prefill/decode input")
        cause = layer0_cause(decoded[0], decoded[1])
        causes.append({"step": step, **cause})
        for name in CHECKPOINTS:
            if name in ("layer0_q", "layer0_k", "layer0_v") and not cause["qkv_compared"]:
                comparisons.append(
                    dict(
                        step=step,
                        checkpoint=name,
                        status="not_compared",
                        reason="RMSNorm outputs differ; projection inputs are not bitwise equal",
                    )
                )
                continue
            a, b = decoded[0][name], decoded[1][name]
            errors = np.abs(a.astype(np.float64) - b.astype(np.float64))
            pos = np.unravel_index(np.argmax(errors), errors.shape)
            comparisons.append(
                dict(
                    step=step,
                    checkpoint=name,
                    shape=list(a.shape),
                    bitwise_equal=tensor_bits_equal(a, b),
                    maximum=float(errors.max()),
                    mean=float(errors.mean()),
                    p95=float(np.percentile(errors, 95)),
                    p99=float(np.percentile(errors, 99)),
                    maximum_position=[int(x) for x in pos],
                    reference=float(a[pos]),
                    candidate=float(b[pos]),
                )
            )
        trace_hashes.update({str(path.relative_to(root)): sha(path) for path in paths})
    return dict(
        diagnostic_only=True,
        accepting=False,
        source=source,
        uninstrumented_candidate_commit=PREVIOUS_COMMIT,
        instrumentation_equivalent=True,
        observer_index_sha256=sha(root / "rust/capture-index.json"),
        candidate_binary_sha256=index["candidate_binary_sha256"],
        reference_metadata_sha256=sha(root / "reference.json"),
        trace_sha256=trace_hashes,
        opened_fixture_ids=["canonical_03"],
        comparisons=comparisons,
        layer0_diagnosis=causes,
    )
