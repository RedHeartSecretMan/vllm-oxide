"""Reproducible CPU fault injections, isolated from every formal measurement file."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from golden_gen.layered_accuracy import compare_case
from golden_gen.layered_artifacts import (
    atomic_json,
    bound_file,
    manifest_artifact_closure,
    sha,
    source_identity,
)
from golden_gen.layered_release import REGISTRY_PATH, Registry, definition_document
from golden_gen.operator_faults import evaluate_distribution_fault, make_fault_models


def global_definitions(registry: Registry) -> dict[str, dict[str, Any]]:
    records = [r for r in registry.fault_matrix if "profile_id" not in r]
    values = {r["fault_id"]: r["input_definition"] for r in records}
    allowed = {
        "wrong_weights_identity",
        "wrong_vocabulary_identity",
        "wrong_prefix_history_or_missing_rows",
        "wrong_partial_prefix_cache",
        "negative_token_id_alias",
        "wrong_raw_argmax",
        "obvious_distribution_swap_v1",
    }
    if (
        len(values) != len(records)
        or not set(values) <= allowed
        or "obvious_distribution_swap_v1" not in values
    ):
        raise ValueError("global fault inventory is incomplete/unsupported")
    return values


def pure_fault_checks(definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    if "negative_token_id_alias" in definitions:
        from golden_gen.oracles.vllm_oracle import _extract_full_logits

        definition = definitions["negative_token_id_alias"]
        if (
            definition.get("vocab_size"),
            definition.get("rows"),
            definition.get("input_dtype"),
        ) != (2, 1, "float32"):
            raise ValueError("unsupported vocabulary alias fault definition")
        row = {
            int(k): SimpleNamespace(logprob=v) for k, v in definition["valid_raw_logit_map"].items()
        }
        if set(row) != {0, 1}:
            raise ValueError("alias fault requires the declared two-token vocabulary")
        row[-1] = row.pop(1)
        try:
            _extract_full_logits(SimpleNamespace(logprobs=[row]), 1, 2)
            detected = False
        except RuntimeError:
            detected = True
        result["negative_token_id_alias"] = dict(detected=detected, gate="identity_INVALID")
    if "wrong_raw_argmax" in definitions:
        definition = definitions["wrong_raw_argmax"]
        arrays = [
            np.asarray(definition[k], dtype=np.float32)
            for k in ("reference_logits", "candidate_logits", "baseline_logits")
        ]
        if (
            definition.get("vocab_size") != 2
            or definition.get("steps") != 1
            or any(a.shape != (1, 2) for a in arrays)
        ):
            raise ValueError("unsupported wrong-argmax fault definition")
        compared = compare_case(arrays[0], arrays[1], arrays[2], definition["predicted_token_ids"])
        result["wrong_raw_argmax"] = dict(
            detected=not compared["behavior_checks"]["prediction_valid"],
            gate="exact_behavior_FAIL",
            g_peak=compared["behavior_checks"]["g_peak"],
        )
    result["obvious_distribution_swap_v1"] = evaluate_distribution_fault(
        None, definitions["obvious_distribution_swap_v1"]
    )
    return result


def structured_fault_checks(
    repo: Path, manifest_path: Path, registry: Registry, source: dict[str, str]
) -> dict[str, Any]:
    from golden_gen.layered_manifest import _evaluate

    definitions = global_definitions(registry)
    experiments: list[tuple[str, str]] = [
        (name, name)
        for name in (
            "wrong_weights_identity",
            "wrong_vocabulary_identity",
            "wrong_partial_prefix_cache",
        )
        if name in definitions
    ]
    if "wrong_prefix_history_or_missing_rows" in definitions:
        experiments.extend(
            (name, "wrong_prefix_history_or_missing_rows")
            for name in ("wrong_prefix_history", "missing_prediction_row")
        )
    original = json.loads(manifest_path.read_text())
    original_manifest_sha = sha(manifest_path)
    groups = [g for g in registry.numerical_cases if g.split == "calibration"]
    if not groups:
        raise ValueError("structural faults need a valid calibration basis")
    results: dict[str, Any] = {}
    # Only hard-link immutable normal files. All mutations use fresh filenames;
    # no linked source inode is opened for writing, including during cleanup.
    with tempfile.TemporaryDirectory(
        prefix="layered-fault-", dir=manifest_path.parent.parent
    ) as temporary:
        root = Path(temporary)
        for path in manifest_artifact_closure(manifest_path):
            destination = root / path.relative_to(manifest_path.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(path, destination)
        for name, rule_id in experiments:
            group = groups[0]
            if name == "wrong_partial_prefix_cache":
                partial_group = next(
                    (
                        g
                        for g in groups
                        if set(g.expected_cached_tokens.get("candidate", {}).values()) == {256, 512}
                    ),
                    None,
                )
                if partial_group is None:
                    raise ValueError("partial-prefix fault lacks its frozen 256/512-token basis")
                group = partial_group
            data = copy.deepcopy(original)
            entry = next(
                e
                for e in data["captures"]
                if e["engine"] == "candidate"
                and e["execution_group_id"] == group.plan.execution_group_id
            )
            receipt = json.loads(bound_file(root, entry["primary_receipt"]).read_text())
            if name == "wrong_weights_identity":
                receipt["model"]["weights_sha256"] = "0" * 64
            elif name == "wrong_vocabulary_identity":
                receipt["model"]["vocab_size"] = 151935
            else:
                raw = json.loads(bound_file(root, entry["primary"]).read_text())
                if name == "wrong_prefix_history":
                    raw["rows"][0]["history_sha256"] = "0" * 64
                elif name == "missing_prediction_row":
                    member = group.plan.members[0]
                    index = next(
                        i
                        for i, r in enumerate(raw["rows"])
                        if r["member_id"] == member.member_id
                        and r["step"] == len(member.continuation) - 1
                    )
                    raw["rows"].pop(index)
                else:
                    member_id = next(
                        m for m, n in group.expected_cached_tokens["candidate"].items() if n == 256
                    )
                    member = next(m for m in group.plan.members if m.member_id == member_id)
                    if len(member.prompt) != 513:
                        raise ValueError(
                            "partial-prefix fault requires its frozen 513-token prompt"
                        )
                    request = next(
                        r["request_id"] for r in raw["rows"] if r["member_id"] == member_id
                    )
                    event, item = next(
                        (event, item)
                        for event in raw["execution_events"]
                        for item in event["members"]
                        if item["request_id"] == request
                    )
                    item.update(
                        cached_range=[0, 512],
                        positions=[512, 513],
                        input_token_ids=member.prompt[512:513],
                        kv_length=513,
                    )
                    event["token_budget"] = sum(len(m["input_token_ids"]) for m in event["members"])
                capture_path = root / f"injected-{name}.capture.json"
                atomic_json(capture_path, raw)
                entry["primary"] = dict(path=capture_path.name, sha256=sha(capture_path))
                receipt["capture_sha256"] = sha(capture_path)
            receipt_path = root / f"injected-{name}.receipt.json"
            atomic_json(receipt_path, receipt)
            entry["primary_receipt"] = dict(path=receipt_path.name, sha256=sha(receipt_path))
            mutated = root / f"injected-{name}.manifest.json"
            atomic_json(mutated, data)
            try:
                _evaluate(repo, mutated, authoritative=False, source_override=source)
                detected = False
                reason = "fault unexpectedly produced complete evidence"
            except ValueError as error:
                reason = str(error)
                patterns = {
                    "wrong_weights_identity": ["source/model/kernel/mode mismatch"],
                    "wrong_vocabulary_identity": ["source/model/kernel/mode mismatch"],
                    "wrong_prefix_history": ["prediction identity/shape/history mismatch"],
                    "missing_prediction_row": ["missing fixed-prefix prediction rows"],
                    "wrong_partial_prefix_cache": ["cached prefix", "KV slots mismatch"],
                }
                detected = any(pattern in reason for pattern in patterns[name])
            results[name] = dict(
                rule_id=rule_id,
                detected=detected,
                gate="identity_INVALID"
                if name in ("wrong_weights_identity", "wrong_vocabulary_identity")
                else "structure_INVALID",
                reason=reason,
            )
    if sha(manifest_path) != original_manifest_sha:
        raise ValueError("fault simulation changed its original calibration manifest")
    manifest_artifact_closure(manifest_path)  # Rehash every original capture/receipt/guard/setup.
    return results


def generate_fault_evidence(repo: Path, manifest_path: Path) -> dict[str, Any]:
    from golden_gen.layered_manifest import evaluate_manifest

    source = source_identity(repo)
    data, registry_sha = definition_document(repo, REGISTRY_PATH)
    registry = Registry.model_validate(data)
    baseline = evaluate_manifest(repo, manifest_path, authoritative=False)
    if not baseline.get("observation_complete"):
        raise ValueError("fault injection requires a complete valid calibration observation")
    definitions = global_definitions(registry)
    records = make_fault_models(registry.operator_profiles)
    records.update(
        source=source,
        registry_sha256=registry_sha,
        calibration_manifest_sha256=sha(manifest_path),
        numerical_fault_rule="obvious_distribution_swap_v1",
        numerical_fault_definition=definitions["obvious_distribution_swap_v1"],
        global_input_definitions=definitions,
        pure_checks=pure_fault_checks(definitions),
        structured_checks=structured_fault_checks(repo, manifest_path, registry, source),
    )
    if source_identity(repo) != source:
        raise ValueError("fault generation source changed")
    return records


def verify_global_fault_evidence(
    repo: Path,
    manifest_path: Path,
    registry: Registry,
    source: dict[str, str],
    record: dict[str, Any],
) -> bool:
    definitions = global_definitions(registry)
    if (
        record.get("global_input_definitions") != definitions
        or record.get("numerical_fault_definition") != definitions["obvious_distribution_swap_v1"]
        or record.get("calibration_manifest_sha256") != sha(manifest_path)
    ):
        raise ValueError("fault report inputs differ from the frozen registry/calibration basis")
    pure = pure_fault_checks(definitions)
    structured = structured_fault_checks(repo, manifest_path, registry, source)
    if record.get("pure_checks") != pure or record.get("structured_checks") != structured:
        raise ValueError("fault detector responses cannot be reproduced from their isolated inputs")
    return all(
        value["detected"] for key, value in pure.items() if key != "obvious_distribution_swap_v1"
    ) and all(value["detected"] for value in structured.values())
