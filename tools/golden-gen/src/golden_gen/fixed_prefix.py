"""Immutable replay inputs and independent validation of raw prediction rows."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from golden_gen.layered_accuracy import PROTOCOL

Token = Annotated[int, Field(strict=True, ge=0, le=4294967295)]


def history_hash(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


class ReplayMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    member_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    prompt: list[Token] = Field(min_length=1)
    continuation: list[Token] = Field(min_length=1)


class ReplayPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["layered-accuracy-v1"]
    schema_version: Literal[1]
    execution_group_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    vocab_size: int = Field(strict=True, gt=0)
    members: list[ReplayMember] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_members(self) -> Self:
        if len({m.member_id for m in self.members}) != len(self.members):
            raise ValueError("duplicate replay member")
        if len({m.case_id for m in self.members}) != len(self.members):
            raise ValueError("duplicate numerical case within execution group")
        if any(t >= self.vocab_size for m in self.members for t in (*m.prompt, *m.continuation)):
            raise ValueError("frozen token outside vocabulary")
        return self

    def history(self, member_id: str, step: int) -> list[int]:
        member = next(m for m in self.members if m.member_id == member_id)
        if type(step) is not int or not 0 <= step < len(member.continuation):
            raise ValueError("replay step outside frozen range")
        return [*member.prompt, *member.continuation[:step]]

    def history_sha256(self, member_id: str, step: int) -> str:
        """Version 1 history hash: concatenated little-endian unsigned 32-bit IDs."""
        tokens = self.history(member_id, step)
        return history_hash(tokens)


def validate_capture(plan: ReplayPlan, capture: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Validate shape/history/identity before any distribution comparison.

    Source, runtime, registry and approval hashes belong to the outer manifest
    validation. This seam never grants release acceptance.
    """
    if (
        capture.get("protocol") != PROTOCOL
        or capture.get("schema_version") != 1
        or capture.get("mode") != "fixed_prefix"
        or capture.get("ignore_eos") is not True
        or capture.get("execution_group_id") != plan.execution_group_id
        or capture.get("call_id") != plan.call_id
        or capture.get("complete") is not True
        or not isinstance(capture.get("rows"), list)
    ):
        raise ValueError("invalid fixed-prefix capture header")
    output: dict[str, list[dict[str, Any]]] = {m.member_id: [] for m in plan.members}
    runtime_ids: dict[str, int] = {}
    members = {m.member_id: m for m in plan.members}
    for row in capture["rows"]:
        member_id = row.get("member_id")
        if member_id not in output:
            raise ValueError("unexpected capture member")
        step = len(output[member_id])
        history = plan.history(member_id, step)
        logits = row.get("logits")
        request = row.get("request_id")
        if (
            row.get("kind") != "prediction"
            or row.get("case_id") != members[member_id].case_id
            or row.get("execution_group_id") != plan.execution_group_id
            or row.get("call_id") != plan.call_id
            or type(row.get("step")) is not int
            or row["step"] != step
            or row.get("history_sha256") != plan.history_sha256(member_id, step)
            or row.get("position") != len(history) - 1
            or row.get("effective_length") != len(history)
            or row.get("phase") not in ("prefill", "decode")
            or row.get("advance_token_id") != members[member_id].continuation[step]
            or type(request) is not int
            or request < 0
            or (member_id in runtime_ids and runtime_ids[member_id] != request)
            or (member_id not in runtime_ids and request in runtime_ids.values())
            or row.get("row_shape") != [plan.vocab_size]
            or not isinstance(logits, list)
            or len(logits) != plan.vocab_size
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)
            or type(row.get("predicted_token_id")) is not int
            or not 0 <= row["predicted_token_id"] < plan.vocab_size
        ):
            raise ValueError("fixed-prefix prediction identity/shape/history mismatch")
        runtime_ids[member_id] = request
        output[member_id].append(row)
    if any(len(output[m.member_id]) != len(m.continuation) for m in plan.members):
        raise ValueError("missing fixed-prefix prediction rows")
    validate_execution_events(plan, capture, output)
    return output


def validate_execution_events(
    plan: ReplayPlan, capture: dict[str, Any], rows: dict[str, list[dict[str, Any]]]
) -> set[str]:
    """Derive mechanisms from actual execution records, never from case names/options."""
    events = capture.get("execution_events")
    if not isinstance(events, list) or not events:
        raise ValueError("fixed replay is missing actual execution events")
    bindings = {member_rows[0]["request_id"]: member for member, member_rows in rows.items()}
    sampled: set[tuple[str, int]] = set()
    mechanisms: set[str] = set()
    seen_requests: set[int] = set()
    prior_plan = -1
    for event in events:
        plan_id = event.get("plan_id")
        if type(plan_id) is not int or plan_id <= prior_plan or not event.get("members"):
            raise ValueError("invalid or duplicated execution step")
        prior_plan = plan_id
        if event.get("token_budget") != sum(len(s["input_token_ids"]) for s in event["members"]):
            raise ValueError("execution token budget differs from actual inputs")
        if len(event["members"]) > 1:
            mechanisms.add("batch")
        if any(s["request_id"] not in seen_requests for s in event["members"]) and any(
            s["request_id"] in seen_requests and s["phase"] == "decode" for s in event["members"]
        ):
            mechanisms.add("waiting_admission")
        event_requests: set[int] = set()
        for item in event["members"]:
            request = item["request_id"]
            if request not in bindings or request in event_requests:
                raise ValueError("unknown/duplicate execution request")
            event_requests.add(request)
            member_id = bindings[request]
            step = item["completion_step"]
            history = plan.history(member_id, step)
            start, end = item["positions"]
            phase = item["phase"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(history)
                or item["input_token_ids"] != history[start:end]
                or item["kv_length"] != end
                or phase not in ("prefill", "decode")
                or type(item["sampling_allowed"]) is not bool
            ):
                raise ValueError("actual execution tokens/positions/lengths mismatch")
            mechanisms.add(phase)
            if "cached_range" in item and item["cached_range"] != [0, start]:
                raise ValueError("executed cached range is not the visible prefix")
            if "slot_mapping" in item:
                table = item["block_table"]
                try:
                    expected_slots = [table[p // 256] * 256 + p % 256 for p in range(start, end)]
                except IndexError as error:
                    raise ValueError("executed block table is too short") from error
                if item["slot_mapping"] != expected_slots:
                    raise ValueError("executed KV slots mismatch logical positions")
            if request not in seen_requests and start > 0:
                mechanisms.add("prefix_hit")
            if phase == "prefill" and step > 0:
                mechanisms.add("recompute")
            if phase == "decode" and start > 0 and start % 256 == 0:
                mechanisms.add("decode_cross_page")
            if not item["sampling_allowed"]:
                if phase != "prefill" or end >= len(history):
                    raise ValueError("unexpected nonsampling execution step")
                mechanisms.add("chunked_prefill")
            else:
                if (
                    end != len(history)
                    or (member_id, step) in sampled
                    or rows[member_id][step]["phase"] != phase
                ):
                    raise ValueError("prediction does not match a unique real sampling boundary")
                sampled.add((member_id, step))
            seen_requests.add(request)
    if len(sampled) != sum(len(r) for r in rows.values()):
        raise ValueError("missing execution evidence for prediction rows")
    return mechanisms


def validate_control_capture(
    plan: ReplayPlan, control: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    if control.get("mode") != "collection_control":
        raise ValueError("collection equivalence requires an explicitly unforced control")
    members = []
    for member in plan.members:
        rows = [r for r in control.get("rows", []) if r.get("member_id") == member.member_id]
        if len(rows) != len(member.continuation) or any(
            r.get("advance_token_id") != r.get("predicted_token_id") for r in rows
        ):
            raise ValueError("unforced control did not advance its own predictions")
        for row in rows:
            logits = row.get("logits", [])
            if not logits or row["predicted_token_id"] != max(
                range(len(logits)), key=lambda i: logits[i]
            ):
                raise ValueError("control raw prediction is not the declared greedy argmax")
        members.append(
            member.model_copy(update={"continuation": [r["advance_token_id"] for r in rows]})
        )
    # Reuse structural validation against the control's *actual* history, not
    # against the frozen continuation. The on-disk control is never relabeled.
    return validate_capture(
        plan.model_copy(update={"members": members}), {**control, "mode": "fixed_prefix"}
    )


def require_collection_equivalence(
    plan: ReplayPlan, forced: dict[str, Any], control: dict[str, Any]
) -> dict[str, int]:
    """Compare only actual shared histories; a control is never fixed-prefix evidence."""
    fixed = validate_capture(plan, forced)
    actual = validate_control_capture(plan, control)
    matched = {}
    for member in plan.members:
        count = 0
        for left, right in zip(fixed[member.member_id], actual[member.member_id], strict=True):
            if left["history_sha256"] != right["history_sha256"]:
                break
            a, b = (
                np.asarray(r["logits"], dtype=np.float32).view(np.uint32) for r in (left, right)
            )
            if (
                not np.array_equal(a, b)
                or left["predicted_token_id"] != right["predicted_token_id"]
            ):
                raise ValueError(
                    "forcing changed raw logits or original prediction on identical history"
                )
            count += 1
        if count == 0:
            raise ValueError("control has no shared prediction history")
        matched[member.member_id] = count
    return matched
