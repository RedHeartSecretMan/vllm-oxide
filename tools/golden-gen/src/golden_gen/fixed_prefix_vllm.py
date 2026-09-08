"""Pinned vLLM V1 forcing adapter, loaded only by an explicitly configured worker."""

from __future__ import annotations

from typing import Any

from vllm.v1.sample.logits_processor import LogitsProcessor

from golden_gen.fixed_prefix import ReplayPlan
from golden_gen.fixed_prefix_oracles import prediction_row


class FixedPrefixProcessor(LogitsProcessor):  # type: ignore[misc]
    def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool) -> None:
        self.state: dict[int, tuple[ReplayPlan, str, list[int], bool]] = {}
        self.execution: dict[int, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.identities: dict[str, dict[str, Any]] = {}

    def request_identity(self, index: int, native: str) -> int:
        """Assign an owner-local ID without interpreting the engine's opaque ID."""
        if not isinstance(native, str) or not native or index not in self.state:
            raise ValueError("vLLM execution has no bound opaque request identity")
        plan, member_id, _, _ = self.state[index]
        member = next(m for m in plan.members if m.member_id == member_id)
        key = (plan.call_id, member_id)
        previous = self.identities.get(native)
        if previous is not None:
            if (previous["call_id"], previous["member_id"]) != key:
                raise ValueError("vLLM native identity aliases two request members")
            return int(previous["request_id"])
        if any((r["call_id"], r["member_id"]) == key for r in self.identities.values()):
            raise ValueError("vLLM request member changed its opaque native identity")
        request_id = len(self.identities)
        value = dict(
            request_id=request_id,
            native_request_id=native,
            execution_group_id=plan.execution_group_id,
            call_id=plan.call_id,
            member_id=member_id,
            case_id=member.case_id,
        )
        self.identities[native] = value
        return request_id

    def request_bindings(self) -> list[dict[str, Any]]:
        active = {r["request_id"] for r in self.rows}
        return [dict(r) for r in self.identities.values() if r["request_id"] in active]

    @classmethod
    def validate_params(cls, params: Any) -> None:
        extra = params.extra_args or {}
        if type(extra.get("fixed_prefix_control", False)) is not bool:
            raise ValueError("invalid vLLM fixed-prefix control flag")
        if "fixed_prefix_plan" not in extra:
            return
        plan = ReplayPlan.model_validate(extra["fixed_prefix_plan"])
        member = next(
            (m for m in plan.members if m.member_id == extra.get("fixed_prefix_member")), None
        )
        if (
            member is None
            or params.temperature != 0
            or not params.ignore_eos
            or params.max_tokens != len(member.continuation)
        ):
            raise ValueError("vLLM fixed-prefix neutral greedy/EOS/horizon contract mismatch")
        if (
            params.presence_penalty != 0
            or params.frequency_penalty != 0
            or params.repetition_penalty != 1
        ):
            raise ValueError("fixed-prefix raw greedy forbids penalties")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Any) -> None:
        if batch_update is None:
            return
        # Pinned 0.18.1 contract: removed, added, then moved (including swaps).
        for index in batch_update.removed:
            self.state.pop(index, None)
        for index, params, prompt, output in batch_update.added:
            self.state.pop(index, None)
            extra = params.extra_args or {}
            if "fixed_prefix_plan" not in extra:
                continue
            self.validate_params(params)
            plan = ReplayPlan.model_validate(extra["fixed_prefix_plan"])
            member_id = extra["fixed_prefix_member"]
            member = next(m for m in plan.members if m.member_id == member_id)
            if prompt != member.prompt:
                raise ValueError("vLLM admitted prompt differs from frozen input")
            self.state[index] = (plan, member_id, output, extra.get("fixed_prefix_control", False))
        for source, destination, direction in batch_update.moved:
            first, second = self.state.pop(source, None), self.state.pop(destination, None)
            if first is not None:
                self.state[destination] = first
            if direction.name == "SWAP" and second is not None:
                self.state[source] = second

    def apply(self, logits: Any) -> Any:
        if not self.state:
            return logits
        forced = logits.clone()
        for index, (plan, member_id, output, control) in self.state.items():
            execution = self.execution.get(index)
            if execution is None:
                raise ValueError("vLLM forcing lacks actual model execution metadata")
            if execution["request_id"] != self.request_identity(
                index, execution["native_request_id"]
            ):
                raise ValueError("vLLM execution request binding differs")
            if not execution["sampling_allowed"]:
                # V1 discards sampled outputs from incomplete prefill chunks.
                continue
            step = len(output)
            member = next(m for m in plan.members if m.member_id == member_id)
            history = plan.history(member_id, step)
            if control:
                history = [*member.prompt, *output]
            start, end = execution["positions"]
            if (
                (not control and output != member.continuation[:step])
                or execution["kv_length"] != len(history)
                or end != len(history)
                or execution["input_token_ids"] != history[start:end]
            ):
                raise ValueError("vLLM executed a different frozen history")
            raw = logits[index].float().cpu().tolist()
            if len(raw) != plan.vocab_size:
                raise ValueError("vLLM full vocabulary capture mismatch")
            self.rows.append(
                prediction_row(
                    plan,
                    member_id,
                    execution["request_id"],
                    step,
                    raw,
                    execution["phase"],
                    history if control else None,
                )
            )
            self.rows[-1]["native_request_id"] = execution["native_request_id"]
            if not control:
                forced[index] = float("-inf")
                forced[index, member.continuation[step]] = 0.0
        return forced
