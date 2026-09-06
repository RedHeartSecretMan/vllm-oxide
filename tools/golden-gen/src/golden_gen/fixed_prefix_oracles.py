"""Fixed-history oracle adapters. No GPU runtime is imported until execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from golden_gen.fixed_prefix import (
    ReplayPlan,
    history_hash,
    validate_capture,
    validate_control_capture,
)
from golden_gen.replay import tensor_bits_equal


def prediction_row(
    plan: ReplayPlan,
    member_id: str,
    request_id: int,
    step: int,
    logits: list[float],
    phase: str,
    control_history: list[int] | None = None,
) -> dict[str, Any]:
    member = next(m for m in plan.members if m.member_id == member_id)
    history = plan.history(member_id, step) if control_history is None else control_history
    predicted = max(range(len(logits)), key=lambda token: logits[token])
    return dict(
        kind="prediction",
        case_id=member.case_id,
        execution_group_id=plan.execution_group_id,
        call_id=plan.call_id,
        member_id=member_id,
        request_id=request_id,
        step=step,
        history_sha256=history_hash(history),
        position=len(history) - 1,
        effective_length=len(history),
        phase=phase,
        row_shape=[len(logits)],
        predicted_token_id=predicted,
        advance_token_id=member.continuation[step] if control_history is None else predicted,
        logits=logits,
    )


class ReferenceForcer:
    """HF saves next_token_logits *after* processors: never modify its input."""

    def __init__(self, plan: ReplayPlan, *, control: bool = False) -> None:
        self.plan = plan
        self.prompt_width = max(len(m.prompt) for m in plan.members)
        self.steps = max(len(m.continuation) for m in plan.members)
        self.rows: list[dict[str, Any]] = []
        self.control = control

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        step = input_ids.shape[1] - self.prompt_width
        if not 0 <= step < self.steps:
            raise ValueError("reference exceeded the registered group horizon")
        if scores.shape != (len(self.plan.members), self.plan.vocab_size):
            raise ValueError("reference fixed replay batch/vocabulary mismatch")
        forced = scores.clone()
        for index, member in enumerate(self.plan.members):
            if step >= len(member.continuation):
                # HF retains rectangular batches. These padding computations
                # are not additional predictions for a completed member.
                forced[index] = float("-inf")
                forced[index, 0] = 0.0
                continue
            expected = self.plan.history(member.member_id, step)
            if self.control:
                expected = [
                    *member.prompt,
                    *[
                        r["advance_token_id"]
                        for r in self.rows
                        if r["member_id"] == member.member_id
                    ],
                ]
            left_padding = self.prompt_width - len(member.prompt)
            if input_ids[index, left_padding:].tolist() != expected:
                raise ValueError("reference consumed a different frozen history")
            row = prediction_row(
                self.plan,
                member.member_id,
                index,
                step,
                scores[index].float().cpu().tolist(),
                "prefill" if step == 0 else "decode",
                expected if self.control else None,
            )
            self.rows.append(row)
            if not self.control:
                forced[index] = float("-inf")
                forced[index, member.continuation[step]] = 0.0
        return forced


def capture_reference(plan: ReplayPlan, oracle: Any, *, control: bool = False) -> dict[str, Any]:
    """One real generate call with incremental KV, never per-step re-prefill."""
    import copy

    import torch
    from transformers import LogitsProcessorList

    forcer = ReferenceForcer(plan, control=control)
    width = forcer.prompt_width
    pad = oracle.tokenizer.pad_token_id
    ids = torch.tensor(
        [[pad] * (width - len(m.prompt)) + m.prompt for m in plan.members], device="cuda"
    )
    mask = torch.tensor(
        [[0] * (width - len(m.prompt)) + [1] * len(m.prompt) for m in plan.members], device="cuda"
    )
    events: list[dict[str, Any]] = []

    def record_execution(_module: Any, _args: Any, kwargs: Any) -> None:
        actual = kwargs["input_ids"]
        positions = kwargs["position_ids"]
        attention_mask = kwargs["attention_mask"]
        members = []
        for index, member in enumerate(plan.members):
            if len(events) >= len(member.continuation):
                continue
            active = attention_mask[index, -actual.shape[1] :].bool()
            tokens = actual[index][active].cpu().tolist()
            pos = positions[index][active].cpu().tolist()
            if not pos or pos != list(range(pos[0], pos[-1] + 1)):
                raise ValueError("reference positions are not a contiguous executed token range")
            members.append(
                dict(
                    request_id=index,
                    input_token_ids=tokens,
                    positions=[pos[0], pos[-1] + 1],
                    kv_length=int(attention_mask[index].sum()),
                    phase="prefill" if len(events) == 0 else "decode",
                    sampling_allowed=True,
                    completion_step=len(events),
                )
            )
        events.append(
            dict(
                plan_id=len(events),
                physical_batch_size=len(plan.members),
                token_budget=sum(len(m["input_token_ids"]) for m in members),
                members=members,
            )
        )

    handle = oracle.model.register_forward_pre_hook(record_execution, with_kwargs=True)
    config = copy.deepcopy(oracle.model.generation_config)
    config.eos_token_id = None
    try:
        with (
            torch.inference_mode(),
            torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]),
        ):
            result = oracle.model.generate(
                ids,
                attention_mask=mask,
                generation_config=config,
                max_new_tokens=forcer.steps,
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                pad_token_id=pad,
                return_dict_in_generate=True,
                output_logits=True,
                logits_processor=LogitsProcessorList([forcer]),
            )
        for index, member in enumerate(plan.members):
            if result.sequences[index, width : width + len(member.continuation)].cpu().tolist() != [
                r["advance_token_id"] for r in forcer.rows if r["member_id"] == member.member_id
            ]:
                raise ValueError("reference did not advance the exact frozen stream")
        for row in forcer.rows:
            saved = result.logits[row["step"]][row["request_id"]].float().cpu().tolist()
            if not tensor_bits_equal(
                np.asarray(saved, dtype=np.float32), np.asarray(row["logits"], dtype=np.float32)
            ):
                raise ValueError("HF output_logits was polluted by forced selection")
    finally:
        handle.remove()
    capture = dict(
        protocol=plan.protocol,
        schema_version=1,
        mode="collection_control" if control else "fixed_prefix",
        ignore_eos=True,
        execution_group_id=plan.execution_group_id,
        call_id=plan.call_id,
        rows=forcer.rows,
        execution_events=events,
        complete=True,
    )
    (validate_control_capture if control else validate_capture)(plan, capture)
    return capture


def capture_baseline(plan: ReplayPlan, oracle: Any, *, control: bool = False) -> dict[str, Any]:
    from vllm import SamplingParams

    from golden_gen.oracles.vllm_oracle import _extract_full_logits

    if not oracle.fixed_prefix:
        raise ValueError(
            "baseline collector requires the explicitly configured fixed-prefix adapter"
        )
    params = [
        SamplingParams(
            temperature=0,
            max_tokens=len(m.continuation),
            ignore_eos=True,
            logprobs=-1,
            extra_args={
                "fixed_prefix_plan": plan.model_dump(),
                "fixed_prefix_member": m.member_id,
                "fixed_prefix_control": control,
            },
        )
        for m in plan.members
    ]
    outputs = oracle.llm.generate([{"prompt_token_ids": m.prompt} for m in plan.members], params)
    evidence = oracle.llm.collective_rpc("release_fixed_prefix_evidence", timeout=30)
    if len(evidence) != 1 or len(outputs) != len(plan.members):
        raise ValueError("baseline fixed replay requires one worker and complete group output")
    rows = evidence[0]["rows"]
    for member, output in zip(plan.members, outputs, strict=True):
        if list(output.prompt_token_ids) != member.prompt or len(output.outputs) != 1:
            raise ValueError("baseline fixed replay prompt/candidate count mismatch")
        completion = output.outputs[0]
        member_rows = [r for r in rows if r["member_id"] == member.member_id]
        if list(completion.token_ids) != [r["advance_token_id"] for r in member_rows]:
            raise ValueError("baseline did not advance the complete frozen stream")
        raw = _extract_full_logits(completion, len(member.continuation), plan.vocab_size)
        member_rows = [r for r in rows if r["member_id"] == member.member_id]
        if len(member_rows) != len(member.continuation) or any(
            r["request_id"] != int(output.request_id) for r in member_rows
        ):
            raise ValueError("baseline worker/consumer request identity mismatch")
        if not tensor_bits_equal(
            raw, np.asarray([r["logits"] for r in member_rows], dtype=np.float32)
        ):
            raise ValueError("vLLM output raw logits differ from the pre-forcing capture")
    capture = dict(
        protocol=plan.protocol,
        schema_version=1,
        mode="collection_control" if control else "fixed_prefix",
        ignore_eos=True,
        execution_group_id=plan.execution_group_id,
        call_id=plan.call_id,
        rows=rows,
        execution_events=evidence[0]["execution_events"],
        allocated_cache_blocks=evidence[0]["allocated_cache_blocks"],
        cache_block_size=evidence[0]["cache_block_size"],
        complete=True,
    )
    (validate_control_capture if control else validate_capture)(plan, capture)
    return capture
