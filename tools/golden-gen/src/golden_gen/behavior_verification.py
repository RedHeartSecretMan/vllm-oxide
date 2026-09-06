"""Recompute public-generation checks from raw calls, never from producer verdicts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from golden_gen.layered_accuracy import PROTOCOL
from golden_gen.layered_release import BehaviorCase


class PublicSampling(BaseModel):
    # Invalid sampler values intentionally survive here: generate must reject them.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_tokens: int = Field(ge=0)
    ignore_eos: bool = False
    temperature: float = 0
    top_k: int | None = None
    top_p: float | None = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    repetition_penalty: float = 0


class PublicCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    call_id: str = Field(min_length=1)
    prompts: list[list[int]]
    params: list[PublicSampling]
    expected: Literal["success", "error"]
    error_contains: str | None = None


class BehaviorScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calls: list[PublicCall] = Field(min_length=1)


def compare_behavior(case: BehaviorCase, raw: dict[str, Any]) -> dict[str, Any]:
    scenario = BehaviorScenario.model_validate(case.scenario)
    if (raw.get("protocol"), raw.get("schema_version"), raw.get("mode")) != (
        PROTOCOL,
        1,
        "free_generation",
    ):
        raise ValueError("behavior capture is not versioned free generation")
    calls = raw.get("calls", [])
    if len(calls) != len(scenario.calls):
        raise ValueError("missing/extra public calls")
    checks = dict.fromkeys(
        (
            "count",
            "order",
            "finished",
            "stop_policy",
            "rejected_before_admission",
            "contextual_error",
        ),
        True,
    )
    eos_observed = False
    errors_observed = False
    next_id = 0
    for expected, actual in zip(scenario.calls, calls, strict=True):
        if actual.get("call_id") != expected.call_id:
            raise ValueError("public call identity/order mismatch")
        if expected.expected != "success":
            if not expected.error_contains:
                raise ValueError("expected error must identify its validation context")
            errors_observed = True
            error = actual.get("error")
            rejected = isinstance(error, str) and bool(error)
            checks["contextual_error"] &= rejected and expected.error_contains in error
            checks["rejected_before_admission"] &= (
                rejected and actual.get("binding") is None and actual.get("outputs") == []
            )
            continue
        binding = actual.get("binding")
        if not isinstance(binding, dict) or (
            binding.get("protocol"),
            binding.get("schema_version"),
            binding.get("mode"),
            binding.get("forcing_enabled"),
        ) != (PROTOCOL, 1, "behavior_binding", False):
            raise ValueError("missing admission binding or forcing enabled")
        n = len(expected.prompts)
        ids = list(range(next_id, next_id + n))
        next_id += n
        if binding.get("request_ids") != ids or binding.get("prompt_lengths") != [
            len(p) for p in expected.prompts
        ]:
            raise ValueError("admission does not match the declared public call")
        eos = binding.get("eos_token_ids")
        if not isinstance(eos, list) or not eos:
            raise ValueError("resolved EOS identity missing")
        outputs = actual.get("outputs", [])
        checks["count"] &= actual.get("error") is None and len(outputs) == n
        checks["order"] &= [o.get("request_id") for o in outputs] == ids
        checks["finished"] &= all(o.get("finished") is True for o in outputs)
        if len(outputs) != len(expected.params):
            checks["stop_policy"] = False
            continue
        for params, output in zip(expected.params, outputs, strict=True):
            tokens = output.get("token_ids")
            if not isinstance(tokens, list) or any(
                type(t) is not int or not 0 <= t < 151936 for t in tokens
            ):
                raise ValueError("invalid public output token IDs")
            if params.ignore_eos:
                checks["stop_policy"] &= len(tokens) == params.max_tokens
            else:
                valid = 0 < len(tokens) <= params.max_tokens
                valid &= not any(t in eos for t in tokens[:-1])
                valid &= bool(tokens) and (tokens[-1] in eos or len(tokens) == params.max_tokens)
                checks["stop_policy"] &= valid
                eos_observed |= valid and bool(tokens) and tokens[-1] in eos
    checks["resolved_eos_stop"] = eos_observed
    checks["rejected_before_admission"] &= errors_observed
    checks["contextual_error"] &= errors_observed
    if (
        len(set(case.required_checks)) != len(case.required_checks)
        or not set(case.required_checks) <= checks.keys()
    ):
        raise ValueError("unsupported/duplicate public behavior check")
    return {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "checks": {key: checks[key] for key in case.required_checks},
    }
