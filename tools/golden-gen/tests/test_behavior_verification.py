import copy


def test_public_behavior_uses_admission_order_and_unforced_eos_evidence() -> None:
    from golden_gen.behavior_verification import compare_behavior
    from golden_gen.layered_release import BehaviorCase

    case = BehaviorCase.model_validate(
        dict(
            case_id="eos",
            required_checks=["count", "order", "finished", "stop_policy", "resolved_eos_stop"],
            scenario={
                "calls": [
                    {
                        "call_id": "first",
                        "prompts": [[4], [5, 6]],
                        "params": [
                            {"max_tokens": 3, "ignore_eos": False},
                            {"max_tokens": 2, "ignore_eos": True},
                        ],
                        "expected": "success",
                    }
                ]
            },
        )
    )
    raw = {
        "protocol": "layered-accuracy-v1",
        "schema_version": 1,
        "mode": "free_generation",
        "calls": [
            {
                "call_id": "first",
                "error": None,
                "binding": {
                    "protocol": "layered-accuracy-v1",
                    "schema_version": 1,
                    "mode": "behavior_binding",
                    "device": "cpu",
                    "request_ids": [0, 1],
                    "prompt_lengths": [1, 2],
                    "eos_token_ids": [9],
                    "max_model_len": 4096,
                    "forcing_enabled": False,
                },
                "outputs": [
                    {"request_id": 0, "token_ids": [8, 9], "text": "a", "finished": True},
                    {"request_id": 1, "token_ids": [9, 7], "text": "b", "finished": True},
                ],
            }
        ],
    }
    assert all(compare_behavior(case, raw)["checks"].values())
    wrong = copy.deepcopy(raw)
    wrong["calls"][0]["outputs"].reverse()
    assert compare_behavior(case, wrong)["checks"]["order"] is False
    wrong = copy.deepcopy(raw)
    wrong["calls"][0]["outputs"][0]["token_ids"] = [9, 8]
    assert compare_behavior(case, wrong)["checks"]["stop_policy"] is False


def test_invalid_call_requires_error_and_no_admission_before_next_success() -> None:
    from golden_gen.behavior_verification import compare_behavior
    from golden_gen.layered_release import BehaviorCase

    case = BehaviorCase.model_validate(
        dict(
            case_id="invalid",
            required_checks=["rejected_before_admission", "contextual_error", "count"],
            scenario={
                "calls": [
                    {
                        "call_id": "bad",
                        "prompts": [[4]],
                        "params": [{"max_tokens": 1, "temperature": -1}],
                        "expected": "error",
                        "error_contains": "temperature",
                    },
                    {
                        "call_id": "good",
                        "prompts": [[4]],
                        "params": [{"max_tokens": 1}],
                        "expected": "success",
                    },
                ]
            },
        )
    )
    raw = {
        "protocol": "layered-accuracy-v1",
        "schema_version": 1,
        "mode": "free_generation",
        "calls": [
            {
                "call_id": "bad",
                "error": "request 0: invalid temperature",
                "binding": None,
                "outputs": [],
            },
            {
                "call_id": "good",
                "error": None,
                "binding": {
                    "protocol": "layered-accuracy-v1",
                    "schema_version": 1,
                    "mode": "behavior_binding",
                    "device": "cpu",
                    "request_ids": [0],
                    "prompt_lengths": [1],
                    "eos_token_ids": [9],
                    "max_model_len": 4096,
                    "forcing_enabled": False,
                },
                "outputs": [{"request_id": 0, "token_ids": [7], "text": "a", "finished": True}],
            },
        ],
    }
    assert all(compare_behavior(case, raw)["checks"].values())
    wrong = copy.deepcopy(raw)
    wrong["calls"][0]["error"] = None
    assert compare_behavior(case, wrong)["checks"]["contextual_error"] is False
