import pytest

from golden_gen.release_performance import summarize_telemetry


def test_raw_synchronized_steps_reproduce_all_latency_samples():
    steps = [
        dict(
            phase="prefill" if i == 0 else "decode",
            started_ns=i * 10,
            ended_ns=(i + 1) * 10,
            prefill_tokens=8 if i == 0 else 0,
            emissions=[dict(request_id=0, completion_step=i, sampled_at_ns=(i + 1) * 10)],
        )
        for i in range(64)
    ]
    result = summarize_telemetry(steps, [0], 8)
    assert result["prefill_tokens_per_second"] == 800_000_000
    assert result["decode_tokens_per_second"] == 100_000_000
    assert result["time_to_first_token_ns"] == [[0, 10]]
    assert result["inter_token_latency_ns"] == [[0, 10]] * 63
    steps[2]["emissions"][0]["completion_step"] = 1
    with pytest.raises(ValueError, match="non-contiguous"):
        summarize_telemetry(steps, [0], 8)


def test_one_request_cannot_emit_63_tokens_in_one_decode_step():
    steps = [
        dict(
            phase="prefill",
            started_ns=0,
            ended_ns=10,
            prefill_tokens=8,
            emissions=[dict(request_id=0, completion_step=0, sampled_at_ns=10)],
        ),
        dict(
            phase="decode",
            started_ns=10,
            ended_ns=20,
            prefill_tokens=0,
            emissions=[
                dict(request_id=0, completion_step=i, sampled_at_ns=20) for i in range(1, 64)
            ],
        ),
    ]
    with pytest.raises(ValueError, match="one token per request"):
        summarize_telemetry(steps, [0], 8)
