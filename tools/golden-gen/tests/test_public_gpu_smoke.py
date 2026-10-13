from __future__ import annotations

import json

import pytest

from golden_gen.public_gpu_smoke import validate_api_output

SOURCE = {"commit": "a" * 40, "tree": "b" * 40}


def record(**changes):
    return {
        "source": SOURCE,
        "warmup": True,
        "repeated_calls": 100,
        "mixed_requests": 3,
        "unique_request_ids": 107,
        "passed": True,
        **changes,
    }


def output(value, summary="test result: ok. 1 passed; 0 failed; 0 ignored;"):
    return f"PUBLIC_GPU_SMOKE {json.dumps(value)}\n{summary}\n"


def test_complete_public_smoke_is_source_bound():
    assert validate_api_output(output(record()), SOURCE) == record()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        output(record(), "test result: ok. 0 passed; 0 failed; 0 ignored;"),
        output(record(), "test result: ok. 0 passed; 0 failed; 1 ignored;"),
        output(record(repeated_calls=99)),
        output(record(repeated_calls=100.0)),
        output(record(mixed_requests=0)),
        output(record(unique_request_ids=106)),
        output(record(warmup=False)),
        output(record(source={"commit": "c" * 40, "tree": "d" * 40})),
        output(record()) + output(record()),
    ],
)
def test_missing_skipped_partial_or_foreign_smoke_cannot_pass(raw):
    with pytest.raises(ValueError):
        validate_api_output(raw, SOURCE)
