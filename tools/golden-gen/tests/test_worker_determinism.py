from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from golden_gen.worker_determinism import BaselineWorkerEvidence


def test_spawn_worker_configures_deterministic_algorithms_before_cuda_initialization():
    python = os.environ.get("GOLDEN_WORKER_TEST_PYTHON")
    if python is None:
        pytest.skip("Set GOLDEN_WORKER_TEST_PYTHON to the prepared release Python (CPU probe only)")
    result = subprocess.run(
        [python, str(Path(__file__).with_name("worker_process_probe.py"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )
    state = json.loads(result.stdout)
    assert state["parent_enabled"] is True
    assert state["child"] == {
        "enabled": True,
        "warn_only": False,
        "cuda_initialized": False,
        "enabled_before_setup": False,
        "constructor_enabled": True,
        "init_device_enabled": True,
    }


def test_worker_evidence_rejects_disabled_flags_wrong_backend_or_late_setup():
    record = {
        "phase": "ready",
        "pid": 1,
        "worker_class": "golden_gen.oracles.vllm_worker.DeterministicWorker",
        "before_cuda": {"enabled": True, "warn_only": False, "cuda_initialized": False},
        "current": {"enabled": True, "warn_only": False, "cuda_initialized": True},
        "attention": [
            {
                "layer": "model.layers.0.self_attn.attn",
                "backend": "FLASH_ATTN",
                "flash_attn_version": 2,
            }
        ],
        "expected_attention_layers": 1,
        "dtype": "torch.bfloat16",
        "seed": 0,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "compilation_mode": "NONE",
        "cudagraph_mode": "NONE",
    }
    BaselineWorkerEvidence.model_validate(record)
    for field, key, bad in (
        ("before_cuda", "cuda_initialized", True),
        ("current", "enabled", False),
        ("current", "warn_only", True),
    ):
        changed = {**record, field: {**record[field], key: bad}}
        with pytest.raises(ValidationError):
            BaselineWorkerEvidence.model_validate(changed)
    wrong_attention = {**record, "attention": [{**record["attention"][0], "flash_attn_version": 3}]}
    with pytest.raises(ValidationError, match="flash_attn_version"):
        BaselineWorkerEvidence.model_validate(wrong_attention)
