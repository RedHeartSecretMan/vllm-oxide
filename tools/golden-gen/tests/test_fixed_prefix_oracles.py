import os
import subprocess
from pathlib import Path

import pytest


def test_reference_forcing_does_not_modify_raw_logits() -> None:
    python = os.environ.get("GOLDEN_WORKER_TEST_PYTHON")
    if python is None:
        pytest.skip("prepared release Python required for CPU Torch adapter test")
    output = subprocess.run(
        [python, str(Path(__file__).with_name("fixed_prefix_process_probe.py"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert "REFERENCE_FORCING_CPU_PASS" in output.stdout


def test_vllm_raw_logits_and_batch_moves_use_the_real_cpu_sampler() -> None:
    python = os.environ.get("GOLDEN_WORKER_TEST_PYTHON")
    if python is None:
        pytest.skip("prepared release Python required for CPU vLLM adapter test")
    output = subprocess.run(
        [python, str(Path(__file__).with_name("vllm_fixed_prefix_process_probe.py"))],
        capture_output=True,
        text=True,
        timeout=45,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "VLLM_PLUGINS": "",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        },
    )
    assert output.returncode == 0, output.stderr
    assert "VLLM_FORCING_CPU_PASS" in output.stdout
