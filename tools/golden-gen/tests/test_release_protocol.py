from pathlib import Path

import pytest
from pydantic import ValidationError

from golden_gen.config import (
    BASELINE_KERNEL_PATH,
    CANDIDATE_KERNEL_PATH,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    REFERENCE_KERNEL_PATH,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
)
from golden_gen.observation import CalibrationAccessPlan
from golden_gen.prompts import discover_fixtures, load_prompts
from golden_gen.release_protocol import CorpusContract
from golden_gen.schema import KernelPaths, ModelInfo, RuntimeInfo, WheelIdentity

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def test_release_corpus_is_exactly_the_approved_28_cases_and_56_assets():
    discovered = discover_fixtures(load_prompts(PROMPTS_DIR))

    contract = CorpusContract.from_discovered(discovered)

    assert contract.case_count == 28
    assert contract.asset_count == 56
    assert contract.assets_by_family == {
        "canonical": 8,
        "batch": 8,
        "regression": 40,
    }
    assert contract.reference_count == 28
    assert contract.reference_l1_count == 28
    assert contract.reference_l2_count == 8
    assert contract.baseline_calibration_count == 28


def test_kernel_paths_are_exact_and_derive_the_comparison_scope():
    paths = KernelPaths(
        reference=REFERENCE_KERNEL_PATH,
        baseline=BASELINE_KERNEL_PATH,
        candidate=CANDIDATE_KERNEL_PATH,
    )

    assert paths.comparison_scope == f"{REFERENCE_KERNEL_PATH}::vs::{CANDIDATE_KERNEL_PATH}"

    with pytest.raises(ValidationError):
        KernelPaths(
            reference="auto-fallback",  # type: ignore[arg-type]
            baseline=BASELINE_KERNEL_PATH,
            candidate=CANDIDATE_KERNEL_PATH,
        )


def test_release_runtime_pins_locked_versions_and_wheel_identities():
    runtime = RuntimeInfo(
        evidence_mode="release",
        registry_install_mode="locked-wheels-only",
        pythonhashseed="0",
        cublas_workspace_config=":4096:8",
        python_version="3.12.13",
        torch_version="2.10.0",
        torch_cuda_version="12.8",
        transformers_version="4.57.6",
        vllm_version="0.18.1",
        xgrammar_version="0.2.3",
        triton_version="3.6.0",
        cuda_toolkit_version="13.2.51",
        rustc_version="rustc 1.89.0",
        nvidia_driver_version="595.71",
        gpu_name="NVIDIA GeForce RTX 4080",
        compute_capability="8.9",
        os_kernel="Linux 6.18.33.2-microsoft-standard-WSL2",
        generator_commit="1" * 40,
        uv_lock_sha256="2" * 64,
        wheels=[
            WheelIdentity(
                name=name,
                version=version,
                filename=f"{name}-{version}-test.whl",
                sha256="3" * 64,
            )
            for name, version in (
                ("torch", "2.10.0"),
                ("transformers", "4.57.6"),
                ("vllm", "0.18.1"),
                ("xgrammar", "0.2.3"),
                ("triton", "3.6.0"),
            )
        ],
    )

    assert runtime.vllm_version == "0.18.1"

    with pytest.raises(ValidationError, match="locked wheel set"):
        RuntimeInfo.model_validate(
            {
                **runtime.model_dump(),
                "wheels": runtime.model_dump()["wheels"][:-1],
            }
        )


def test_model_identity_binds_tokenizer_and_exact_artifact_hashes():
    model = ModelInfo(
        id="Qwen/Qwen3-0.6B",
        revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        config_sha256=MODEL_CONFIG_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
        weights_sha256=MODEL_WEIGHTS_SHA256,
        arch="Qwen3ForCausalLM",
        dtype="bfloat16",
        vocab_size=151936,
    )

    assert model.tokenizer_revision == model.revision

    payload = model.model_dump()
    del payload["tokenizer_revision"]
    with pytest.raises(ValidationError, match="tokenizer_revision"):
        ModelInfo.model_validate(payload)


def test_calibration_access_plan_opens_only_four_candidates_and_seals_holdout():
    plan = CalibrationAccessPlan()

    assert plan.candidate_ids == (
        "canonical_01",
        "canonical_02",
        "canonical_03",
        "canonical_05a",
    )
    assert plan.holdout_ids == (
        "canonical_04",
        "canonical_05b",
        "canonical_05c",
        "canonical_05d",
    )
    for fixture_id in plan.candidate_ids:
        plan.record_open(fixture_id)

    with pytest.raises(ValueError, match="sealed holdout"):
        plan.record_open("canonical_04")

    assert plan.opened_ids == plan.candidate_ids
