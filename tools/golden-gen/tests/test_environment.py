from __future__ import annotations

import hashlib

import pytest

from golden_gen.environment import (
    InstallEvidence,
    validate_deterministic_environment,
    validate_model_artifacts,
    validate_wheel_only_install,
)


def test_environment_requires_deterministic_variables_before_runtime_import():
    validate_deterministic_environment(
        {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    )

    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        validate_deterministic_environment({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


def test_model_preflight_hashes_config_tokenizer_and_weights(tmp_path):
    payloads = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "model.safetensors": b"weights",
    }
    for filename, payload in payloads.items():
        (tmp_path / filename).write_bytes(payload)
    expected = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}

    assert validate_model_artifacts(tmp_path, expected) == expected

    (tmp_path / "tokenizer.json").write_bytes(b"moving tokenizer")
    with pytest.raises(ValueError, match="tokenizer.json SHA-256"):
        validate_model_artifacts(tmp_path, expected)


def test_registry_install_allows_only_locked_wheels_and_local_project_build():
    validate_wheel_only_install(
        InstallEvidence(
            frozen=True,
            registry_no_build=True,
            source_builds=("golden-gen",),
        )
    )

    with pytest.raises(ValueError, match="third-party source build"):
        validate_wheel_only_install(
            InstallEvidence(
                frozen=True,
                registry_no_build=True,
                source_builds=("golden-gen", "flash-attn"),
            )
        )
