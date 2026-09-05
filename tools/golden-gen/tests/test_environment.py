from __future__ import annotations

import hashlib
from email.message import Message

import pytest

from golden_gen.environment import (
    InstallEvidence,
    collect_installed_wheels,
    validate_deterministic_environment,
    validate_model_artifacts,
    validate_wheel_only_install,
)


def test_live_wheel_provenance_rejects_unknown_and_source_distributions(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("""[[package]]
name = "example"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
[[package.wheels]]
url = "https://files.pythonhosted.org/example-1.0-py3-none-any.whl"
hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
""")

    class Distribution:
        def __init__(self, name="example", direct=None):
            self.metadata = Message()
            self.metadata["Name"] = name
            self.version = "1.0"
            self.direct = direct

        def read_text(self, name):
            return {
                "WHEEL": "Wheel-Version: 1.0\nTag: py3-none-any\n",
                "INSTALLER": "uv\n",
                "direct_url.json": self.direct,
            }.get(name)

    wheels = collect_installed_wheels(lock, distributions=[Distribution()])
    assert [(wheel.name, wheel.filename) for wheel in wheels] == [
        ("example", "example-1.0-py3-none-any.whl")
    ]
    with pytest.raises(ValueError, match="not a unique locked registry"):
        collect_installed_wheels(lock, distributions=[Distribution("unlocked")])
    with pytest.raises(ValueError, match="direct/source"):
        collect_installed_wheels(lock, distributions=[Distribution(direct='{"url":"file:///src"}')])


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


@pytest.mark.parametrize("oracle", ["transformers", "vllm"])
def test_each_oracle_rechecks_its_actual_local_model_before_runtime_import(tmp_path, oracle):
    from golden_gen.oracles.transformers_oracle import TransformersOracle
    from golden_gen.oracles.vllm_oracle import VllmOracle

    (tmp_path / "config.json").write_bytes(b"changed after environment preflight")
    constructor = TransformersOracle if oracle == "transformers" else VllmOracle
    with pytest.raises(ValueError, match="config.json SHA-256"):
        constructor(tmp_path)


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
