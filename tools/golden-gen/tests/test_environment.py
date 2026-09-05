from __future__ import annotations

import hashlib
import os
from email.message import Message
from pathlib import Path

import pytest

from golden_gen.environment import (
    InstallEvidence,
    collect_installed_wheels,
    validate_deterministic_environment,
    validate_model_artifacts,
    validate_wheel_only_install,
)


@pytest.mark.parametrize("python_tag", ["py3", "py2.py3"])
def test_live_wheel_provenance_rejects_unknown_and_source_distributions(tmp_path, python_tag):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """[[package]]
name = "example"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
[[package.wheels]]
url = "https://files.pythonhosted.org/example-1.0-py3-none-any.whl"
hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
""".replace("py3", python_tag)
    )

    class Distribution:
        def __init__(self, name="example", direct=None):
            self.metadata = Message()
            self.metadata["Name"] = name
            self.version = "1.0"
            self.direct = direct

        def read_text(self, name):
            return {
                "WHEEL": f"Wheel-Version: 1.0\nTag: {python_tag}-none-any\n",
                "INSTALLER": "uv\n",
                "direct_url.json": self.direct,
            }.get(name)

    wheels = collect_installed_wheels(lock, distributions=[Distribution()])
    assert [(wheel.name, wheel.filename) for wheel in wheels] == [
        ("example", f"example-1.0-{python_tag}-none-any.whl")
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


def test_vendor_retagged_wheel_requires_actual_cache_hash_and_installed_hardlink(tmp_path):
    filename = "vllm-0.18.1-cp38-abi3-manylinux_2_31_x86_64.whl"
    digest = "33d6d81299dedd45dde43fef261fbad2d513ed80e7cf7e64fb5880e8cf8ea105"
    lock = tmp_path / "uv.lock"
    lock.write_text(f"""[[package]]
name = "vllm"
version = "0.18.1"
source = {{ registry = "https://pypi.org/simple" }}
[[package.wheels]]
url = "https://files.pythonhosted.org/{filename}"
hash = "sha256:{digest}"
""")
    cache = tmp_path / "cache"
    archive = cache / "archive-v0/hSDctOFdnLr4X1ryZFkzZ"
    relative = Path("vllm-0.18.1.dist-info/WHEEL")
    (archive / relative).parent.mkdir(parents=True)
    (archive / relative).write_text("Wheel-Version: 1.0\nTag: cp38-abi3-linux_x86_64\n")
    installed = tmp_path / "installed" / relative
    installed.parent.mkdir(parents=True)
    os.link(archive / relative, installed)
    entry = cache / "wheels-v6/pypi/vllm" / filename.removeprefix("vllm-").removesuffix(".whl")
    entry.parent.mkdir(parents=True)
    entry.symlink_to(archive, target_is_directory=True)
    origin = entry.with_name(entry.name + ".http")
    # Actual uv wheels-v6 first record observed on the release host, not derived
    # by the implementation under test; trailing HTTP policy is irrelevant.
    prefix = bytes.fromhex(
        "94b568534463744f46646e4c7234583172795a466b7a5a9192a6536861323536"
        "d940333364366438313239396465646434356464653433666566323631666261"
        "6432643531336564383065376366376536346662353838306538636638656131"
        "3035d92f766c6c6d2d302e31382e312d637033382d616269332d6d616e796c69"
        "6e75785f325f33315f7838365f36342e77686c00"
    )
    origin.write_bytes(prefix)

    class Distribution:
        metadata = {"Name": "vllm"}
        version = "0.18.1"
        files = [relative]

        def locate_file(self, path):
            return tmp_path / "installed" / path

        def read_text(self, name):
            return {"WHEEL": installed.read_text(), "INSTALLER": "uv\n"}.get(name)

    wheels = collect_installed_wheels(lock, distributions=[Distribution()], cache_root=cache)
    assert wheels[0].filename == filename
    assert wheels[0].sha256 == digest
    origin.write_bytes(prefix.replace(b"33d6", b"44d6"))
    with pytest.raises(ValueError, match="hash-bound locked archive"):
        collect_installed_wheels(lock, distributions=[Distribution()], cache_root=cache)
