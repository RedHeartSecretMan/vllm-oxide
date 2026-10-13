"""Source-role CPU envelope fixtures contain synthetic source-bound logs only."""

import json
from pathlib import Path

import pytest

from golden_gen.layered_artifacts import sha
from golden_gen.release_cpu import PRESSURE, gate_commands, validate_release_cpu_roles
from golden_gen.release_roles import SourceRoles


def cpu_document(root: Path, name: str, source: dict) -> Path:
    directory = root / name
    directory.mkdir()
    outputs = dict.fromkeys(gate_commands("/synthetic/python"), "")
    for key in ("default-core", "workspace", "internal-golden"):
        outputs[key] = (
            f"test llm::{PRESSURE} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored\n"
        )
    outputs.update(
        python="1 passed in 0.01s",
        ruff="All checks passed!",
        mypy="Success: no issues found in 1 source file",
        dependencies="advisories ok, bans ok, licenses ok, sources ok",
    )
    records = []
    for gate, command in gate_commands("/synthetic/python").items():
        refs = {}
        for channel, content in (("stdout", outputs[gate]), ("stderr", "")):
            path = directory / f"{gate}.{channel}.log"
            path.write_text(content)
            refs[channel] = dict(path=path.name, sha256=sha(path))
        records.append(dict(gate=gate, command=command, returncode=0, **refs))
    path = directory / "evidence.json"
    path.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                kind="recorded_cpu_gates",
                producer="golden_gen.release_cpu-v1",
                source=source,
                python="/synthetic/python",
                environment=dict(
                    CARGO_BUILD_JOBS="1",
                    CARGO_NET_OFFLINE="true",
                    GOLDEN_WORKER_TEST_PYTHON="/synthetic/worker",
                    GOLDEN_TRANSPORT_TEST_BINARY="/synthetic/consumer",
                    CARGO_TARGET_DIR="/synthetic/target",
                    PYTHONPATH="/synthetic/src",
                ),
                toolchain=dict(rustc="synthetic", cargo="synthetic", python="synthetic"),
                gates=records,
            )
        )
    )
    return path


def test_four_role_cpu_requires_each_actual_source_and_preserves_original_envelope(tmp_path):
    roles = SourceRoles.model_validate(
        {
            key: dict(commit=value * 40, tree=value * 40)
            for key, value in [
                ("numerical_measurement", "a"),
                ("numerical_evaluator", "b"),
                ("performance_producer", "c"),
                ("release_evaluator", "d"),
            ]
        }
    )
    sources = roles.model_dump()
    docs = {key: cpu_document(tmp_path, key, source) for key, source in sources.items()}

    def ref(path):
        return dict(path=path.relative_to(tmp_path).as_posix(), sha256=sha(path))

    original = tmp_path / "original-cpu-roles.json"
    original.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                kind="role_cpu_gates",
                measurement_source=sources["numerical_measurement"],
                supervision_source=sources["numerical_evaluator"],
                measurement=ref(docs["numerical_measurement"]),
                supervision=ref(docs["numerical_evaluator"]),
            )
        )
    )
    original_bytes = original.read_bytes()
    policy = tmp_path / "policy.json"
    policy.write_text('{"synthetic_policy_fixture":true}')
    envelope = dict(
        protocol="layered-accuracy-v1",
        schema_version=2,
        kind="release_role_cpu_gates",
        sources=sources,
        original_cpu_roles=ref(original),
        performance_cpu=ref(docs["performance_producer"]),
        release_cpu=ref(docs["release_evaluator"]),
        source_role_policy=ref(policy),
    )
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps(envelope))
    closure = validate_release_cpu_roles(path, roles, sha(policy))
    assert docs["performance_producer"] in closure and docs["release_evaluator"] in closure
    assert original.read_bytes() == original_bytes
    envelope["performance_cpu"] = ref(docs["release_evaluator"])
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="CPU"):
        validate_release_cpu_roles(path, roles, sha(policy))
