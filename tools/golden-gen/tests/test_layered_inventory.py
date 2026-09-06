import pytest


def test_frozen_owner_inventory_rejects_missing_owner_without_double_counting_shared_uses() -> None:
    from golden_gen.layered_inventory import frozen_owner_inventory
    from golden_gen.layered_release import Registry

    owners = [
        dict(
            kind="execution_group",
            execution_group_id="g",
            engine=e,
            variant=v,
            target_rows=1,
            setup_calls=0,
            setup_rows=0,
        )
        for e in ("reference", "baseline", "candidate")
        for v in ("primary", "replay", "control")
    ]
    owners += [
        dict(
            kind="standalone_behavior", verification_id="b", engine="candidate", variant=v, calls=1
        )
        for v in ("primary", "replay")
    ]
    data = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        numerical_cases=[
            dict(
                split="calibration",
                required_mechanisms={"candidate": ["prefill"]},
                engine_options={},
                plan=dict(
                    protocol="layered-accuracy-v1",
                    schema_version=1,
                    execution_group_id="g",
                    call_id="call",
                    vocab_size=3,
                    members=[dict(case_id="a", member_id="a", prompt=[1], continuation=[2])],
                ),
            )
        ],
        operator_profiles=[
            dict(
                profile_id="rms",
                operator="rmsnorm",
                dtype="bfloat16",
                shape=[1, 4],
                input_rule="materialized_halfway_sum_v1",
                required_faults=["wrong_epsilon"],
            )
        ],
        behavior_cases=[
            dict(
                case_id="b",
                required_checks=["count"],
                scenario={
                    "calls": [
                        dict(
                            call_id="b",
                            prompts=[[1]],
                            params=[dict(max_tokens=1)],
                            expected="success",
                        )
                    ]
                },
            )
        ],
        expected_counts={"calibration": {"owner_inventory": owners, "unique_gpu_owners": 11}},
        auxiliary_operators={
            "owner_inventory": [
                dict(
                    kind="operator_suite",
                    verification_id="operators",
                    engine="candidate",
                    variant=v,
                )
                for v in ("primary", "replay")
            ]
        },
    )
    registry = Registry.model_validate(data)
    assert len(frozen_owner_inventory(registry, authoritative=False)) == 13
    data["expected_counts"]["calibration"]["owner_inventory"].pop()
    with pytest.raises(ValueError, match="owner inventory"):
        frozen_owner_inventory(Registry.model_validate(data), authoritative=False)


def test_assembly_hashes_exact_owner_files_and_refuses_partial_receipts(tmp_path) -> None:
    from golden_gen.layered_workflow import assemble_entries

    owners = []
    for engine in ("reference", "baseline", "candidate"):
        for variant in ("primary", "replay", "control"):
            owners.append(
                dict(kind="execution_group", execution_group_id="g", engine=engine, variant=variant)
            )
            directory = tmp_path / f"g-{engine}-{variant}"
            directory.mkdir()
            (directory / "capture.json").write_text("{}")
            (directory / "receipt.json").write_text("{}")
    for kind, name in (("operator_suite", "operators"), ("standalone_behavior", "b")):
        for variant in ("primary", "replay"):
            owners.append(
                dict(kind=kind, verification_id=name, engine="candidate", variant=variant)
            )
            directory = tmp_path / f"aux-{name}-candidate-{variant}"
            directory.mkdir()
            (directory / "capture.json").write_text("{}")
            (directory / "receipt.json").write_text("{}")
    entries = assemble_entries(tmp_path, owners)
    assert len(entries["captures"]) == 3
    assert entries["captures"][2]["primary"]["path"] == "g-candidate-primary/capture.json"
    assert len(entries["operator_checks"]) == len(entries["behavior_checks"]) == 1
    (tmp_path / "g-candidate-control/receipt.json").unlink()
    with pytest.raises((OSError, ValueError)):
        assemble_entries(tmp_path, owners)
    assert not (tmp_path / "manifest.json").exists()
