import json
import subprocess
from pathlib import Path


def test_authoritative_pending_budgets_stop_before_opening_manifest_or_holdout(
    tmp_path: Path,
) -> None:
    from golden_gen.layered_manifest import evaluate_manifest
    from golden_gen.layered_release import POLICY_PATH, REGISTRY_PATH

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CPU Test")
    git("config", "user.email", "cpu@example.invalid")
    (tmp_path / "docs/validation").mkdir(parents=True)
    (tmp_path / ".dag").mkdir()
    registry = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        numerical_cases=[
            dict(
                split="acceptance",
                required_mechanisms={"candidate": ["prefill"]},
                engine_options={},
                plan=dict(
                    protocol="layered-accuracy-v1",
                    schema_version=1,
                    execution_group_id="g",
                    call_id="c",
                    vocab_size=3,
                    members=[dict(case_id="new", member_id="a", prompt=[1], continuation=[2])],
                ),
            )
        ],
        operator_profiles=[
            dict(
                profile_id="r",
                operator="rmsnorm",
                dtype="bfloat16",
                shape=[1, 2],
                input_rule="literal-v1",
                required_faults=["bad-scale"],
            )
        ],
        behavior_cases=[dict(case_id="order", required_checks=["order"], scenario={})],
    )
    (tmp_path / REGISTRY_PATH).write_text(json.dumps(registry))
    (tmp_path / POLICY_PATH).write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                algorithm="fp64-logsoftmax-fsum-underflow-recorded-p95-linear-v1",
                values=None,
            )
        )
    )
    (tmp_path / ".dag/definition-index.json").write_text(
        json.dumps(
            dict(
                inputs=[
                    dict(path=p, blob_oid=git("hash-object", p))
                    for p in (REGISTRY_PATH, POLICY_PATH)
                ]
            )
        )
    )
    git("add", ".")
    git("commit", "-qm", "synthetic approved definitions")
    result = evaluate_manifest(tmp_path, tmp_path / "must-not-open.json", authoritative=True)
    assert result["verdict"] == "INVALID"
    assert result["accepting"] is False
    assert "budgets_pending" in result["reasons"]
    assert not (tmp_path / "layered-markers").exists()
    # The real outer CLI must also stop before guard/model ownership or output creation.
    import sys

    attempted = subprocess.run(
        [
            sys.executable,
            "-m",
            "golden_gen.layered_cli",
            "collect",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            str(tmp_path.parent / "uncreated-layered-run"),
            "--model-dir",
            str(tmp_path / "unopened-model"),
            "--group",
            "g",
            "--engine",
            "reference",
            "--variant",
            "primary",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert attempted.returncode == 2
    assert json.loads(attempted.stdout)["verdict"] == "INVALID"
    assert not (tmp_path.parent / "uncreated-layered-run").exists()
    auxiliary_command = list(attempted.args)
    auxiliary_command[3] = "collect-aux"
    auxiliary_command[auxiliary_command.index("--group") + 1] = "unknown"
    auxiliary_command[auxiliary_command.index("--engine") + 1] = "candidate"
    rejected = subprocess.run(auxiliary_command, capture_output=True, text=True, check=False)
    assert rejected.returncode == 2
    assert "unknown auxiliary verification" in json.loads(rejected.stdout)["reasons"][0]
    assert not (tmp_path.parent / "uncreated-layered-run").exists()


def test_complete_synthetic_three_engine_io_produces_only_nonaccepting_observation(
    tmp_path: Path,
) -> None:
    from golden_gen import config
    from golden_gen.fixed_prefix import ReplayPlan
    from golden_gen.layered_artifacts import sha
    from golden_gen.layered_manifest import evaluate_manifest
    from golden_gen.layered_release import REGISTRY_PATH

    repo, run = tmp_path / "repo", tmp_path / "run"
    repo.mkdir()
    run.mkdir()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CPU Test")
    git("config", "user.email", "cpu@example.invalid")
    (repo / "docs/validation").mkdir(parents=True)
    (repo / ".dag").mkdir()
    plan = ReplayPlan.model_validate(
        dict(
            protocol="layered-accuracy-v1",
            schema_version=1,
            execution_group_id="g",
            call_id="c",
            vocab_size=config.VOCAB_SIZE,
            members=[dict(case_id="new", member_id="a", prompt=[1], continuation=[2])],
        )
    )
    registry = dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        numerical_cases=[
            dict(
                split="calibration",
                required_mechanisms={"candidate": ["prefill"]},
                engine_options={},
                plan=plan.model_dump(),
            )
        ],
        operator_profiles=[
            dict(
                profile_id="r",
                operator="rmsnorm",
                dtype="bfloat16",
                shape=[1, 4],
                input_rule="materialized_halfway_sum_v1",
                required_faults=["bad-scale"],
            )
        ],
        behavior_cases=[
            dict(
                case_id="order",
                required_checks=["order"],
                scenario={
                    "calls": [
                        dict(
                            call_id="first",
                            prompts=[[1]],
                            params=[dict(max_tokens=1)],
                            expected="success",
                        )
                    ]
                },
            )
        ],
    )
    (repo / REGISTRY_PATH).write_text(json.dumps(registry))
    (repo / ".dag/definition-index.json").write_text(
        json.dumps(
            dict(inputs=[dict(path=REGISTRY_PATH, blob_oid=git("hash-object", REGISTRY_PATH))])
        )
    )
    git("add", ".")
    git("commit", "-qm", "synthetic definitions")
    source = dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
    registry_sha = sha(repo / REGISTRY_PATH)
    runtime = json.loads((Path(__file__).parent / "fixtures/manifest-v4.json").read_text())[
        "runtime"
    ]
    runtime["generator_commit"] = source["commit"]

    def store(name: str, data: dict) -> dict:
        path = run / name
        path.write_text(json.dumps(data))
        return dict(path=name, sha256=sha(path))

    entries = []
    for e, (engine, kernel) in enumerate(
        zip(
            ("reference", "baseline", "candidate"),
            (
                config.REFERENCE_KERNEL_PATH,
                config.BASELINE_KERNEL_PATH,
                config.CANDIDATE_KERNEL_PATH,
            ),
            strict=True,
        )
    ):
        entry = dict(execution_group_id="g", engine=engine)
        for n, variant in enumerate(("primary", "replay", "control")):
            mode = "collection_control" if variant == "control" else "fixed_prefix"
            row = dict(
                kind="prediction",
                case_id="new",
                execution_group_id="g",
                call_id="c",
                member_id="a",
                request_id=7,
                step=0,
                history_sha256=plan.history_sha256("a", 0),
                position=0,
                effective_length=1,
                phase="prefill",
                row_shape=[config.VOCAB_SIZE],
                predicted_token_id=0,
                advance_token_id=0 if variant == "control" else 2,
                logits=[0.0] * config.VOCAB_SIZE,
            )
            capture = dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                mode=mode,
                ignore_eos=True,
                execution_group_id="g",
                call_id="c",
                rows=[row],
                complete=True,
                execution_events=[
                    dict(
                        plan_id=1,
                        token_budget=1,
                        members=[
                            dict(
                                request_id=7,
                                completion_step=0,
                                sampling_allowed=True,
                                input_token_ids=[1],
                                positions=[0, 1],
                                kv_length=1,
                                phase="prefill",
                            )
                        ],
                    )
                ],
            )
            capref = store(f"{engine}-{variant}.json", capture)
            pid = 100 + 10 * e + n
            guard = store(
                f"{engine}-{variant}-guard.json",
                dict(
                    child_returncode=0,
                    failure=None,
                    minimum_available_ram_bytes=20 * 1024**3,
                    before=dict(compute_processes=""),
                    after=dict(compute_processes=""),
                    child_pid=pid,
                ),
            )
            receipt = dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                source=source,
                registry_sha256=registry_sha,
                engine=engine,
                mode=mode,
                capture_sha256=capref["sha256"],
                kernel=kernel,
                runtime=runtime,
                guard=guard,
                driver_pid=pid,
                model=dict(
                    revision=config.MODEL_REVISION,
                    config_sha256=config.MODEL_CONFIG_SHA256,
                    tokenizer_sha256=config.TOKENIZER_SHA256,
                    weights_sha256=config.MODEL_WEIGHTS_SHA256,
                    dtype="bfloat16",
                    vocab_size=config.VOCAB_SIZE,
                ),
            )
            if engine == "candidate":
                receipt.update(binary_sha256="d" * 64, build_source_id="e" * 40)
            entry[variant] = capref
            entry[variant + "_receipt"] = store(f"{engine}-{variant}-receipt.json", receipt)
        entries.append(entry)
    auxiliary = dict(verification_id="operators")
    for n, variant in enumerate(("primary", "replay")):
        capref = store(
            f"operators-{variant}.json",
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                mode="operator_verification",
                complete=True,
                accepting=False,
                device="cuda:0",
                operator_checks=[
                    dict(
                        profile_id="r",
                        rule_id="materialized_halfway_sum_v1",
                        input_shape=[1, 4],
                        input_dtype="bfloat16",
                        output_shape=[1, 4],
                        values=[1, 1, 1, 1],
                    )
                ],
            ),
        )
        guard = store(
            f"operators-{variant}-guard.json",
            dict(
                child_returncode=0,
                failure=None,
                minimum_available_ram_bytes=20 * 1024**3,
                before=dict(compute_processes=""),
                after=dict(compute_processes=""),
                child_pid=200 + n,
            ),
        )
        auxreceipt = {
            **receipt,
            "mode": "operator_verification",
            "capture_sha256": capref["sha256"],
            "guard": guard,
            "driver_pid": 200 + n,
        }
        auxiliary[variant] = capref
        auxiliary[variant + "_receipt"] = store(f"operators-{variant}-receipt.json", auxreceipt)
    behavior = dict(verification_id="order")
    for n, variant in enumerate(("primary", "replay")):
        capref = store(
            f"behavior-{variant}.json",
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                mode="free_generation",
                calls=[
                    dict(
                        call_id="first",
                        error=None,
                        binding=dict(
                            protocol="layered-accuracy-v1",
                            schema_version=1,
                            mode="behavior_binding",
                            device="cuda:0",
                            request_ids=[0],
                            prompt_lengths=[1],
                            eos_token_ids=[9],
                            max_model_len=4096,
                            forcing_enabled=False,
                        ),
                        outputs=[dict(request_id=0, token_ids=[7], text="a", finished=True)],
                    )
                ],
            ),
        )
        guard = store(
            f"behavior-{variant}-guard.json",
            dict(
                child_returncode=0,
                failure=None,
                minimum_available_ram_bytes=20 * 1024**3,
                before=dict(compute_processes=""),
                after=dict(compute_processes=""),
                child_pid=300 + n,
            ),
        )
        auxreceipt = {
            **receipt,
            "mode": "free_generation",
            "capture_sha256": capref["sha256"],
            "guard": guard,
            "driver_pid": 300 + n,
        }
        behavior[variant] = capref
        behavior[variant + "_receipt"] = store(f"behavior-{variant}-receipt.json", auxreceipt)
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                protocol="layered-accuracy-v1",
                schema_version=1,
                source=source,
                registry_sha256=registry_sha,
                purpose="observation",
                captures=entries,
                operator_checks=[auxiliary],
                behavior_checks=[behavior],
            )
        )
    )
    result = evaluate_manifest(repo, manifest, authoritative=False)
    assert result["observation_complete"] is True, result
    assert result["accepting"] is False and result["verdict"] == "INVALID"
    assert result["compared_capture_groups"] == 3
    assert result["numerical_checks"][0]["numerical_checks"]["k_mean"] == 0
    assert result["operator_checks"][0]["max_abs_error"] == 0
    assert result["behavior_checks"][0]["checks"]["order"] is True
    # A receipt cannot smuggle unregistered setup calls into the same owner.
    manifest_data = json.loads(manifest.read_text())
    extra_receipt_path = run / entries[2]["primary_receipt"]["path"]
    original_receipt = extra_receipt_path.read_text()
    extra_receipt = json.loads(original_receipt)
    extra_receipt["setup_captures"] = [entries[2]["primary"]]
    extra_receipt_path.write_text(json.dumps(extra_receipt))
    manifest_data["captures"][2]["primary_receipt"]["sha256"] = sha(extra_receipt_path)
    original_manifest = manifest.read_text()
    manifest.write_text(json.dumps(manifest_data))
    invalid = evaluate_manifest(repo, manifest, authoritative=False)
    assert invalid["observation_complete"] is False
    assert "setup capture count" in invalid["reasons"][0]
    extra_receipt_path.write_text(original_receipt)
    manifest.write_text(original_manifest)
    (run / "candidate-replay.json").write_text("{}")
    assert evaluate_manifest(repo, manifest, authoritative=False)["observation_complete"] is False
