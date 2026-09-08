"""Separate layered workflow. Every GPU owner is a fresh guarded worker process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from golden_gen import config
from golden_gen.fixed_prefix import validate_capture, validate_control_capture
from golden_gen.layered_accuracy import PROTOCOL
from golden_gen.layered_artifacts import atomic_json, bound_file, sha, source_identity, write_marker
from golden_gen.layered_manifest import evaluate_manifest
from golden_gen.layered_release import (
    POLICY_PATH,
    REGISTRY_PATH,
    BudgetPolicy,
    NumericalCase,
    Registry,
    definition_document,
)


def _group(repo: Path, group_id: str) -> tuple[NumericalCase, str]:
    data, digest = definition_document(repo, REGISTRY_PATH)
    registry = Registry.model_validate(data)
    group = next(
        (g for g in registry.numerical_cases if g.plan.execution_group_id == group_id), None
    )
    if group is None:
        raise ValueError("execution group is absent from the approved registry")
    if group.split == "acceptance":
        values, _ = definition_document(repo, POLICY_PATH)
        policy = BudgetPolicy.model_validate(values)
        if (
            policy.values is None
            or policy.registry_sha256 != digest
            or any(
                policy.operator_budgets.get(p.profile_id) is None
                for p in registry.operator_profiles
            )
        ):
            raise ValueError("budgets_pending: independent acceptance inputs remain sealed")
    if group.plan.vocab_size != config.VOCAB_SIZE or any(
        p.vocab_size != config.VOCAB_SIZE for p in group.setup_calls
    ):
        raise ValueError("release replay requires the full pinned vocabulary")
    return group, digest


def _artifact(path: Path, root: Path) -> dict[str, str]:
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path))


def run_candidate_capture(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("candidate metadata must be a JSON object")
    return value


def _registered_owner(
    repo: Path, group_id: str, engine: str, variant: str, auxiliary: bool
) -> None:
    from golden_gen.layered_inventory import frozen_owner_inventory, owner_key

    registry = Registry.model_validate(definition_document(repo, REGISTRY_PATH)[0])
    kind = (
        ("operator_suite" if group_id == "operators" else "standalone_behavior")
        if auxiliary
        else "execution_group"
    )
    if (kind, group_id, engine, variant) not in {
        owner_key(o) for o in frozen_owner_inventory(registry, authoritative=True)
    }:
        raise ValueError("collector owner is absent from the frozen inventory")


def _auxiliary_definition(
    repo: Path, verification_id: str
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    data, digest = definition_document(repo, REGISTRY_PATH)
    registry = Registry.model_validate(data)
    if verification_id == "operators":
        from golden_gen.operator_verification import reference_rule

        for profile in registry.operator_profiles:
            reference_rule(profile.input_rule)
        return (
            "operator_verification",
            dict(
                protocol=PROTOCOL,
                schema_version=1,
                profiles=[
                    dict(profile_id=p.profile_id, rule_id=p.input_rule)
                    for p in registry.operator_profiles
                ],
            ),
            {},
            digest,
        )
    case = next((c for c in registry.behavior_cases if c.case_id == verification_id), None)
    if case is None:
        raise ValueError("unknown auxiliary verification")
    if case.mode != "free_generation":
        raise ValueError("execution behavior is derived from its guarded numerical groups")
    if case.split == "acceptance":
        values, _ = definition_document(repo, POLICY_PATH)
        policy = BudgetPolicy.model_validate(values)
        if (
            policy.values is None
            or policy.registry_sha256 != digest
            or any(
                policy.operator_budgets.get(p.profile_id) is None
                for p in registry.operator_profiles
            )
        ):
            raise ValueError("budgets_pending: independent behavior acceptance remains sealed")
    from golden_gen.behavior_verification import BehaviorScenario

    scenario = BehaviorScenario.model_validate(case.scenario)
    return "free_generation", scenario.model_dump(), case.engine_options, digest


def collect_group(
    repo: Path,
    run_dir: Path,
    model: Path,
    group_id: str,
    engine: str,
    variant: str,
    binary: Path | None = None,
    *,
    auxiliary: bool = False,
) -> dict[str, Any]:
    from golden_gen.guard import run_guarded

    source = source_identity(repo)
    # Freeze/approval check before creating files or importing a GPU runtime.
    if auxiliary:
        _auxiliary_definition(repo, group_id)
        if engine != "candidate" or variant not in ("primary", "replay"):
            raise ValueError("auxiliary verification requires candidate primary/replay")
    else:
        _group(repo, group_id)
    if engine not in ("reference", "baseline", "candidate") or variant not in (
        "primary",
        "replay",
        "control",
        "control-replay",
    ):
        raise ValueError("invalid layered collector engine/variant")
    if variant == "control-replay" and engine != "candidate":
        raise ValueError("public control replay is candidate-only")
    _registered_owner(repo, group_id, engine, variant, auxiliary)
    if (
        not group_id
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for c in group_id
        )
        or group_id in (".", "..")
    ):
        raise ValueError("unsafe execution group identifier")
    approved_root = Path("/tmp/vllm-oxide-dag-v0.2.0/t45-artifacts")
    if not run_dir.resolve().is_relative_to(approved_root) or run_dir.resolve() == approved_root:
        raise ValueError("layered run must use a dedicated ticket artifact directory")
    run_dir.mkdir(mode=0o700, exist_ok=True)
    output = run_dir / f"{'aux-' if auxiliary else ''}{group_id}-{engine}-{variant}"
    output.mkdir(mode=0o700)  # never resume or overwrite an incomplete owner
    command = [
        sys.executable,
        "-m",
        "golden_gen.layered_cli",
        "worker-aux" if auxiliary else "worker",
        "--repo-root",
        str(repo),
        "--run-dir",
        str(run_dir),
        "--model-dir",
        str(model),
        "--group",
        group_id,
        "--engine",
        engine,
        "--variant",
        variant,
    ]
    if binary is not None:
        command.extend(["--candidate-binary", str(binary)])
    guard_path = output / "guard.json"
    run_guarded(command, guard_path)
    guard = json.loads(guard_path.read_text())
    if guard["after"]["compute_processes"]:
        raise ValueError("GPU owner left an active compute process")
    metadata = json.loads((output / "worker.json").read_text())
    if (
        metadata["source"] != source
        or metadata["driver_pid"] != guard["child_pid"]
        or source_identity(repo) != source
    ):
        raise ValueError("worker metadata does not belong to this guarded source/process")
    metadata["guard"] = _artifact(guard_path, run_dir)
    atomic_json(output / "receipt.json", metadata)
    return dict(
        protocol=PROTOCOL,
        schema_version=1,
        accepting=False,
        execution_group_id=group_id,
        engine=engine,
        variant=variant,
        capture=_artifact(output / "capture.json", run_dir),
        receipt=_artifact(output / "receipt.json", run_dir),
    )


def _worker(args: argparse.Namespace) -> None:
    # The outer process owns the guard and logs remain complete even on failure.
    auxiliary = args.action == "worker-aux"
    _registered_owner(args.repo_root, args.group, args.engine, args.variant, auxiliary)
    output = (
        args.run_dir / f"{'aux-' if auxiliary else ''}{args.group}-{args.engine}-{args.variant}"
    )
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        os.dup2(stdout.fileno(), 1)
        os.dup2(stderr.fileno(), 2)
        if auxiliary:
            _worker_auxiliary(args, output)
        else:
            _worker_capture(args, output)


def _worker_auxiliary(args: argparse.Namespace, output: Path) -> None:
    from golden_gen.environment import collect_release_runtime, validate_deterministic_environment

    validate_deterministic_environment(os.environ)
    source = source_identity(args.repo_root)
    mode, request, options, registry_sha = _auxiliary_definition(args.repo_root, args.group)
    if (
        args.engine != "candidate"
        or args.variant not in ("primary", "replay")
        or args.candidate_binary is None
    ):
        raise ValueError("auxiliary verification requires a reviewed candidate binary")
    runtime = collect_release_runtime(args.model_dir, args.repo_root)
    plan_path, options_path = output / "plan.json", output / "options.json"
    atomic_json(plan_path, request)
    atomic_json(options_path, options)
    evidence = run_candidate_capture(
        [
            str(args.candidate_binary),
            "--repo-root",
            str(args.repo_root),
            "--measurement-commit",
            source["commit"],
            "--measurement-tree",
            source["tree"],
            "--model-path",
            str(args.model_dir),
            "--output",
            str(output / "capture.json"),
            "--options",
            str(options_path),
            "--operator-plan" if mode == "operator_verification" else "--behavior-plan",
            str(plan_path),
        ]
    )
    if evidence.get("source") != source or evidence.get("cuda_feature_enabled") is not True:
        raise ValueError("auxiliary candidate must be a CUDA build of the measured source")
    registry = Registry.model_validate(definition_document(args.repo_root, REGISTRY_PATH)[0])
    raw = json.loads((output / "capture.json").read_text())
    if mode == "operator_verification":
        from golden_gen.operator_verification import verify_operator_capture

        verify_operator_capture(registry.operator_profiles, raw, require_cuda=True)
    else:
        from golden_gen.behavior_verification import compare_behavior

        compare_behavior(next(c for c in registry.behavior_cases if c.case_id == args.group), raw)
    atomic_json(
        output / "worker.json",
        dict(
            protocol=PROTOCOL,
            schema_version=1,
            source=source,
            registry_sha256=registry_sha,
            engine="candidate",
            mode=mode,
            driver_pid=os.getpid(),
            capture_sha256=sha(output / "capture.json"),
            runtime=runtime.model_dump(),
            engine_evidence=evidence,
            kernel=config.CANDIDATE_KERNEL_PATH,
            binary_sha256=sha(args.candidate_binary),
            build_source_id=evidence["build_source_id"],
            model=dict(
                revision=config.MODEL_REVISION,
                config_sha256=config.MODEL_CONFIG_SHA256,
                tokenizer_sha256=config.TOKENIZER_SHA256,
                weights_sha256=config.MODEL_WEIGHTS_SHA256,
                dtype="bfloat16",
                vocab_size=config.VOCAB_SIZE,
            ),
        ),
    )


def _worker_capture(args: argparse.Namespace, output: Path) -> None:
    from golden_gen.environment import collect_release_runtime, validate_deterministic_environment
    from golden_gen.fixed_prefix import validate_baseline_owner_bindings
    from golden_gen.fixed_prefix_oracles import capture_baseline, capture_reference

    validate_deterministic_environment(os.environ)
    source = source_identity(args.repo_root)
    group, registry_sha = _group(args.repo_root, args.group)
    runtime = collect_release_runtime(args.model_dir, args.repo_root)
    control = args.variant in ("control", "control-replay")
    evidence: dict[str, Any] = {}
    setup_paths = []
    if args.engine == "candidate":
        if args.candidate_binary is None:
            raise ValueError("candidate collection requires an already-built reviewed binary")
        plan_path, options_path = output / "plan.json", output / "options.json"
        atomic_json(plan_path, group.plan.model_dump())
        atomic_json(options_path, group.engine_options)
        command = [
            str(args.candidate_binary),
            "--repo-root",
            str(args.repo_root),
            "--measurement-commit",
            source["commit"],
            "--measurement-tree",
            source["tree"],
            "--model-path",
            str(args.model_dir),
            "--plan",
            str(plan_path),
            "--output",
            str(output / "capture.json"),
            "--options",
            str(options_path),
        ]
        for index, setup in enumerate(group.setup_calls):
            path = output / f"setup-{index}.plan.json"
            atomic_json(path, setup.model_dump())
            command.extend(["--setup-plan", str(path)])
            setup_paths.append(output / f"setup-{index}.capture.json")
        if control:
            command.append("--control")
        evidence = run_candidate_capture(command)
        if evidence["source"] != source or evidence.get("cuda_feature_enabled") is not True:
            raise ValueError("candidate build source metadata mismatch")
        capture = json.loads((output / "capture.json").read_text())
    else:
        if args.engine == "reference":
            from golden_gen.oracles.transformers_oracle import TransformersOracle

            oracle: Any = TransformersOracle(args.model_dir)
            collector = capture_reference
        else:
            from golden_gen.oracles.vllm_oracle import VllmOracle

            oracle = VllmOracle(
                args.model_dir, fixed_prefix=True, execution_options=group.engine_options
            )
            collector = capture_baseline
        try:
            for index, setup in enumerate(group.setup_calls):
                path = output / f"setup-{index}.capture.json"
                atomic_json(path, collector(setup, oracle, control=control))
                setup_paths.append(path)
            capture = collector(group.plan, oracle, control=control)
            if args.engine == "baseline":
                # Retain identities, not all persisted setup logits, in the validator.
                validate_baseline_owner_bindings(
                    (plan, capture if path is None else json.loads(path.read_text()))
                    for plan, path in zip(
                        [*group.setup_calls, group.plan], [*setup_paths, None], strict=True
                    )
                )
                evidence["worker_states"] = [x.model_dump() for x in oracle.protocol_evidence()]
            else:
                import torch

                evidence.update(
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                    attention_backend="SDPBackend.MATH",
                )
            atomic_json(output / "capture.json", capture)
        finally:
            oracle.close()
    (validate_control_capture if control else validate_capture)(group.plan, capture)
    if (
        args.engine == "candidate"
        and "test_kv_blocks" in group.engine_options
        and capture.get("allocated_cache_blocks") != group.engine_options["test_kv_blocks"]
    ):
        raise ValueError("candidate did not use the requested private KV capacity")
    if (
        args.engine == "baseline"
        and "baseline_blocks" in group.engine_options
        and capture.get("allocated_cache_blocks") != group.engine_options["baseline_blocks"]
    ):
        raise ValueError("baseline did not use its declared KV capacity")
    metadata = dict(
        protocol=PROTOCOL,
        schema_version=1,
        source=source,
        registry_sha256=registry_sha,
        engine=args.engine,
        mode="collection_control" if control else "fixed_prefix",
        driver_pid=os.getpid(),
        capture_sha256=sha(output / "capture.json"),
        runtime=runtime.model_dump(),
        engine_evidence=evidence,
        kernel={
            "reference": config.REFERENCE_KERNEL_PATH,
            "baseline": config.BASELINE_KERNEL_PATH,
            "candidate": config.CANDIDATE_KERNEL_PATH,
        }[args.engine],
        model=dict(
            revision=config.MODEL_REVISION,
            config_sha256=config.MODEL_CONFIG_SHA256,
            tokenizer_sha256=config.TOKENIZER_SHA256,
            weights_sha256=config.MODEL_WEIGHTS_SHA256,
            dtype="bfloat16",
            vocab_size=config.VOCAB_SIZE,
        ),
        setup_captures=[_artifact(p, args.run_dir) for p in setup_paths],
    )
    if args.engine == "candidate":
        metadata.update(
            binary_sha256=sha(args.candidate_binary), build_source_id=evidence["build_source_id"]
        )
    atomic_json(output / "worker.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "collect",
            "worker",
            "collect-aux",
            "worker-aux",
            "assemble-observation",
            "assemble-authoritative",
            "faults",
            "observe",
            "authoritative",
        ],
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--group")
    parser.add_argument("--engine", choices=["reference", "baseline", "candidate"])
    parser.add_argument("--variant", choices=["primary", "replay", "control", "control-replay"])
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    for name in (
        "calibration-evidence",
        "calibration-manifest",
        "calibration-marker",
        "fault-evidence",
    ):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.action in ("collect", "worker", "collect-aux", "worker-aux") and (
        args.model_dir is None or args.group is None or args.engine is None or args.variant is None
    ):
        parser.error("collection requires model-dir, group, engine and variant")
    if args.action in ("worker", "worker-aux"):
        _worker(args)
        return
    try:
        if args.action in ("collect", "collect-aux"):
            result = collect_group(
                args.repo_root,
                args.run_dir,
                args.model_dir,
                args.group,
                args.engine,
                args.variant,
                args.candidate_binary,
                auxiliary=args.action == "collect-aux",
            )
        elif args.action == "faults":
            from golden_gen.layered_faults import generate_fault_evidence

            if args.manifest is None:
                parser.error("faults requires a complete calibration --manifest")
            evidence = generate_fault_evidence(args.repo_root, args.manifest)
            output = args.output or args.run_dir / "fault-evidence.json"
            if output.parent.resolve() != args.run_dir.resolve():
                raise ValueError("fault output must remain inside its dedicated run root")
            atomic_json(output, evidence)
            result = dict(
                protocol=PROTOCOL,
                schema_version=1,
                accepting=False,
                fault_observation_complete=True,
                evidence=_artifact(output, args.run_dir),
            )
        elif args.action in ("assemble-observation", "assemble-authoritative"):
            from golden_gen.layered_workflow import assemble_manifest

            names = (
                "calibration_evidence",
                "calibration_manifest",
                "calibration_marker",
                "fault_evidence",
            )
            calibration = {
                name: getattr(args, name) for name in names if getattr(args, name) is not None
            }
            prepared = assemble_manifest(
                args.repo_root,
                args.run_dir,
                authoritative=args.action == "assemble-authoritative",
                calibration=calibration or None,
            )
            output = args.manifest or args.run_dir / "manifest.json"
            if output.parent.resolve() != args.run_dir.resolve():
                raise ValueError("assembled manifest must be rooted beside its owner artifacts")
            atomic_json(output, prepared)
            result = dict(
                protocol=PROTOCOL,
                schema_version=1,
                accepting=False,
                assembly_complete=True,
                manifest=_artifact(output, args.run_dir),
            )
        else:
            if args.manifest is None:
                parser.error("comparison requires --manifest")
            authoritative = args.action == "authoritative"
            result = evaluate_manifest(args.repo_root, args.manifest, authoritative=authoritative)
            output = args.run_dir / f"{args.action}.json"
            atomic_json(output, result)
            if result.get("observation_complete") or result.get("accepting"):
                source = source_identity(args.repo_root)
                prior = (
                    bound_file(
                        args.manifest.parent,
                        json.loads(args.manifest.read_text())["calibration_marker"],
                    )
                    if authoritative
                    else None
                )
                write_marker(
                    args.run_dir,
                    "authoritative" if authoritative else "observation",
                    source,
                    [output, args.manifest],
                    prior,
                )
        print(json.dumps(result, sort_keys=True))
        if args.action == "authoritative" and result.get("verdict") != "PASS":
            raise SystemExit(2)
        if args.action == "observe" and not result.get("observation_complete"):
            raise SystemExit(2)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                dict(
                    protocol=PROTOCOL,
                    schema_version=1,
                    verdict="INVALID",
                    accepting=False,
                    reasons=[str(error)],
                )
            )
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
