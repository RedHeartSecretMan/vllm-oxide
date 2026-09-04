"""Thin argparse adapter for golden generation and release-evidence modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from golden_gen.assets import publish_release_bundle
from golden_gen.calibrate import calibrate_from_fixtures, validate_calibration_coverage
from golden_gen.dry_run import run_dry_generate
from golden_gen.environment import collect_release_runtime
from golden_gen.manifest import write_manifest
from golden_gen.observation import (
    approve_manifest_policy,
    canonical_observation_json,
    observation_exit_code,
    observe_capture_replays,
    verify_candidate_capture_replay,
)
from golden_gen.oracle_run import (
    assemble_release_fixtures,
    generate_oracle_run,
    verify_fresh_replay,
)
from golden_gen.oracles.fake import FakeOracle
from golden_gen.report import ReleaseReportInput, render_release_report
from golden_gen.schema import Manifest
from golden_gen.stages import write_stage_marker


def _resolve_prompts_dir() -> Path:
    package = Path(__file__).resolve().parent
    for candidate in (package.parent.parent.parent / "prompts", package.parent.parent / "prompts"):
        if candidate.exists():
            return candidate
    return Path.cwd() / "prompts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate golden fixtures and validate vllm-oxide release evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="Record the pinned release environment")
    preflight.add_argument("--model-dir", type=Path, required=True)
    preflight.add_argument("--repo-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    generate = commands.add_parser("generate", help="Synthetic lifecycle smoke test only")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--output-dir", type=Path, default=Path("./output"))
    generate.add_argument("--only-category", choices=["canonical", "regression"])

    one = commands.add_parser("generate-oracle", help="Generate one fresh oracle corpus")
    one.add_argument("--oracle", choices=["transformers", "vllm"], required=True)
    one.add_argument("--runtime-record", type=Path, required=True)
    one.add_argument("--prompts-dir", type=Path, default=Path("./prompts"))
    one.add_argument("--output-dir", type=Path, required=True)

    replay = commands.add_parser("verify-replay", help="Verify one oracle replay")
    replay.add_argument("--oracle", choices=["transformers", "vllm"], required=True)
    replay.add_argument("--primary-dir", type=Path, required=True)
    replay.add_argument("--replay-dir", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)

    assemble = commands.add_parser("assemble", help="Assemble the pending schema-v4 corpus")
    assemble.add_argument("--runtime-record", type=Path, required=True)
    assemble.add_argument("--reference-dir", type=Path, required=True)
    assemble.add_argument("--baseline-dir", type=Path, required=True)
    assemble.add_argument("--prompts-dir", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)

    baseline = commands.add_parser(
        "calibrate-baseline", help="Record baseline evidence without choosing thresholds"
    )
    baseline.add_argument("--manifest-dir", type=Path, required=True)

    observe = commands.add_parser("observe", help="Write a non-accepting calibration observation")
    observe.add_argument("--manifest", type=Path, required=True)
    observe.add_argument("--primary-dir", type=Path, required=True)
    observe.add_argument("--replay-dir", type=Path, required=True)
    observe.add_argument("--output", type=Path, required=True)

    candidate = commands.add_parser(
        "verify-candidate-replay", help="Verify all candidate captures bit-identically"
    )
    candidate.add_argument("--primary-dir", type=Path, required=True)
    candidate.add_argument("--replay-dir", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)

    approve = commands.add_parser("approve-policy", help="Apply the tracked mechanical proposal")
    approve.add_argument("--manifest", type=Path, required=True)
    approve.add_argument("--observation", type=Path, required=True)
    approve.add_argument("--repo-root", type=Path, required=True)
    approve.add_argument("--rationale", required=True)

    report = commands.add_parser("report", help="Render the evidence-only release report")
    for name in ("manifest", "observation", "comparison", "benchmark"):
        report.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "observation-commit",
        "observation-tree",
        "policy-checkpoint-commit",
        "policy-checkpoint-tree",
        "measurement-commit",
        "measurement-tree",
    ):
        report.add_argument(f"--{name}", required=True)
    report.add_argument("--limitation", action="append", required=True)
    report.add_argument("--output", type=Path, required=True)

    bundle = commands.add_parser("bundle", help="Build the exact two-file local bundle")
    bundle.add_argument("--fixture-dir", type=Path, required=True)
    bundle.add_argument("--release-dir", type=Path, required=True)

    marker = commands.add_parser("stage-marker", help="Write a content-bound stage marker")
    marker.add_argument("--run-root", type=Path, required=True)
    marker.add_argument("--stage", required=True)
    marker.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "preflight": _run_preflight,
        "generate": _run_generate,
        "generate-oracle": _run_generate_oracle,
        "verify-replay": _run_verify_replay,
        "assemble": _run_assemble,
        "calibrate-baseline": _run_calibrate_baseline,
        "observe": _run_observe,
        "verify-candidate-replay": _run_verify_candidate_replay,
        "approve-policy": _run_approve_policy,
        "report": _run_report,
        "bundle": _run_bundle,
        "stage-marker": _run_stage_marker,
    }
    return handlers[args.command](args)


def _fresh_output(path: Path, description: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{description} must be fresh and non-existing")


def _run_preflight(args: argparse.Namespace) -> int:
    try:
        _fresh_output(args.output, "runtime record output")
        runtime = collect_release_runtime(args.model_dir, args.repo_root)
        with args.output.open("xb") as output:
            output.write(runtime.model_dump_json(indent=2).encode())
            output.write(b"\n")
    except (OSError, ValueError) as error:
        print(f"ERROR: environment preflight failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    if not args.dry_run:
        print(
            "ERROR: release generation must use separate generate-oracle/replay/assemble stages",
            file=sys.stderr,
        )
        return 1
    return run_dry_generate(args, _resolve_prompts_dir(), FakeOracle)


def _run_generate_oracle(args: argparse.Namespace) -> int:
    try:
        generate_oracle_run(args.oracle, args.runtime_record, args.prompts_dir, args.output_dir)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {args.oracle} oracle generation failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_verify_replay(args: argparse.Namespace) -> int:
    try:
        verify_fresh_replay(args.oracle, args.primary_dir, args.replay_dir, args.output)
    except (OSError, ValueError) as error:
        print(f"ERROR: {args.oracle} replay verification failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_assemble(args: argparse.Namespace) -> int:
    try:
        assemble_release_fixtures(
            args.runtime_record,
            args.reference_dir,
            args.baseline_dir,
            args.prompts_dir,
            args.output_dir,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: fixture assembly failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_calibrate_baseline(args: argparse.Namespace) -> int:
    manifest_path = args.manifest_dir / "manifest.json"
    try:
        manifest = Manifest.from_json(manifest_path)
        policy = manifest.tolerance_policy
        if policy.evidence or policy.l1_near_tie_max_abs_logit_gap or policy.l2_atol:
            raise ValueError("baseline calibration requires a pending zero-threshold manifest")
        manifest.calibrated_fixtures = validate_calibration_coverage(args.manifest_dir, manifest)
        manifest.baseline_calibration = calibrate_from_fixtures(args.manifest_dir)
        manifest = Manifest.model_validate(manifest.model_dump())
        write_manifest(manifest, manifest_path)
    except (OSError, ValueError) as error:
        print(f"ERROR: baseline calibration failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_observe(args: argparse.Namespace) -> int:
    try:
        _fresh_output(args.output, "observation output")
        record = observe_capture_replays(args.manifest, args.primary_dir, args.replay_dir)
        with args.output.open("xb") as output:
            output.write(canonical_observation_json(record))
    except (OSError, ValueError) as error:
        print(f"ERROR: calibration observation failed: {error}", file=sys.stderr)
        return 1
    return observation_exit_code(record)


def _run_verify_candidate_replay(args: argparse.Namespace) -> int:
    try:
        _fresh_output(args.output, "candidate replay output")
        evidence = verify_candidate_capture_replay(args.primary_dir, args.replay_dir)
        with args.output.open("xb") as output:
            output.write((json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
    except (OSError, ValueError) as error:
        print(f"ERROR: candidate replay verification failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_approve_policy(args: argparse.Namespace) -> int:
    try:
        approve_manifest_policy(args.manifest, args.observation, args.repo_root, args.rationale)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: tolerance policy approval failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_report(args: argparse.Namespace) -> int:
    try:
        _fresh_output(args.output, "release report output")
        manifest_bytes = args.manifest.read_bytes()
        manifest = Manifest.model_validate_json(manifest_bytes)
        observation_bytes = args.observation.read_bytes()
        observation = json.loads(observation_bytes)
        comparison = json.loads(args.comparison.read_bytes())
        benchmark = json.loads(args.benchmark.read_bytes())
        lifecycle = comparison["lifecycle"]
        if comparison.get("overall") is not True or lifecycle.get("compared") != 56:
            raise ValueError("authoritative comparison is not an exact complete pass")
        evidence = ReleaseReportInput(
            observation_commit=args.observation_commit,
            observation_tree=args.observation_tree,
            policy_checkpoint_commit=args.policy_checkpoint_commit,
            policy_checkpoint_tree=args.policy_checkpoint_tree,
            measurement_commit=args.measurement_commit,
            measurement_tree=args.measurement_tree,
            runtime=manifest.runtime,
            kernel_paths=manifest.kernel_paths,
            lifecycle={
                "expected": lifecycle["expected"],
                "discovered": lifecycle["discovered"],
                "generated": lifecycle["generated"],
                "calibration_compared": lifecycle["compared"],
                "reference_compared": len(comparison["reference_correctness"]["l1"]),
                "missing": lifecycle["missing"],
                "unexpected": lifecycle["unexpected"],
                "skipped": lifecycle["skipped"],
                "failed": lifecycle["failed"],
                "duplicate": 0,
                "stale": 0,
                "unmatched": 0,
            },
            tolerance={
                "l1_near_tie_max_abs_logit_gap": (
                    manifest.tolerance_policy.l1_near_tie_max_abs_logit_gap
                ),
                "l2_atol": manifest.tolerance_policy.l2_atol,
                "observation_sha256": hashlib.sha256(observation_bytes).hexdigest(),
                "raw_evidence_sha256": observation["identity"]["raw_evidence_sha256"],
            },
            benchmark=benchmark["workloads"],
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            archive_sha256=manifest.archive.sha256,
            limitations=args.limitation,
        )
        with args.output.open("xb") as output:
            output.write(render_release_report(evidence).encode())
    except (KeyError, OSError, ValueError) as error:
        print(f"ERROR: release report generation failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_bundle(args: argparse.Namespace) -> int:
    try:
        publish_release_bundle(args.fixture_dir, args.release_dir)
    except (OSError, ValueError) as error:
        print(f"ERROR: release bundle failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_stage_marker(args: argparse.Namespace) -> int:
    try:
        write_stage_marker(args.run_root, args.stage, args.repo_root)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: stage marker failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
