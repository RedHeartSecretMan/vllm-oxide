"""CLI entrypoint for golden-gen harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from golden_gen.assets import build_fixture_archive, publish_release_bundle
from golden_gen.calibrate import (
    calibrate_from_fixtures,
    validate_calibration_coverage,
)
from golden_gen.config import ATTN_IMPLEMENTATION, MODEL_DTYPE
from golden_gen.generate import run_all
from golden_gen.manifest import build_expected_fixtures, build_manifest, write_manifest
from golden_gen.oracles.fake import FakeOracle
from golden_gen.prompts import discover_fixtures, load_prompts
from golden_gen.schema import BaselineCalibration, Manifest, PromptCategory, TolerancePolicy


def _resolve_prompts_dir() -> Path:
    """Find the prompts/ directory relative to this package."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent.parent / "prompts",
        pkg_dir.parent.parent / "prompts",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd() / "prompts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate golden fixtures for vllm-oxide from two oracles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Run oracles and produce fixtures")
    gen.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use fake oracle (no GPU, no model download). For local pipeline smoke "
            "testing ONLY -- NOT for CI. Goldens produced in --dry-run are synthetic "
            "and must never be published as release assets."
        ),
    )
    gen.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Where to write fixtures + manifest.json. Default: ./output",
    )
    gen.add_argument(
        "--only-category",
        type=str,
        choices=["canonical", "regression"],
        help="Run only canonical or only regression prompts.",
    )

    cal = subparsers.add_parser(
        "calibrate",
        help="Record baseline observations and a reviewed tolerance policy",
    )
    cal.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Path to directory containing manifest.json + .safetensors fixtures",
    )
    cal.add_argument(
        "--tolerance-policy-version",
        choices=["same-prefix-v1"],
        required=True,
        help="Versioned comparison semantics to record in the manifest",
    )
    cal.add_argument(
        "--l1-near-tie-max-abs-logit-gap",
        type=float,
        required=True,
        help="Reviewed L1 expected/actual candidate-logit gap threshold",
    )
    cal.add_argument(
        "--l2-atol",
        type=float,
        required=True,
        help="Reviewed absolute tolerance for same-prefix L2 comparison",
    )
    cal.add_argument(
        "--tolerance-policy-rationale",
        required=True,
        help="Reviewed rationale for selecting the acceptance thresholds",
    )
    cal.add_argument(
        "--tolerance-policy-evidence",
        action="append",
        required=True,
        help="Evidence identifier or URI; repeat for every reviewed source",
    )

    bundle = subparsers.add_parser(
        "bundle",
        help="Build the exact two-file local release bundle from calibrated fixtures",
    )
    bundle.add_argument(
        "--fixture-dir",
        type=Path,
        required=True,
        help="Directory containing schema-v4 manifest and declared fixtures",
    )
    bundle.add_argument(
        "--release-dir",
        type=Path,
        required=True,
        help="New independent directory to contain only the two release assets",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return _run_generate(args)
    elif args.command == "calibrate":
        return _run_calibrate(args)
    elif args.command == "bundle":
        return _run_bundle(args)
    else:
        parser.print_help()
        return 1


def _run_generate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = _resolve_prompts_dir()
    all_prompts = load_prompts(prompts_dir)
    expected_fixtures = build_expected_fixtures(discover_fixtures(all_prompts))

    only_category: PromptCategory | None = args.only_category
    selected_expected_fixtures = [
        fixture
        for fixture in expected_fixtures
        if only_category is None
        or (only_category == "canonical" and fixture.family in ("canonical", "batch"))
        or (only_category == "regression" and fixture.family == "regression")
    ]
    if only_category:
        all_prompts = [p for p in all_prompts if p.category == only_category]

    if not all_prompts:
        print("ERROR: No prompts loaded.", file=sys.stderr)
        return 1

    staging = TemporaryDirectory(prefix="golden-gen-", dir=output_dir.parent)
    staging_dir = Path(staging.name)

    oracle_specs: list[tuple[str, type[Any] | type[FakeOracle]]] = []

    if args.dry_run:
        oracle_specs = [
            ("transformers", FakeOracle),
            ("vllm", FakeOracle),
        ]
    else:
        from golden_gen.oracles.transformers_oracle import TransformersOracle
        from golden_gen.oracles.vllm_oracle import VllmOracle

        oracle_specs = [
            ("vllm", VllmOracle),
            ("transformers", TransformersOracle),
        ]

    all_fixtures: list[Any] = []

    existing_manifest_path = output_dir / "manifest.json"
    existing_calibration: BaselineCalibration | None = None
    existing_policy: TolerancePolicy | None = None
    if existing_manifest_path.exists():
        from golden_gen.manifest import read_manifest

        existing = read_manifest(existing_manifest_path)
        all_fixtures = list(existing.fixtures)
        if existing.baseline_calibration.candidate_atol > 0.0:
            existing_calibration = existing.baseline_calibration
            existing_policy = existing.tolerance_policy

    failed_oracles: list[str] = []
    generated_this_run: set[str] = set()
    for name, oracle_cls in oracle_specs:
        oracle = oracle_cls()
        if args.dry_run:
            oracle.name = name
        try:
            fixtures = run_all(
                [oracle],
                all_prompts,
                staging_dir,
                only_category=only_category,
            )
            new_keys = {(f.oracle, f.prompt_id) for f in fixtures}
            generated_this_run.update(
                f"{fixture.prompt_id}.{fixture.oracle}" for fixture in fixtures
            )
            all_fixtures = [f for f in all_fixtures if (f.oracle, f.prompt_id) not in new_keys]
            all_fixtures.extend(fixtures)
        except Exception as e:
            print(f"ERROR: oracle {name} failed: {e}", file=sys.stderr)
            failed_oracles.append(name)
        finally:
            oracle.close()

    if failed_oracles:
        failed = sum(
            1 for fixture in selected_expected_fixtures if fixture.oracle in failed_oracles
        )
        skipped = len(expected_fixtures) - len(selected_expected_fixtures)
        print(
            "Lifecycle totals: "
            f"expected={len(expected_fixtures)} discovered={len(expected_fixtures)} "
            f"generated={len(generated_this_run)} compared=0 skipped={skipped} failed={failed}",
            file=sys.stderr,
        )
        print(
            "ERROR: fixture generation incomplete; manifest was not published "
            f"(failed oracles: {', '.join(failed_oracles)})",
            file=sys.stderr,
        )
        staging.cleanup()
        return 1

    generated_ids = {f"{fixture.prompt_id}.{fixture.oracle}" for fixture in all_fixtures}
    expected_ids = {fixture.fixture_id for fixture in expected_fixtures}
    if only_category is None and generated_ids != expected_ids:
        missing = len(expected_ids - generated_ids)
        unexpected = len(generated_ids - expected_ids)
        print(
            "Lifecycle totals: "
            f"expected={len(expected_ids)} discovered={len(expected_ids)} "
            f"generated={len(generated_ids & expected_ids)} compared=0 skipped=0 "
            f"failed={missing + unexpected}",
            file=sys.stderr,
        )
        print(
            "ERROR: full fixture generation did not match the expected manifest contract; "
            "manifest was not published",
            file=sys.stderr,
        )
        staging.cleanup()
        return 1

    print(f"Generated {len(all_fixtures)} fixtures in {output_dir}")

    if existing_calibration is not None:
        baseline_calibration = existing_calibration
        assert existing_policy is not None
        tolerance_policy = existing_policy
        print(
            "Reusing existing baseline calibration: "
            f"candidate_atol={baseline_calibration.candidate_atol:.6f}"
        )
    else:
        baseline_calibration = BaselineCalibration(
            candidate_atol=0.0,
            observed_max_abs_diff=0.0,
            calibration_factor=2.0,
            method="pending -- run `golden-gen calibrate` to compute",
        )
        tolerance_policy = TolerancePolicy(
            version="same-prefix-v1",
            dtype=MODEL_DTYPE,
            kernel=ATTN_IMPLEMENTATION,
            l1_near_tie_max_abs_logit_gap=0.0,
            l2_atol=0.0,
            rationale="pending reviewed policy selection",
            evidence=[],
        )
    for fixture in all_fixtures:
        staged_path = staging_dir / fixture.filename
        if staged_path.exists():
            continue
        source_path = output_dir / fixture.filename
        if not source_path.is_file() or source_path.is_symlink():
            print(f"ERROR: regular existing fixture not found at {source_path}", file=sys.stderr)
            staging.cleanup()
            return 1
        staged_path.hardlink_to(source_path)
    staged_archive_path = staging_dir / "goldens-v0.2.tar.gz"
    try:
        archive = build_fixture_archive(staging_dir, all_fixtures, staged_archive_path)
    except (OSError, ValueError) as error:
        print(f"ERROR: fixture archive build failed: {error}", file=sys.stderr)
        staging.cleanup()
        return 1

    manifest = build_manifest(
        fixtures=all_fixtures,
        baseline_calibration=baseline_calibration,
        archive=archive,
        tolerance_policy=tolerance_policy,
        expected_fixtures=expected_fixtures,
    )
    manifest_path = output_dir / "manifest.json"
    staged_manifest_path = staging_dir / "manifest.json"
    write_manifest(manifest, staged_manifest_path)
    generated_filenames = {f"{fixture_id}.safetensors" for fixture_id in generated_this_run}
    for fixture_path in staging_dir.glob("*.safetensors"):
        if fixture_path.name in generated_filenames:
            fixture_path.replace(output_dir / fixture_path.name)
    staged_archive_path.replace(output_dir / archive.filename)
    staged_manifest_path.replace(manifest_path)
    staging.cleanup()
    print(f"Manifest written to {manifest_path}")
    generated = len(generated_this_run)
    expected = len(expected_fixtures)
    skipped = expected - len(selected_expected_fixtures)
    print(
        "Lifecycle totals: "
        f"expected={expected} discovered={expected} generated={generated} "
        f"compared=0 skipped={skipped} failed=0"
    )
    if existing_calibration is None:
        print(
            "NOTE: baseline calibration and tolerance policy are pending. Run "
            "`golden-gen calibrate --help` for the required reviewed inputs."
        )

    return 0


def _run_bundle(args: argparse.Namespace) -> int:
    try:
        release_dir = publish_release_bundle(Path(args.fixture_dir), Path(args.release_dir))
    except (OSError, ValueError) as error:
        print(f"ERROR: release bundle failed: {error}", file=sys.stderr)
        return 1
    print(f"Release bundle written to {release_dir}")
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    manifest_dir = Path(args.manifest_dir)
    manifest_path = manifest_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    from golden_gen.manifest import read_manifest

    manifest = read_manifest(manifest_path)
    calibrated_fixtures = validate_calibration_coverage(manifest_dir, manifest)
    calibration = calibrate_from_fixtures(manifest_dir)
    print(
        "Baseline calibration observed: "
        f"candidate_atol={calibration.candidate_atol:.6f}, "
        f"observed_max_abs_diff={calibration.observed_max_abs_diff:.6f}"
    )

    manifest.baseline_calibration = calibration
    manifest.tolerance_policy = TolerancePolicy(
        version=args.tolerance_policy_version,
        dtype=manifest.model.dtype,
        kernel=manifest.generation.attn_implementation,
        l1_near_tie_max_abs_logit_gap=args.l1_near_tie_max_abs_logit_gap,
        l2_atol=args.l2_atol,
        rationale=args.tolerance_policy_rationale,
        evidence=args.tolerance_policy_evidence,
    )
    manifest.calibrated_fixtures = calibrated_fixtures
    manifest = Manifest.model_validate(manifest.model_dump())
    write_manifest(manifest, manifest_path)
    print(f"Updated manifest written to {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
