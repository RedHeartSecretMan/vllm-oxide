"""Synthetic-only fixture generation used to test lifecycle plumbing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from golden_gen.assets import build_fixture_archive
from golden_gen.config import MODEL_DTYPE
from golden_gen.generate import run_all
from golden_gen.manifest import build_expected_fixtures, build_manifest, write_manifest
from golden_gen.oracles.fake import FakeOracle
from golden_gen.prompts import discover_fixtures, load_prompts
from golden_gen.release_protocol import dry_run_runtime_info, pinned_kernel_paths
from golden_gen.schema import BaselineCalibration, PromptCategory, TolerancePolicy


def run_dry_generate(
    args: argparse.Namespace,
    prompts_dir: Path,
    oracle_type: type[FakeOracle] = FakeOracle,
) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = dry_run_runtime_info()
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
        all_prompts = [prompt for prompt in all_prompts if prompt.category == only_category]
    if not all_prompts:
        print("ERROR: No prompts loaded.", file=sys.stderr)
        return 1

    staging = TemporaryDirectory(prefix="golden-gen-", dir=output_dir.parent)
    staging_dir = Path(staging.name)
    oracle_specs: list[tuple[str, type[Any]]] = [
        ("transformers", oracle_type),
        ("vllm", oracle_type),
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
        oracle.name = name
        try:
            fixtures = run_all(
                [oracle],
                all_prompts,
                staging_dir,
                only_category=only_category,
            )
            new_keys = {(fixture.oracle, fixture.prompt_id) for fixture in fixtures}
            generated_this_run.update(
                f"{fixture.prompt_id}.{fixture.oracle}" for fixture in fixtures
            )
            all_fixtures = [
                fixture
                for fixture in all_fixtures
                if (fixture.oracle, fixture.prompt_id) not in new_keys
            ]
            all_fixtures.extend(fixtures)
        except Exception as error:
            print(f"ERROR: oracle {name} failed: {error}", file=sys.stderr)
            failed_oracles.append(name)
        finally:
            oracle.close()

    if failed_oracles:
        failed = sum(fixture.oracle in failed_oracles for fixture in selected_expected_fixtures)
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

    if existing_calibration is not None:
        baseline_calibration = existing_calibration
        assert existing_policy is not None
        tolerance_policy = existing_policy
    else:
        baseline_calibration = BaselineCalibration(
            candidate_atol=0.0,
            observed_max_abs_diff=0.0,
            calibration_factor=2.0,
            method="pending -- run the isolated calibrate-baseline stage",
        )
        tolerance_policy = TolerancePolicy(
            version="same-prefix-v1",
            dtype=MODEL_DTYPE,
            kernel=pinned_kernel_paths().comparison_scope,
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
        runtime=runtime,
        kernel_paths=pinned_kernel_paths(),
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
    expected = len(expected_fixtures)
    skipped = expected - len(selected_expected_fixtures)
    print(f"Generated {len(all_fixtures)} fixtures in {output_dir}")
    print(f"Manifest written to {manifest_path}")
    print(
        "Lifecycle totals: "
        f"expected={expected} discovered={expected} generated={len(generated_this_run)} "
        f"compared=0 skipped={skipped} failed=0"
    )
    return 0
