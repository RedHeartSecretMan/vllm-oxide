"""CLI entrypoint for golden-gen harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from golden_gen.cross_validate import (
    calibrate_tolerance,
    cross_validate_all,
    flag_suspicious_divergence,
)
from golden_gen.generate import run_all
from golden_gen.manifest import build_manifest, write_manifest
from golden_gen.oracles.fake import FakeOracle
from golden_gen.prompts import load_prompts
from golden_gen.schema import PromptCategory


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
        description=(
            "Generate golden fixtures for vllm-oxide from the oracle triangle. "
            "Runs HF Transformers, nano-vllm, and vLLM V1 on a fixed set of "
            "prompts, cross-validates outputs, and writes manifest.json + "
            ".safetensors fixtures."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use fake oracle (no GPU, no model download). For local pipeline smoke "
            "testing ONLY -- NOT for CI. Goldens produced in --dry-run are synthetic "
            "and must never be published as release assets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Where to write fixtures + manifest.json. Default: ./output",
    )
    parser.add_argument(
        "--only-oracle",
        type=str,
        action="append",
        choices=["transformers", "nanovllm", "vllm_v1"],
        help="Run only one oracle. Repeatable.",
        dest="only_oracles",
    )
    parser.add_argument(
        "--only-category",
        type=str,
        choices=["canonical", "regression"],
        help="Run only canonical or only regression prompts.",
    )
    parser.add_argument(
        "--no-cross-validate",
        action="store_true",
        help="Skip cross-validation step.",
    )
    parser.add_argument(
        "--cross-validate-only",
        action="store_true",
        help="Only run cross-validation on existing fixtures (skip oracle generation).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts_dir = _resolve_prompts_dir()
    all_prompts = load_prompts(prompts_dir)

    only_category: PromptCategory | None = args.only_category
    if only_category:
        all_prompts = [p for p in all_prompts if p.category == only_category]

    if not all_prompts:
        print("ERROR: No prompts loaded.", file=sys.stderr)
        return 1

    oracle_specs: list[tuple[str, type[Any] | type[FakeOracle]]] = []

    if args.dry_run:
        oracle_specs = [
            ("transformers", FakeOracle),
            ("nanovllm", FakeOracle),
            ("vllm_v1", FakeOracle),
        ]
    elif args.only_oracles:
        for name in args.only_oracles:
            if name == "transformers":
                from golden_gen.oracles.transformers_oracle import TransformersOracle
                oracle_specs.append((name, TransformersOracle))
            elif name == "nanovllm":
                from golden_gen.oracles.nanovllm_oracle import NanovllmOracle
                oracle_specs.append((name, NanovllmOracle))
            elif name == "vllm_v1":
                from golden_gen.oracles.vllm_v1_oracle import VllmV1Oracle
                oracle_specs.append((name, VllmV1Oracle))
    else:
        from golden_gen.oracles.nanovllm_oracle import NanovllmOracle
        from golden_gen.oracles.transformers_oracle import TransformersOracle
        from golden_gen.oracles.vllm_v1_oracle import VllmV1Oracle

        oracle_specs = [
            ("vllm_v1", VllmV1Oracle),
            ("nanovllm", NanovllmOracle),
            ("transformers", TransformersOracle),
        ]

    # Run oracles one at a time to avoid GPU memory exhaustion
    all_fixtures: list[Any] = []

    # If manifest.json already exists (e.g. from a previous --only-oracle run),
    # merge its fixtures so cross-validation sees the full oracle triangle.
    existing_manifest_path = output_dir / "manifest.json"
    if existing_manifest_path.exists():
        from golden_gen.manifest import read_manifest

        existing = read_manifest(existing_manifest_path)
        all_fixtures = list(existing.fixtures)

    if not args.cross_validate_only:
        for name, oracle_cls in oracle_specs:
            oracle = oracle_cls()
            if args.dry_run:
                oracle.name = name
            try:
                fixtures = run_all(
                    [oracle], all_prompts, output_dir, only_category=only_category,
                )
                # Replace any existing fixtures for this oracle+prompt combo
                # to avoid duplicates when re-running the same oracle.
                new_keys = {(f.oracle, f.prompt_id) for f in fixtures}
                all_fixtures = [f for f in all_fixtures if (f.oracle, f.prompt_id) not in new_keys]
                all_fixtures.extend(fixtures)
            except Exception as e:
                print(f"ERROR: oracle {name} failed: {e}", file=sys.stderr)
            finally:
                oracle.close()

    if not args.cross_validate_only:
        print(f"Generated {len(all_fixtures)} fixtures in {output_dir}")

    suspect_prompt_ids: list[str] = []

    # Cross-validate
    cross_validation: list[Any] = []
    tolerance = calibrate_tolerance({})

    if not args.no_cross_validate:
        canonical_prompts = [p for p in all_prompts if p.category == "canonical"]
        if canonical_prompts:
            results = {}
            for f in all_fixtures:
                if f.category == "canonical":
                    from golden_gen.io import load_fixture

                    data = load_fixture(output_dir / f.filename)
                    if "logits" in data:
                        results[(f.oracle, f.prompt_id)] = (
                            data["token_ids"],
                            data["logits"],
                        )
            if results:
                deviations, per_prompt_l2, per_prompt_max_abs = cross_validate_all(results)
                cross_validation = deviations
                tolerance = calibrate_tolerance(per_prompt_l2, per_prompt_max_abs)
                print(
                    f"Cross-validation: {len(deviations)} deviations, "
                    f"tolerance calibrated to atol={tolerance.atol:.6f}"
                )

                suspect_prompt_ids = flag_suspicious_divergence(per_prompt_l2)
                if suspect_prompt_ids:
                    print(
                        f"WARNING: oracle divergence > 0.1 on prompts {suspect_prompt_ids}.",
                        file=sys.stderr,
                    )
                    print(
                        "Per spec T8 Q8.2, these fixtures are SUSPECT -- "
                        "investigate the oracle before trusting these goldens. "
                        "Recorded as 'suspect' in manifest.",
                        file=sys.stderr,
                    )

    # Build and write manifest
    manifest = build_manifest(
        fixtures=all_fixtures,
        tolerance=tolerance,
        cross_validation=cross_validation,
        suspect_prompt_ids=suspect_prompt_ids,
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"Manifest written to {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
