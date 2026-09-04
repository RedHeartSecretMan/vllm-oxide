"""Load canonical and regression prompts from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from golden_gen.schema import DiscoveredFixture, PromptSpec


def load_prompts(prompts_dir: str | Path) -> list[PromptSpec]:
    """Load all prompt specs from prompts_dir (canonical.jsonl + regression.jsonl).

    Args:
        prompts_dir: Directory containing canonical.jsonl and regression.jsonl.

    Returns:
        List of PromptSpec in file order (canonical first, then regression).
    """
    prompts_dir = Path(prompts_dir)
    prompts: list[PromptSpec] = []

    canonical_path = prompts_dir / "canonical.jsonl"
    if canonical_path.exists():
        prompts.extend(_load_jsonl(canonical_path))

    regression_path = prompts_dir / "regression.jsonl"
    if regression_path.exists():
        prompts.extend(_load_jsonl(regression_path))

    return prompts


def load_canonical(prompts_dir: str | Path) -> list[PromptSpec]:
    """Load only canonical prompts."""
    path = Path(prompts_dir) / "canonical.jsonl"
    return _load_jsonl(path) if path.exists() else []


def load_regression(prompts_dir: str | Path) -> list[PromptSpec]:
    """Load only regression prompts."""
    path = Path(prompts_dir) / "regression.jsonl"
    return _load_jsonl(path) if path.exists() else []


def discover_fixtures(prompts: list[PromptSpec]) -> list[DiscoveredFixture]:
    """Flatten prompt corpora into concrete canonical, batch, and regression cases."""
    discovered: list[DiscoveredFixture] = []
    for spec in prompts:
        if spec.is_batch:
            assert spec.sub_prompts is not None
            for index, prompt in enumerate(spec.sub_prompts):
                suffix = chr(ord("a") + index)
                discovered.append(
                    DiscoveredFixture(
                        prompt_id=f"{spec.id}{suffix}",
                        family="batch",
                        prompt=prompt,
                    )
                )
        else:
            discovered.append(
                DiscoveredFixture(
                    prompt_id=spec.id,
                    family=spec.category,
                    prompt=spec.prompt,
                )
            )
    required_families = {"canonical", "batch", "regression"}
    discovered_families = {case.family for case in discovered}
    missing = sorted(required_families - discovered_families)
    if missing:
        raise ValueError(f"missing required fixture families: {', '.join(missing)}")
    seen: set[str] = set()
    for case in discovered:
        if case.prompt_id in seen:
            raise ValueError(f"duplicate fixture identifier: {case.prompt_id}")
        seen.add(case.prompt_id)
    return discovered


def _load_jsonl(path: Path) -> list[PromptSpec]:
    """Parse a JSONL file into PromptSpec objects."""
    prompts: list[PromptSpec] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompts.append(PromptSpec.model_validate(data))
    return prompts
