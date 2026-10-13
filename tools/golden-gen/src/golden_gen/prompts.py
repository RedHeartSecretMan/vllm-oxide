"""Load canonical and regression prompts from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from golden_gen.schema import PromptSpec, validate_fixture_component


def validate_prompt_inventory(prompts: list[PromptSpec]) -> None:
    """Reject collisions after batch expansion before invoking any oracle."""
    seen: set[str] = set()
    for original in prompts:
        spec = PromptSpec.model_validate(original.model_dump())
        count = len(spec.sub_prompts or []) if spec.is_batch else 1
        identifiers = (
            [f"{spec.id}{chr(ord('a') + i)}" for i in range(count)] if spec.is_batch else [spec.id]
        )
        for identifier in identifiers:
            validate_fixture_component(identifier)
            key = identifier.casefold()
            if key in seen:
                raise ValueError(
                    f"duplicate fixture identifier: {identifier} (after batch expansion)"
                )
            seen.add(key)


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

    validate_prompt_inventory(prompts)
    return prompts


def load_canonical(prompts_dir: str | Path) -> list[PromptSpec]:
    """Load only canonical prompts."""
    path = Path(prompts_dir) / "canonical.jsonl"
    return _load_jsonl(path) if path.exists() else []


def load_regression(prompts_dir: str | Path) -> list[PromptSpec]:
    """Load only regression prompts."""
    path = Path(prompts_dir) / "regression.jsonl"
    return _load_jsonl(path) if path.exists() else []


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
    validate_prompt_inventory(prompts)
    return prompts
