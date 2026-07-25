"""Load canonical and regression prompts from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from golden_gen.schema import PromptSpec


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
