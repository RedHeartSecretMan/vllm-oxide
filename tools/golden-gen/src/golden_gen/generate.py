from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from golden_gen.config import VOCAB_SIZE
from golden_gen.io import save_fixture
from golden_gen.oracles.base import Oracle, OracleResult
from golden_gen.schema import FixtureMetadata, OracleName, PromptCategory, PromptSpec


def _validate_oracle_result(
    result: OracleResult,
    category: PromptCategory,
    *,
    oracle_name: OracleName,
    prompt_id: str,
) -> None:
    """Reject internally inconsistent oracle output before writing any fixture."""
    token_ids = result.token_ids
    valid_tokens = (
        token_ids.ndim == 1
        and token_ids.dtype == np.int64
        and len(token_ids) > 0
        and bool(np.all((token_ids >= 0) & (token_ids < VOCAB_SIZE)))
        and result.n_prompt_tokens > 0
    )
    if not valid_tokens:
        raise ValueError(f"invalid {category} oracle result for {prompt_id}.{oracle_name}: tokens")

    n_tokens = len(token_ids)
    if category == "canonical":
        valid_shape = (
            result.logits_per_step.shape == (n_tokens, VOCAB_SIZE)
            and result.logits_per_step.dtype == np.float32
            and result.top5_indices.shape == (0, 5)
            and result.top5_indices.dtype == np.int64
            and result.top5_logits.shape == (0, 5)
            and result.top5_logits.dtype == np.float32
        )
    else:
        valid_shape = (
            result.logits_per_step.shape == (0, 0)
            and result.logits_per_step.dtype == np.float32
            and result.top5_indices.shape == (n_tokens, 5)
            and result.top5_indices.dtype == np.int64
            and result.top5_logits.shape == (n_tokens, 5)
            and result.top5_logits.dtype == np.float32
            and bool(np.all((result.top5_indices >= 0) & (result.top5_indices < VOCAB_SIZE)))
        )
    if not valid_shape:
        raise ValueError(
            f"invalid {category} oracle result for {prompt_id}.{oracle_name}: tensor shape"
        )


def _save_fixture(
    result: OracleResult,
    prompt_id: str,
    oracle_name: OracleName,
    category: PromptCategory,
    output_dir: Path,
) -> FixtureMetadata:
    """Save a single fixture file and return its metadata."""
    filename = f"{prompt_id}.{oracle_name}.safetensors"
    filepath = output_dir / filename

    logits: NDArray[np.float32] | None = None
    top5_indices: NDArray[np.int64] | None = None
    top5_logits: NDArray[np.float32] | None = None

    if category == "canonical":
        logits = result.logits_per_step
    else:
        top5_indices = result.top5_indices
        top5_logits = result.top5_logits

    sha256 = save_fixture(
        path=filepath,
        token_ids=result.token_ids,
        logits=logits,
        top5_indices=top5_indices,
        top5_logits=top5_logits,
        n_prompt_tokens=result.n_prompt_tokens,
    )

    n_tokens = len(result.token_ids)
    shape: tuple[int, int] = (n_tokens, VOCAB_SIZE) if category == "canonical" else (0, 0)

    return FixtureMetadata(
        prompt_id=prompt_id,
        category=category,
        oracle=oracle_name,
        num_tokens=n_tokens,
        logits_dtype="float32",
        logits_shape=shape,
        sha256=sha256,
        filename=filename,
    )


def run_all(
    oracles: list[Oracle],
    prompts: list[PromptSpec],
    output_dir: Path,
    *,
    only_category: PromptCategory | None = None,
    resource_guard: Callable[[], None] | None = None,
) -> list[FixtureMetadata]:
    """For each (oracle, prompt): generate, save .safetensors fixture, return metadata.

    For batch prompts (prompt.is_batch), one fixture per sub-prompt is saved
    with a suffixed ID (e.g., canonical_05a, canonical_05b, ...).

    Args:
        oracles: List of Oracle instances.
        prompts: List of PromptSpec to generate.
        output_dir: Directory to write fixture files.
        only_category: If set, only generate prompts of this category.

    Returns:
        List of FixtureMetadata for all generated fixtures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures: list[FixtureMetadata] = []
    oracle_names: list[OracleName] = []
    for o in oracles:
        if o.name == "transformers":
            oracle_names.append("transformers")
        elif o.name == "vllm":
            oracle_names.append("vllm")
        elif o.name == "fake":
            oracle_names.append("fake")
        else:
            oracle_names.append(o.name)  # type: ignore[arg-type]

    for prompt in prompts:
        if resource_guard is not None:
            resource_guard()
        if only_category and prompt.category != only_category:
            continue

        for oracle, oname in zip(oracles, oracle_names, strict=True):
            results = oracle.generate(prompt)
            expected_results = len(prompt.sub_prompts or []) if prompt.is_batch else 1
            if len(results) != expected_results:
                raise ValueError(
                    f"oracle {oname} returned {len(results)} results for {prompt.id}; "
                    f"expected {expected_results}"
                )
            for result in results:
                _validate_oracle_result(
                    result,
                    prompt.category,
                    oracle_name=oname,
                    prompt_id=prompt.id,
                )

            if prompt.is_batch:
                # Save one fixture per sub-prompt: canonical_05a, canonical_05b, ...
                for i, result in enumerate(results):
                    suffix = chr(ord("a") + i)
                    sub_id = f"{prompt.id}{suffix}"
                    fixture = _save_fixture(
                        result,
                        prompt_id=sub_id,
                        oracle_name=oname,
                        category=prompt.category,
                        output_dir=output_dir,
                    )
                    fixtures.append(fixture)
            else:
                fixture = _save_fixture(
                    results[0],
                    prompt_id=prompt.id,
                    oracle_name=oname,
                    category=prompt.category,
                    output_dir=output_dir,
                )
                fixtures.append(fixture)
        if resource_guard is not None:
            resource_guard()

    return fixtures
