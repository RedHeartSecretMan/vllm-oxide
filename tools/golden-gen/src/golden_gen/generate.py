from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from golden_gen.config import VOCAB_SIZE
from golden_gen.io import save_fixture
from golden_gen.oracles.base import Oracle
from golden_gen.schema import FixtureMetadata, OracleName, PromptCategory, PromptSpec


def run_all(
    oracles: list[Oracle],
    prompts: list[PromptSpec],
    output_dir: Path,
    *,
    only_category: PromptCategory | None = None,
) -> list[FixtureMetadata]:
    """For each (oracle, prompt): generate, save .safetensors fixture, return metadata.

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
        elif o.name == "nanovllm":
            oracle_names.append("nanovllm")
        elif o.name == "vllm_v1":
            oracle_names.append("vllm_v1")
        elif o.name == "fake":
            oracle_names.append("fake")
        else:
            oracle_names.append(o.name)  # type: ignore[arg-type]

    for prompt in prompts:
        if only_category and prompt.category != only_category:
            continue

        for oracle, oname in zip(oracles, oracle_names, strict=True):
            result = oracle.generate(prompt)

            filename = f"{prompt.id}.{oname}.safetensors"
            filepath = output_dir / filename

            logits: NDArray[np.float32] | None = None
            top5_indices: NDArray[np.int64] | None = None
            top5_logits: NDArray[np.float32] | None = None

            if prompt.category == "canonical":
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
            shape: tuple[int, int] = (
                (n_tokens, VOCAB_SIZE) if prompt.category == "canonical" else (0, 0)
            )

            fixtures.append(
                FixtureMetadata(
                    prompt_id=prompt.id,
                    category=prompt.category,
                    oracle=oname,
                    num_tokens=n_tokens,
                    logits_dtype="float32",
                    logits_shape=shape,
                    sha256=sha256,
                    filename=filename,
                )
            )

    return fixtures
