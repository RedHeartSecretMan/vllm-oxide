from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from golden_gen.schema import PromptSpec


@dataclass
class OracleResult:
    """Output of one oracle on one prompt."""

    token_ids: NDArray[np.int64]
    logits_per_step: NDArray[np.float32]
    top5_indices: NDArray[np.int64]
    top5_logits: NDArray[np.float32]
    n_prompt_tokens: int

    @classmethod
    def for_canonical(
        cls,
        token_ids: NDArray[np.int64],
        logits_per_step: NDArray[np.float32],
        n_prompt_tokens: int,
    ) -> OracleResult:
        """Create a canonical result (full logits, empty top-5)."""
        return cls(
            token_ids=token_ids,
            logits_per_step=logits_per_step,
            top5_indices=np.empty((0, 5), dtype=np.int64),
            top5_logits=np.empty((0, 5), dtype=np.float32),
            n_prompt_tokens=n_prompt_tokens,
        )

    @classmethod
    def for_regression(
        cls,
        token_ids: NDArray[np.int64],
        top5_indices: NDArray[np.int64],
        top5_logits: NDArray[np.float32],
        n_prompt_tokens: int,
    ) -> OracleResult:
        """Create a regression result (top-5 only, empty full logits)."""
        return cls(
            token_ids=token_ids,
            logits_per_step=np.empty((0, 0), dtype=np.float32),
            top5_indices=top5_indices,
            top5_logits=top5_logits,
            n_prompt_tokens=n_prompt_tokens,
        )


class Oracle(Protocol):
    """Protocol for oracle engines that generate text and return logits."""

    name: str

    def generate(self, prompt: PromptSpec) -> list[OracleResult]:
        """Run generation on a prompt and return results.

        For single prompts: returns a list of 1.
        For batch prompts (prompt.is_batch): returns a list of N, one per sub-prompt.
        """
        ...

    def close(self) -> None:
        """Release CUDA resources."""
        ...
