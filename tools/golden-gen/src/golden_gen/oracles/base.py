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
        *,
        token_ids: NDArray[np.int64],
        logits_per_step: NDArray[np.float32],
        n_prompt_tokens: int,
    ) -> OracleResult:
        """Build a result for a canonical prompt (full logits, no top-5)."""
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
        *,
        token_ids: NDArray[np.int64],
        top5_indices: NDArray[np.int64],
        top5_logits: NDArray[np.float32],
        n_prompt_tokens: int,
    ) -> OracleResult:
        """Build a result for a regression prompt (top-5 only, no full logits)."""
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

    def generate(self, prompt: PromptSpec) -> OracleResult:
        """Run generation on a single prompt and return logits + token IDs."""
        ...

    def close(self) -> None:
        """Release CUDA resources."""
        ...
