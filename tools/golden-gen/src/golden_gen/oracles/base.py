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


class Oracle(Protocol):
    """Protocol for oracle engines that generate text and return logits."""

    name: str

    def generate(self, prompt: PromptSpec) -> OracleResult:
        """Run generation on a single prompt and return logits + token IDs."""
        ...

    def close(self) -> None:
        """Release CUDA resources."""
        ...
