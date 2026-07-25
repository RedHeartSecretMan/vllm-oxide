from __future__ import annotations

import numpy as np

from golden_gen.config import CANONICAL_MAX_TOKENS, REGRESSION_MAX_TOKENS, VOCAB_SIZE
from golden_gen.oracles.base import OracleResult
from golden_gen.schema import PromptSpec


class FakeOracle:
    """Deterministic fake oracle for --dry-run mode.

    Generates synthetic logits from a seeded RNG for each prompt.
    Different oracle NAME values produce slightly different logits
    (via perturbation at ~1e-3 scale), so cross-validation detects
    realistic non-zero deviations.

    Shape matches the real oracle contract exactly.
    """

    name = "fake"

    def generate(self, prompt: PromptSpec) -> OracleResult:
        """Generate fake deterministic output for a prompt.

        Uses base seed from prompt.id (shared across oracles) plus
        a small per-oracle perturbation seeded from (prompt.id, self.name).
        """
        base_rng = np.random.default_rng(seed=hash(prompt.id) & 0xFFFFFFFF)
        perturb_rng = np.random.default_rng(seed=hash((prompt.id, self.name)) & 0xFFFFFFFF)

        if prompt.category == "canonical":
            max_tokens = CANONICAL_MAX_TOKENS
            n_tokens = int(base_rng.integers(16, max_tokens + 1))
            base_logits = base_rng.standard_normal((n_tokens, VOCAB_SIZE), dtype=np.float32) * 5.0
            perturb = perturb_rng.standard_normal((n_tokens, VOCAB_SIZE), dtype=np.float32) * 1e-3
            logits = base_logits + perturb
            token_ids = np.argmax(logits, axis=1).astype(np.int64)
            return OracleResult(
                token_ids=token_ids,
                logits_per_step=logits,
                top5_indices=np.empty((0, 5), dtype=np.int64),
                top5_logits=np.empty((0, 5), dtype=np.float32),
                n_prompt_tokens=10,
            )
        else:
            max_tokens = REGRESSION_MAX_TOKENS
            n_tokens = int(base_rng.integers(8, max_tokens + 1))
            base_logits = base_rng.standard_normal((n_tokens, VOCAB_SIZE), dtype=np.float32) * 5.0
            perturb = perturb_rng.standard_normal((n_tokens, VOCAB_SIZE), dtype=np.float32) * 1e-3
            logits = base_logits + perturb
            token_ids = np.argmax(logits, axis=1).astype(np.int64)
            top5_indices = np.argsort(-logits, axis=1)[:, :5].astype(np.int64)
            top5_logits = np.take_along_axis(logits, top5_indices, axis=1)
            return OracleResult(
                token_ids=token_ids,
                logits_per_step=np.empty((0, 0), dtype=np.float32),
                top5_indices=top5_indices,
                top5_logits=top5_logits,
                n_prompt_tokens=10,
            )

    def close(self) -> None:
        """Fake oracle has no resources to close."""
        pass
