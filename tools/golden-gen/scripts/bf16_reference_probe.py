"""Tiny non-accepting Qwen3 operator probe; never loads model weights or fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    from golden_gen.environment import validate_deterministic_environment

    validate_deterministic_environment(os.environ)
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb

    from golden_gen.worker_determinism import seed_and_enable_determinism

    seed_and_enable_determinism(torch)
    device = torch.device(args.device)
    with torch.inference_mode():
        x = torch.ones(4, dtype=torch.bfloat16, device=device)
        residual = torch.tensor([1 / 256, 1 / 256, 1 / 256, 0], dtype=torch.bfloat16, device=device)
        norm = Qwen3RMSNorm(4, eps=1e-6).to(dtype=torch.bfloat16, device=device)
        unrounded = x.float() + residual.float()
        legacy_norm = (unrounded / torch.sqrt(unrounded.square().mean() + 1e-6)).bfloat16()
        query = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16, device=device)
        angle = torch.ones((1, 1, 2), dtype=torch.float32, device=device)
        cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
        rotated, _ = apply_rotary_pos_emb(query, query, cos, sin)
        legacy_rope = torch.cat(
            (
                angle.cos()[..., :1] - angle.sin()[..., :1],
                angle.cos()[..., :1] + angle.sin()[..., :1],
            ),
            -1,
        ).bfloat16()
        result = {
            "diagnostic_only": True,
            "accepting": False,
            "model_weights_loaded": False,
            "fixtures_opened": [],
            "device": str(x.device),
            "torch_version": torch.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bf16_residual_sum": (x + residual).float().tolist(),
            "rmsnorm_reference": norm(x + residual).float().tolist(),
            "legacy_unrounded_sum_emulation": legacy_norm.float().tolist(),
            "rope_cos_bf16": cos.float().flatten().tolist(),
            "rope_sin_bf16": sin.float().flatten().tolist(),
            "rope_reference": rotated.float().flatten().tolist(),
            "legacy_fp32_rotation_emulation": legacy_rope.float().flatten().tolist(),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
