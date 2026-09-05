"""Fixed BF16 SiLU diagnostic; no model weights or fixtures are opened."""

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

    from golden_gen.worker_determinism import seed_and_enable_determinism

    seed_and_enable_determinism(torch)
    with torch.inference_mode():
        gate = torch.tensor([2.0], dtype=torch.bfloat16, device=args.device)
        up = torch.tensor([0.515625], dtype=torch.bfloat16, device=args.device)
        silu = torch.nn.functional.silu(gate)
        result = silu * up
        legacy = (gate / (1 + (-gate).exp())) * up
        late_cast = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
        assert silu.float().tolist() == [1.7578125]
        assert result.float().tolist() == [0.90625]
        assert legacy.float().tolist() == late_cast.float().tolist() == [0.91015625]
        report = {
            "diagnostic_only": True,
            "accepting": False,
            "model_weights_loaded": False,
            "fixtures_opened": [],
            "device": str(gate.device),
            "torch_version": torch.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gate": gate.float().tolist(),
            "up": up.float().tolist(),
            "silu_reference": silu.float().tolist(),
            "gated_reference": result.float().tolist(),
            "legacy_low_precision_emulation": legacy.float().tolist(),
            "late_cast_emulation": late_cast.float().tolist(),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
