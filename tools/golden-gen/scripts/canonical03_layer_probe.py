"""Private two-step localization. Not a fixture generator or acceptance stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from golden_gen.layer_diagnostic import (
    CHECKPOINTS,
    MANIFEST_SHA,
    compare,
    identity,
    require_prefix_equivalence,
    sha,
)


def reference(root: Path, repo: Path, model_path: Path, manifest_path: Path) -> None:
    from golden_gen.environment import validate_deterministic_environment
    from golden_gen.io import load_fixture
    from golden_gen.oracles.transformers_oracle import TransformersOracle

    validate_deterministic_environment(os.environ)
    source = identity(repo)
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("wrong BC reference manifest")
    manifest = json.loads(manifest_path.read_text())
    fixture = next(
        x
        for x in manifest["fixtures"]
        if x["prompt_id"] == "canonical_03" and x["oracle"] == "transformers"
    )
    fixture_path = manifest_path.parent / fixture["filename"]
    if sha(fixture_path) != fixture["sha256"]:
        raise ValueError("reference fixture changed")
    previous = load_fixture(fixture_path)
    prompt = next(
        json.loads(line)
        for line in (repo / "tools/golden-gen/prompts/canonical.jsonl").read_text().splitlines()
        if json.loads(line)["id"] == "canonical_03"
    )
    root.mkdir(mode=0o700)
    (root / "torch").mkdir(mode=0o700)
    import torch

    oracle = TransformersOracle(model_path)
    input_ids = oracle.tokenizer(prompt["prompt"], return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to("cuda")
    token_ids = input_ids[0].cpu().tolist()
    if not 0 < len(token_ids) <= 1024:
        raise ValueError("canonical_03 request length outside diagnostic bound")
    (root / "request.json").write_text(
        json.dumps(dict(prompt_id="canonical_03", token_ids=token_ids, decode_token=151667))
    )
    state: dict[str, Any] = {"step": -1, "records": [], "pending": None, "tokens": None}

    def checkpoint(name: str, output: Any) -> dict[str, Any]:
        if output.dtype != torch.bfloat16 or output.shape[0] != 1:
            raise ValueError("unexpected reference checkpoint dtype/batch")
        host = output[0].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        return dict(
            kind="checkpoint",
            name=name,
            dtype="BF16",
            shape=list(host.shape),
            bf16_bits=host.flatten().tolist(),
        )

    def embedding_hook(_module: Any, inputs: Any, output: Any) -> None:
        state["step"] += 1
        if state["step"] not in (0, 1):
            raise ValueError("reference exceeded the two-step diagnostic")
        state["tokens"] = inputs[0][0].cpu().tolist()
        state["pending"] = checkpoint("embedding", output)

    def rotary_hook(_module: Any, inputs: Any) -> None:
        positions = inputs[1][0].cpu().tolist()
        expected = token_ids if state["step"] == 0 else [151667]
        expected_positions = list(range(len(token_ids))) if state["step"] == 0 else [len(token_ids)]
        if state["tokens"] != expected or positions != expected_positions:
            raise ValueError("reference input differs from shared prefix")
        state["records"] = [
            dict(
                kind="header",
                diagnostic_only=True,
                accepting=False,
                prompt_id="canonical_03",
                step=state["step"],
                token_ids=expected,
                positions=positions,
            ),
            state.pop("pending"),
        ]
        flush_records()

    def flush_records() -> None:
        path = root / "torch" / f"step-{state['step']}.jsonl"
        with path.open("a") as output:
            for row in state["records"]:
                output.write(json.dumps(row) + "\n")
        state["records"] = []

    def hook(name: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            state["records"].append(checkpoint(name, output))
            if name == "final_norm":
                state["records"].append(dict(kind="trailer", complete=True, checkpoints=30))
            flush_records()

        return record

    handles = [
        oracle.model.model.embed_tokens.register_forward_hook(embedding_hook),
        oracle.model.model.rotary_emb.register_forward_pre_hook(rotary_hook),
    ]
    handles.extend(
        layer.register_forward_hook(hook(f"layer_{i}"))
        for i, layer in enumerate(oracle.model.model.layers)
    )
    handles.append(oracle.model.model.norm.register_forward_hook(hook("final_norm")))
    try:
        with (
            torch.inference_mode(),
            torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]),
        ):
            out = oracle.model.generate(
                input_ids,
                max_new_tokens=2,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=oracle.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )
        logits = np.stack([row[0].float().cpu().numpy() for row in out.logits])
        tokens = out.sequences[0, len(token_ids) :].cpu().numpy().astype(np.int64)
        require_prefix_equivalence(
            logits,
            tokens,
            previous["logits"].astype(np.float32),
            previous["token_ids"].astype(np.int64),
        )
        np.savez(root / "torch/logits.npz", logits=logits, tokens=tokens)
        report = dict(
            diagnostic_only=True,
            accepting=False,
            source=source,
            equivalent_to_bc_reference=True,
            reference_fixture_sha256=fixture["sha256"],
            script_sha256=sha(Path(__file__)),
            torch_version=torch.__version__,
            transformers_version=__import__("transformers").__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            attention_backend="SDPBackend.MATH",
            device=str(input_ids.device),
            checkpoints=CHECKPOINTS,
            artifacts={
                name: sha(root / name)
                for name in (
                    "request.json",
                    "torch/step-0.jsonl",
                    "torch/step-1.jsonl",
                    "torch/logits.npz",
                )
            },
        )
        (root / "reference.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        for handle in handles:
            handle.remove()
        oracle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "compare"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--previous-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "reference":
        if args.model_path is None or args.manifest is None:
            parser.error("reference requires --model-path and --manifest")
        reference(args.output_dir, args.repo_root, args.model_path, args.manifest)
    else:
        if args.previous_dir is None or args.manifest is None:
            parser.error("compare requires --previous-dir and --manifest")
        print(
            json.dumps(
                compare(args.output_dir, args.repo_root, args.previous_dir, args.manifest),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
