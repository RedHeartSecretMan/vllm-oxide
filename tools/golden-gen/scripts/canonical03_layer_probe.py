"""Private fixed-case localization. Not a fixture generator or acceptance stage."""

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
    REGRESSION11_MANIFEST_SHA,
    REGRESSION11_PREFIX,
    REGRESSION11_PROMPT,
    compare,
    identity,
    require_prefix_equivalence,
    require_regression11_equivalence,
    sha,
)


def reference(
    root: Path, repo: Path, model_path: Path, manifest_path: Path, regression11: bool = False
) -> None:
    from golden_gen.environment import validate_deterministic_environment
    from golden_gen.io import load_fixture
    from golden_gen.oracles.transformers_oracle import TransformersOracle

    validate_deterministic_environment(os.environ)
    source = identity(repo)
    prompt_id = "regression_11" if regression11 else "canonical_03"
    steps = 12 if regression11 else 2
    checkpoints = CHECKPOINTS[:11] if regression11 else CHECKPOINTS
    prefix = REGRESSION11_PREFIX if regression11 else [151667]
    if sha(manifest_path) != (REGRESSION11_MANIFEST_SHA if regression11 else MANIFEST_SHA):
        raise ValueError("wrong BC reference manifest")
    manifest = json.loads(manifest_path.read_text())
    fixture = next(
        x
        for x in manifest["fixtures"]
        if x["prompt_id"] == prompt_id and x["oracle"] == "transformers"
    )
    fixture_path = manifest_path.parent / fixture["filename"]
    if sha(fixture_path) != fixture["sha256"]:
        raise ValueError("reference fixture changed")
    previous = load_fixture(fixture_path)
    prompt_file = "regression.jsonl" if regression11 else "canonical.jsonl"
    prompt = next(
        json.loads(line)
        for line in (repo / "tools/golden-gen/prompts" / prompt_file).read_text().splitlines()
        if json.loads(line)["id"] == prompt_id
    )
    root.mkdir(mode=0o700)
    (root / "torch").mkdir(mode=0o700)
    import torch

    oracle = TransformersOracle(model_path)
    input_ids = oracle.tokenizer(
        prompt["prompt"], return_tensors="pt", add_special_tokens=regression11
    )["input_ids"].to("cuda")
    token_ids = input_ids[0].cpu().tolist()
    if not 0 < len(token_ids) <= 1024:
        raise ValueError("canonical_03 request length outside diagnostic bound")
    if regression11 and token_ids != REGRESSION11_PROMPT:
        raise ValueError("regression_11 original prompt tokens changed")
    (root / "request.json").write_text(
        json.dumps(
            dict(
                prompt_id=prompt_id,
                token_ids=token_ids,
                **({"decode_tokens": prefix} if regression11 else {"decode_token": 151667}),
            )
        )
    )
    state: dict[str, Any] = {
        "step": -1,
        "records": [],
        "pending": None,
        "tokens": None,
        "active_attention": False,
        "pending_qk_norm": {},
    }

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
        if state["step"] not in range(steps):
            raise ValueError("reference exceeded the fixed diagnostic")
        state["tokens"] = inputs[0][0].cpu().tolist()
        state["pending"] = checkpoint("embedding", output)

    def rotary_hook(_module: Any, inputs: Any) -> None:
        positions = inputs[1][0].cpu().tolist()
        expected = token_ids if state["step"] == 0 else [prefix[state["step"] - 1]]
        expected_positions = (
            list(range(len(token_ids)))
            if state["step"] == 0
            else [len(token_ids) + state["step"] - 1]
        )
        if state["tokens"] != expected or positions != expected_positions:
            raise ValueError("reference input differs from shared prefix")
        state["records"] = [
            dict(
                kind="header",
                diagnostic_only=True,
                accepting=False,
                prompt_id=prompt_id,
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
            if name == "layer0_v":
                # HF interleaves Q projection/norm and K projection/norm.
                # Only the two CPU snapshots wait for logical trace order.
                state["records"].extend(
                    state["pending_qk_norm"].pop(key) for key in ("layer0_q_norm", "layer0_k_norm")
                )
            if name == checkpoints[-1]:
                state["records"].append(
                    dict(kind="trailer", complete=True, checkpoints=len(checkpoints))
                )
            flush_records()

        return record

    def norm_hook(name: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            flat = output.reshape(output.shape[0], output.shape[1], -1)
            state["pending_qk_norm"][name] = checkpoint(name, flat)

        return record

    def active(_module: Any, _inputs: Any) -> None:
        state["active_attention"] = True
        state["repeat_events"] = []

    def inactive(_module: Any, _inputs: Any, _output: Any) -> None:
        state["active_attention"] = False

    def context_hook(module: Any, inputs: Any) -> None:
        hook("layer0_attention_context")(module, (), inputs[0])

    from importlib import import_module

    from transformers.models.qwen3 import modeling_qwen3

    sdpa_module: Any = import_module("transformers.integrations.sdpa_attention")

    original_rope = modeling_qwen3.apply_rotary_pos_emb
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    original_repeat = sdpa_module.repeat_kv

    def repeat_call(hidden: Any, n_rep: int) -> Any:
        output = original_repeat(hidden, n_rep)
        if state["active_attention"]:
            state["repeat_events"].append(
                dict(
                    n_rep=n_rep,
                    input_heads=hidden.shape[1],
                    output_heads=output.shape[1],
                    output_to_input_head=[head // n_rep for head in range(output.shape[1])],
                )
            )
        return output

    def rope_call(*args: Any, **kwargs: Any) -> Any:
        query, key = original_rope(*args, **kwargs)
        if state["active_attention"]:
            for name, tensor in (("layer0_q_rope", query), ("layer0_k_rope", key)):
                flat = tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)
                state["records"].append(checkpoint(name, flat))
            flush_records()
        return query, key

    def sdpa_call(
        query: Any,
        key: Any,
        value: Any,
        *,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Any:
        if state["active_attention"]:
            q_length, k_length = query.shape[-2], key.shape[-2]
            raw_mask = None
            visible = (
                np.tri(q_length, k_length, dtype=bool)
                if is_causal
                else np.ones((q_length, k_length), dtype=bool)
            )
            mask_contract: dict[str, Any] = dict(
                kind="visibility", shape=[q_length, k_length], allowed=visible.flatten().tolist()
            )
            if attn_mask is not None:
                host = attn_mask.detach().cpu().contiguous()
                raw_mask = dict(
                    dtype=str(host.dtype),
                    shape=list(host.shape),
                    bytes_hex=host.view(torch.uint8).numpy().tobytes().hex(),
                )
                array = host.double().numpy()
                expanded = np.broadcast_to(array, (1, query.shape[1], q_length, k_length))
                uniform = np.array_equal(expanded, np.broadcast_to(expanded[:, :1], expanded.shape))
                if host.dtype == torch.bool and uniform:
                    visible &= expanded[0, 0].astype(bool)
                elif uniform and np.all((array == 0) | np.isneginf(array)):
                    visible &= expanded[0, 0] == 0
                else:
                    mask_contract = dict(kind="unreduced_additive_or_head_specific", raw=raw_mask)
                if mask_contract["kind"] == "visibility":
                    mask_contract["allowed"] = visible.flatten().tolist()
            evidence = dict(
                diagnostic_only=True,
                accepting=False,
                step=state["step"],
                raw=dict(
                    backend="SDPA_MATH",
                    query_shape=list(query.shape),
                    key_shape=list(key.shape),
                    value_shape=list(value.shape),
                    is_causal=is_causal,
                    scale_argument_hex=None if scale is None else scale.hex(),
                    enable_gqa=enable_gqa,
                    repeat_kv_calls=state["repeat_events"],
                    mask=raw_mask,
                ),
                common=dict(
                    q_length=q_length,
                    k_length=k_length,
                    q_heads=query.shape[1],
                    kv_heads=key.shape[1],
                    head_dim=query.shape[-1],
                    head_mapping=[
                        head // (query.shape[1] // key.shape[1]) for head in range(query.shape[1])
                    ]
                    if enable_gqa
                    else list(range(query.shape[1])),
                    scale_argument=scale,
                    dropout_p=dropout_p,
                    mask=mask_contract,
                ),
            )
            with (root / "torch" / f"attention-{state['step']}.json").open("x") as output:
                json.dump(evidence, output, sort_keys=True)
        return original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    handles = [
        oracle.model.model.embed_tokens.register_forward_hook(embedding_hook),
        oracle.model.model.rotary_emb.register_forward_pre_hook(rotary_hook),
    ]
    first_layer = oracle.model.model.layers[0]
    handles.extend(
        (
            first_layer.self_attn.register_forward_pre_hook(active),
            first_layer.self_attn.register_forward_hook(inactive),
            first_layer.self_attn.q_norm.register_forward_hook(norm_hook("layer0_q_norm")),
            first_layer.self_attn.k_norm.register_forward_hook(norm_hook("layer0_k_norm")),
            first_layer.self_attn.o_proj.register_forward_pre_hook(context_hook),
        )
    )
    handles.append(first_layer.input_layernorm.register_forward_hook(hook("layer0_input_norm")))
    handles.extend(
        getattr(first_layer.self_attn, f"{name}_proj").register_forward_hook(hook(f"layer0_{name}"))
        for name in ("q", "k", "v")
    )
    if regression11:
        handles.append(first_layer.register_forward_hook(hook("layer_0")))
    else:
        handles.extend(
            layer.register_forward_hook(hook(f"layer_{i}"))
            for i, layer in enumerate(oracle.model.model.layers)
        )
        handles.append(oracle.model.model.norm.register_forward_hook(hook("final_norm")))
    modeling_qwen3.apply_rotary_pos_emb = rope_call
    torch.nn.functional.scaled_dot_product_attention = sdpa_call
    sdpa_module.repeat_kv = repeat_call
    try:
        with (
            torch.inference_mode(),
            torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]),
        ):
            out = oracle.model.generate(
                input_ids,
                max_new_tokens=steps,
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
        if regression11:
            # Match the original regression producer's CPU top-k, including tie ordering.
            top5 = torch.topk(torch.from_numpy(logits), k=5, dim=-1)
            indices = top5.indices.numpy().astype(np.int64)
            values = top5.values.numpy().astype(np.float32)
            require_regression11_equivalence(tokens, indices, values, previous)
            np.savez(
                root / "torch/logits.npz", tokens=tokens, top5_indices=indices, top5_logits=values
            )
        else:
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
            checkpoints=checkpoints,
            artifacts={
                name: sha(root / name)
                for name in [
                    "request.json",
                    "torch/logits.npz",
                    *(f"torch/step-{step}.jsonl" for step in range(steps)),
                    *(f"torch/attention-{step}.json" for step in range(steps)),
                ]
            },
        )
        if regression11:
            report.pop("equivalent_to_bc_reference")
            report["equivalent_to_original_regression_tokens_top5"] = True
            report["manifest_sha256"] = sha(manifest_path)
        (root / "reference.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        modeling_qwen3.apply_rotary_pos_emb = original_rope
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        sdpa_module.repeat_kv = original_repeat
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
    parser.add_argument(
        "--regression11",
        action="store_true",
        help="fixed original regression_11, layer zero only, rows 0..11",
    )
    args = parser.parse_args()
    if args.mode == "reference":
        if args.model_path is None or args.manifest is None:
            parser.error("reference requires --model-path and --manifest")
        reference(
            args.output_dir, args.repo_root, args.model_path, args.manifest, args.regression11
        )
    else:
        if args.previous_dir is None or args.manifest is None:
            parser.error("compare requires --previous-dir and --manifest")
        if args.regression11:
            from golden_gen.layer_diagnostic import compare_regression11_collection

            print(
                json.dumps(
                    compare_regression11_collection(
                        args.output_dir, args.repo_root, args.previous_dir, args.manifest
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        print(
            json.dumps(
                compare(args.output_dir, args.repo_root, args.previous_dir, args.manifest),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
