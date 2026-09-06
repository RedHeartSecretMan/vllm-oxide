"""Self-contained normative data definitions carried by the approved registry."""

from __future__ import annotations

from typing import Any


def input_definition(rule: str) -> dict[str, Any]:
    rounding = "round-to-nearest ties-to-even; each named BF16 materialization is observable"
    if rule == "materialized_halfway_sum_v1":
        return dict(
            x=[[1, 1, 1, 1]],
            residual=[[1 / 256, 1 / 256, 1 / 256, 0]],
            weight=[1, 1, 1, 1],
            epsilon=1e-6,
            input_dtype="bfloat16",
            rounding=rounding,
            computation=(
                "s=BF16(x+residual); variance=mean(FP32(s)^2); "
                "y=BF16(FP32(s)/sqrt(variance+epsilon)); output=BF16(y*weight)"
            ),
            faults=dict(
                missing_bf16_residual_round=(
                    "omit s materialization before the variance and normalization"
                ),
                wrong_epsilon="replace epsilon with 0.1, retain every materialization",
            ),
        )
    if rule == "gate_2_up_0.515625_v1":
        return dict(
            x=[[2, 0.515625]],
            layout="last axis: gate then up",
            input_dtype="bfloat16",
            rounding=rounding,
            computation=(
                "gate_activation=BF16(FP32(gate)/(1+exp(-FP32(gate)))); "
                "output=BF16(gate_activation*up)"
            ),
            faults=dict(
                bf16_opmath=(
                    "BF16 exp(-gate), BF16(1+exp), BF16(gate/denominator), then BF16 multiply by up"
                ),
                missing_intermediate_round=(
                    "compute gate/(1+exp(-gate))*up in FP32; round only the final product"
                ),
            ),
        )
    if rule == "nonconsecutive_positions_and_half_rotation_v1":
        return dict(
            shape=[3, 2, 128],
            positions=[0, 7, 255],
            theta=1000000,
            input_dtype="bfloat16",
            rounding=rounding,
            input=(
                "x[row,head,dimension]=(dimension mod 7-3)/4+head/8; row in [0,3), "
                "head in [0,2), dimension in [0,128)"
            ),
            angles=(
                "FP32(position)/FP32(theta^(FP32(2*j)/128)), j in [0,64); sine and "
                "cosine cast to BF16"
            ),
            computation=(
                "split x into L=x[...,0:64], R=x[...,64:128]; concatenate "
                "BF16(BF16(L*cos)-BF16(R*sin)), BF16(BF16(R*cos)+BF16(L*sin))"
            ),
            faults=dict(
                position_shift="add exactly one to each declared position before angles",
                wrong_half_rotation="negate the sine table, retain original positions",
            ),
        )
    if rule in (
        "asymmetric_qkv_causal_gqa_v1",
        "257_visible_tokens_noncontiguous_pages_v1",
        "noncontiguous_block_readback_v1",
    ):
        prefill = rule == "asymmetric_qkv_causal_gqa_v1"
        value = dict(
            query_tokens=5 if prefill else 1,
            kv_tokens=5 if prefill else 257,
            query_heads=16,
            kv_heads=8,
            head_dim=128,
            input_dtype="bfloat16",
            rounding=rounding,
            q="Q[t,h,0]=(t+1)*(h mod 3+1)/4; Q[t,h,d>0]=0",
            k="K[t,h,0]=(t mod 7+1)*(h mod 3+1)/8; K[t,h,d>0]=0",
            v="V[t,h,d]=(t mod 11)/8+h/16+(d mod 13)/64",
            construction=(
                "evaluate input formulas in FP32, cast each Q/K/V tensor to BF16 before use"
            ),
            head_mapping="KV head=floor(query head/2)",
            scale="FP32(1)/sqrt(FP32(128))",
            visible_tokens="keys 0..query_position inclusive"
            if prefill
            else "all 257 keys, query logical position 256",
            attention_reference=(
                "FP64 dot/scale, stable max-centered softmax with accurate "
                "summation, FP64 weighted V sum, final BF16 output"
            ),
            cache=dict(
                layers=1,
                blocks=2,
                page_size=256,
                block_table=[1, 0],
                plane_storage_shape=[2, 256, 8, 128],
                plane_storage_axes=["blocks", "page_tokens", "kv_heads", "head_dim"],
                separate_planes=["K", "V"],
                logical_readback_shape=[2, 257, 8, 128],
                logical_readback_axes=["K_or_V", "logical_tokens", "kv_heads", "head_dim"],
                logical_to_physical_slot="256+t for t in [0,256); slot 0 for t=256",
                readback=(
                    "stack K and V planes in that order; logical token order 0..256; "
                    "shape [2,257,8,128]"
                ),
            ),
        )
        if prefill:
            value["cache"] = dict(used=False, slot_mapping=[0, 1, 2, 3, 4], block_table=[])
            value["faults"] = dict(
                future_mask="each query sees all 5 keys",
                head_mapping="KV head=query head mod 8",
                scale="replace scale with 1",
            )
        elif rule == "noncontiguous_block_readback_v1":
            value["faults"] = dict(
                slot_swap="swap complete K and V plane outputs",
                stale_owner="return all-zero storage from a different owner",
            )
        else:
            value["faults"] = dict(
                missing_history="only key/value at logical position 256 remain visible",
                wrong_slot="use K plane values for the entire V plane",
            )
        return value
    if rule == "unique_and_multiway_max_with_filter_penalties_v1":
        return dict(
            shape=[3, 151936],
            input_dtype="float32",
            background_logit=-100,
            overrides=[{"7": 4, "12": 4}, {"7": 4, "12": 3}, {"29": 4, "30": 4, "31": 4}],
            histories=[[], [7], []],
            seed=0,
            params=[
                dict(
                    temperature=0,
                    top_k=None,
                    top_p=None,
                    presence_penalty=0,
                    frequency_penalty=0,
                    repetition_penalty=0,
                ),
                dict(
                    temperature=0,
                    top_k=None,
                    top_p=None,
                    presence_penalty=0,
                    frequency_penalty=2,
                    repetition_penalty=0,
                ),
                dict(
                    temperature=1,
                    top_k=1,
                    top_p=None,
                    presence_penalty=0,
                    frequency_penalty=0,
                    repetition_penalty=0,
                ),
            ],
            expected_token_ids=[7, 12, 29],
            tie_rule="lowest vocabulary token ID, not sorted rank",
            faults=dict(
                wrong_token_id="return the winning sorted rank zero for each row",
                wrong_tie="choose greatest token ID among equal maximal logits",
                ignored_penalty="skip the frequency penalty for the second row",
            ),
        )
    raise ValueError("no complete normative operator input definition")
