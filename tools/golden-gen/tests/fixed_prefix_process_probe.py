"""Real CPU Torch tensors at the forcing adapter seam; never initialize CUDA."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from golden_gen.execution_checks import verify_prefix_reuse
from golden_gen.fixed_prefix import ReplayPlan, validate_capture, validate_control_capture
from golden_gen.fixed_prefix_oracles import ReferenceForcer, capture_reference
from golden_gen.layered_release import NumericalCase

plan = ReplayPlan.model_validate(
    dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        execution_group_id="cpu-group",
        call_id="call",
        vocab_size=3,
        members=[dict(case_id="cpu", member_id="a", prompt=[2], continuation=[0, 2])],
    )
)
forcer = ReferenceForcer(plan)
raw = torch.tensor([[1.0, 3.0, 2.0]])
before = raw.clone()
forced = forcer(torch.tensor([[2]]), raw)
assert torch.equal(raw, before), "forcing polluted the tensor HF stores as output_logits"
assert forced.data_ptr() != raw.data_ptr()
assert forced.argmax(-1).tolist() == [0]
forcer(torch.tensor([[2, 0]]), raw)
assert [r["predicted_token_id"] for r in forcer.rows] == [1, 1]
assert [r["advance_token_id"] for r in forcer.rows] == [0, 2]
try:
    forcer(torch.tensor([[2, 1]]), raw)
except ValueError:
    pass
else:
    raise AssertionError("wrong frozen history was accepted")
assert not torch.cuda.is_initialized()
mixed = ReplayPlan.model_validate(
    dict(
        protocol="layered-accuracy-v1",
        schema_version=1,
        execution_group_id="mixed",
        call_id="call",
        vocab_size=3,
        members=[
            dict(case_id="short", member_id="a", prompt=[2], continuation=[0]),
            dict(case_id="long", member_id="b", prompt=[1], continuation=[2, 1]),
        ],
    )
)
mixed_forcer = ReferenceForcer(mixed)
mixed_forcer(torch.tensor([[2], [1]]), torch.tensor([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]]))
mixed_forcer(torch.tensor([[2, 0], [1, 2]]), torch.tensor([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]]))
assert [row["case_id"] for row in mixed_forcer.rows] == ["short", "long", "long"]

# Exercise the real HF generation/forward hooks, not just its logits processor.
# Only the external device allocator is redirected; release construction remains CUDA-only.
torch.manual_seed(0)
model = Qwen3ForCausalLM(
    Qwen3Config(
        vocab_size=3,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        attn_implementation="sdpa",
    )
).to(dtype=torch.bfloat16).eval()
oracle = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=0))
tensor = torch.tensor


def cpu_tensor(*args, **kwargs):
    if kwargs.get("device") == "cuda":
        kwargs["device"] = "cpu"
    return tensor(*args, **kwargs)


case = NumericalCase(
    split="calibration",
    plan=mixed,
    required_mechanisms={"reference": ["prefill", "decode"]},
    engine_options={},
    expected_cached_tokens={"reference": {"a": 0, "b": 0}},
)
with patch.object(torch, "tensor", side_effect=cpu_tensor):
    for control in (False, True):
        capture = capture_reference(mixed, oracle, control=control)
        rows = (validate_control_capture if control else validate_capture)(mixed, capture)
        assert verify_prefix_reuse(case, "reference", capture, rows) == {"a": 0, "b": 0}
        assert [event["members"][0]["cached_range"] for event in capture["execution_events"]] == [
            [0, 0],
            [0, 1],
        ]
assert not torch.cuda.is_initialized()
print("REFERENCE_FORCING_CPU_PASS")
