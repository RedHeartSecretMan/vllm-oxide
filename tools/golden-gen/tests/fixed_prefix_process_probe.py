"""Real CPU Torch tensors at the forcing adapter seam; never initialize CUDA."""

import torch

from golden_gen.fixed_prefix import ReplayPlan
from golden_gen.fixed_prefix_oracles import ReferenceForcer

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
print("REFERENCE_FORCING_CPU_PASS")
