"""Exercise the installed V1 sampler and BatchUpdate with CPU tensors only."""

from types import SimpleNamespace

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessors, MoveDirectionality
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from golden_gen.fixed_prefix import ReplayPlan
from golden_gen.fixed_prefix_oracles import capture_baseline
from golden_gen.fixed_prefix_vllm import FixedPrefixProcessor

events = []

plan = dict(
    protocol="layered-accuracy-v1",
    schema_version=1,
    execution_group_id="g",
    call_id="c",
    vocab_size=3,
    members=[dict(case_id="a", member_id="a", prompt=[2, 1], continuation=[0, 2])],
)
params = SamplingParams(
    temperature=0,
    max_tokens=2,
    ignore_eos=True,
    logprobs=-1,
    extra_args={"fixed_prefix_plan": plan, "fixed_prefix_member": "a"},
)
history = []
processor = FixedPrefixProcessor(None, torch.device("cpu"), False)
processor.update_state(
    BatchUpdate(batch_size=1, removed=[], added=[(0, params, [2, 1], history)], moved=[])
)
native_id = "0-8b3db2f6"
request_id = processor.request_identity(0, native_id)
assert request_id == 0
processor.execution = {
    0: dict(
        request_id=request_id,
        native_request_id=native_id,
        input_token_ids=[2],
        positions=[0, 1],
        kv_length=1,
        phase="prefill",
        sampling_allowed=False,
    )
}
partial = torch.tensor([[1.0, 3.0, 2.0]])
events.append(
    dict(plan_id=0, token_budget=1, members=[{**processor.execution[0], "completion_step": 0}])
)
assert torch.equal(processor.apply(partial), partial)
assert not processor.rows and not history
processor.execution = {
    0: dict(
        request_id=request_id,
        native_request_id=native_id,
        input_token_ids=[1],
        positions=[1, 2],
        kv_length=2,
        phase="prefill",
        sampling_allowed=True,
    )
}
metadata = SamplingMetadata(
    temperature=None,
    all_greedy=True,
    all_random=False,
    top_p=None,
    top_k=None,
    generators={},
    max_num_logprobs=-1,
    no_penalties=True,
    prompt_token_ids=None,
    frequency_penalties=torch.zeros(1),
    presence_penalties=torch.zeros(1),
    repetition_penalties=torch.ones(1),
    output_token_ids=[history],
    allowed_token_ids_mask=None,
    bad_words_token_ids={},
    logitsprocs=LogitsProcessors([processor]),
)
raw = torch.tensor([[1.0, 3.0, 2.0]])
events.append(
    dict(plan_id=1, token_budget=1, members=[{**processor.execution[0], "completion_step": 0}])
)
out = Sampler(logprobs_mode="raw_logits")(raw, metadata)
assert out.sampled_token_ids.tolist() == [[0]]
assert torch.equal(out.logprobs_tensors.logprobs, torch.tensor([[1.0, 3.0, 2.0]]))
assert processor.rows[0]["predicted_token_id"] == 1
history.append(0)
processor.update_state(
    BatchUpdate(
        batch_size=2, removed=[], added=[], moved=[(0, 1, MoveDirectionality.UNIDIRECTIONAL)]
    )
)
assert processor.request_identity(1, native_id) == request_id
try:
    processor.request_identity(1, "0-another-suffix")
except ValueError:
    pass
else:
    raise AssertionError("a member cannot silently switch opaque runtime identities")
processor.execution = {
    1: dict(
        request_id=request_id,
        native_request_id=native_id,
        input_token_ids=[0],
        positions=[2, 3],
        kv_length=3,
        phase="decode",
        sampling_allowed=True,
    )
}
forced = processor.apply(torch.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]))
events.append(
    dict(plan_id=2, token_budget=1, members=[{**processor.execution[1], "completion_step": 1}])
)
assert forced.argmax(-1).tolist() == [2, 2]
assert processor.rows[1]["advance_token_id"] == 2


class ReturnedOutput:
    def generate(self, prompts, params):
        assert prompts == [dict(prompt_token_ids=[2, 1])]
        assert len(params) == 1
        return [
            SimpleNamespace(
                request_id="public/opaque",
                prompt_token_ids=[2, 1],
                outputs=[
                    SimpleNamespace(
                        token_ids=[0, 2],
                        logprobs=[
                            {i: SimpleNamespace(logprob=v) for i, v in enumerate(row)}
                            for row in ([1.0, 3.0, 2.0], [3.0, 2.0, 1.0])
                        ],
                    )
                ],
            )
        ]

    def collective_rpc(self, method, timeout):
        assert method == "release_fixed_prefix_evidence"
        return [
            dict(
                rows=processor.rows,
                execution_events=events,
                request_bindings=processor.request_bindings(),
                allocated_cache_blocks=32,
                cache_block_size=16,
            )
        ]


captured = capture_baseline(
    ReplayPlan.model_validate(plan), SimpleNamespace(fixed_prefix=True, llm=ReturnedOutput())
)
assert captured["request_bindings"][0]["native_request_id"] == native_id
assert captured["request_bindings"][0]["external_request_id"] == "public/opaque"
processor.update_state(BatchUpdate(batch_size=0, removed=[1], added=[], moved=[]))
assert not processor.state
next_plan = {**plan, "call_id": "next-call"}
next_params = SamplingParams(
    temperature=0,
    max_tokens=2,
    ignore_eos=True,
    extra_args={"fixed_prefix_plan": next_plan, "fixed_prefix_member": "a"},
)
processor.update_state(
    BatchUpdate(batch_size=1, removed=[], added=[(0, next_params, [2, 1], [])], moved=[])
)
assert processor.request_identity(0, "opaque-next-call") == 1
assert not torch.cuda.is_initialized()
print("VLLM_FORCING_CPU_PASS")
