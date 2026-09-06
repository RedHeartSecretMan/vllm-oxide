"""Exercise the installed V1 sampler and BatchUpdate with CPU tensors only."""

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessors, MoveDirectionality
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from golden_gen.fixed_prefix_vllm import FixedPrefixProcessor

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
processor.execution = {
    0: dict(
        request_id=7,
        input_token_ids=[2],
        positions=[0, 1],
        kv_length=1,
        phase="prefill",
        sampling_allowed=False,
    )
}
partial = torch.tensor([[1.0, 3.0, 2.0]])
assert torch.equal(processor.apply(partial), partial)
assert not processor.rows and not history
processor.execution = {
    0: dict(
        request_id=7,
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
processor.execution = {
    1: dict(
        request_id=7,
        input_token_ids=[0],
        positions=[2, 3],
        kv_length=3,
        phase="decode",
        sampling_allowed=True,
    )
}
forced = processor.apply(torch.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]))
assert forced.argmax(-1).tolist() == [2, 2]
assert processor.rows[1]["advance_token_id"] == 2
processor.update_state(BatchUpdate(batch_size=0, removed=[1], added=[], moved=[]))
assert not processor.state
assert not torch.cuda.is_initialized()
print("VLLM_FORCING_CPU_PASS")
