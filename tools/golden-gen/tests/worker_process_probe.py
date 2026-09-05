"""CPU-only spawn probe; run with the already prepared release Python interpreter."""

import json
import multiprocessing
import sys
import types


def child_probe(connection):
    import torch

    before = torch.are_deterministic_algorithms_enabled()
    observed = {}

    class ExternalWorker:
        def __init__(self):
            observed["constructor_enabled"] = torch.are_deterministic_algorithms_enabled()
            assert not torch.cuda.is_initialized()

        def init_device(self):
            observed["init_device_enabled"] = torch.are_deterministic_algorithms_enabled()
            assert not torch.cuda.is_initialized()

    # The sole fake is the external GPU-owning adapter; execute our real worker
    # constructor and init_device hook in a real torch spawn process without CUDA.
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.v1",
        "vllm.v1.worker",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    gpu_module = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_module.Worker = ExternalWorker
    sys.modules[gpu_module.__name__] = gpu_module
    attention = types.ModuleType("vllm.model_executor.layers.attention")
    attention.Attention = type("Attention", (), {})
    sys.modules[attention.__name__] = attention
    from golden_gen.oracles.vllm_oracle import baseline_engine_kwargs
    from golden_gen.oracles.vllm_worker import DeterministicWorker

    assert baseline_engine_kwargs()["worker_cls"] == (
        f"{DeterministicWorker.__module__}.{DeterministicWorker.__qualname__}"
    )
    worker = DeterministicWorker()
    worker.init_device()

    connection.send(
        {
            "enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "enabled_before_setup": before,
            **observed,
        }
    )
    connection.close()


if __name__ == "__main__":
    import torch

    from golden_gen.oracles.vllm_oracle import _configure_determinism

    _configure_determinism(torch)
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    child = context.Process(target=child_probe, args=(sending,))
    child.start()
    sending.close()
    state = receiving.recv()
    child.join(10)
    if child.exitcode != 0:
        raise RuntimeError(f"spawn worker failed: {child.exitcode}")
    print(
        json.dumps({"parent_enabled": torch.are_deterministic_algorithms_enabled(), "child": state})
    )
