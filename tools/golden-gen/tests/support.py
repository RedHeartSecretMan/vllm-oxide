from __future__ import annotations

from golden_gen.release_protocol import pinned_kernel_paths
from golden_gen.schema import RuntimeInfo, WheelIdentity


def release_runtime() -> RuntimeInfo:
    versions = {
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "vllm": "0.18.1",
        "xgrammar": "0.2.3",
        "triton": "3.6.0",
    }
    return RuntimeInfo(
        evidence_mode="release",
        registry_install_mode="locked-wheels-only",
        pythonhashseed="0",
        cublas_workspace_config=":4096:8",
        python_version="3.12.13",
        torch_version="2.10.0",
        torch_cuda_version="12.8",
        transformers_version="4.57.6",
        vllm_version="0.18.1",
        xgrammar_version="0.2.3",
        triton_version="3.6.0",
        cuda_toolkit_version="13.2.51",
        rustc_version="rustc 1.89.0",
        nvidia_driver_version="595.71",
        gpu_name="NVIDIA GeForce RTX 4080",
        compute_capability="8.9",
        os_kernel="Linux 6.18.33.2-microsoft-standard-WSL2",
        generator_commit="1" * 40,
        uv_lock_sha256="2" * 64,
        wheels=[
            WheelIdentity(
                name=name,
                version=version,
                filename=f"{name}-{version}-test.whl",
                sha256="3" * 64,
            )
            for name, version in versions.items()
        ],
    )


__all__ = ["pinned_kernel_paths", "release_runtime"]
