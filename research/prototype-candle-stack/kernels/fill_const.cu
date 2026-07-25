// Throwaway prototype kernel — validates the cudarc seam (T10 part a).
// Writes a constant into every element of an f32 buffer so we can read it back
// via candle and confirm the kernel actually ran on the GPU.
//
// The `extern "C"` linkage is the whole point: mistral.rs / candle-flash-attn
// expose undecorated C symbols so the Rust side can declare them in an
// `extern "C" {}` block with no #[link_name].

#include <cstdint>
#include <cuda_runtime.h>

__global__ void fill_const_f32_kernel(float* ptr, int64_t n, float val) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

extern "C" void fill_const_f32_launch(float* ptr, int64_t n, float val, void* stream) {
    int64_t threads = 256;
    int64_t blocks  = (n + threads - 1) / threads;
    cudaStream_t s  = (cudaStream_t)stream;
    fill_const_f32_kernel<<<(unsigned)blocks, (unsigned)threads, 0, s>>>(ptr, n, val);
}
