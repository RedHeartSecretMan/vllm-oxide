#include <cstddef>
#include <cuda_runtime.h>

__global__ void rms_rsqrt_f32_kernel(const float* input, float* output, size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = rsqrtf(input[index]);
}

extern "C" int rms_rsqrt_f32(const float* input, float* output, size_t count, void* stream) {
    rms_rsqrt_f32_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(input, output, count);
    return static_cast<int>(cudaGetLastError());
}
