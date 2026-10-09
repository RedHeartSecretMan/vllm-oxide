// F32 mean for the two pinned Qwen3 normalization widths.
// Four independent, adjacent-element accumulators precede a fixed reduction.
#include <cuda_runtime.h>
#include <cstddef>

__global__ void rms_mean_f32_kernel(const float* input, float* output, size_t columns) {
    const size_t row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    float accum[4] = {0.f, 0.f, 0.f, 0.f};
    for (size_t vector = tid; vector < columns / 4; vector += blockDim.x) {
        for (unsigned lane = 0; lane < 4; ++lane)
            accum[lane] += input[row * columns + vector * 4 + lane];
    }
    float total = ((accum[0] + accum[1]) + accum[2]) + accum[3];
    __shared__ float partial[256];
    partial[tid] = total;
    for (unsigned offset = blockDim.x / 2; offset >= 32; offset /= 2) {
        __syncthreads();
        if (tid < offset) {
            total += partial[tid + offset];
            partial[tid] = total;
        }
    }
    // Only the first warp contains the final row reduction.
    if (tid < 32) {
        for (unsigned offset = 16; offset; offset /= 2)
            total += __shfl_down_sync(0xffffffff, total, offset);
        if (tid == 0) output[row] = total * (1.0f / static_cast<float>(columns));
    }
}

extern "C" int rms_mean_f32(const float* input, float* output, size_t rows,
                          size_t columns, void* stream) {
    if (!rows || (columns != 128 && columns != 1024)) return cudaErrorInvalidValue;
    unsigned row_power = 1;
    while (row_power < 512 && row_power * 2 <= rows) row_power *= 2;
    const unsigned height = row_power < 16 ? row_power : 16;
    const unsigned vector_width = static_cast<unsigned>(columns / 4);
    const unsigned width = vector_width < 512 / height ? vector_width : 512 / height;
    rms_mean_f32_kernel<<<static_cast<unsigned>(rows), width, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(input, output, columns);
    return static_cast<int>(cudaGetLastError());
}
