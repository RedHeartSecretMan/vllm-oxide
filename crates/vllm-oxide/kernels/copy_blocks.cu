#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int COPY_BLOCKS_THREADS = 256;

template <typename T>
__global__ void copy_blocks_kernel(
    T* key_cache,
    T* value_cache,
    const int64_t* __restrict__ block_mapping,
    int64_t block_stride
) {
    int64_t pair = blockIdx.x;
    int64_t element_idx = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;

    if (element_idx >= block_stride) return;

    int64_t src_block = block_mapping[pair * 2];
    int64_t dst_block = block_mapping[pair * 2 + 1];

    int64_t src_off = src_block * block_stride + element_idx;
    int64_t dst_off = dst_block * block_stride + element_idx;

    key_cache[dst_off] = key_cache[src_off];
    value_cache[dst_off] = value_cache[src_off];
}

extern "C" void copy_blocks(
    void* key_cache,
    void* value_cache,
    const int64_t* block_mapping,
    int64_t num_pairs,
    int64_t block_stride,
    int dtype,
    void* stream
) {
    unsigned y_blocks = (unsigned)((block_stride + COPY_BLOCKS_THREADS - 1) / COPY_BLOCKS_THREADS);
    dim3 grid((unsigned)num_pairs, y_blocks);
    dim3 block(COPY_BLOCKS_THREADS);
    cudaStream_t s = (cudaStream_t)stream;

    switch (dtype) {
        case 0: {
            copy_blocks_kernel<__half><<<grid, block, 0, s>>>(
                (__half*)key_cache, (__half*)value_cache, block_mapping, block_stride);
            break;
        }
        case 1: {
            copy_blocks_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
                (__nv_bfloat16*)key_cache, (__nv_bfloat16*)value_cache, block_mapping, block_stride);
            break;
        }
        case 2: {
            copy_blocks_kernel<float><<<grid, block, 0, s>>>(
                (float*)key_cache, (float*)value_cache, block_mapping, block_stride);
            break;
        }
    }
}
