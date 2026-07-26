#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void reshape_and_cache_kernel(
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* key_cache,
    T* value_cache,
    const int64_t* __restrict__ slot_mapping,
    int64_t D
) {
    int64_t idx = blockIdx.x;
    int64_t slot = slot_mapping[idx];
    if (slot == -1) return;

    int64_t tid = threadIdx.x;
    int64_t nthreads = blockDim.x;

    for (int64_t d = tid; d < D; d += nthreads) {
        int64_t src_off = idx * D + d;
        int64_t dst_off = slot * D + d;
        key_cache[dst_off] = key[src_off];
        value_cache[dst_off] = value[src_off];
    }
}

extern "C" void reshape_and_cache(
    const void* key,
    const void* value,
    void* key_cache,
    void* value_cache,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int dtype,
    void* stream
) {
    int64_t D = num_kv_heads * head_dim;
    unsigned block_size = (unsigned)(D < 1024 ? D : 1024);
    dim3 grid((unsigned)num_tokens);
    dim3 block(block_size);
    cudaStream_t s = (cudaStream_t)stream;

    switch (dtype) {
        case 0: {
            reshape_and_cache_kernel<__half><<<grid, block, 0, s>>>(
                (const __half*)key, (const __half*)value,
                (__half*)key_cache, (__half*)value_cache,
                slot_mapping, D);
            break;
        }
        case 1: {
            reshape_and_cache_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
                (const __nv_bfloat16*)key, (const __nv_bfloat16*)value,
                (__nv_bfloat16*)key_cache, (__nv_bfloat16*)value_cache,
                slot_mapping, D);
            break;
        }
        case 2: {
            reshape_and_cache_kernel<float><<<grid, block, 0, s>>>(
                (const float*)key, (const float*)value,
                (float*)key_cache, (float*)value_cache,
                slot_mapping, D);
            break;
        }
    }
}
