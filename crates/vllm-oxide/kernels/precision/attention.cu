// FP32 attention helpers. This translation unit is compiled without fast math.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

struct AttentionMask {
    size_t queries, keys, heads, query_start, chunk_rows, member_offset;
};

__device__ bool query_valid(size_t row, const uint32_t* cu_q, AttentionMask mask) {
    const size_t member = mask.member_offset + row / (mask.heads * mask.chunk_rows);
    const size_t length = cu_q[member + 1] - cu_q[member];
    const size_t query = mask.query_start + row % mask.chunk_rows;
    return query >= mask.queries - length && query < mask.queries;
}

__device__ float masked_value(const float* input, size_t row, size_t column,
                              const uint32_t* cu_q, const uint32_t* cu_k,
                              AttentionMask mask) {
    const size_t member = mask.member_offset + row / (mask.heads * mask.chunk_rows);
    const size_t keys = cu_k[member + 1] - cu_k[member];
    const size_t query = mask.query_start + row % mask.chunk_rows;
    if (!query_valid(row, cu_q, mask) || column < mask.keys - keys
        || column > mask.keys - mask.queries + query) return -INFINITY;
    return input[row * mask.keys + column] + 0.0f;
}

__global__ void causal_softmax_kernel(const float* input, float* output,
                                      const uint32_t* cu_q, const uint32_t* cu_k,
                                      size_t rows, AttentionMask mask) {
    const size_t row = static_cast<size_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % 32;
    const unsigned iterations = static_cast<unsigned>((mask.keys + 31) / 32);
    float values[32];
    float maximum = -INFINITY;
    for (unsigned tile = 0; tile < iterations; ++tile) {
        const size_t column = lane + tile * 32;
        const float value = column < mask.keys
            ? masked_value(input, row, column, cu_q, cu_k, mask) : -INFINITY;
        values[tile] = value;
        maximum = maximum < value ? value : maximum;
    }
    for (unsigned offset = 16; offset; offset /= 2) {
        const float other = __shfl_xor_sync(0xffffffff, maximum, offset);
        maximum = maximum < other ? other : maximum;
    }
    if (maximum == -INFINITY) {
        for (unsigned tile = 0; tile < iterations; ++tile) {
            const size_t column = lane + tile * 32;
            if (column < mask.keys) output[row * mask.keys + column] = 0.0f;
        }
        return;
    }
    float total = 0.0f;
    for (unsigned tile = 0; tile < iterations; ++tile) {
        values[tile] = expf(values[tile] - maximum);
        total += values[tile];
    }
    for (unsigned offset = 16; offset; offset /= 2)
        total += __shfl_xor_sync(0xffffffff, total, offset);
    for (unsigned tile = 0; tile < iterations; ++tile) {
        const size_t column = lane + tile * 32;
        if (column < mask.keys) output[row * mask.keys + column] = values[tile] / total;
    }
}

__global__ void causal_mask_kernel(const float* input, float* output,
                                   const uint32_t* cu_q, const uint32_t* cu_k,
                                   size_t count, AttentionMask mask, bool zero_padding) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t row = index / mask.keys;
    output[index] = zero_padding
        ? (query_valid(row, cu_q, mask) ? input[index] : 0.0f)
        : masked_value(input, row, index % mask.keys, cu_q, cu_k, mask);
}

extern "C" int attention_rows_f32(const float* input, float* output,
                                  const uint32_t* cu_q, const uint32_t* cu_k,
                                  size_t rows, AttentionMask mask, int operation, void* stream) {
    if (!rows || !mask.keys || !mask.queries || !mask.heads || !mask.chunk_rows)
        return cudaErrorInvalidValue;
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (operation == 0) {
        if (mask.keys > 1024) return cudaErrorInvalidValue;
        causal_softmax_kernel<<<static_cast<unsigned>((rows + 3) / 4), 128, 0, cuda_stream>>>
            (input, output, cu_q, cu_k, rows, mask);
    } else if (operation == 1 || operation == 2) {
        const size_t count = rows * mask.keys;
        causal_mask_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, cuda_stream>>>
            (input, output, cu_q, cu_k, count, mask, operation == 2);
    } else return cudaErrorInvalidValue;
    return static_cast<int>(cudaGetLastError());
}

struct CacheGather {
    size_t batch, keys, kv_heads, head_dim, block_size, table_width, num_blocks, member_offset;
};

template<typename T>
__global__ void cache_gather_kernel(const T* input, T* output, const uint32_t* cu_k,
                                     const uint32_t* block_table, size_t count,
                                     CacheGather shape) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t width = shape.kv_heads * shape.head_dim;
    const size_t token = index / width;
    const size_t member = shape.member_offset + token / shape.keys;
    const size_t column = token % shape.keys;
    const size_t length = cu_k[member + 1] - cu_k[member];
    if (column < shape.keys - length) { output[index] = T(0.0f); return; }
    const size_t logical = column - (shape.keys - length);
    const uint32_t block = block_table[member * shape.table_width + logical / shape.block_size];
    assert(static_cast<size_t>(block) < shape.num_blocks);
    const size_t slot = static_cast<size_t>(block) * shape.block_size + logical % shape.block_size;
    output[index] = input[slot * width + index % width];
}

extern "C" int attention_gather(const void* input, void* output, const uint32_t* cu_k,
                                 const uint32_t* block_table, CacheGather shape,
                                 int dtype, void* stream) {
    const size_t count = shape.batch * shape.keys * shape.kv_heads * shape.head_dim;
    if (!count || !shape.block_size || !shape.table_width || !shape.num_blocks)
        return cudaErrorInvalidValue;
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto blocks = static_cast<unsigned>((count + 255) / 256);
    switch (dtype) {
        case 0: cache_gather_kernel<<<blocks,256,0,cuda_stream>>>(static_cast<const __half*>(input),static_cast<__half*>(output),cu_k,block_table,count,shape); break;
        case 1: cache_gather_kernel<<<blocks,256,0,cuda_stream>>>(static_cast<const __nv_bfloat16*>(input),static_cast<__nv_bfloat16*>(output),cu_k,block_table,count,shape); break;
        case 2: cache_gather_kernel<<<blocks,256,0,cuda_stream>>>(static_cast<const float*>(input),static_cast<float*>(output),cu_k,block_table,count,shape); break;
        default: return cudaErrorInvalidValue;
    }
    return static_cast<int>(cudaGetLastError());
}
