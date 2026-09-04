#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/device/device_radix_sort.cuh>

namespace {

constexpr int kThreads = 256;

enum SamplingStage : int {
    kHistoryH2D = 1,
    kRowD2D = 2,
    kPenalty = 3,
    kGreedy = 4,
    kTemperature = 5,
    kTokenIndices = 6,
    kRadixSort = 7,
    kTopK = 8,
    kTopP = 9,
    kCategorical = 10,
    kSynchronize = 11,
};

__device__ inline bool better_candidate(
    float score,
    uint32_t token,
    float best_score,
    uint32_t best_token
) {
    if (isnan(score)) {
        return false;
    }
    return best_token == UINT32_MAX || score > best_score ||
           (score == best_score && token < best_token);
}

__device__ inline uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ inline float gumbel_noise(uint64_t row_seed, uint32_t token) {
    const uint64_t bits = splitmix64(
        row_seed ^ (static_cast<uint64_t>(token) * 0xd6e8feb86659fd93ULL));
    // Strictly inside (0, 1), including at both integer endpoints.
    constexpr double denominator = 9007199254740993.0;  // 2^53 + 1
    const double uniform = static_cast<double>((bits >> 11) + 1ULL) / denominator;
    return static_cast<float>(-log(-log(uniform)));
}

__global__ void apply_penalties_kernel(
    float* logits,
    const uint32_t* history_tokens,
    const uint32_t* history_counts,
    uint32_t history_len,
    float presence_penalty,
    float frequency_penalty,
    float repetition_penalty
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= history_len) {
        return;
    }
    const uint32_t token = history_tokens[index];
    float value = logits[token];
    // Order is load-bearing and matches the accepted host contract:
    // repetition first, then presence, then frequency.
    if (repetition_penalty != 0.0f) {
        value /= repetition_penalty;
    }
    value -= presence_penalty;
    value -= frequency_penalty * static_cast<float>(history_counts[index]);
    logits[token] = value;
}

__global__ void scale_temperature_kernel(
    const float* input,
    float* output,
    uint32_t vocab_size,
    float temperature
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vocab_size) {
        return;
    }
    // ADR-0009 keeps +inf valid as a uniform pre-filter distribution.
    output[index] = isinf(temperature) ? 0.0f : input[index] / temperature;
}

__global__ void init_token_indices_kernel(uint32_t* token_ids, uint32_t vocab_size) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < vocab_size) {
        token_ids[index] = index;
    }
}

__global__ void greedy_argmax_kernel(
    const float* logits,
    uint32_t vocab_size,
    uint32_t* selected_tokens,
    uint32_t row
) {
    __shared__ float scores[kThreads];
    __shared__ uint32_t tokens[kThreads];

    float best_score = -CUDART_INF_F;
    uint32_t best_token = UINT32_MAX;
    for (uint32_t token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        const float score = logits[token];
        if (better_candidate(score, token, best_score, best_token)) {
            best_score = score;
            best_token = token;
        }
    }
    scores[threadIdx.x] = best_score;
    tokens[threadIdx.x] = best_token;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            better_candidate(
                scores[threadIdx.x + stride],
                tokens[threadIdx.x + stride],
                scores[threadIdx.x],
                tokens[threadIdx.x])) {
            scores[threadIdx.x] = scores[threadIdx.x + stride];
            tokens[threadIdx.x] = tokens[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        selected_tokens[row] = tokens[0] == UINT32_MAX ? 0 : tokens[0];
    }
}

__global__ void top_k_limit_kernel(
    const float* sorted_logits,
    uint32_t requested_k,
    uint32_t vocab_size,
    uint32_t* limit
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    if (requested_k >= vocab_size) {
        *limit = vocab_size;
        return;
    }
    const float threshold = sorted_logits[requested_k - 1];
    uint32_t active = requested_k;
    // Keep every threshold tie, exactly like the host mask's `value <
    // threshold` rule. The serial tail is normally tiny; +inf intentionally
    // makes the whole row tied and therefore preserves the no-op semantics.
    while (active < vocab_size && sorted_logits[active] == threshold) {
        ++active;
    }
    *limit = active;
}

__global__ void top_p_cutoff_kernel(
    const float* sorted_logits,
    const uint32_t* top_k_limit,
    float top_p,
    uint32_t* cutoff
) {
    __shared__ double chunk_sums[kThreads];
    __shared__ uint32_t chunk_size;
    __shared__ double maximum;

    const uint32_t active = *top_k_limit;
    if (threadIdx.x == 0) {
        chunk_size = (active + blockDim.x - 1) / blockDim.x;
        maximum = static_cast<double>(sorted_logits[0]);
    }
    __syncthreads();

    const uint32_t begin = threadIdx.x * chunk_size;
    const uint32_t end = min(begin + chunk_size, active);
    double sum = 0.0;
    for (uint32_t index = begin; index < end; ++index) {
        sum += exp(static_cast<double>(sorted_logits[index]) - maximum);
    }
    chunk_sums[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    double total = 0.0;
    for (uint32_t chunk = 0; chunk < blockDim.x; ++chunk) {
        total += chunk_sums[chunk];
    }
    const double target = static_cast<double>(top_p) * total;
    double cumulative = 0.0;
    uint32_t selected_chunk = 0;
    for (; selected_chunk < blockDim.x; ++selected_chunk) {
        const double next = cumulative + chunk_sums[selected_chunk];
        if (next >= target) {
            break;
        }
        cumulative = next;
    }

    uint32_t selected_cutoff = active;
    if (selected_chunk < blockDim.x) {
        const uint32_t selected_begin = selected_chunk * chunk_size;
        const uint32_t selected_end = min(selected_begin + chunk_size, active);
        for (uint32_t index = selected_begin; index < selected_end; ++index) {
            cumulative += exp(static_cast<double>(sorted_logits[index]) - maximum);
            if (cumulative >= target) {
                selected_cutoff = index + 1;
                break;
            }
        }
    }
    *cutoff = selected_cutoff;
}

__global__ void copy_limit_kernel(const uint32_t* source, uint32_t* destination) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *destination = *source;
    }
}

__global__ void categorical_gumbel_kernel(
    const float* logits,
    const uint32_t* sorted_token_ids,
    const uint32_t* device_limit,
    uint32_t host_limit,
    float temperature,
    bool already_scaled,
    uint64_t row_seed,
    uint32_t* selected_tokens,
    uint32_t row
) {
    __shared__ float scores[kThreads];
    __shared__ uint32_t tokens[kThreads];

    const uint32_t limit = device_limit == nullptr ? host_limit : *device_limit;
    float best_score = -CUDART_INF_F;
    uint32_t best_token = UINT32_MAX;
    for (uint32_t index = threadIdx.x; index < limit; index += blockDim.x) {
        const uint32_t token = sorted_token_ids == nullptr ? index : sorted_token_ids[index];
        const float scaled = already_scaled
            ? logits[index]
            : (isinf(temperature) ? 0.0f : logits[index] / temperature);
        const float score = scaled + gumbel_noise(row_seed, token);
        if (better_candidate(score, token, best_score, best_token)) {
            best_score = score;
            best_token = token;
        }
    }
    scores[threadIdx.x] = best_score;
    tokens[threadIdx.x] = best_token;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            better_candidate(
                scores[threadIdx.x + stride],
                tokens[threadIdx.x + stride],
                scores[threadIdx.x],
                tokens[threadIdx.x])) {
            scores[threadIdx.x] = scores[threadIdx.x + stride];
            tokens[threadIdx.x] = tokens[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        selected_tokens[row] = tokens[0] == UINT32_MAX ? 0 : tokens[0];
    }
}

inline uint32_t blocks_for(uint32_t elements) {
    return (elements + kThreads - 1) / kThreads;
}

}  // namespace

extern "C" int vllm_oxide_sampling_workspace_bytes(
    uint32_t vocab_size,
    size_t* temp_storage_bytes
) {
    if (vocab_size == 0 || temp_storage_bytes == nullptr ||
        vocab_size > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    size_t bytes = 0;
    const cudaError_t status = cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        bytes,
        static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr),
        static_cast<const uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr),
        static_cast<int>(vocab_size));
    if (status == cudaSuccess) {
        *temp_storage_bytes = bytes;
    }
    return static_cast<int>(status);
}

extern "C" const char* vllm_oxide_sampling_cuda_error_string(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

extern "C" int vllm_oxide_sample_f32(
    const float* logits,
    uint32_t* selected_tokens,
    const float* temperatures,
    const uint32_t* top_ks,
    const float* top_ps,
    const float* presence_penalties,
    const float* frequency_penalties,
    const float* repetition_penalties,
    const uint64_t* row_seeds,
    const uint32_t* history_tokens_host,
    const uint32_t* history_counts_host,
    const uint32_t* history_offsets_host,
    uint32_t* history_tokens_device,
    uint32_t* history_counts_device,
    float* keys_in,
    float* keys_out,
    uint32_t* ids_in,
    uint32_t* ids_out,
    uint32_t* limits,
    void* temp_storage,
    size_t temp_storage_bytes,
    uint32_t batch_size,
    uint32_t vocab_size,
    int* failed_stage,
    int* failed_row,
    cudaStream_t stream
) {
    if (logits == nullptr || selected_tokens == nullptr || temperatures == nullptr ||
        top_ks == nullptr || top_ps == nullptr || presence_penalties == nullptr ||
        frequency_penalties == nullptr || repetition_penalties == nullptr ||
        row_seeds == nullptr || history_offsets_host == nullptr || keys_in == nullptr ||
        keys_out == nullptr || ids_in == nullptr || ids_out == nullptr || limits == nullptr ||
        temp_storage == nullptr || failed_stage == nullptr || failed_row == nullptr ||
        batch_size == 0 || vocab_size == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t status = cudaSuccess;
    *failed_stage = 0;
    *failed_row = -1;

#define TRY_STAGE(expression, stage, row)         \
    do {                                           \
        *failed_stage = (stage);                   \
        *failed_row = (row);                       \
        status = (expression);                     \
        if (status != cudaSuccess) goto finish;    \
    } while (false)

    const uint32_t history_len = history_offsets_host[batch_size];
    if (history_len > 0) {
        if (history_tokens_host == nullptr || history_counts_host == nullptr ||
            history_tokens_device == nullptr || history_counts_device == nullptr) {
            status = cudaErrorInvalidValue;
            *failed_stage = kHistoryH2D;
            goto finish;
        }
        TRY_STAGE(
            cudaMemcpyAsync(
                history_tokens_device,
                history_tokens_host,
                static_cast<size_t>(history_len) * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream),
            kHistoryH2D,
            -1);
        TRY_STAGE(
            cudaMemcpyAsync(
                history_counts_device,
                history_counts_host,
                static_cast<size_t>(history_len) * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream),
            kHistoryH2D,
            -1);
    }

    for (uint32_t row = 0; row < batch_size; ++row) {
        const float temperature = temperatures[row];
        const uint32_t top_k = top_ks[row];
        const float top_p = top_ps[row];
        if (isnan(temperature) || temperature < 0.0f || top_k == 0 || top_k > vocab_size ||
            !isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f) {
            status = cudaErrorInvalidValue;
            *failed_stage = 0;
            *failed_row = static_cast<int>(row);
            goto finish;
        }

        const float* row_logits = logits + static_cast<size_t>(row) * vocab_size;
        const uint32_t history_begin = history_offsets_host[row];
        const uint32_t row_history_len = history_offsets_host[row + 1] - history_begin;
        const bool has_penalties = presence_penalties[row] != 0.0f ||
                                   frequency_penalties[row] != 0.0f ||
                                   repetition_penalties[row] != 0.0f;
        const float* source = row_logits;
        if (has_penalties && row_history_len > 0) {
            TRY_STAGE(
                cudaMemcpyAsync(
                    keys_in,
                    row_logits,
                    static_cast<size_t>(vocab_size) * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream),
                kRowD2D,
                static_cast<int>(row));
            apply_penalties_kernel<<<blocks_for(row_history_len), kThreads, 0, stream>>>(
                keys_in,
                history_tokens_device + history_begin,
                history_counts_device + history_begin,
                row_history_len,
                presence_penalties[row],
                frequency_penalties[row],
                repetition_penalties[row]);
            TRY_STAGE(cudaGetLastError(), kPenalty, static_cast<int>(row));
            source = keys_in;
        }

        const bool greedy = temperature == 0.0f || top_k == 1;
        if (greedy) {
            greedy_argmax_kernel<<<1, kThreads, 0, stream>>>(
                source, vocab_size, selected_tokens, row);
            TRY_STAGE(cudaGetLastError(), kGreedy, static_cast<int>(row));
            continue;
        }

        const bool filtered = top_k < vocab_size || top_p < 1.0f;
        if (!filtered) {
            categorical_gumbel_kernel<<<1, kThreads, 0, stream>>>(
                source,
                nullptr,
                nullptr,
                vocab_size,
                temperature,
                false,
                row_seeds[row],
                selected_tokens,
                row);
            TRY_STAGE(cudaGetLastError(), kCategorical, static_cast<int>(row));
            continue;
        }

        scale_temperature_kernel<<<blocks_for(vocab_size), kThreads, 0, stream>>>(
            source, keys_in, vocab_size, temperature);
        TRY_STAGE(cudaGetLastError(), kTemperature, static_cast<int>(row));
        init_token_indices_kernel<<<blocks_for(vocab_size), kThreads, 0, stream>>>(
            ids_in, vocab_size);
        TRY_STAGE(cudaGetLastError(), kTokenIndices, static_cast<int>(row));

        size_t available_temp_storage = temp_storage_bytes;
        TRY_STAGE(
            cub::DeviceRadixSort::SortPairsDescending(
                temp_storage,
                available_temp_storage,
                keys_in,
                keys_out,
                ids_in,
                ids_out,
                static_cast<int>(vocab_size),
                0,
                static_cast<int>(sizeof(float) * 8),
                stream),
            kRadixSort,
            static_cast<int>(row));

        top_k_limit_kernel<<<1, 1, 0, stream>>>(keys_out, top_k, vocab_size, limits);
        TRY_STAGE(cudaGetLastError(), kTopK, static_cast<int>(row));
        if (top_p < 1.0f) {
            top_p_cutoff_kernel<<<1, kThreads, 0, stream>>>(
                keys_out, limits, top_p, limits + 1);
            TRY_STAGE(cudaGetLastError(), kTopP, static_cast<int>(row));
        } else {
            copy_limit_kernel<<<1, 1, 0, stream>>>(limits, limits + 1);
            TRY_STAGE(cudaGetLastError(), kTopP, static_cast<int>(row));
        }

        categorical_gumbel_kernel<<<1, kThreads, 0, stream>>>(
            keys_out,
            ids_out,
            limits + 1,
            0,
            temperature,
            true,
            row_seeds[row],
            selected_tokens,
            row);
        TRY_STAGE(cudaGetLastError(), kCategorical, static_cast<int>(row));
    }

finish:
    {
        const cudaError_t synchronize_status = cudaStreamSynchronize(stream);
        if (status == cudaSuccess && synchronize_status != cudaSuccess) {
            status = synchronize_status;
            *failed_stage = kSynchronize;
        }
    }
#undef TRY_STAGE
    return static_cast<int>(status);
}
