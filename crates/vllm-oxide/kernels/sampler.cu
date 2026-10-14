#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
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
    // 2^53 + 1 rounds to 2^53 in F64. Preserve existing draws, but clamp
    // the single upper endpoint so log(-log(u)) cannot produce infinity.
    constexpr double denominator = 9007199254740993.0;
    const double uniform = fmin(
        static_cast<double>((bits >> 11) + 1ULL) / denominator,
        0x1.fffffffffffffp-1);
    return static_cast<float>(-log(-log(uniform)));
}

// The CUDA build enables fast math, which flushes F32 subnormals. Decode the
// original F32 bits into F64 so every accepted positive divisor stays positive.
__device__ inline double wide_float(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000U) == 0) {
        const double magnitude = static_cast<double>(bits & 0x007fffffU) * 0x1p-149;
        return (bits & 0x80000000U) == 0 ? magnitude : -magnitude;
    }
    return static_cast<double>(value);
}

// Retain counts rather than folding presence and frequency into an absolute
// score. Common terms cancel before small penalty differences are evaluated.
struct WideScore {
    double base;
    uint32_t count;
};

__device__ inline double wide_two_sum(double left, double right, double* error) {
    const double sum = __dadd_rn(left, right);
    const double right_part = __dsub_rn(sum, left);
    *error = __dadd_rn(__dsub_rn(left, __dsub_rn(sum, right_part)),
                      __dsub_rn(right, right_part));
    return sum;
}

__device__ inline double wide_difference(
    WideScore value, WideScore other, float presence_penalty, float frequency_penalty
) {
    const double primary = value.base - other.base;
    // The -inf argmax sentinel is not an input to compensated addition.
    if (!isfinite(primary)) {
        return primary;
    }
    const double presence = (value.count > 0) == (other.count > 0) ? 0.0
        : (value.count > 0 ? -wide_float(presence_penalty) : wide_float(presence_penalty));
    const double count_difference = value.count >= other.count
        ? static_cast<double>(value.count - other.count)
        : -static_cast<double>(other.count - value.count);
    const double frequency = -wide_float(frequency_penalty) * count_difference;
    double error = 0.0;
    double next_error = 0.0;
    const double sum = wide_two_sum(primary, frequency, &error);
    const double total = wide_two_sum(sum, presence, &next_error);
    return __dadd_rn(total, __dadd_rn(error, next_error));
}

__device__ inline bool better_wide_candidate(
    WideScore score, uint32_t token, WideScore best_score, uint32_t best_token,
    float presence_penalty, float frequency_penalty
) {
    const double difference = wide_difference(score, best_score, presence_penalty, frequency_penalty);
    return !isnan(difference) && (best_token == UINT32_MAX || difference > 0.0 ||
                                (difference == 0.0 && token < best_token));
}

__device__ inline WideScore wide_penalized_score(
    const float* logits,
    uint32_t token,
    const uint32_t* history_tokens,
    const uint32_t* history_counts,
    uint32_t history_len,
    float repetition_penalty
) {
    WideScore value{wide_float(logits[token]), 0};
    // Rust prepares each row's history in ascending token order.
    uint32_t begin = 0;
    uint32_t end = history_len;
    while (begin < end) {
        const uint32_t middle = begin + (end - begin) / 2;
        if (history_tokens[middle] < token) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    if (begin < history_len && history_tokens[begin] == token) {
        const double repetition = wide_float(repetition_penalty);
        if (repetition != 0.0) {
            value.base /= repetition;
        }
        value.count = history_counts[begin];
    }
    return value;
}

__global__ void wide_max_kernel(
    const float* logits,
    uint32_t vocab_size,
    const uint32_t* history_tokens,
    const uint32_t* history_counts,
    uint32_t history_len,
    float presence_penalty,
    float frequency_penalty,
    float repetition_penalty,
    const uint32_t* wide_required,
    double* maximum
) {
    if (*wide_required == 0) {
        return;
    }
    __shared__ WideScore maxima[kThreads];
    WideScore best{-CUDART_INF, 0};
    for (uint32_t token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        const WideScore value = wide_penalized_score(
            logits, token, history_tokens, history_counts, history_len,
            repetition_penalty);
        if (wide_difference(value, best, presence_penalty, frequency_penalty) > 0.0) {
            best = value;
        }
    }
    maxima[threadIdx.x] = best;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (wide_difference(maxima[threadIdx.x + stride], maxima[threadIdx.x], presence_penalty, frequency_penalty) > 0.0) {
                maxima[threadIdx.x] = maxima[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maximum[0] = maxima[0].base;
        maximum[1] = static_cast<double>(maxima[0].count);
    }
}

__global__ void normalize_wide_kernel(
    const float* logits,
    float* normalized,
    uint32_t vocab_size,
    const uint32_t* history_tokens,
    const uint32_t* history_counts,
    uint32_t history_len,
    float presence_penalty,
    float frequency_penalty,
    float repetition_penalty,
    float temperature,
    const uint32_t* wide_required,
    const double* maximum
) {
    if (*wide_required == 0) {
        return;
    }
    const uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= vocab_size) {
        return;
    }
    const WideScore value = wide_penalized_score(
        logits, token, history_tokens, history_counts, history_len,
        repetition_penalty);
    // A constant logit shift preserves categorical and nucleus probabilities.
    // Very negative F32 overflow becomes a zero-mass tail, never a false +inf tie.
    normalized[token] = isinf(temperature) ? 0.0f
        : static_cast<float>(wide_difference(value, WideScore{maximum[0], static_cast<uint32_t>(maximum[1])}, presence_penalty, frequency_penalty) / wide_float(temperature));
}

__global__ void select_wide_kernel(
    const float* logits,
    uint32_t vocab_size,
    const uint32_t* history_tokens,
    const uint32_t* history_counts,
    uint32_t history_len,
    float presence_penalty,
    float frequency_penalty,
    float repetition_penalty,
    float temperature,
    bool greedy,
    uint64_t row_seed,
    const uint32_t* wide_required,
    uint32_t* selected_tokens,
    uint32_t row
) {
    if (*wide_required == 0) {
        return;
    }
    __shared__ WideScore maxima[kThreads];
    __shared__ float scores[kThreads];
    __shared__ uint32_t tokens[kThreads];
    WideScore best{-CUDART_INF, 0};
    uint32_t best_token = UINT32_MAX;
    for (uint32_t token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        const WideScore value = wide_penalized_score(
            logits, token, history_tokens, history_counts, history_len,
            repetition_penalty);
        if (better_wide_candidate(value, token, best, best_token, presence_penalty, frequency_penalty)) {
            best = value;
            best_token = token;
        }
    }
    maxima[threadIdx.x] = best;
    tokens[threadIdx.x] = best_token;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && better_wide_candidate(
                maxima[threadIdx.x + stride], tokens[threadIdx.x + stride],
                maxima[threadIdx.x], tokens[threadIdx.x], presence_penalty, frequency_penalty)) {
            maxima[threadIdx.x] = maxima[threadIdx.x + stride];
            tokens[threadIdx.x] = tokens[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (greedy) {
        if (threadIdx.x == 0) {
            selected_tokens[row] = tokens[0] == UINT32_MAX ? 0 : tokens[0];
        }
        return;
    }
    const WideScore maximum = maxima[0];
    // All threads read the reduced maximum before reusing token scratch.
    __syncthreads();
    float best_score = -CUDART_INF_F;
    best_token = UINT32_MAX;
    for (uint32_t token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        const WideScore value = wide_penalized_score(
            logits, token, history_tokens, history_counts, history_len,
            repetition_penalty);
        const float scaled = isinf(temperature) ? 0.0f
            : static_cast<float>(wide_difference(value, maximum, presence_penalty, frequency_penalty) / wide_float(temperature));
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
        if (threadIdx.x < stride && better_candidate(
                scores[threadIdx.x + stride], tokens[threadIdx.x + stride],
                scores[threadIdx.x], tokens[threadIdx.x])) {
            scores[threadIdx.x] = scores[threadIdx.x + stride];
            tokens[threadIdx.x] = tokens[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        selected_tokens[row] = tokens[0] == UINT32_MAX ? 0 : tokens[0];
    }
}

__global__ void apply_penalties_kernel(
    float* logits,
    const uint32_t* history_tokens,
    const uint32_t* history_counts,
    uint32_t history_len,
    float presence_penalty,
    float frequency_penalty,
    float repetition_penalty,
    uint32_t* wide_required
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
    if (isfinite(logits[token]) && !isfinite(value)) {
        atomicExch(wide_required, 1U);
    }
    logits[token] = value;
}

__global__ void scale_temperature_kernel(
    const float* input,
    float* output,
    uint32_t vocab_size,
    float temperature,
    uint32_t* wide_required
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vocab_size) {
        return;
    }
    // ADR-0009 keeps +inf valid as a uniform pre-filter distribution.
    const float value = input[index];
    const float scaled = isinf(temperature) ? 0.0f : value / temperature;
    if (isfinite(value) && (!isfinite(scaled) || fabsf(scaled) >= 0x1p24f)) {
        atomicExch(wide_required, 1U);
    }
    output[index] = scaled;
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

    // One block per row. The mixed-path caller still passes a single row
    // with an explicit output offset; plain greedy batches share one launch.
    logits += static_cast<size_t>(blockIdx.x) * vocab_size;
    row += blockIdx.x;
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
    uint32_t* wide_required,
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
        if (!already_scaled && isfinite(logits[index]) &&
            (!isfinite(scaled) || fabsf(scaled) >= 0x1p24f)) {
            atomicExch(wide_required, 1U);
        }
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

inline bool is_positive_subnormal(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits > 0 && bits < 0x00800000U;
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
    double* wide_maximum,
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
        wide_maximum == nullptr || temp_storage == nullptr || failed_stage == nullptr || failed_row == nullptr ||
        batch_size == 0 || vocab_size == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t status = cudaSuccess;
    *failed_stage = 0;
    *failed_row = -1;

#define TRY_STAGE(expression, stage)              \
    do {                                           \
        /* CUDA APIs may surface an earlier asynchronous error here. */ \
        *failed_stage = (stage);                   \
        *failed_row = -1;                          \
        status = (expression);                     \
        if (status != cudaSuccess) goto finish;    \
    } while (false)

    const uint32_t history_len = history_offsets_host[batch_size];
    bool plain_greedy_batch = true;
    for (uint32_t row = 0; row < batch_size; ++row) {
        const float temperature = temperatures[row];
        const uint32_t top_k = top_ks[row];
        const float top_p = top_ps[row];
        if (isnan(temperature) || temperature < 0.0f || top_k == 0 || top_k > vocab_size ||
            !isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f) {
            status = cudaErrorInvalidValue;
            // Rust performs the authoritative synchronous per-row validation.
            // This defensive guard must not claim an origin row: runtime APIs
            // may surface an earlier asynchronous failure here.
            goto finish;
        }
        plain_greedy_batch = plain_greedy_batch &&
            (temperature == 0.0f || top_k == 1) &&
            presence_penalties[row] == 0.0f && frequency_penalties[row] == 0.0f &&
            repetition_penalties[row] == 0.0f;
    }
    if (plain_greedy_batch) {
        greedy_argmax_kernel<<<batch_size, kThreads, 0, stream>>>(
            logits, vocab_size, selected_tokens, 0);
        TRY_STAGE(cudaGetLastError(), kGreedy);
        goto finish;
    }

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
            kHistoryH2D);
        TRY_STAGE(
            cudaMemcpyAsync(
                history_counts_device,
                history_counts_host,
                static_cast<size_t>(history_len) * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream),
            kHistoryH2D);
    }

    for (uint32_t row = 0; row < batch_size; ++row) {
        const float temperature = temperatures[row];
        const uint32_t top_k = top_ks[row];
        const float top_p = top_ps[row];
        const float* row_logits = logits + static_cast<size_t>(row) * vocab_size;
        const uint32_t history_begin = history_offsets_host[row];
        const uint32_t row_history_len = history_offsets_host[row + 1] - history_begin;
        const bool has_penalties = presence_penalties[row] != 0.0f ||
                                   frequency_penalties[row] != 0.0f ||
                                   repetition_penalties[row] != 0.0f;
        // limits[2] is a private per-row fallback flag. A nonzero byte fill
        // also requests wide arithmetic when fast math would flush a divisor.
        const bool greedy = temperature == 0.0f || top_k == 1;
        const bool subnormal_divisor =
            (!greedy && is_positive_subnormal(temperature)) ||
            (row_history_len > 0 && is_positive_subnormal(repetition_penalties[row]));
        TRY_STAGE(cudaMemsetAsync(limits + 2, subnormal_divisor ? 1 : 0,
                                  sizeof(uint32_t), stream), kRowD2D);
        const float* source = row_logits;
        if (has_penalties && row_history_len > 0) {
            TRY_STAGE(
                cudaMemcpyAsync(
                    keys_in,
                    row_logits,
                    static_cast<size_t>(vocab_size) * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream),
                kRowD2D);
            apply_penalties_kernel<<<blocks_for(row_history_len), kThreads, 0, stream>>>(
                keys_in,
                history_tokens_device + history_begin,
                history_counts_device + history_begin,
                row_history_len,
                presence_penalties[row],
                frequency_penalties[row],
                repetition_penalties[row],
                limits + 2);
            TRY_STAGE(cudaGetLastError(), kPenalty);
            source = keys_in;
        }

        if (greedy) {
            greedy_argmax_kernel<<<1, kThreads, 0, stream>>>(
                source, vocab_size, selected_tokens, row);
            TRY_STAGE(cudaGetLastError(), kGreedy);
            select_wide_kernel<<<1, kThreads, 0, stream>>>(
                row_logits, vocab_size,
                history_tokens_device + history_begin,
                history_counts_device + history_begin, row_history_len,
                presence_penalties[row], frequency_penalties[row], repetition_penalties[row],
                temperature, true, row_seeds[row], limits + 2, selected_tokens, row);
            TRY_STAGE(cudaGetLastError(), kGreedy);
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
                limits + 2,
                row_seeds[row],
                selected_tokens,
                row);
            TRY_STAGE(cudaGetLastError(), kCategorical);
            select_wide_kernel<<<1, kThreads, 0, stream>>>(
                row_logits, vocab_size,
                history_tokens_device + history_begin,
                history_counts_device + history_begin, row_history_len,
                presence_penalties[row], frequency_penalties[row], repetition_penalties[row],
                temperature, false, row_seeds[row], limits + 2, selected_tokens, row);
            TRY_STAGE(cudaGetLastError(), kCategorical);
            continue;
        }

        scale_temperature_kernel<<<blocks_for(vocab_size), kThreads, 0, stream>>>(
            source, keys_in, vocab_size, temperature, limits + 2);
        TRY_STAGE(cudaGetLastError(), kTemperature);
        wide_max_kernel<<<1, kThreads, 0, stream>>>(
            row_logits, vocab_size,
            history_tokens_device + history_begin,
            history_counts_device + history_begin, row_history_len,
            presence_penalties[row], frequency_penalties[row], repetition_penalties[row],
            limits + 2, wide_maximum);
        TRY_STAGE(cudaGetLastError(), kTemperature);
        normalize_wide_kernel<<<blocks_for(vocab_size), kThreads, 0, stream>>>(
            row_logits, keys_in, vocab_size,
            history_tokens_device + history_begin,
            history_counts_device + history_begin, row_history_len,
            presence_penalties[row], frequency_penalties[row], repetition_penalties[row],
            temperature, limits + 2, wide_maximum);
        TRY_STAGE(cudaGetLastError(), kTemperature);
        init_token_indices_kernel<<<blocks_for(vocab_size), kThreads, 0, stream>>>(
            ids_in, vocab_size);
        TRY_STAGE(cudaGetLastError(), kTokenIndices);

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
            kRadixSort);

        top_k_limit_kernel<<<1, 1, 0, stream>>>(keys_out, top_k, vocab_size, limits);
        TRY_STAGE(cudaGetLastError(), kTopK);
        if (top_p < 1.0f) {
            top_p_cutoff_kernel<<<1, kThreads, 0, stream>>>(
                keys_out, limits, top_p, limits + 1);
            TRY_STAGE(cudaGetLastError(), kTopP);
        } else {
            copy_limit_kernel<<<1, 1, 0, stream>>>(limits, limits + 1);
            TRY_STAGE(cudaGetLastError(), kTopP);
        }

        categorical_gumbel_kernel<<<1, kThreads, 0, stream>>>(
            keys_out,
            ids_out,
            limits + 1,
            0,
            temperature,
            true,
            limits + 2,
            row_seeds[row],
            selected_tokens,
            row);
        TRY_STAGE(cudaGetLastError(), kCategorical);
    }

finish:
    {
        const cudaError_t synchronize_status = cudaStreamSynchronize(stream);
        if (status == cudaSuccess && synchronize_status != cudaSuccess) {
            status = synchronize_status;
            *failed_stage = kSynchronize;
            // A stream-wide asynchronous failure cannot be attributed to the
            // most recently launched row. Preserve correct diagnostics by
            // reporting the row as unknown instead of inventing causality.
            *failed_row = -1;
        }
    }
#undef TRY_STAGE
    return static_cast<int>(status);
}
