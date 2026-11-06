// Fused BERT post-attention / post-MLP epilogues.
//
// `add_layernorm`: `layer_norm(x + residual + bias, weight, ln_bias)` in one pass. The
// three rows of `wb` are the projection bias, the LayerNorm weight and the LayerNorm
// bias. One threadgroup per row, strided per thread, reduced in threadgroup memory.
//
// `bias_gelu`: `gelu(x + bias)` elementwise, bias broadcast over the row.
#include <metal_stdlib>
using namespace metal;

#define TG 256

template <typename T>
METAL_FUNC void add_layernorm(
    device const T *x,
    device const T *residual,
    device const T *wb,
    device T *dst,
    constant uint &cols,
    constant float &eps,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float *sh_sum,
    threadgroup float *sh_sq
) {
    const uint base = gid * cols;
    const device T *bias = wb;
    const device T *weight = wb + cols;
    const device T *ln_bias = wb + 2 * cols;

    float s = 0.0f;
    float q = 0.0f;
    for (uint i = tid; i < cols; i += tg) {
        const float v = static_cast<float>(x[base + i])
                      + static_cast<float>(residual[base + i])
                      + static_cast<float>(bias[i]);
        s += v;
        q += v * v;
    }
    sh_sum[tid] = s;
    sh_sq[tid] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = tg / 2; step > 0; step >>= 1) {
        if (tid < step) {
            sh_sum[tid] += sh_sum[tid + step];
            sh_sq[tid] += sh_sq[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = sh_sum[0] / static_cast<float>(cols);
    const float var = max(sh_sq[0] / static_cast<float>(cols) - mean * mean, 0.0f);
    const float inv_std = rsqrt(var + eps);
    for (uint i = tid; i < cols; i += tg) {
        const float v = static_cast<float>(x[base + i])
                      + static_cast<float>(residual[base + i])
                      + static_cast<float>(bias[i]);
        const float n = (v - mean) * inv_std * static_cast<float>(weight[i])
                      + static_cast<float>(ln_bias[i]);
        dst[base + i] = static_cast<T>(n);
    }
}

// Metal has no `erf` in the standard library namespace on every SDK; this is
// Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7, below f16 resolution).
METAL_FUNC float erf_approx(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y = 1.0f
        - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
            - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return sign * y;
}

template <typename T>
METAL_FUNC void bias_gelu(
    constant uint &n,
    device const T *x,
    device const T *bias,
    device T *dst,
    constant uint &cols,
    uint i
) {
    if (i >= n) {
        return;
    }
    const float v = static_cast<float>(x[i]) + static_cast<float>(bias[i % cols]);
    // Exact GELU, the activation both checkpoints were trained with.
    dst[i] = static_cast<T>(0.5f * v * (1.0f + erf_approx(v * 0.7071067811865476f)));
}

#define ADD_LAYERNORM(NAME, T)                                                     \
kernel void NAME(                                                                  \
    device const T *x,                                                             \
    device const T *residual,                                                      \
    device const T *wb,                                                            \
    device T *dst,                                                                 \
    constant uint &cols,                                                           \
    constant float &eps,                                                           \
    uint gid [[threadgroup_position_in_grid]],                                     \
    uint tid [[thread_index_in_threadgroup]],                                      \
    uint tg [[threads_per_threadgroup]]                                            \
) {                                                                                \
    threadgroup float sh_sum[TG];                                                  \
    threadgroup float sh_sq[TG];                                                   \
    add_layernorm<T>(x, residual, wb, dst, cols, eps, gid, tid, tg, sh_sum, sh_sq);\
}

#define BIAS_GELU(NAME, T)                                                         \
kernel void NAME(                                                                  \
    constant uint &n,                                                              \
    device const T *x,                                                             \
    device const T *bias,                                                          \
    device T *dst,                                                                 \
    constant uint &cols,                                                           \
    uint i [[thread_position_in_grid]]                                             \
) {                                                                                \
    bias_gelu<T>(n, x, bias, dst, cols, i);                                        \
}

ADD_LAYERNORM(add_layernorm_f32, float)
ADD_LAYERNORM(add_layernorm_f16, half)
#if defined(__HAVE_BFLOAT__)
ADD_LAYERNORM(add_layernorm_bf16, bfloat)
#endif

BIAS_GELU(bias_gelu_f32, float)
BIAS_GELU(bias_gelu_f16, half)
#if defined(__HAVE_BFLOAT__)
BIAS_GELU(bias_gelu_bf16, bfloat)
#endif
