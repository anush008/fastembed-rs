// Qwen3's q_norm / k_norm followed by the half-split rope, in one pass over q or k.
//
// One threadgroup per `[D]` row of a contiguous `[B, T, H, D]` tensor, D/2 threads: thread i
// holds the pair (x[i], x[i + D/2]) the rotation mixes. Everything runs in f32 (the sum of
// squares, the row's scale, the weight multiply and the rotation) and each value is rounded to T
// once, at the store. In f32 that is candle's `rmsnorm` then `rope_thd` up to the order of the
// sum; in bf16 and f16 it is closer to the f32 result than those two kernels, which round the
// scale, the normed value and the rotated value each to T.
#include <metal_stdlib>
using namespace metal;

template <typename T>
METAL_FUNC void qk_norm_rope(
    constant uint &heads,
    constant uint &seq_len,
    constant uint &half_d,
    constant float &eps,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device const T *weight,
    device T *dst,
    threadgroup float *partial,
    uint row,
    uint tid,
    uint lane,
    uint simd_id,
    uint n_simd
) {
    const uint d = 2 * half_d;
    const uint i1 = row * d + tid;
    const uint i2 = i1 + half_d;
    const T x1 = src[i1];
    const T x2 = src[i2];

    const float f1 = static_cast<float>(x1);
    const float f2 = static_cast<float>(x2);
    const float sum = simd_sum(f1 * f1 + f2 * f2);
    if (lane == 0) {
        partial[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = 0;
    for (uint s = 0; s < n_simd; ++s) {
        sum_sq += partial[s];
    }
    const float total = rsqrt(sum_sq / float(d) + eps);

    const float n1 = f1 * total * static_cast<float>(weight[tid]);
    const float n2 = f2 * total * static_cast<float>(weight[tid + half_d]);

    const uint i_cs = ((row / heads) % seq_len) * half_d + tid;
    const float c = static_cast<float>(cos[i_cs]);
    const float s = static_cast<float>(sin[i_cs]);
    dst[i1] = static_cast<T>(n1 * c - n2 * s);
    dst[i2] = static_cast<T>(n1 * s + n2 * c);
}

#define QK_NORM_ROPE(NAME, T)                                   \
kernel void NAME(                                               \
    constant uint &heads,                                       \
    constant uint &seq_len,                                     \
    constant uint &half_d,                                      \
    constant float &eps,                                        \
    device const T *src,                                        \
    device const T *cos,                                        \
    device const T *sin,                                        \
    device const T *weight,                                     \
    device T *dst,                                              \
    uint row [[ threadgroup_position_in_grid ]],                \
    uint tid [[ thread_position_in_threadgroup ]],              \
    uint lane [[ thread_index_in_simdgroup ]],                  \
    uint simd_id [[ simdgroup_index_in_threadgroup ]],          \
    uint n_simd [[ simdgroups_per_threadgroup ]]                \
) {                                                             \
    threadgroup float partial[32];                              \
    qk_norm_rope<T>(heads, seq_len, half_d, eps, src, cos, sin, \
        weight, dst, partial, row, tid, lane, simd_id, n_simd); \
}

QK_NORM_ROPE(qk_norm_rope_f32, float)
QK_NORM_ROPE(qk_norm_rope_f16, half)
#if defined(__HAVE_BFLOAT__)
QK_NORM_ROPE(qk_norm_rope_bf16, bfloat)
#endif
