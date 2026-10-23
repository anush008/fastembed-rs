// Qwen3's MLP activation, `silu(gate) * up`, in one pass.
//
// One thread per element of two contiguous tensors of the same shape. Both values are read as
// float, `gate / (1 + exp(-gate)) * up` is computed in float, and the result is rounded once to
// T, where candle's `silu` then `mul` round twice and write the [T, intermediate] silu output
// back to memory in between.
#include <metal_stdlib>
using namespace metal;

template <typename T>
METAL_FUNC void silu_mul(
    constant uint &n,
    device const T *gate,
    device const T *up,
    device T *dst,
    uint i
) {
    if (i >= n) {
        return;
    }
    const float g = static_cast<float>(gate[i]);
    const float u = static_cast<float>(up[i]);
    dst[i] = static_cast<T>(g / (1.0f + exp(-g)) * u);
}

#define SILU_MUL(NAME, T)                                   \
kernel void NAME(                                           \
    constant uint &n,                                       \
    device const T *gate,                                   \
    device const T *up,                                     \
    device T *dst,                                          \
    uint i [[ thread_position_in_grid ]]                    \
) {                                                         \
    silu_mul<T>(n, gate, up, dst, i);                       \
}

SILU_MUL(silu_mul_f32, float)
SILU_MUL(silu_mul_f16, half)
#if defined(__HAVE_BFLOAT__)
SILU_MUL(silu_mul_bf16, bfloat)
#endif
