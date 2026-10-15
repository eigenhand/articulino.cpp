// Fused CPU ops used through ggml custom ops (multi-threaded by ggml):
//  - depthwise Conv1d (ConvNeXt, k 7, zero padding) in one pass instead of 14 element-wise graph nodes,
//  - exact GELU with a vectorized erf (Abramowitz-Stegun 7.1.26, |error| <= 1.5e-7, the same formula PyTorch's
//    vectorized CPU kernel uses) instead of a scalar erff call per element.
#include "ops_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ARTICULINO_AVX2 1
#endif

namespace articulino {

// x [C, T], w [C, K], b [C] -> y [C, T]; y[:, t] = b + sum_k w[:, k] * x[:, t + k - K/2]
static void dwconv_op(ggml_tensor * dst, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * b, int ith,
                      int nth, void *) {
    const int C = (int) x->ne[0], T = (int) x->ne[1], K = (int) w->ne[1], pad = K / 2;
    const float * __restrict X = (const float *) x->data;
    const float * __restrict Wt = (const float *) w->data;
    const float * __restrict B = (const float *) b->data;
    float * __restrict Y = (float *) dst->data;
    const int per = (T + nth - 1) / nth, t0 = ith * per, t1 = std::min(T, t0 + per);
    for (int t = t0; t < t1; t++) {
        float * __restrict y = Y + (size_t) t * C;
        std::memcpy(y, B, (size_t) C * sizeof(float));
        for (int k = 0; k < K; k++) {
            const int s = t + k - pad;
            if (s < 0 || s >= T) continue;
            const float * __restrict xs = X + (size_t) s * C;
            const float * __restrict wk = Wt + (size_t) k * C;
            for (int c = 0; c < C; c++) y[c] += wk[c] * xs[c];
        }
    }
}

#ifdef ARTICULINO_AVX2
static inline __m256 exp256(__m256 x) {   // Cephes-style expf
    const __m256 hi = _mm256_set1_ps(88.3762626647949f), lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i n = _mm256_cvttps_epi32(fx);
    n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

static inline __m256 gelu256(__m256 v) {
    const __m256 z = _mm256_mul_ps(v, _mm256_set1_ps(0.70710678118654752f));
    const __m256 sign = _mm256_and_ps(z, _mm256_set1_ps(-0.0f));
    const __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), z);
    const __m256 t = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), a, _mm256_set1_ps(1.0f)));
    __m256 p = _mm256_set1_ps(1.061405429f);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453152027f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254829592f));
    p = _mm256_mul_ps(p, t);
    const __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(a, a)));
    __m256 erf = _mm256_fnmadd_ps(p, e, _mm256_set1_ps(1.0f));
    erf = _mm256_or_ps(erf, sign);
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), v), _mm256_add_ps(_mm256_set1_ps(1.0f), erf));
}
#endif

static void gelu_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    const int64_t n = ggml_nelements(a);
    const int64_t per = ((n + nth - 1) / nth + 7) / 8 * 8;
    const int64_t i0 = std::min(n, ith * per), i1 = std::min(n, i0 + per);
    const float * __restrict x = (const float *) a->data;
    float * __restrict y = (float *) dst->data;
    int64_t i = i0;
#ifdef ARTICULINO_AVX2
    for (; i + 8 <= i1; i += 8) _mm256_storeu_ps(y + i, gelu256(_mm256_loadu_ps(x + i)));
#endif
    for (; i < i1; i++) y[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * 0.70710678118654752f));
}

ggml_tensor * dwconv_fused(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_map_custom3(ctx, x, w, b, dwconv_op, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor * gelu_fast(ggml_context * ctx, ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, gelu_op, GGML_N_TASKS_MAX, nullptr);
}

}  // namespace articulino
