// Fused CPU ops (ggml custom ops), see ops_cpu.cpp.
#pragma once

#include "ggml.h"

namespace articulino {

// depthwise Conv1d with zero padding K/2: x [C, T], w [C, K] (tap k = row k), b [C] -> [C, T]
ggml_tensor * dwconv_fused(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
// exact (erf) GELU, vectorized
ggml_tensor * gelu_fast(ggml_context * ctx, ggml_tensor * x);

}  // namespace articulino
