#include "common.cuh"
#include "ggml.h"

#define CUDA_HC_COMBINE_BLOCK_SIZE 256

// [johnv8] Fuzja lancucha qwen4exp build_hc_combine w jedno jadro:
//
//   SCALE -> UNARY(SIGMOID) -> SCALE -> RESHAPE -> RESHAPE -> REPEAT -> MUL -> ADD
//
//   out[i, c, t] = residual[i, c, t] + block_out[i, t] * (2 / (1 + exp(-inject[c, t] / hc)))
//
// GGML_JOHNV8_HC_FUSE=0 wylacza fuzje (domyslnie wlaczona).

// czy przelacznik pozwala na fuzje
bool ggml_cuda_hc_fuse_enabled();

// czy wezly cgraph->nodes[node_idx .. node_idx+7] to naprawde nasz przypadek
bool ggml_cuda_hc_combine_ok(const ggml_cgraph * cgraph, int node_idx);

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
