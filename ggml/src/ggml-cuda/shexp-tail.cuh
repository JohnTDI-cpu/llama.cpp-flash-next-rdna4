#include "common.cuh"
#include "ggml.h"

#define CUDA_SHEXP_TAIL_BLOCK_SIZE 256

// [johnv8] E6b: fuzja ogona eksperta wspoldzielonego w qwen4exp build_layer_ffn w jedno jadro:
//
//   UNARY(SIGMOID)[1, nt] -> MUL[n_embd, nt] -> ADD[n_embd, nt]
//
//   out[i, t] = moe_out[i, t] + shexp[i, t] * (1 / (1 + exp(-gate[t])))
//
// Stock: 3 jadra na warstwe (unary, binbcast MUL, binbcast ADD) = 144 uruchomien na token.
// GGML_JOHNV8_SHEXP_FUSE=0 wylacza (domyslnie wlaczona).

bool ggml_cuda_shexp_fuse_enabled();
bool ggml_cuda_shexp_tail_ok(const ggml_cgraph * cgraph, int node_idx);
void ggml_cuda_op_shexp_tail(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
