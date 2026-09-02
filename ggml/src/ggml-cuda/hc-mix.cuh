#include "common.cuh"
#include "ggml.h"

#define CUDA_HC_MIX_BLOCK_SIZE 256

// [johnv8] Fuzja lancuchow qwen4exp build_hc_mix (src/models/qwen4exp.cpp).
//
// 1) kolaps strumieni hyper-connections - 2*hc+3 wezlow w jedno jadro:
//
//      UNARY(SIGMOID) -> MUL -> RESHAPE -> hc x VIEW -> (hc-1) x ADD -> SCALE
//
//      mixed[i, t] = (1/hc) * SUM_c ( xn[i + n_embd*c, t] * sigmoid(up[i + n_embd*c, t]) )
//
//    Sumowanie idzie DOKLADNIE w kolejnosci lancucha ADD: ((m0+m1)+m2)+m3.
//    Tensor posredni `gated` (hc*n_embd*nt floatow) w ogole nie powstaje.
//
// 2) SCALE -> UNARY(SILU) w jedno jadro:  dst = silu(scale*x + bias), bias == 0
//
// GGML_JOHNV8_MIX_FUSE=0 wylacza obie fuzje (domyslnie wlaczone).

// czy przelacznik pozwala na fuzje
bool ggml_cuda_mix_fuse_enabled();

// kolaps strumieni: czy wezly od node_idx to naprawde nasz przypadek.
// Przy sukcesie *n_nodes dostaje dlugosc dopasowanego podgrafu (2*hc+3).
bool ggml_cuda_hc_mix_collapse_ok(const ggml_cgraph * cgraph, int node_idx, int * n_nodes);

void ggml_cuda_op_hc_mix_collapse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// SCALE + UNARY(SILU)
bool ggml_cuda_scale_silu_ok(const ggml_cgraph * cgraph, int node_idx);

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
