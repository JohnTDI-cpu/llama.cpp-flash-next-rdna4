#include "common.cuh"
#include "ggml.h"

// [johnv8] E10c: caly blok hyper-connection przed podblokiem (attn/ffn) w DWOCH jadrach zamiast siedmiu:
//   A: RMS_NORM(x) -> RESHAPE -> MUL(w_norm) -> MUL_MAT(hc_down Q8_0) -> SCALE(1/hc) -> SILU
//      (kazdy blok CUDA liczy norme i kwantyzacje Q8_1 od nowa, potem swoje wiersze hc_down; blok 0 zapisuje xn)
//   B: MUL_MAT(hc_up Q8_0) -> SIGMOID -> MUL(xn) -> RESHAPE -> VIEW*hc -> ADD*(hc-1) -> SCALE(1/hc)
// Arytmetyka replikuje bit w bit: norm.cu (rms_norm_f32<1024>), binbcast MUL, quantize.cu (quantize_q8_1),
// mmvq.cu (mul_mat_vec_q<Q8_0, ncols_dst, nwarps, rows_per_block> z tablicy RDNA4), hc-mix.cu (scale_silu, collapse).
// Warunek: nt <= 4 (paczka MTP), stock uzywalby mmvq. GGML_JOHNV8_HCBLOCK=0 wylacza.
bool ggml_cuda_hcblock_enabled();
bool ggml_cuda_hcblock_a_ok(const ggml_cgraph * cgraph, int node_idx);
void ggml_cuda_op_hcblock_a(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
bool ggml_cuda_hcblock_b_ok(const ggml_cgraph * cgraph, int node_idx, int * n_extra);
void ggml_cuda_op_hcblock_b(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int n_extra);
