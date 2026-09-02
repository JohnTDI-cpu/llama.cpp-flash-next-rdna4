#include "hc-block.cuh"
#include "hc-mix.cuh"
#include "mmvq.cuh"
#include "vecdotq.cuh"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <cstdio>
#include <cstdlib>

#define HCB_THREADS 1024
#define HCB_MAX_NT  4

bool ggml_cuda_hcblock_enabled() {
    static const bool v = []() { const char * e = getenv("GGML_JOHNV8_HCBLOCK"); return e == nullptr || atoi(e) != 0; }();
    return v;
}
// te same przelaczniki co mmvq.cu (sciezka small-K na RDNA4)
static bool hcb_rdna4_small_k() {
    static const bool v = []() { const char * e = getenv("GGML_CUDA_RDNA4_SMALL_K"); return e == nullptr || atoi(e) != 0; }();
    return v;
}
static int hcb_small_k_mult() {
    static const int v = []() { const char * e = getenv("GGML_JOHNV8_SMALLK_MULT"); const int m = e ? atoi(e) : 1; return m < 1 ? 1 : m; }();
    return v;
}
static void hcb_dbg(const char * what, int step) {
    static int left = getenv("GGML_CUDA_DUMP_DISPATCH") ? 60 : 0;
    if (left > 0) { left--; fprintf(stderr, "[johnv8] hcblock_%s: odrzucone w kroku %d\n", what, step); }
}
#define HCB_REJECT(what, n) do { hcb_dbg(what, n); return false; } while (0)

// ---------------------------------------------------------------------------------------------
// replika block_reduce<SUM, 1024> z common.cuh (warp_reduce_sum, s_sum[32], warp_reduce_sum)
static __device__ __forceinline__ float hcb_block_reduce_sum_1024(float val, float * s_sum) {
    val = warp_reduce_sum(val);
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        s_sum[warp_id] = val;
    }
    __syncthreads();
    val = 0.0f;
    if (lane_id < HCB_THREADS / WARP_SIZE) {
        val = s_sum[lane_id];
    }
    val = warp_reduce_sum(val);
    __syncthreads();   // s_sum zostanie uzyte ponownie przy nastepnym wierszu
    return val;
}
// replika quantize_q8_1: jeden warp = jeden blok 32 elementow, lane = element
static __device__ __forceinline__ void hcb_quantize_block(const float xi, block_q8_1 * y, const int lane) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y->qs[lane] = q;
    if (lane == 0) {
        y->ds = make_half2(d, sum);
    }
}
// element xn = (scale_norm * x) * w  — dwa osobne zaokraglenia jak rms_norm_f32<1024,false> + binbcast MUL
static __device__ __forceinline__ float hcb_xn_elem(const float x, const float scale_norm, const float w) {
    const float r = johnv8_mul_nc(scale_norm, x);
    return johnv8_mul_nc(r, w);
}

// ---------------------------------------------------------------------------------------------
// Jadro A. grid: ceil(R / (VB*rows)), blok 1024. Kazdy blok: norma (nt*hc wierszy), kwantyzacja xn (do LDS),
// potem VB wirtualnych blokow mmvq po `rows` wierszy hc_down; wynik przez scale_silu do lo.
template <int ncols_dst, int nwarps, int rows>
static __global__ void __launch_bounds__(HCB_THREADS, 1)
hcblock_a_f32(const float * __restrict__ x,        // [n_embd, hc, nt]
              const float * __restrict__ w_norm,   // [hc_dim]
              const void  * __restrict__ Wd,       // Q8_0 [hc_dim x R]: R wierszy po hc_dim
              float       * __restrict__ xn_out,   // [hc_dim, nt]
              float       * __restrict__ lo_out,   // [R, nt]
              const int n_embd, const int hc, const int R, const float eps,
              const float silu_scale, const float silu_bias) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int vthreads  = nwarps * warp_size;
    constexpr int VB        = HCB_THREADS / vthreads;
    constexpr int qk  = QK8_0;
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    extern __shared__ char hcb_smem[];
    const int hc_dim = n_embd * hc;
    const int nblk   = hc_dim / QK8_1;                   // blokow q8_1 na token
    block_q8_1 * yq  = (block_q8_1 *) hcb_smem;          // [ncols_dst][nblk]
    float * tmp_shared = (float *) (hcb_smem + (size_t) ncols_dst * nblk * sizeof(block_q8_1)); // [VB][nwarps-1][ncols_dst][rows][warp_size]
    float * s_sum  = tmp_shared + (size_t) VB * (nwarps > 1 ? nwarps - 1 : 1) * ncols_dst * rows * warp_size;
    float * scales = s_sum + 32;                          // [ncols_dst*hc]

    const int tid = threadIdx.x;
    // --- A1: normy (rms_norm_f32<1024, false>)
    for (int t = 0; t < ncols_dst; ++t) {
        for (int c = 0; c < hc; ++c) {
            const float * xr = x + ((int64_t) t * hc + c) * n_embd;
            float tmp = 0.0f;
            for (int col = tid; col < n_embd; col += HCB_THREADS) {
                const float xi = xr[col];
                tmp += xi * xi;
            }
            tmp = hcb_block_reduce_sum_1024(tmp, s_sum);
            const float mean  = tmp / n_embd;
            const float scale = rsqrtf(mean + eps);
            if (tid == 0) {
                scales[t * hc + c] = scale;
            }
        }
    }
    __syncthreads();
    // --- A1b: blok 0 zapisuje xn do pamieci globalnej (potrzebne dla hc_inject i jadra B)
    if (blockIdx.x == 0) {
        for (int t = 0; t < ncols_dst; ++t) {
            for (int idx = tid; idx < hc_dim; idx += HCB_THREADS) {
                const int c   = idx / n_embd;
                const int col = idx - c * n_embd;
                xn_out[(int64_t) t * hc_dim + idx] = hcb_xn_elem(x[((int64_t) t * hc + c) * n_embd + col], scales[t * hc + c], w_norm[idx]);
            }
        }
    }
    // --- A2: kwantyzacja xn -> Q8_1 (LDS); warp = blok 32 elementow
    {
        const int warp = tid / warp_size;
        const int lane = tid % warp_size;
        for (int t = 0; t < ncols_dst; ++t) {
            for (int b = warp; b < nblk; b += HCB_THREADS / warp_size) {
                const int idx = b * QK8_1 + lane;
                const int c   = idx / n_embd;
                const int col = idx - c * n_embd;
                const float xi = hcb_xn_elem(x[((int64_t) t * hc + c) * n_embd + col], scales[t * hc + c], w_norm[idx]);
                hcb_quantize_block(xi, &yq[t * nblk + b], lane);
            }
        }
    }
    __syncthreads();
    // --- A3: mmvq (Q8_0 x Q8_1) dla wierszy tego bloku, konfiguracja jak mul_mat_vec_q<Q8_0, ncols_dst, nwarps, rows>
    const int v     = tid / vthreads;
    const int vtid  = tid - v * vthreads;
    const int vwarp = vtid / warp_size;
    const int vlane = vtid - vwarp * warp_size;
    const int row0  = (blockIdx.x * VB + v) * rows;
    const int blocks_per_row_x = hc_dim / qk;
    float tmp[ncols_dst][rows];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            tmp[j][i] = 0.0f;
        }
    }
    if (row0 < R) {
        const int kqs = vdr * (vtid % (qi / vdr));
        for (int kbx = vtid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    tmp[j][i] += vec_dot_q8_0_q8_1(Wd, &yq[j * nblk + kbx], (row0 + i) * blocks_per_row_x + kbx, kqs);
                }
            }
        }
    }
    // redukcja jak w mmvq: warpy > 0 do LDS, warp 0 sumuje w kolejnosci l = 0..nwarps-2, potem warp_reduce
    float * ts = tmp_shared + (size_t) v * (nwarps > 1 ? nwarps - 1 : 1) * ncols_dst * rows * warp_size;
    if (nwarps > 1 && vwarp > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                ts[((vwarp - 1) * ncols_dst * rows + j * rows + i) * warp_size + vlane] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (vwarp == 0 && row0 < R) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps - 1; ++l) {
                    tmp[j][i] += ts[(l * ncols_dst * rows + j * rows + i) * warp_size + vlane];
                }
                tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (vlane == i && row0 + i < R) {
                    // scale_silu_f32: s = scale*x + bias (bez kontrakcji), silu = s / (1 + expf(-s))
                    float s = johnv8_add_nc(johnv8_mul_nc(silu_scale, tmp[j][i]), silu_bias);
#if defined(__AMDGCN__)
                    __asm__("" : "+v"(s));
#endif
                    lo_out[(int64_t) j * R + row0 + i] = s / (1.0f + expf(-s));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Jadro B. grid: n_embd/32, blok 1024. Blok obsluguje 32 elementy i (wszystkie strumienie hc) = hc*32 wierszy hc_up.
template <int ncols_dst, int nwarps, int rows>
static __global__ void __launch_bounds__(HCB_THREADS, 1)
hcblock_b_f32(const float * __restrict__ lo,       // [R, nt]
              const void  * __restrict__ Wu,       // Q8_0 [R x hc_dim]: hc_dim wierszy po R
              const float * __restrict__ xn,       // [hc_dim, nt]
              float       * __restrict__ mixed,    // [n_embd, nt]
              const int n_embd, const int hc, const int R,
              const float mix_scale, const float mix_bias) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int vthreads  = nwarps * warp_size;
    constexpr int VB        = HCB_THREADS / vthreads;
    constexpr int qk  = QK8_0;
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
    constexpr int E = 32;                                  // elementow na blok
    static_assert(VB * rows == E, "VB*rows musi byc 32");

    extern __shared__ char hcb_smem[];
    const int hc_dim = n_embd * hc;
    const int nblk   = R / QK8_1;                          // blokow q8_1 w lo na token
    block_q8_1 * yq  = (block_q8_1 *) hcb_smem;            // [ncols_dst][nblk]
    float * tmp_shared = (float *) (hcb_smem + (size_t) ncols_dst * nblk * sizeof(block_q8_1));
    float * gated_s = tmp_shared + (size_t) VB * (nwarps > 1 ? nwarps - 1 : 1) * ncols_dst * rows * warp_size; // [ncols_dst][hc][E]

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * E;
    // --- B1: kwantyzacja lo (Q8_1)
    {
        const int warp = tid / warp_size;
        const int lane = tid % warp_size;
        for (int t = 0; t < ncols_dst; ++t) {
            for (int b = warp; b < nblk; b += HCB_THREADS / warp_size) {
                hcb_quantize_block(lo[(int64_t) t * R + b * QK8_1 + lane], &yq[t * nblk + b], lane);
            }
        }
    }
    __syncthreads();
    // --- B2: mmvq hc_up dla wierszy c*n_embd + i0 + [0,E), po jednym strumieniu na przebieg
    const int v     = tid / vthreads;
    const int vtid  = tid - v * vthreads;
    const int vwarp = vtid / warp_size;
    const int vlane = vtid - vwarp * warp_size;
    const int blocks_per_row_x = R / qk;
    float * ts = tmp_shared + (size_t) v * (nwarps > 1 ? nwarps - 1 : 1) * ncols_dst * rows * warp_size;
    for (int c = 0; c < hc; ++c) {
        const int row0 = c * n_embd + i0 + v * rows;
        float tmp[ncols_dst][rows];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                tmp[j][i] = 0.0f;
            }
        }
        const int kqs = vdr * (vtid % (qi / vdr));
        for (int kbx = vtid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    tmp[j][i] += vec_dot_q8_0_q8_1(Wu, &yq[j * nblk + kbx], (row0 + i) * blocks_per_row_x + kbx, kqs);
                }
            }
        }
        if (nwarps > 1 && vwarp > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    ts[((vwarp - 1) * ncols_dst * rows + j * rows + i) * warp_size + vlane] = tmp[j][i];
                }
            }
        }
        __syncthreads();
        if (vwarp == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
#pragma unroll
                    for (int l = 0; l < nwarps - 1; ++l) {
                        tmp[j][i] += ts[(l * ncols_dst * rows + j * rows + i) * warp_size + vlane];
                    }
                    tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    if (vlane == i) {
                        // hc_mix_collapse: xn * (1 / (1 + expf(-g))) — exp przez VALU
                        float g = tmp[j][i];
#if defined(__AMDGCN__)
                        __asm__("" : "+v"(g));
#endif
                        const float sg = 1.0f / (1.0f + expf(-g));
                        const int   e  = v * rows + i;                // element w bloku
                        gated_s[(j * hc + c) * E + e] = johnv8_mul_nc(xn[(int64_t) j * hc_dim + row0 + i], sg);
                    }
                }
            }
        }
        __syncthreads();
    }
    // --- B3: ((m0 + m1) + m2) + m3, potem scale*acc + bias
    if (tid < ncols_dst * E) {
        const int j = tid / E;
        const int e = tid - j * E;
        float acc = gated_s[(j * hc + 0) * E + e];
        for (int c = 1; c < hc; ++c) {
            acc = johnv8_add_nc(acc, gated_s[(j * hc + c) * E + e]);
        }
        mixed[(int64_t) j * n_embd + i0 + e] = johnv8_add_nc(johnv8_mul_nc(mix_scale, acc), mix_bias);
    }
}

// ---------------------------------------------------------------------------------------------
static bool hcb_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a->buffer || !b->buffer) {
        return true;
    }
    const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return (b0 <= a0 && a0 < b1) || (a0 <= b0 && b0 < a1);
}
// konfiguracja mmvq jak w mmvq.cu dla RDNA4 / Q8_0
static void hcb_mmvq_cfg(const int ncols_dst, const int K, int * nwarps, int * rows) {
    const int nw = ncols_dst == 1 ? 4 : 2;
    const int blocks_per_row = K / QK8_0;
    const int blocks_per_iter_1warp = VDR_Q8_0_Q8_1_MMVQ * 32 / QI8_0;
    const bool small_k = hcb_rdna4_small_k() && nw > 1 && blocks_per_row < nw * blocks_per_iter_1warp * hcb_small_k_mult();
    *nwarps = nw;
    *rows   = ncols_dst == 1 ? (small_k ? nw : 1) : 2;
}

bool ggml_cuda_hcblock_a_ok(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_hcblock_enabled()) { return false; }
    if (node_idx < 0 || node_idx + 5 >= cgraph->n_nodes) { return false; }
    const ggml_tensor * rms  = cgraph->nodes[node_idx + 0];
    const ggml_tensor * rs   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * mul  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mm   = cgraph->nodes[node_idx + 3];
    const ggml_tensor * scl  = cgraph->nodes[node_idx + 4];
    const ggml_tensor * silu = cgraph->nodes[node_idx + 5];
    if (rms->op != GGML_OP_RMS_NORM || rs->op != GGML_OP_RESHAPE || mul->op != GGML_OP_MUL ||
        mm->op != GGML_OP_MUL_MAT || scl->op != GGML_OP_SCALE || silu->op != GGML_OP_UNARY) { return false; }
    if (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) { HCB_REJECT("a", 1); }
    if (rs->src[0] != rms || mul->src[0] != rs || mm->src[1] != mul || scl->src[0] != mm || silu->src[0] != scl) { HCB_REJECT("a", 2); }
    const ggml_tensor * x  = rms->src[0];
    const ggml_tensor * w  = mul->src[1];
    const ggml_tensor * Wd = mm->src[0];
    if (!x || !w || !Wd) { HCB_REJECT("a", 3); }
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || Wd->type != GGML_TYPE_Q8_0 ||
        rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) { HCB_REJECT("a", 4); }
    const int64_t n_embd = x->ne[0], hc = x->ne[1], nt = x->ne[2];
    const int64_t hc_dim = n_embd * hc;
    if (x->ne[3] != 1 || nt < 1 || nt > HCB_MAX_NT || hc < 2 || hc > 8) { HCB_REJECT("a", 5); }
    if (n_embd % 32 != 0 || n_embd < HCB_THREADS) { HCB_REJECT("a", 6); }   // norma liczona blokiem 1024 jak stock (ncols >= 1024)
    if (rs->ne[0] != hc_dim || rs->ne[1] != nt || rs->ne[2] != 1 || rs->ne[3] != 1) { HCB_REJECT("a", 7); }
    if (w->ne[0] != hc_dim || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) { HCB_REJECT("a", 8); }
    if (!ggml_are_same_shape(mul, rs)) { HCB_REJECT("a", 9); }
    const int64_t R = Wd->ne[1];
    if (Wd->ne[0] != hc_dim || Wd->ne[2] != 1 || Wd->ne[3] != 1 || R % 32 != 0 || R < 32) { HCB_REJECT("a", 10); }
    if (mm->ne[0] != R || mm->ne[1] != nt || mm->ne[2] != 1 || mm->ne[3] != 1) { HCB_REJECT("a", 11); }
    if (!ggml_are_same_shape(scl, mm) || !ggml_are_same_shape(silu, mm)) { HCB_REJECT("a", 12); }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(Wd) || !ggml_is_contiguous(mul) || !ggml_is_contiguous(silu)) { HCB_REJECT("a", 13); }
    if (ggml_get_op_params_f32(scl, 1) != 0.0f) { HCB_REJECT("a", 14); }
    // stock: mmvq dla tej macierzy
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || !ggml_cuda_should_use_mmvq(GGML_TYPE_Q8_0, cc, nt)) { HCB_REJECT("a", 15); }
    // wyjscia podgrafu: mul (xn) i silu (lo); reszta tylko wewnetrzna
    const enum ggml_op ops[6] = { GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY };
    const int32_t outs[2] = { node_idx + 2, node_idx + 5 };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, 6, ops, outs, 2)) { HCB_REJECT("a", 16); }
    if (hcb_overlap(mul, x) || hcb_overlap(silu, x) || hcb_overlap(silu, mul) || hcb_overlap(mul, w) || hcb_overlap(silu, w)) { HCB_REJECT("a", 17); }
    return true;
}

bool ggml_cuda_hcblock_b_ok(const ggml_cgraph * cgraph, int node_idx, int * n_extra) {
    if (!ggml_cuda_hcblock_enabled()) { return false; }
    if (node_idx < 0 || node_idx + 3 >= cgraph->n_nodes) { return false; }
    const ggml_tensor * mm = cgraph->nodes[node_idx];
    if (mm->op != GGML_OP_MUL_MAT) { return false; }
    int n_mix = 0;
    if (!ggml_cuda_hc_mix_collapse_ok(cgraph, node_idx + 1, &n_mix)) { return false; }
    const ggml_tensor * sigmoid = cgraph->nodes[node_idx + 1];
    const ggml_tensor * mul     = cgraph->nodes[node_idx + 2];
    const ggml_tensor * scl     = cgraph->nodes[node_idx + n_mix];   // ostatni wezel lancucha mix (n_mix wezlow od sigmoid)
    if (sigmoid->src[0] != mm) { HCB_REJECT("b", 1); }
    if (mul->op != GGML_OP_MUL || mul->src[1] != sigmoid) { HCB_REJECT("b", 2); }
    if (scl->op != GGML_OP_SCALE) { HCB_REJECT("b", 3); }
    const ggml_tensor * Wu = mm->src[0];
    const ggml_tensor * lo = mm->src[1];
    const ggml_tensor * xn = mul->src[0];
    if (!Wu || !lo || !xn) { HCB_REJECT("b", 4); }
    if (Wu->type != GGML_TYPE_Q8_0 || lo->type != GGML_TYPE_F32 || xn->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || scl->type != GGML_TYPE_F32) { HCB_REJECT("b", 5); }
    const int64_t R = Wu->ne[0], hc_dim = Wu->ne[1], nt = lo->ne[1];
    const int64_t n_embd = scl->ne[0];
    if (n_embd <= 0 || hc_dim % n_embd != 0) { HCB_REJECT("b", 6); }
    const int64_t hc = hc_dim / n_embd;
    if (nt < 1 || nt > HCB_MAX_NT || hc < 2 || hc > 8 || n_embd % 32 != 0 || R % 32 != 0) { HCB_REJECT("b", 7); }
    if (lo->ne[0] != R || lo->ne[2] != 1 || lo->ne[3] != 1 || Wu->ne[2] != 1 || Wu->ne[3] != 1) { HCB_REJECT("b", 8); }
    if (mm->ne[0] != hc_dim || mm->ne[1] != nt || mm->ne[2] != 1 || mm->ne[3] != 1) { HCB_REJECT("b", 9); }
    if (xn->ne[0] != hc_dim || xn->ne[1] != nt || xn->ne[2] != 1 || xn->ne[3] != 1) { HCB_REJECT("b", 10); }
    if (scl->ne[1] != nt || scl->ne[2] != 1 || scl->ne[3] != 1) { HCB_REJECT("b", 11); }
    if (!ggml_is_contiguous(Wu) || !ggml_is_contiguous(lo) || !ggml_is_contiguous(xn) || !ggml_is_contiguous(scl)) { HCB_REJECT("b", 12); }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || !ggml_cuda_should_use_mmvq(GGML_TYPE_Q8_0, cc, nt)) { HCB_REJECT("b", 13); }
    // wynik mm uzywany tylko przez sigmoid
    {
        const int n_nodes = 1 + n_mix;
        std::vector<enum ggml_op> ops(n_nodes);
        for (int k = 0; k < n_nodes; ++k) { ops[k] = cgraph->nodes[node_idx + k]->op; }
        const int32_t out = node_idx + n_mix;
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, n_nodes, ops.data(), &out, 1)) { HCB_REJECT("b", 14); }
    }
    // dst (scale) nie moze nachodzic na lo/xn/Wu (lo bywa martwe po mm -> alokator moglby je nalozyc; model pinuje lo jako output)
    if (hcb_overlap(scl, lo) || hcb_overlap(scl, xn) || hcb_overlap(scl, Wu)) { HCB_REJECT("b", 15); }
    *n_extra = n_mix;
    return true;
}

template <int ncols_dst>
static void hcb_launch_a(ggml_backend_cuda_context & ctx, const float * x, const float * w, const void * Wd, float * xn, float * lo,
                         int n_embd, int hc, int R, float eps, float s_scale, float s_bias) {
    int nwarps = 0, rows = 0;
    hcb_mmvq_cfg(ncols_dst, n_embd * hc, &nwarps, &rows);
    const int hc_dim = n_embd * hc;
    const int nblk   = hc_dim / QK8_1;
    auto launch = [&](auto kern, int nw, int rw) {
        const int VB = HCB_THREADS / (nw * 32);
        const size_t smem = (size_t) ncols_dst * nblk * sizeof(block_q8_1) + (size_t) VB * (nw > 1 ? nw - 1 : 1) * ncols_dst * rw * 32 * sizeof(float) + (32 + ncols_dst * hc) * sizeof(float);
        const int grid = (R + VB * rw - 1) / (VB * rw);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(grid, 1, 1), dim3(HCB_THREADS, 1, 1), smem, ctx.stream());
        ggml_cuda_kernel_launch(kern, lp, x, w, Wd, xn, lo, n_embd, hc, R, eps, s_scale, s_bias);
    };
    if (nwarps == 4 && rows == 1)      { launch(hcblock_a_f32<ncols_dst, 4, 1>, 4, 1); }
    else if (nwarps == 2 && rows == 2) { launch(hcblock_a_f32<ncols_dst, 2, 2>, 2, 2); }
    else if (nwarps == 4 && rows == 4) { launch(hcblock_a_f32<ncols_dst, 4, 4>, 4, 4); }
    else { GGML_ABORT("hcblock_a: nieobslugiwana konfiguracja mmvq nwarps=%d rows=%d", nwarps, rows); }
}
template <int ncols_dst>
static void hcb_launch_b(ggml_backend_cuda_context & ctx, const float * lo, const void * Wu, const float * xn, float * mixed,
                         int n_embd, int hc, int R, float m_scale, float m_bias) {
    int nwarps = 0, rows = 0;
    hcb_mmvq_cfg(ncols_dst, R, &nwarps, &rows);
    const int nblk = R / QK8_1;
    auto launch = [&](auto kern, int nw, int rw) {
        const int VB = HCB_THREADS / (nw * 32);
        GGML_ASSERT(VB * rw == 32);
        const size_t smem = (size_t) ncols_dst * nblk * sizeof(block_q8_1) + (size_t) VB * (nw > 1 ? nw - 1 : 1) * ncols_dst * rw * 32 * sizeof(float) + (size_t) ncols_dst * hc * 32 * sizeof(float);
        const int grid = n_embd / 32;
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(grid, 1, 1), dim3(HCB_THREADS, 1, 1), smem, ctx.stream());
        ggml_cuda_kernel_launch(kern, lp, lo, Wu, xn, mixed, n_embd, hc, R, m_scale, m_bias);
    };
    if (nwarps == 4 && rows == 4)      { launch(hcblock_b_f32<ncols_dst, 4, 4>, 4, 4); }
    else if (nwarps == 2 && rows == 2) { launch(hcblock_b_f32<ncols_dst, 2, 2>, 2, 2); }
    else { GGML_ABORT("hcblock_b: nieobslugiwana konfiguracja mmvq nwarps=%d rows=%d (VB*rows musi byc 32)", nwarps, rows); }
}

void ggml_cuda_op_hcblock_a(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * rms  = cgraph->nodes[node_idx + 0];
    const ggml_tensor * mul  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mm   = cgraph->nodes[node_idx + 3];
    const ggml_tensor * scl  = cgraph->nodes[node_idx + 4];
    const ggml_tensor * silu = cgraph->nodes[node_idx + 5];
    const ggml_tensor * x = rms->src[0]; const ggml_tensor * w = mul->src[1]; const ggml_tensor * Wd = mm->src[0];
    const int n_embd = (int) x->ne[0], hc = (int) x->ne[1], nt = (int) x->ne[2], R = (int) Wd->ne[1];
    float eps = 0.0f; memcpy(&eps, rms->op_params, sizeof(float));
    const float s_scale = ggml_get_op_params_f32(scl, 0), s_bias = ggml_get_op_params_f32(scl, 1);
    static bool logged = false;
    if (!logged && getenv("GGML_CUDA_DUMP_DISPATCH")) { logged = true; fprintf(stderr, "[johnv8] hcblock A fused: n_embd=%d hc=%d nt=%d R=%d (6 wezlow -> 1 jadro)\n", n_embd, hc, nt, R); }
    const float * x_d = (const float *) x->data; const float * w_d = (const float *) w->data;
    float * xn_d = (float *) mul->data; float * lo_d = (float *) silu->data;
    switch (nt) {
        case 1: hcb_launch_a<1>(ctx, x_d, w_d, Wd->data, xn_d, lo_d, n_embd, hc, R, eps, s_scale, s_bias); break;
        case 2: hcb_launch_a<2>(ctx, x_d, w_d, Wd->data, xn_d, lo_d, n_embd, hc, R, eps, s_scale, s_bias); break;
        case 3: hcb_launch_a<3>(ctx, x_d, w_d, Wd->data, xn_d, lo_d, n_embd, hc, R, eps, s_scale, s_bias); break;
        case 4: hcb_launch_a<4>(ctx, x_d, w_d, Wd->data, xn_d, lo_d, n_embd, hc, R, eps, s_scale, s_bias); break;
        default: GGML_ABORT("hcblock_a: nt=%d", nt);
    }
}
void ggml_cuda_op_hcblock_b(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int n_extra) {
    const ggml_tensor * mm  = cgraph->nodes[node_idx];
    const ggml_tensor * mul = cgraph->nodes[node_idx + 2];
    const ggml_tensor * scl = cgraph->nodes[node_idx + n_extra];
    const ggml_tensor * Wu = mm->src[0]; const ggml_tensor * lo = mm->src[1]; const ggml_tensor * xn = mul->src[0];
    const int R = (int) Wu->ne[0], hc_dim = (int) Wu->ne[1], nt = (int) lo->ne[1], n_embd = (int) scl->ne[0];
    const int hc = hc_dim / n_embd;
    const float m_scale = ggml_get_op_params_f32(scl, 0), m_bias = ggml_get_op_params_f32(scl, 1);
    static bool logged = false;
    if (!logged && getenv("GGML_CUDA_DUMP_DISPATCH")) { logged = true; fprintf(stderr, "[johnv8] hcblock B fused: n_embd=%d hc=%d nt=%d R=%d (%d wezlow -> 1 jadro)\n", n_embd, hc, nt, R, n_extra + 1); }
    const float * lo_d = (const float *) lo->data; const float * xn_d = (const float *) xn->data; float * mixed_d = (float *) scl->data;
    switch (nt) {
        case 1: hcb_launch_b<1>(ctx, lo_d, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        case 2: hcb_launch_b<2>(ctx, lo_d, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        case 3: hcb_launch_b<3>(ctx, lo_d, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        case 4: hcb_launch_b<4>(ctx, lo_d, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        default: GGML_ABORT("hcblock_b: nt=%d", nt);
    }
}
