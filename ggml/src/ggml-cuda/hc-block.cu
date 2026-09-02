#include "hc-block.cuh"
#include "hc-mix.cuh"
#include "mmvq.cuh"
#include "vecdotq.cuh"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <cstdio>
#include <cstdlib>

// [johnv8] E10c v2 (A0 / A1 / B1): trzy jadra zamiast siedmiu, kazde z rownolegloscia jak stock.
//   A0: normy + zapis xn + kwantyzacja Q8_1 xn -> bufor kontekstu (grid = nt blokow po 1024 watkow)
//   A1: mmvq hc_down (siatka i blok jak mul_mat_vec_q) + scale_silu -> lo (tensor + kopia); OSTATNI blok kwantyzuje lo
//   B1: mmvq hc_up (wirtualne bloki jak stock) + sigmoid + mix -> mixed (blok = 8 elementow x hc strumieni)
// Arytmetyka replikuje bit w bit: norm.cu, binbcast MUL, quantize.cu, mmvq.cu (tablica RDNA4), hc-mix.cu.
#define HCB_NORM_THREADS 1024
#define HCB_MAX_NT       4
#define HCB_B_THREADS    256
#define HCB_B_ELEMS      8

static int ggml_cuda_hcblock_mode() {
    // domyslnie OFF do walidacji v2 (E39); 1 = oba jadra, 2 = tylko A (A0+A1), 3 = tylko B (wymaga A dla kwantyzacji lo -> w praktyce = 1)
    static const int v = []() { const char * e = getenv("GGML_JOHNV8_HCBLOCK"); return e == nullptr ? 0 : atoi(e); }();
    return v;
}
bool ggml_cuda_hcblock_enabled() { return ggml_cuda_hcblock_mode() != 0; }
static bool hcb_a_enabled() { const int m = ggml_cuda_hcblock_mode(); return m == 1 || m == 2 || m == 3; }
static bool hcb_b_enabled() { const int m = ggml_cuda_hcblock_mode(); return m == 1 || m == 3; }
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
static thread_local const ggml_tensor * hcb_last_lo = nullptr;   // lo, dla ktorego A1 zapisal kwantyzacje do bufora kontekstu

// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ float hcb_block_reduce_sum_1024(float val, float * s_sum) {
    val = warp_reduce_sum(val);
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) { s_sum[warp_id] = val; }
    __syncthreads();
    val = 0.0f;
    if (lane_id < HCB_NORM_THREADS / WARP_SIZE) { val = s_sum[lane_id]; }
    val = warp_reduce_sum(val);
    __syncthreads();
    return val;
}
static __device__ __forceinline__ void hcb_quantize_block(const float xi, block_q8_1 * y, const int lane) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y->qs[lane] = q;
    if (lane == 0) { y->ds = make_half2(d, sum); }
}
static __device__ __forceinline__ float hcb_xn_elem(const float x, const float scale_norm, const float w) {
    const float r = johnv8_mul_nc(scale_norm, x);
    return johnv8_mul_nc(r, w);
}

// ---- A0: grid = nt, blok 1024
static __global__ void __launch_bounds__(HCB_NORM_THREADS, 1)
hcblock_a0_f32(const float * __restrict__ x, const float * __restrict__ w_norm, float * __restrict__ xn_out,
               block_q8_1 * __restrict__ yq, const int n_embd, const int hc, const float eps) {
    __shared__ float s_sum[32];
    __shared__ float scales[8];
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    const int hc_dim = n_embd * hc;
    const int nblk   = hc_dim / QK8_1;
    for (int c = 0; c < hc; ++c) {
        const float * xr = x + ((int64_t) t * hc + c) * n_embd;
        float tmp = 0.0f;
        for (int col = tid; col < n_embd; col += HCB_NORM_THREADS) { const float xi = xr[col]; tmp += xi * xi; }
        tmp = hcb_block_reduce_sum_1024(tmp, s_sum);
        const float mean  = tmp / n_embd;
        const float scale = rsqrtf(mean + eps);
        if (tid == 0) { scales[c] = scale; }
    }
    __syncthreads();
    for (int idx = tid; idx < hc_dim; idx += HCB_NORM_THREADS) {
        const int c = idx / n_embd; const int col = idx - c * n_embd;
        xn_out[(int64_t) t * hc_dim + idx] = hcb_xn_elem(x[((int64_t) t * hc + c) * n_embd + col], scales[c], w_norm[idx]);
    }
    const int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    for (int b = warp; b < nblk; b += HCB_NORM_THREADS / WARP_SIZE) {
        const int idx = b * QK8_1 + lane; const int c = idx / n_embd; const int col = idx - c * n_embd;
        const float xi = hcb_xn_elem(x[((int64_t) t * hc + c) * n_embd + col], scales[c], w_norm[idx]);
        hcb_quantize_block(xi, &yq[t * nblk + b], lane);
    }
}

// ---- A1: replika mul_mat_vec_q<Q8_0, ncols_dst, nwarps, rows>: grid = R/rows, blok = nwarps*32
template <int ncols_dst, int nwarps, int rows>
static __global__ void __launch_bounds__(nwarps * 32, 1)
hcblock_a1_f32(const void * __restrict__ Wd, const block_q8_1 * __restrict__ yq, float * __restrict__ lo_out,
               float * __restrict__ lo_scr, block_q8_1 * __restrict__ yq2, int * __restrict__ counter,
               const int K, const int R, const float silu_scale, const float silu_bias) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk = QK8_0, qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
    __shared__ float tmp_shared[nwarps > 1 ? nwarps - 1 : 1][ncols_dst][rows][warp_size];
    __shared__ int   s_last;
    const int tid  = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = rows * blockIdx.x;
    const int blocks_per_row_x = K / qk;
    const int nblk = blocks_per_row_x;   // q8_1 blokow na kolumne yq
    float tmp[ncols_dst][rows];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows; ++i) { tmp[j][i] = 0.0f; }
    }
    const int kqs = vdr * (tid % (qi / vdr));
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                tmp[j][i] += vec_dot_q8_0_q8_1(Wd, &yq[j * nblk + kbx], (row0 + i) * blocks_per_row_x + kbx, kqs);
            }
        }
    }
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) { tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i]; }
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps - 1; ++l) { tmp[j][i] += tmp_shared[l][j][i][threadIdx.x]; }
                tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (threadIdx.x == i && row0 + i < R) {
                    float s = johnv8_add_nc(johnv8_mul_nc(silu_scale, tmp[j][i]), silu_bias);
#if defined(__AMDGCN__)
                    __asm__("" : "+v"(s));
#endif
                    const float sv = s / (1.0f + expf(-s));
                    lo_out[(int64_t) j * R + row0 + i] = sv;
                    lo_scr[(int64_t) j * R + row0 + i] = sv;
                }
            }
        }
    }
    // ostatni blok kwantyzuje lo (wynik kwantyzacji nie zalezy od tego, ktory blok ja robi)
    __threadfence();
    __syncthreads();
    if (tid == 0) { s_last = (atomicAdd(counter, 1) == (int) gridDim.x - 1) ? 1 : 0; }
    __syncthreads();
    if (s_last) {
        __threadfence();
        const int nblk2 = R / QK8_1;
        const int warp = tid / warp_size, lane = tid % warp_size;
        for (int t = 0; t < ncols_dst; ++t) {
            for (int b = warp; b < nblk2; b += nwarps) {
                hcb_quantize_block(lo_scr[(int64_t) t * R + b * QK8_1 + lane], &yq2[t * nblk2 + b], lane);
            }
        }
        __threadfence();
        if (tid == 0) { *counter = 0; }
    }
}

// ---- B1: grid = n_embd/8, blok 256 = VB wirtualnych blokow po nwarps*32
template <int ncols_dst, int nwarps, int rows>
static __global__ void __launch_bounds__(HCB_B_THREADS, 1)
hcblock_b1_f32(const block_q8_1 * __restrict__ yq2, const void * __restrict__ Wu, const float * __restrict__ xn,
               float * __restrict__ mixed, const int n_embd, const int hc, const int R, const float mix_scale, const float mix_bias) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int vthreads  = nwarps * warp_size;
    constexpr int VB        = HCB_B_THREADS / vthreads;
    constexpr int E         = HCB_B_ELEMS;
    static_assert(VB * rows == E, "VB*rows musi byc 8");
    constexpr int qk = QK8_0, qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
    __shared__ float tmp_shared[VB][nwarps > 1 ? nwarps - 1 : 1][ncols_dst][rows][warp_size];
    __shared__ float gated_s[ncols_dst][8][E];
    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * E;
    const int hc_dim = n_embd * hc;
    const int nblk2 = R / QK8_1;
    const int v = tid / vthreads, vtid = tid - v * vthreads, vwarp = vtid / warp_size, vlane = vtid - vwarp * warp_size;
    const int blocks_per_row_x = R / qk;
    const int kqs = vdr * (vtid % (qi / vdr));
    for (int c = 0; c < hc; ++c) {
        const int row0 = c * n_embd + i0 + v * rows;
        float tmp[ncols_dst][rows];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) { tmp[j][i] = 0.0f; }
        }
        for (int kbx = vtid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    tmp[j][i] += vec_dot_q8_0_q8_1(Wu, &yq2[j * nblk2 + kbx], (row0 + i) * blocks_per_row_x + kbx, kqs);
                }
            }
        }
        if (vwarp > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) { tmp_shared[v][vwarp - 1][j][i][vlane] = tmp[j][i]; }
            }
        }
        __syncthreads();
        if (vwarp == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
#pragma unroll
                    for (int l = 0; l < nwarps - 1; ++l) { tmp[j][i] += tmp_shared[v][l][j][i][vlane]; }
                    tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    if (vlane == i) {
                        float g = tmp[j][i];
#if defined(__AMDGCN__)
                        __asm__("" : "+v"(g));
#endif
                        const float sg = 1.0f / (1.0f + expf(-g));
                        gated_s[j][c][v * rows + i] = johnv8_mul_nc(xn[(int64_t) j * hc_dim + row0 + i], sg);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (tid < ncols_dst * E) {
        const int j = tid / E, e = tid - j * E;
        float acc = gated_s[j][0][e];
        for (int c = 1; c < hc; ++c) { acc = johnv8_add_nc(acc, gated_s[j][c][e]); }
        mixed[(int64_t) j * n_embd + i0 + e] = johnv8_add_nc(johnv8_mul_nc(mix_scale, acc), mix_bias);
    }
}

// ---------------------------------------------------------------------------------------------
static bool hcb_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a->buffer || !b->buffer) { return true; }
    const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return (b0 <= a0 && a0 < b1) || (a0 <= b0 && b0 < a1);
}
static void hcb_mmvq_cfg(const int ncols_dst, const int K, int * nwarps, int * rows) {
    const int nw = ncols_dst == 1 ? 4 : 2;
    const int blocks_per_row = K / QK8_0;
    const int blocks_per_iter_1warp = VDR_Q8_0_Q8_1_MMVQ * 32 / QI8_0;
    const bool small_k = hcb_rdna4_small_k() && nw > 1 && blocks_per_row < nw * blocks_per_iter_1warp * hcb_small_k_mult();
    *nwarps = nw;
    *rows   = ncols_dst == 1 ? (small_k ? nw : 1) : 2;
}

bool ggml_cuda_hcblock_a_ok(const ggml_cgraph * cgraph, int node_idx) {
    if (!hcb_a_enabled()) { return false; }
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
    const ggml_tensor * x = rms->src[0]; const ggml_tensor * w = mul->src[1]; const ggml_tensor * Wd = mm->src[0];
    if (!x || !w || !Wd) { HCB_REJECT("a", 3); }
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || Wd->type != GGML_TYPE_Q8_0 ||
        rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) { HCB_REJECT("a", 4); }
    const int64_t n_embd = x->ne[0], hc = x->ne[1], nt = x->ne[2], hc_dim = n_embd * hc;
    if (x->ne[3] != 1 || nt < 1 || nt > HCB_MAX_NT || hc < 2 || hc > 8) { HCB_REJECT("a", 5); }
    if (n_embd % 32 != 0 || n_embd < HCB_NORM_THREADS || n_embd % HCB_B_ELEMS != 0) { HCB_REJECT("a", 6); }
    if (rs->ne[0] != hc_dim || rs->ne[1] != nt || rs->ne[2] != 1 || rs->ne[3] != 1) { HCB_REJECT("a", 7); }
    if (w->ne[0] != hc_dim || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) { HCB_REJECT("a", 8); }
    if (!ggml_are_same_shape(mul, rs)) { HCB_REJECT("a", 9); }
    const int64_t R = Wd->ne[1];
    if (Wd->ne[0] != hc_dim || Wd->ne[2] != 1 || Wd->ne[3] != 1 || R % 32 != 0 || R < 32) { HCB_REJECT("a", 10); }
    if (mm->ne[0] != R || mm->ne[1] != nt || mm->ne[2] != 1 || mm->ne[3] != 1) { HCB_REJECT("a", 11); }
    if (!ggml_are_same_shape(scl, mm) || !ggml_are_same_shape(silu, mm)) { HCB_REJECT("a", 12); }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(Wd) || !ggml_is_contiguous(mul) || !ggml_is_contiguous(silu)) { HCB_REJECT("a", 13); }
    if (ggml_get_op_params_f32(scl, 1) != 0.0f) { HCB_REJECT("a", 14); }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || !ggml_cuda_should_use_mmvq(GGML_TYPE_Q8_0, cc, nt)) { HCB_REJECT("a", 15); }
    {
        int nw = 0, rw = 0; hcb_mmvq_cfg((int) nt, (int) hc_dim, &nw, &rw);
        if (R % rw != 0) { HCB_REJECT("a", 18); }
    }
    const enum ggml_op ops[6] = { GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY };
    const int32_t outs[2] = { node_idx + 2, node_idx + 5 };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, 6, ops, outs, 2)) { HCB_REJECT("a", 16); }
    if (hcb_overlap(mul, x) || hcb_overlap(silu, x) || hcb_overlap(silu, mul) || hcb_overlap(mul, w) || hcb_overlap(silu, w)) { HCB_REJECT("a", 17); }
    return true;
}

bool ggml_cuda_hcblock_b_ok(const ggml_cgraph * cgraph, int node_idx, int * n_extra) {
    if (!hcb_b_enabled()) { return false; }
    if (node_idx < 0 || node_idx + 3 >= cgraph->n_nodes) { return false; }
    const ggml_tensor * mm = cgraph->nodes[node_idx];
    if (mm->op != GGML_OP_MUL_MAT) { return false; }
    int n_mix = 0;
    if (!ggml_cuda_hc_mix_collapse_ok(cgraph, node_idx + 1, &n_mix)) { return false; }
    const ggml_tensor * sigmoid = cgraph->nodes[node_idx + 1];
    const ggml_tensor * mul     = cgraph->nodes[node_idx + 2];
    const ggml_tensor * scl     = cgraph->nodes[node_idx + n_mix];
    if (sigmoid->src[0] != mm) { HCB_REJECT("b", 1); }
    if (mul->op != GGML_OP_MUL || mul->src[1] != sigmoid) { HCB_REJECT("b", 2); }
    if (scl->op != GGML_OP_SCALE) { HCB_REJECT("b", 3); }
    const ggml_tensor * Wu = mm->src[0]; const ggml_tensor * lo = mm->src[1]; const ggml_tensor * xn = mul->src[0];
    if (!Wu || !lo || !xn) { HCB_REJECT("b", 4); }
    if (Wu->type != GGML_TYPE_Q8_0 || lo->type != GGML_TYPE_F32 || xn->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || scl->type != GGML_TYPE_F32) { HCB_REJECT("b", 5); }
    const int64_t R = Wu->ne[0], hc_dim = Wu->ne[1], nt = lo->ne[1], n_embd = scl->ne[0];
    if (n_embd <= 0 || hc_dim % n_embd != 0) { HCB_REJECT("b", 6); }
    const int64_t hc = hc_dim / n_embd;
    if (nt < 1 || nt > HCB_MAX_NT || hc < 2 || hc > 8 || n_embd % HCB_B_ELEMS != 0 || R % 32 != 0) { HCB_REJECT("b", 7); }
    if (lo->ne[0] != R || lo->ne[2] != 1 || lo->ne[3] != 1 || Wu->ne[2] != 1 || Wu->ne[3] != 1) { HCB_REJECT("b", 8); }
    if (mm->ne[0] != hc_dim || mm->ne[1] != nt || mm->ne[2] != 1 || mm->ne[3] != 1) { HCB_REJECT("b", 9); }
    if (xn->ne[0] != hc_dim || xn->ne[1] != nt || xn->ne[2] != 1 || xn->ne[3] != 1) { HCB_REJECT("b", 10); }
    if (scl->ne[1] != nt || scl->ne[2] != 1 || scl->ne[3] != 1) { HCB_REJECT("b", 11); }
    if (!ggml_is_contiguous(Wu) || !ggml_is_contiguous(lo) || !ggml_is_contiguous(xn) || !ggml_is_contiguous(scl)) { HCB_REJECT("b", 12); }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || !ggml_cuda_should_use_mmvq(GGML_TYPE_Q8_0, cc, nt)) { HCB_REJECT("b", 13); }
    {
        const int n_nodes = 1 + n_mix;
        std::vector<enum ggml_op> ops(n_nodes);
        for (int k = 0; k < n_nodes; ++k) { ops[k] = cgraph->nodes[node_idx + k]->op; }
        const int32_t out = node_idx + n_mix;
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, n_nodes, ops.data(), &out, 1)) { HCB_REJECT("b", 14); }
    }
    if (hcb_overlap(scl, xn) || hcb_overlap(scl, Wu)) { HCB_REJECT("b", 15); }
    // B czyta lo (skwantyzowane) z bufora kontekstu zapisanego przez A1 dla tego samego tensora
    if (hcb_last_lo != lo) { HCB_REJECT("b", 16); }
    *n_extra = n_mix;
    return true;
}

template <int ncols_dst>
static void hcb_launch_a1(ggml_backend_cuda_context & ctx, const void * Wd, const block_q8_1 * yq, float * lo, float * lo_scr,
                          block_q8_1 * yq2, int * counter, int K, int R, float s_scale, float s_bias) {
    int nwarps = 0, rows = 0; hcb_mmvq_cfg(ncols_dst, K, &nwarps, &rows);
    auto launch = [&](auto kern, int nw, int rw) {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(R / rw, 1, 1), dim3(32, nw, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(kern, lp, Wd, yq, lo, lo_scr, yq2, counter, K, R, s_scale, s_bias);
    };
    if (nwarps == 4 && rows == 1)      { launch(hcblock_a1_f32<ncols_dst, 4, 1>, 4, 1); }
    else if (nwarps == 2 && rows == 2) { launch(hcblock_a1_f32<ncols_dst, 2, 2>, 2, 2); }
    else if (nwarps == 4 && rows == 4) { launch(hcblock_a1_f32<ncols_dst, 4, 4>, 4, 4); }
    else { GGML_ABORT("hcblock_a1: konfiguracja nwarps=%d rows=%d", nwarps, rows); }
}
template <int ncols_dst>
static void hcb_launch_b1(ggml_backend_cuda_context & ctx, const block_q8_1 * yq2, const void * Wu, const float * xn, float * mixed,
                          int n_embd, int hc, int R, float m_scale, float m_bias) {
    int nwarps = 0, rows = 0; hcb_mmvq_cfg(ncols_dst, R, &nwarps, &rows);
    auto launch = [&](auto kern) {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(n_embd / HCB_B_ELEMS, 1, 1), dim3(HCB_B_THREADS, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(kern, lp, yq2, Wu, xn, mixed, n_embd, hc, R, m_scale, m_bias);
    };
    if (nwarps == 4 && rows == 4)      { launch(hcblock_b1_f32<ncols_dst, 4, 4>); }   // VB = 2, 2*4 = 8
    else if (nwarps == 2 && rows == 2) { launch(hcblock_b1_f32<ncols_dst, 2, 2>); }   // VB = 4, 4*2 = 8
    else { GGML_ABORT("hcblock_b1: konfiguracja nwarps=%d rows=%d", nwarps, rows); }
}

void ggml_cuda_op_hcblock_a(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * rms  = cgraph->nodes[node_idx + 0];
    const ggml_tensor * mul  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mm   = cgraph->nodes[node_idx + 3];
    const ggml_tensor * scl  = cgraph->nodes[node_idx + 4];
    const ggml_tensor * silu = cgraph->nodes[node_idx + 5];
    const ggml_tensor * x = rms->src[0]; const ggml_tensor * w = mul->src[1]; const ggml_tensor * Wd = mm->src[0];
    const int n_embd = (int) x->ne[0], hc = (int) x->ne[1], nt = (int) x->ne[2], R = (int) Wd->ne[1], hc_dim = n_embd * hc;
    float eps = 0.0f; memcpy(&eps, rms->op_params, sizeof(float));
    const float s_scale = ggml_get_op_params_f32(scl, 0), s_bias = ggml_get_op_params_f32(scl, 1);
    static bool logged = false;
    if (!logged && getenv("GGML_CUDA_DUMP_DISPATCH")) { logged = true; fprintf(stderr, "[johnv8] hcblock A0+A1 fused: n_embd=%d hc=%d nt=%d R=%d (6 wezlow -> 2 jadra)\n", n_embd, hc, nt, R); }
    auto & H = ctx.hcb;
    const size_t yq_bytes  = (size_t) nt * (hc_dim / QK8_1) * sizeof(block_q8_1);
    const size_t yq2_bytes = (size_t) nt * (R / QK8_1) * sizeof(block_q8_1);
    const size_t lo_bytes  = (size_t) nt * R * sizeof(float);
    block_q8_1 * yq  = (block_q8_1 *) H.ensure_buf(&H.yq,  &H.yq_cap,  yq_bytes);
    block_q8_1 * yq2 = (block_q8_1 *) H.ensure_buf(&H.yq2, &H.yq2_cap, yq2_bytes);
    float      * los = (float *)      H.ensure_buf(&H.buf, &H.cap,     lo_bytes);
    int        * cnt = (int *)        H.ensure_buf(&H.counter, &H.counter_cap, 256, true);
    if (!yq || !yq2 || !los || !cnt) { GGML_ABORT("hcblock: brak pamieci na bufory kontekstu"); }
    const float * x_d = (const float *) x->data; const float * w_d = (const float *) w->data;
    float * xn_d = (float *) mul->data; float * lo_d = (float *) silu->data;
    {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(HCB_NORM_THREADS, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(hcblock_a0_f32, lp, x_d, w_d, xn_d, yq, n_embd, hc, eps);
    }
    switch (nt) {
        case 1: hcb_launch_a1<1>(ctx, Wd->data, yq, lo_d, los, yq2, cnt, hc_dim, R, s_scale, s_bias); break;
        case 2: hcb_launch_a1<2>(ctx, Wd->data, yq, lo_d, los, yq2, cnt, hc_dim, R, s_scale, s_bias); break;
        case 3: hcb_launch_a1<3>(ctx, Wd->data, yq, lo_d, los, yq2, cnt, hc_dim, R, s_scale, s_bias); break;
        case 4: hcb_launch_a1<4>(ctx, Wd->data, yq, lo_d, los, yq2, cnt, hc_dim, R, s_scale, s_bias); break;
        default: GGML_ABORT("hcblock_a: nt=%d", nt);
    }
    H.lo = silu; H.eval = ctx.q8cache.eval; hcb_last_lo = silu;
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
    if (!logged && getenv("GGML_CUDA_DUMP_DISPATCH")) { logged = true; fprintf(stderr, "[johnv8] hcblock B1 fused: n_embd=%d hc=%d nt=%d R=%d (%d wezlow -> 1 jadro)\n", n_embd, hc, nt, R, n_extra + 1); }
    auto & H = ctx.hcb;
    GGML_ASSERT(H.lo == lo && H.eval == ctx.q8cache.eval && H.yq2 != nullptr);
    const block_q8_1 * yq2 = (const block_q8_1 *) H.yq2;
    const float * xn_d = (const float *) xn->data; float * mixed_d = (float *) scl->data;
    switch (nt) {
        case 1: hcb_launch_b1<1>(ctx, yq2, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        case 2: hcb_launch_b1<2>(ctx, yq2, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        case 3: hcb_launch_b1<3>(ctx, yq2, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        case 4: hcb_launch_b1<4>(ctx, yq2, Wu->data, xn_d, mixed_d, n_embd, hc, R, m_scale, m_bias); break;
        default: GGML_ABORT("hcblock_b: nt=%d", nt);
    }
    hcb_last_lo = nullptr;
}
