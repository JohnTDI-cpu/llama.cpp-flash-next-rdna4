#include "hc-combine.cuh"

#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>

// [johnv8] Fuzja lancucha qwen4exp build_hc_combine (src/models/qwen4exp.cpp) w jedno jadro.
//
// Stock generuje 6 jader na wywolanie (2 x na warstwe, 48 warstw = 576 uruchomien na token),
// z czego dwa licza na hc=4 liczbach, a REPEAT materializuje n_embd*hc floatow tylko po to,
// zeby rozmnozyc wektor.
//
// Bit-identycznosc: jadro powtarza *dokladnie* te same operacje f32 w tej samej kolejnosci co
// sciezka stockowa (scale.cu: dst = scale*x + bias, unary.cu op_sigmoid: 1/(1+expf(-x)),
// binbcast MUL, binbcast ADD). Kontrakcja mnozenia z dodawaniem jest wylaczona, bo stock liczy
// (residual + round(b*w)) w dwoch osobnych jadrach i fma zmienialoby ostatnie zaokraglenie.

bool ggml_cuda_hc_fuse_enabled() {
    static const bool v = []() {
        const char * e  = getenv("GGML_JOHNV8_HC_FUSE");
        const bool   on = (e == nullptr) || (atoi(e) != 0);
        if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
            fprintf(stderr, "[johnv8] hc_fuse = %d\n", (int) on);
        }
        return on;
    }();
    return v;
}

static __global__ void hc_combine_f32(
        const float * __restrict__ residual,
        const float * __restrict__ block_out,
        const float * __restrict__ inject,
        float       * __restrict__ dst,
        const float                scale_in,
        const float                bias_in,
        const float                scale_w,
        const float                bias_w,
        const int                  n_embd,
        const int                  hc) {
#if defined(__clang__)
    // bez tego kompilator zwinalby `residual + b*w` w jedno fma i wynik przestalby byc
    // bit-identyczny ze sciezka stockowa (osobne jadra MUL i ADD)
#pragma clang fp contract(off)
#endif
    ggml_cuda_pdl_lc();

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    const int t = blockIdx.z;

    if (i >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    // GGML_OP_SCALE -> GGML_UNARY_OP_SIGMOID -> GGML_OP_SCALE, wszystko na [hc, nt]
    float in = inject[(int64_t) c + (int64_t) hc*t];
#if defined(__AMDGCN__)
    // adres inject nie zalezy od threadIdx, wiec kompilator scalaryzuje cala sigmoide na SALU
    // (s_rndne_f32 ... v_s_exp_f32). Sekwencja arytmetyczna jest ta sama co w stocku, ale
    // skalarny i wektorowy exp to osobne jednostki i nikt nie gwarantuje identycznych bitow.
    // Przypiecie do VGPR wymusza dokladnie te sciezke, ktora liczy unary.cu (v_exp_f32).
    __asm__("" : "+v"(in));
#endif
    const float s  = scale_in * in + bias_in;
    const float sg = 1.0f / (1.0f + expf(-s));
    const float w  = scale_w * sg + bias_w;

    // GGML_OP_RESHAPE + GGML_OP_REPEAT: block_out[i, t] rozmnozony po hc
    const int64_t ib = (int64_t) i + (int64_t) n_embd*t;
    const int64_t id = (int64_t) i + (int64_t) n_embd*((int64_t) c + (int64_t) hc*t);

    // GGML_OP_MUL, potem GGML_OP_ADD - dwa osobne zaokraglenia, jak w stocku
    const float m = block_out[ib] * w;

    dst[id] = residual[id] + m;
}

// czy dwa tensory dziela chocby bajt pamieci
static bool ggml_cuda_hc_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a->buffer || !b->buffer) {
        return true; // nie wiemy - zakladamy najgorsze
    }

    const char * a_start = (const char *) a->data;
    const char * a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

    const char * b_start = (const char *) b->data;
    const char * b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

    return (b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end);
}

bool ggml_cuda_hc_combine_ok(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_hc_fuse_enabled()) {
        return false;
    }

    if (node_idx < 0 || node_idx + 7 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * scale_in = cgraph->nodes[node_idx + 0];
    const ggml_tensor * sigmoid  = cgraph->nodes[node_idx + 1];
    const ggml_tensor * scale_w  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * rs_w     = cgraph->nodes[node_idx + 3];
    const ggml_tensor * rs_b     = cgraph->nodes[node_idx + 4];
    const ggml_tensor * repeat   = cgraph->nodes[node_idx + 5];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 6];
    const ggml_tensor * add      = cgraph->nodes[node_idx + 7];

    if (scale_in->op != GGML_OP_SCALE   || sigmoid->op != GGML_OP_UNARY  ||
        scale_w->op  != GGML_OP_SCALE   || rs_w->op    != GGML_OP_RESHAPE ||
        rs_b->op     != GGML_OP_RESHAPE || repeat->op  != GGML_OP_REPEAT  ||
        mul->op      != GGML_OP_MUL     || add->op     != GGML_OP_ADD) {
        return false;
    }

    if (ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }

    // krawedzie: dokladnie ten ksztalt lancucha, ktory buduje build_hc_combine
    if (sigmoid->src[0] != scale_in || scale_w->src[0] != sigmoid || rs_w->src[0] != scale_w) {
        return false;
    }
    if (repeat->src[0] != rs_b) {
        return false;
    }
    if (mul->src[0] != repeat || mul->src[1] != rs_w) {
        return false;
    }
    // residual musi byc src0 dodawania, a wynik mnozenia src1
    if (add->src[1] != mul || add->src[0] == mul) {
        return false;
    }

    const ggml_tensor * inject    = scale_in->src[0];
    const ggml_tensor * block_out = rs_b->src[0];
    const ggml_tensor * residual  = add->src[0];

    if (!inject || !block_out || !residual) {
        return false;
    }

    // wszystko f32
    const ggml_tensor * f32_nodes[] = { scale_in, sigmoid, scale_w, rs_w, rs_b, repeat, mul, add,
                                        inject, block_out, residual };
    for (const ggml_tensor * t : f32_nodes) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }

    const int64_t n_embd = add->ne[0];
    const int64_t hc     = add->ne[1];
    const int64_t nt     = add->ne[2];

    if (n_embd <= 0 || hc <= 0 || nt <= 0 || add->ne[3] != 1) {
        return false;
    }

    // siatka: y po hc, z po nt
    if (hc > 65535 || nt > 65535) {
        return false;
    }

    // ksztalty
    if (!ggml_are_same_shape(add, residual) || !ggml_are_same_shape(add, mul) ||
        !ggml_are_same_shape(mul, repeat)) {
        return false;
    }
    if (inject->ne[0] != hc || inject->ne[1] != nt || inject->ne[2] != 1 || inject->ne[3] != 1) {
        return false;
    }
    if (!ggml_are_same_shape(scale_in, inject) || !ggml_are_same_shape(sigmoid, inject) ||
        !ggml_are_same_shape(scale_w, inject)) {
        return false;
    }
    if (rs_w->ne[0] != 1 || rs_w->ne[1] != hc || rs_w->ne[2] != nt || rs_w->ne[3] != 1) {
        return false;
    }
    if (block_out->ne[0] != n_embd || block_out->ne[1] != nt ||
        block_out->ne[2] != 1      || block_out->ne[3] != 1) {
        return false;
    }
    if (rs_b->ne[0] != n_embd || rs_b->ne[1] != 1 || rs_b->ne[2] != nt || rs_b->ne[3] != 1) {
        return false;
    }

    // jadro indeksuje plaskie bufory
    if (!ggml_is_contiguous(inject) || !ggml_is_contiguous(block_out) ||
        !ggml_is_contiguous(residual) || !ggml_is_contiguous(add)) {
        return false;
    }

    // parametry scale musza byc dokladnie te, ktore sklada build_hc_combine
    if (ggml_get_op_params_f32(scale_in, 1) != 0.0f || ggml_get_op_params_f32(scale_w, 1) != 0.0f) {
        return false;
    }
    if (ggml_get_op_params_f32(scale_in, 0) != 1.0f / (float) hc) {
        return false;
    }
    if (ggml_get_op_params_f32(scale_w, 0) != 2.0f) {
        return false;
    }

    // aliasowanie: kazdy element block_out i inject czyta hc (badz n_embd) watkow, wiec
    // nadpisywanie ich wyjsciem byloby wyscigiem. residual jest czytany pod tym samym
    // indeksem co zapis, wiec dokladny alias (add w miejscu) jest bezpieczny.
    if (ggml_cuda_hc_overlap(add, block_out) || ggml_cuda_hc_overlap(add, inject)) {
        return false;
    }
    if (add->data != residual->data && ggml_cuda_hc_overlap(add, residual)) {
        return false;
    }

    return true;
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * scale_in = cgraph->nodes[node_idx + 0];
    const ggml_tensor * scale_w  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * rs_b     = cgraph->nodes[node_idx + 4];
    const ggml_tensor * add      = cgraph->nodes[node_idx + 7];

    const ggml_tensor * inject    = scale_in->src[0];
    const ggml_tensor * block_out = rs_b->src[0];
    const ggml_tensor * residual  = add->src[0];

    const int64_t n_embd = add->ne[0];
    const int64_t hc     = add->ne[1];
    const int64_t nt     = add->ne[2];

    GGML_ASSERT(add->type == GGML_TYPE_F32);

    const float s_in = ggml_get_op_params_f32(scale_in, 0);
    const float b_in = ggml_get_op_params_f32(scale_in, 1);
    const float s_w  = ggml_get_op_params_f32(scale_w,  0);
    const float b_w  = ggml_get_op_params_f32(scale_w,  1);

    if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[johnv8] hc_combine fused: n_embd=%d hc=%d nt=%d (8 wezlow -> 1 jadro)\n",
                    (int) n_embd, (int) hc, (int) nt);
        }
    }

    const int64_t num_blocks_x = (n_embd + CUDA_HC_COMBINE_BLOCK_SIZE - 1) / CUDA_HC_COMBINE_BLOCK_SIZE;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3((unsigned int) num_blocks_x, (unsigned int) hc, (unsigned int) nt),
            dim3(CUDA_HC_COMBINE_BLOCK_SIZE, 1, 1),
            (size_t) 0,
            ctx.stream());

    ggml_cuda_kernel_launch(hc_combine_f32, launch_params,
            (const float *) residual->data,
            (const float *) block_out->data,
            (const float *) inject->data,
            (float *) add->data,
            s_in, b_in, s_w, b_w,
            (int) n_embd, (int) hc);
}

// ============================================================================
// [johnv8] E10a: hc_inject wciagniety do combine
// ============================================================================
#include "mmvf.cuh"

static bool ggml_cuda_inject_fuse_enabled() {
    static const bool v = []() { const char * e = getenv("GGML_JOHNV8_INJECT_FUSE"); return e == nullptr || atoi(e) != 0; }();
    return v;
}
static int ggml_cuda_inject_use_fma() {
    static const int v = []() { const char * e = getenv("GGML_JOHNV8_INJECT_FMA"); return e == nullptr ? 1 : atoi(e); }();
    return v;
}

template <int block_size>
static __global__ void hc_combine_inject_f32(
        const float * __restrict__ W,          // [ncols, hc] f32, wiersz c = W + c*ncols
        const float * __restrict__ hnorm,      // [ncols, nt] f32
        const float * __restrict__ residual,   // [n_embd, hc, nt]
        const float * __restrict__ block_out,  // [n_embd, nt]
        float       * __restrict__ dst,        // [n_embd, hc, nt]
        const float                scale_in,
        const float                bias_in,
        const float                scale_w,
        const float                bias_w,
        const int                  n_embd,
        const int                  hc,
        const int                  ncols,
        const int                  use_fma) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int c   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float buf_iw[warp_size];
    __shared__ float s_val;
    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }
    // --- iloczyn skalarny jak mul_mat_vec_f<float, float, ncols_dst, block_size> dla wiersza c, kolumny t
    const float2 * x2 = (const float2 *) (W     + (int64_t) c * ncols);
    const float2 * y2 = (const float2 *) (hnorm + (int64_t) t * ncols);
    const int ncols2 = ncols / 2;
    float sumf = 0.0f;
    if (use_fma) {
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tx = x2[col2];
            const float2 ty = y2[col2];
            sumf = fmaf(tx.x, ty.x, sumf);
            sumf = fmaf(tx.y, ty.y, sumf);
        }
    } else {
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tx = x2[col2];
            const float2 ty = y2[col2];
            sumf += tx.x * ty.x;
            sumf += tx.y * ty.y;
        }
    }
    sumf = warp_reduce_sum<warp_size>(sumf);
    if (block_size > warp_size) {
        buf_iw[tid / warp_size] = sumf;
        __syncthreads();
        if (tid < warp_size) {
            sumf = buf_iw[tid];
            sumf = warp_reduce_sum<warp_size>(sumf);
        }
    }
    if (tid == 0) {
        s_val = sumf;
    }
    __syncthreads();
    // --- reszta jak hc_combine_f32
    float in = s_val;
#if defined(__AMDGCN__)
    __asm__("" : "+v"(in));
#endif
    const float s  = scale_in * in + bias_in;
    const float sg = 1.0f / (1.0f + expf(-s));
    const float w  = scale_w * sg + bias_w;
    for (int i = tid; i < n_embd; i += block_size) {
        const int64_t ib = (int64_t) i + (int64_t) n_embd*t;
        const int64_t id = (int64_t) i + (int64_t) n_embd*((int64_t) c + (int64_t) hc*t);
        const float m = block_out[ib] * w;
        dst[id] = residual[id] + m;
    }
}

bool ggml_cuda_hc_combine_inject_ok(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_inject_fuse_enabled()) {
        return false;
    }
    if (node_idx < 0 || node_idx + 8 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * mm = cgraph->nodes[node_idx];
    if (mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_cuda_hc_combine_ok(cgraph, node_idx + 1)) {
        return false;
    }
    const ggml_tensor * scale_in = cgraph->nodes[node_idx + 1];
    if (scale_in->src[0] != mm) {
        return false;
    }
    const ggml_tensor * W  = mm->src[0];
    const ggml_tensor * hn = mm->src[1];
    if (!W || !hn || W->type != GGML_TYPE_F32 || hn->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[node_idx + 8];
    const int64_t n_embd = add->ne[0];
    const int64_t hc     = add->ne[1];
    const int64_t nt     = add->ne[2];
    const int64_t ncols  = n_embd * hc;
    if (W->ne[0] != ncols || W->ne[1] != hc || W->ne[2] != 1 || W->ne[3] != 1) {
        return false;
    }
    if (hn->ne[0] != ncols || hn->ne[1] != nt || hn->ne[2] != 1 || hn->ne[3] != 1) {
        return false;
    }
    if (mm->ne[0] != hc || mm->ne[1] != nt || mm->ne[2] != 1 || mm->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(W) || !ggml_is_contiguous(hn) || !ggml_is_contiguous(mm)) {
        return false;
    }
    if (ncols % 2 != 0 || ncols > INT32_MAX || n_embd > INT32_MAX) {
        return false;
    }
    // stock liczy ten matmul przez mul_mat_vec_f (nt <= 8 na AMD) - replikujemy dokladnie te sciezke
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_should_use_mmvf(GGML_TYPE_F32, cc, W->ne, W->nb, nt)) {
        return false;
    }
    // wynik MUL_MAT i posrednie wezly nie moga byc uzywane poza lancuchem
    const enum ggml_op ops[9] = { GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_RESHAPE,
                                  cgraph->nodes[node_idx + 5]->op, GGML_OP_REPEAT, GGML_OP_MUL, GGML_OP_ADD };
    const int32_t out_idx = node_idx + 8;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, 9, ops, &out_idx, 1)) {
        return false;
    }
    if (ggml_cuda_hc_overlap(add, W) || ggml_cuda_hc_overlap(add, hn)) {
        return false;
    }
    return true;
}

void ggml_cuda_op_hc_combine_inject(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * mm       = cgraph->nodes[node_idx + 0];
    const ggml_tensor * scale_in = cgraph->nodes[node_idx + 1];
    const ggml_tensor * scale_w  = cgraph->nodes[node_idx + 3];
    const ggml_tensor * rs_b     = cgraph->nodes[node_idx + 5];
    const ggml_tensor * add      = cgraph->nodes[node_idx + 8];
    const ggml_tensor * W         = mm->src[0];
    const ggml_tensor * hn        = mm->src[1];
    const ggml_tensor * block_out = rs_b->src[0];
    const ggml_tensor * residual  = add->src[0];

    const int64_t n_embd = add->ne[0];
    const int64_t hc     = add->ne[1];
    const int64_t nt     = add->ne[2];
    const int64_t ncols  = n_embd * hc;

    const float s_in = ggml_get_op_params_f32(scale_in, 0);
    const float b_in = ggml_get_op_params_f32(scale_in, 1);
    const float s_w  = ggml_get_op_params_f32(scale_w,  0);
    const float b_w  = ggml_get_op_params_f32(scale_w,  1);

    // block_size dokladnie jak mul_mat_vec_f_cuda
    const int device    = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const int cc        = ggml_cuda_info().devices[device].cc;
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    static bool logged = false;
    if (!logged && getenv("GGML_CUDA_DUMP_DISPATCH")) {
        logged = true;
        fprintf(stderr, "[johnv8] hc_combine_inject fused: ncols=%d hc=%d nt=%d block=%d fma=%d (9 wezlow -> 1 jadro)\n",
                (int) ncols, (int) hc, (int) nt, (int) block_size_best, ggml_cuda_inject_use_fma());
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3((unsigned int) hc, (unsigned int) nt, 1), dim3((unsigned int) block_size_best, 1, 1), 0, ctx.stream());
    const float * W_d  = (const float *) W->data;
    const float * hn_d = (const float *) hn->data;
    const float * r_d  = (const float *) residual->data;
    const float * b_d  = (const float *) block_out->data;
    float       * d_d  = (float *) add->data;
    const int fma = ggml_cuda_inject_use_fma();
#define JOHNV8_INJECT_LAUNCH(BS) \
        ggml_cuda_kernel_launch(hc_combine_inject_f32<BS>, launch_params, W_d, hn_d, r_d, b_d, d_d, s_in, b_in, s_w, b_w, \
                                (int) n_embd, (int) hc, (int) ncols, fma)
    switch (block_size_best) {
        case  32: JOHNV8_INJECT_LAUNCH(32);  break;
        case  64: JOHNV8_INJECT_LAUNCH(64);  break;
        case  96: JOHNV8_INJECT_LAUNCH(96);  break;
        case 128: JOHNV8_INJECT_LAUNCH(128); break;
        case 160: JOHNV8_INJECT_LAUNCH(160); break;
        case 192: JOHNV8_INJECT_LAUNCH(192); break;
        case 224: JOHNV8_INJECT_LAUNCH(224); break;
        case 256: JOHNV8_INJECT_LAUNCH(256); break;
        default: GGML_ABORT("hc_combine_inject: nieobslugiwany block_size %d", (int) block_size_best);
    }
#undef JOHNV8_INJECT_LAUNCH
}
