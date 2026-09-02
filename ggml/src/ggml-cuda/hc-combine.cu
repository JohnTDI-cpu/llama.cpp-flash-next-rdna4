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
