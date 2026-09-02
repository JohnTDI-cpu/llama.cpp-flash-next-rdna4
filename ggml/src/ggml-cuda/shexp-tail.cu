#include "shexp-tail.cuh"

#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>

// [johnv8] E6b: sigmoid(gate[t]) * shexp[i,t] + moe_out[i,t] w jednym jadrze.
// Bit-identycznosc ze stockiem: sigmoid jak unary.cu (1/(1+expf(-x))), MUL i ADD jak binbcast,
// kontrakcja fma wylaczona (stock liczy mul i add w osobnych jadrach z zaokragleniem miedzy nimi).

bool ggml_cuda_shexp_fuse_enabled() {
    static const bool v = []() {
        const char * e  = getenv("GGML_JOHNV8_SHEXP_FUSE");
        const bool   on = (e == nullptr) || (atoi(e) != 0);   // domyslnie ON (bit-exact po zamianie __fmul_rn/__fadd_rn na zwykle operatory)
        if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
            fprintf(stderr, "[johnv8] shexp_fuse = %d\n", (int) on);
        }
        return on;
    }();
    return v;
}

static __global__ void shexp_tail_f32(
        const float * __restrict__ moe_out,
        const float * __restrict__ shexp,
        const float * __restrict__ gate,
        float       * __restrict__ dst,
        const int                  n_embd) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    ggml_cuda_pdl_lc();

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;

    if (i >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    float g = gate[t];
#if defined(__AMDGCN__)
    __asm__("" : "+v"(g));
#endif
    const float sg = 1.0f / (1.0f + expf(-g));
    const int64_t id = (int64_t) i + (int64_t) n_embd*t;
    const float m = shexp[id] * sg;      // jak binbcast MUL (pragma contract(off) w tym jadrze; bez __fmul_rn!)
    dst[id] = moe_out[id] + m;           // jak binbcast ADD
}

static bool ggml_cuda_shexp_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a->buffer || !b->buffer) {
        return true;
    }
    const char * a_start = (const char *) a->data;
    const char * a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const char * b_start = (const char *) b->data;
    const char * b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return (b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end);
}

bool ggml_cuda_shexp_tail_ok(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_shexp_fuse_enabled()) {
        return false;
    }
    if (node_idx < 0 || node_idx + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * sigmoid = cgraph->nodes[node_idx + 0];
    const ggml_tensor * mul     = cgraph->nodes[node_idx + 1];
    const ggml_tensor * add     = cgraph->nodes[node_idx + 2];

    if (sigmoid->op != GGML_OP_UNARY || mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD) {
        return false;
    }
    if (ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    // mul = shexp * sigmoid ; add = moe_out + mul   (kolejnosc jak w qwen4exp.cpp build_layer_ffn)
    if (mul->src[1] != sigmoid || mul->src[0] == sigmoid) {
        return false;
    }
    if (add->src[1] != mul || add->src[0] == mul) {
        return false;
    }
    const ggml_tensor * gate    = sigmoid->src[0];
    const ggml_tensor * shexp   = mul->src[0];
    const ggml_tensor * moe_out = add->src[0];
    if (!gate || !shexp || !moe_out) {
        return false;
    }
    const ggml_tensor * f32_nodes[] = { sigmoid, mul, add, gate, shexp, moe_out };
    for (const ggml_tensor * t : f32_nodes) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const int64_t n_embd = add->ne[0];
    const int64_t nt     = add->ne[1];
    if (n_embd <= 0 || nt <= 0 || add->ne[2] != 1 || add->ne[3] != 1 || nt > 65535) {
        return false;
    }
    if (!ggml_are_same_shape(add, moe_out) || !ggml_are_same_shape(add, mul) || !ggml_are_same_shape(mul, shexp)) {
        return false;
    }
    // bramka: jedna liczba na token
    if (gate->ne[0] != 1 || gate->ne[1] != nt || gate->ne[2] != 1 || gate->ne[3] != 1) {
        return false;
    }
    if (!ggml_are_same_shape(sigmoid, gate)) {
        return false;
    }
    if (!ggml_is_contiguous(gate) || !ggml_is_contiguous(shexp) || !ggml_is_contiguous(moe_out) || !ggml_is_contiguous(add)) {
        return false;
    }
    if (ggml_cuda_shexp_overlap(add, shexp) || ggml_cuda_shexp_overlap(add, gate)) {
        return false;
    }
    if (add->data != moe_out->data && ggml_cuda_shexp_overlap(add, moe_out)) {
        return false;
    }
    return true;
}

void ggml_cuda_op_shexp_tail(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * sigmoid = cgraph->nodes[node_idx + 0];
    const ggml_tensor * mul     = cgraph->nodes[node_idx + 1];
    const ggml_tensor * add     = cgraph->nodes[node_idx + 2];
    const ggml_tensor * gate    = sigmoid->src[0];
    const ggml_tensor * shexp   = mul->src[0];
    const ggml_tensor * moe_out = add->src[0];

    const int64_t n_embd = add->ne[0];
    const int64_t nt     = add->ne[1];
    GGML_ASSERT(add->type == GGML_TYPE_F32);

    if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[johnv8] shexp_tail fused: n_embd=%d nt=%d (3 wezly -> 1 jadro)\n", (int) n_embd, (int) nt);
        }
    }

    const int64_t num_blocks_x = (n_embd + CUDA_SHEXP_TAIL_BLOCK_SIZE - 1) / CUDA_SHEXP_TAIL_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3((unsigned int) num_blocks_x, (unsigned int) nt, 1),
            dim3(CUDA_SHEXP_TAIL_BLOCK_SIZE, 1, 1),
            0, ctx.stream());
    ggml_cuda_kernel_launch(shexp_tail_f32, launch_params,
            (const float *) moe_out->data,
            (const float *) shexp->data,
            (const float *) gate->data,
            (float *) add->data,
            (int) n_embd);
}
