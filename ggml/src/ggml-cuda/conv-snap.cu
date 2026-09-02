#include "conv-snap.cuh"

#include <cstdio>
#include <cstdlib>

bool ggml_cuda_conv_snap_enabled() {
    static const bool v = []() {
        const char * e  = getenv("GGML_JOHNV8_CONV_SNAP");
        const bool   on = (e == nullptr) || (atoi(e) != 0);
        if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
            fprintf(stderr, "[johnv8] conv_snap = %d\n", (int) on);
        }
        return on;
    }();
    return v;
}

struct conv_snap_slots {
    int64_t src_off[CUDA_CONV_SNAP_MAX_SLOTS]; // bajty wzgledem src_base
    int64_t dst_off[CUDA_CONV_SNAP_MAX_SLOTS]; // bajty wzgledem dst_base
};

// src: [state_cols, channels, n_seqs] F32, kroki nb1_src/nb2_src (bajty), ne0 ciagle
// dst: [state_cols*channels, n_seqs] F32, krok nb1_dst (bajty), wiersz ciagly
static __global__ void conv_snap_f32(
        const char * __restrict__ src_base, char * __restrict__ dst_base,
        const conv_snap_slots slots,
        const int state_cols, const int channels,
        const int64_t nb1_src, const int64_t nb2_src, const int64_t nb1_dst) {
    ggml_cuda_pdl_lc();
    const int e = blockIdx.x*blockDim.x + threadIdx.x; // 0..state_cols*channels
    const int n = blockIdx.y;                          // sekwencja
    const int s = blockIdx.z;                          // slot
    if (e >= state_cols*channels) {
        return;
    }
    ggml_cuda_pdl_sync();
    const int c = e / state_cols;
    const int j = e - c*state_cols;
    const float * src = (const float *) (src_base + slots.src_off[s] + (int64_t) j*sizeof(float) + (int64_t) c*nb1_src + (int64_t) n*nb2_src);
    float       * dst = (float *)       (dst_base + slots.dst_off[s] + (int64_t) n*nb1_dst + (int64_t) e*sizeof(float));
    *dst = *src;
}

bool ggml_cuda_conv_snap_ok(const ggml_cgraph * cgraph, int node_idx, int * n_nodes) {
    *n_nodes = 0;
    if (!ggml_cuda_conv_snap_enabled() || node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_CPY) {
        return false;
    }
    const ggml_tensor * src0 = first->src[0];
    const ggml_tensor * dst0 = first->src[1];
    if (!src0 || !dst0 || !src0->view_src || !dst0->view_src) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst0->type != GGML_TYPE_F32) {
        return false;
    }
    // src: 3D ogon [state_cols, channels, n_seqs] z ciaglym ne0; dst: 2D wiersz [state_cols*channels, n_seqs] ciagly w ne0
    if (src0->ne[3] != 1 || dst0->ne[2] != 1 || dst0->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || dst0->nb[0] != sizeof(float)) {
        return false;
    }
    if (dst0->ne[0] != src0->ne[0]*src0->ne[1] || dst0->ne[1] != src0->ne[2]) {
        return false;
    }
    int n = 1;
    while (node_idx + n < cgraph->n_nodes && n < CUDA_CONV_SNAP_MAX_SLOTS) {
        const ggml_tensor * t = cgraph->nodes[node_idx + n];
        if (t->op != GGML_OP_CPY || !t->src[0] || !t->src[1]) break;
        const ggml_tensor * s = t->src[0];
        const ggml_tensor * d = t->src[1];
        if (s->view_src != src0->view_src || d->view_src != dst0->view_src) break;
        if (!ggml_are_same_shape(s, src0) || !ggml_are_same_shape(d, dst0)) break;
        if (s->nb[1] != src0->nb[1] || s->nb[2] != src0->nb[2] || d->nb[1] != dst0->nb[1]) break;
        if (s->type != GGML_TYPE_F32 || d->type != GGML_TYPE_F32) break;
        n++;
    }
    if (n < 2) {
        return false; // jeden slot = zwykly cpy, nic do sklejania
    }
    if (src0->ne[0]*src0->ne[1] > (int64_t) INT32_MAX || src0->ne[2] > 65535) {
        return false;
    }
    *n_nodes = n;
    return true;
}

void ggml_cuda_op_conv_snap(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int n_nodes) {
    const ggml_tensor * src0 = cgraph->nodes[node_idx]->src[0];
    const ggml_tensor * dst0 = cgraph->nodes[node_idx]->src[1];
    const char * src_base = (const char *) src0->view_src->data;
    char       * dst_base = (char *)       dst0->view_src->data;
    conv_snap_slots slots = {};
    for (int s = 0; s < n_nodes; ++s) {
        const ggml_tensor * t = cgraph->nodes[node_idx + s];
        slots.src_off[s] = (const char *) t->src[0]->data - src_base;
        slots.dst_off[s] = (char *) t->src[1]->data - dst_base;
    }
    const int state_cols = (int) src0->ne[0];
    const int channels   = (int) src0->ne[1];
    const int n_seqs     = (int) src0->ne[2];
    if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
        static bool logged = false;
        if (!logged) { logged = true; fprintf(stderr, "[johnv8] conv_snap fused: %d slotow x [%d,%d,%d] (%d wezlow CPY -> 1 jadro)\n", n_nodes, state_cols, channels, n_seqs, n_nodes); }
    }
    const int64_t ne = (int64_t) state_cols*channels;
    const int64_t nbx = (ne + CUDA_CONV_SNAP_BLOCK_SIZE - 1) / CUDA_CONV_SNAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3((unsigned int) nbx, (unsigned int) n_seqs, (unsigned int) n_nodes),
            dim3(CUDA_CONV_SNAP_BLOCK_SIZE, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(conv_snap_f32, launch_params,
            src_base, dst_base, slots, state_cols, channels,
            (int64_t) src0->nb[1], (int64_t) src0->nb[2], (int64_t) dst0->nb[1]);
}
