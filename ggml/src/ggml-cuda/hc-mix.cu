#include "hc-mix.cuh"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "scale.cuh"

#include <cstdio>
#include <cstdlib>

// [johnv8] Fuzja lancuchow qwen4exp build_hc_mix (src/models/qwen4exp.cpp), ta sama recepta
// co w hc-combine.cu. build_hc_mix jest wywolywana 97 razy na token (2 na warstwe x 48 + glowa),
// wiec kazdy skasowany dispatch to ~97 uruchomien jadra mniej.
//
// Bit-identycznosc: jadra powtarzaja *dokladnie* te same operacje f32 w tej samej kolejnosci co
// sciezka stockowa (unary.cu op_sigmoid: 1/(1+expf(-x)), unary.cuh op_silu: x/(1+expf(-x)),
// binbcast MUL, binbcast ADD, scale.cu: dst = scale*x + bias). Kontrakcja mnozenia z dodawaniem
// jest wylaczona, bo stock liczy kolejne zaokraglenia w osobnych jadrach.

bool ggml_cuda_mix_fuse_enabled() {
    static const bool v = []() {
        const char * e  = getenv("GGML_JOHNV8_MIX_FUSE");
        const bool   on = (e == nullptr) || (atoi(e) != 0);
        if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
            fprintf(stderr, "[johnv8] mix_fuse = %d\n", (int) on);
        }
        return on;
    }();
    return v;
}

// czy dwa tensory dziela chocby bajt pamieci
static bool ggml_cuda_mix_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a->buffer || !b->buffer) {
        return true; // nie wiemy - zakladamy najgorsze
    }

    const char * a_start = (const char *) a->data;
    const char * a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

    const char * b_start = (const char *) b->data;
    const char * b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

    return (b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end);
}

// ---------------------------------------------------------------------------------------------
// 1) kolaps strumieni hyper-connections
// ---------------------------------------------------------------------------------------------

static __global__ void hc_mix_collapse_f32(
        const float * __restrict__ xn,
        const float * __restrict__ up,
        float       * __restrict__ dst,
        const float                scale,
        const float                bias,
        const int                  n_embd,
        const int                  hc) {
#if defined(__clang__)
    // bez tego kompilator zwinalby `acc + xn*sigmoid(up)` w jedno fma i wynik przestalby byc
    // bit-identyczny ze sciezka stockowa (osobne jadra MUL i ADD, kazde z wlasnym zaokragleniem)
#pragma clang fp contract(off)
#endif
    ggml_cuda_pdl_lc();

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;

    if (i >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    // xn i up sa ciagle jako [n_embd*hc, nt]; widok strumienia c to offset n_embd*c
    const int64_t base = (int64_t) i + (int64_t) n_embd*(int64_t) hc*(int64_t) t;

    // GGML_UNARY_OP_SIGMOID + GGML_OP_MUL dla strumienia 0
    float g0 = up[base];
#if defined(__AMDGCN__)
    // pilnujemy, zeby exp poszedl przez wektorowe v_exp_f32 (tak jak unary.cu), a nie zostal
    // przez kompilator zeskalaryzowany na v_s_exp_f32 - to osobna jednostka i osobne bity
    __asm__("" : "+v"(g0));
#endif
    float acc = xn[base] * (1.0f / (1.0f + expf(-g0)));

    // lancuch GGML_OP_ADD: ((m0 + m1) + m2) + m3 - kolejnosc musi byc dokladnie taka jak
    // w lancuchu ggml_add, zadnego sumowania "drzewkiem"
    for (int c = 1; c < hc; ++c) {
        const int64_t idx = base + (int64_t) n_embd*(int64_t) c;

        float g = up[idx];
#if defined(__AMDGCN__)
        __asm__("" : "+v"(g));
#endif
        const float m = xn[idx] * (1.0f / (1.0f + expf(-g)));

        acc = acc + m;
    }

    // GGML_OP_SCALE
    dst[(int64_t) i + (int64_t) n_embd*(int64_t) t] = scale * acc + bias;
}

// wyciaga geometrie podgrafu; zwraca false gdy cokolwiek sie nie zgadza
static bool hc_mix_collapse_match(const ggml_cgraph * cgraph, int node_idx, int * n_nodes_out,
                                  const ggml_tensor ** xn_out, const ggml_tensor ** up_out) {
    // minimalny prefiks: UNARY, MUL, RESHAPE
    if (node_idx < 0 || node_idx + 2 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * sigmoid = cgraph->nodes[node_idx + 0];
    const ggml_tensor * mul     = cgraph->nodes[node_idx + 1];
    const ggml_tensor * rs      = cgraph->nodes[node_idx + 2];

    if (sigmoid->op != GGML_OP_UNARY || mul->op != GGML_OP_MUL) {
        return false;
    }
    // reshape gated bywa RESHAPE albo VIEW, zaleznie od tego czy zrodlo bylo juz widokiem
    if (rs->op != GGML_OP_RESHAPE && rs->op != GGML_OP_VIEW) {
        return false;
    }
    if (ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }

    // MUL musi czytac wyjscie sigmoidy; drugi operand to xn
    const ggml_tensor * up = sigmoid->src[0];
    const ggml_tensor * xn = nullptr;
    if (mul->src[0] == sigmoid) {
        xn = mul->src[1];
    } else if (mul->src[1] == sigmoid) {
        xn = mul->src[0];
    } else {
        return false;
    }
    if (!up || !xn) {
        return false;
    }

    if (rs->src[0] != mul) {
        return false;
    }

    const int64_t n_embd = rs->ne[0];
    const int64_t hc     = rs->ne[1];
    const int64_t nt     = rs->ne[2];

    if (n_embd <= 0 || nt <= 0 || rs->ne[3] != 1) {
        return false;
    }
    // hc widokow + (hc-1) dodawan - trzymamy sie zakresu, ktory buduje build_hc_mix
    if (hc < 2 || hc > 8) {
        return false;
    }

    // siatka: y po nt
    if (nt > 65535) {
        return false;
    }

    const int n_nodes = (int) (2*hc + 3);
    if (node_idx + n_nodes > cgraph->n_nodes) {
        return false;
    }

    const int idx_view = node_idx + 3;
    const int idx_add  = idx_view + (int) hc;
    const int idx_scl  = idx_add + (int) hc - 1;

    const ggml_tensor * scl = cgraph->nodes[idx_scl];

    // ksztalty i typy calego lancucha
    if (sigmoid->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        up->type      != GGML_TYPE_F32 || xn->type  != GGML_TYPE_F32 ||
        scl->type     != GGML_TYPE_F32) {
        return false;
    }

    const int64_t hc_dim = n_embd*hc;

    // sigmoid, mul, xn, up: wszystko [hc_dim, nt] i ciagle - jadro indeksuje plaskie bufory
    const ggml_tensor * flat[] = { sigmoid, mul, up, xn };
    for (const ggml_tensor * f : flat) {
        if (f->ne[0] != hc_dim || f->ne[1] != nt || f->ne[2] != 1 || f->ne[3] != 1) {
            return false;
        }
        if (!ggml_is_contiguous(f)) {
            return false;
        }
    }

    // hc widokow strumieni: [n_embd, nt], krok wiersza n_embd*hc, offset n_embd*c
    const size_t esz = sizeof(float);
    for (int64_t c = 0; c < hc; ++c) {
        const ggml_tensor * v = cgraph->nodes[idx_view + (int) c];

        if (v->op != GGML_OP_VIEW || v->src[0] != rs || v->type != GGML_TYPE_F32) {
            return false;
        }
        if (v->ne[0] != n_embd || v->ne[1] != nt || v->ne[2] != 1 || v->ne[3] != 1) {
            return false;
        }
        if (v->nb[0] != esz || v->nb[1] != (size_t) hc_dim*esz) {
            return false;
        }
        // widok musi lezec dokladnie tam, gdzie zaklada jadro
        if ((const char *) v->data != (const char *) mul->data + (size_t) (n_embd*c)*esz) {
            return false;
        }
    }

    // lancuch dodawan: add[0] = v0 + v1, add[k] = add[k-1] + v[k+1]
    for (int64_t k = 0; k + 1 < hc; ++k) {
        const ggml_tensor * a = cgraph->nodes[idx_add + (int) k];

        if (a->op != GGML_OP_ADD || a->type != GGML_TYPE_F32) {
            return false;
        }
        if (a->ne[0] != n_embd || a->ne[1] != nt || a->ne[2] != 1 || a->ne[3] != 1) {
            return false;
        }

        const ggml_tensor * lhs = (k == 0) ? cgraph->nodes[idx_view] : cgraph->nodes[idx_add + (int) k - 1];
        const ggml_tensor * rhs = cgraph->nodes[idx_view + (int) k + 1];

        if (a->src[0] != lhs || a->src[1] != rhs) {
            return false;
        }
    }

    // domykajacy SCALE
    if (scl->op != GGML_OP_SCALE || scl->src[0] != cgraph->nodes[idx_scl - 1]) {
        return false;
    }
    if (scl->ne[0] != n_embd || scl->ne[1] != nt || scl->ne[2] != 1 || scl->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(scl)) {
        return false;
    }
    // 1/hc, bez biasu. bias == 0 jest istotny: stock liczy scale*x+bias jednym v_fma_f32,
    // a nasze jadro ma wylaczona kontrakcje - przy zerowym biasie oba daja te same bity.
    if (ggml_get_op_params_f32(scl, 0) != 1.0f / (float) hc) {
        return false;
    }
    if (ggml_get_op_params_f32(scl, 1) != 0.0f) {
        return false;
    }

    // aliasowanie: kazdy element xn i up jest czytany raz, ale pod INNYM indeksem niz zapis
    // (zwijamy hc strumieni w jeden), wiec jakiekolwiek nachodzenie wyjscia na wejscia
    // byloby wyscigiem miedzy blokami
    if (ggml_cuda_mix_overlap(scl, xn) || ggml_cuda_mix_overlap(scl, up)) {
        return false;
    }

    *n_nodes_out = n_nodes;
    *xn_out      = xn;
    *up_out      = up;

    return true;
}

bool ggml_cuda_hc_mix_collapse_ok(const ggml_cgraph * cgraph, int node_idx, int * n_nodes) {
    if (!ggml_cuda_mix_fuse_enabled()) {
        return false;
    }

    int                 n  = 0;
    const ggml_tensor * xn = nullptr;
    const ggml_tensor * up = nullptr;

    if (!hc_mix_collapse_match(cgraph, node_idx, &n, &xn, &up)) {
        return false;
    }

    // Sprawdzenie domkniecia podgrafu (liczba uzyc, flagi, lancuch view_src) robimy tym samym
    // kodem co reszta backendu. Liste opow bierzemy z samych wezlow - dopasowanie opow zrobil
    // juz hc_mix_collapse_match, a tutaj chodzi wylacznie o to, czy wyniki posrednie nie
    // wyciekaja poza podgraf. Wyjsciem jest domykajacy SCALE.
    enum ggml_op ops[32];
    int          idxs[32];

    GGML_ASSERT(n <= 32);

    for (int k = 0; k < n; ++k) {
        idxs[k] = node_idx + k;
        ops[k]  = cgraph->nodes[node_idx + k]->op;
    }

    const int outputs[] = { node_idx + n - 1 };

    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, n, ops, outputs, 1)) {
        return false;
    }

    // UWAGA: bez tego przypisania wywolujacy dostawal n_nodes == 0 i liczyl
    // "return n_nodes - 1" = -1, czyli petla ewaluacji cofala sie o wezel i w kolko
    // uruchamiala to samo jadro. Zapisujemy dopiero na sciezce sukcesu.
    *n_nodes = n;

    return true;
}

void ggml_cuda_op_hc_mix_collapse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    int                 n  = 0;
    const ggml_tensor * xn = nullptr;
    const ggml_tensor * up = nullptr;

    const bool ok = hc_mix_collapse_match(cgraph, node_idx, &n, &xn, &up);
    GGML_ASSERT(ok);

    const ggml_tensor * rs  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * scl = cgraph->nodes[node_idx + n - 1];

    const int64_t n_embd = rs->ne[0];
    const int64_t hc     = rs->ne[1];
    const int64_t nt     = rs->ne[2];

    const float scale = ggml_get_op_params_f32(scl, 0);
    const float bias  = ggml_get_op_params_f32(scl, 1);

    if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[johnv8] hc_mix_collapse fused: n_embd=%d hc=%d nt=%d (%d wezlow -> 1 jadro)\n",
                    (int) n_embd, (int) hc, (int) nt, n);
        }
    }

    const int64_t num_blocks_x = (n_embd + CUDA_HC_MIX_BLOCK_SIZE - 1) / CUDA_HC_MIX_BLOCK_SIZE;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3((unsigned int) num_blocks_x, (unsigned int) nt, 1),
            dim3(CUDA_HC_MIX_BLOCK_SIZE, 1, 1),
            (size_t) 0,
            ctx.stream());

    ggml_cuda_kernel_launch(hc_mix_collapse_f32, launch_params,
            (const float *) xn->data,
            (const float *) up->data,
            (float *) scl->data,
            scale, bias,
            (int) n_embd, (int) hc);
}

// ---------------------------------------------------------------------------------------------
// 2) SCALE + UNARY(SILU)
// ---------------------------------------------------------------------------------------------

static __global__ void scale_silu_f32(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const float                scale,
        const float                bias,
        const int64_t              nelements) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    ggml_cuda_pdl_lc();

    const int64_t tid    = (int64_t) blockIdx.x*(int64_t) blockDim.x + (int64_t) threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x*(int64_t) gridDim.x;

    ggml_cuda_pdl_sync();

    for (int64_t i = tid; i < nelements; i += stride) {
        // GGML_OP_SCALE (bias == 0, wiec fma i mul+add daja te same bity)
        float s = scale * x[i] + bias;
#if defined(__AMDGCN__)
        // jak wyzej: wymuszamy wektorowy v_exp_f32
        __asm__("" : "+v"(s));
#endif
        // GGML_UNARY_OP_SILU, unary.cuh ggml_cuda_op_silu_single
        dst[i] = s / (1.0f + expf(-s));
    }
}

bool ggml_cuda_scale_silu_ok(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_mix_fuse_enabled()) {
        return false;
    }

    if (node_idx < 0 || node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * scl  = cgraph->nodes[node_idx + 0];
    const ggml_tensor * silu = cgraph->nodes[node_idx + 1];

    if (scl->op != GGML_OP_SCALE || silu->op != GGML_OP_UNARY) {
        return false;
    }
    if (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return false;
    }
    if (silu->src[0] != scl) {
        return false;
    }

    const ggml_tensor * src = scl->src[0];
    if (!src) {
        return false;
    }

    if (src->type != GGML_TYPE_F32 || scl->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        return false;
    }

    // scale.cu i unary.cu jada po plaskiej liscie elementow
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(silu)) {
        return false;
    }
    if (!ggml_are_same_shape(src, scl) || !ggml_are_same_shape(scl, silu)) {
        return false;
    }

    // bias != 0 wymagalby powtorzenia kontrakcji scale.cu (v_fma_f32); przy zerze nie ma roznicy
    if (ggml_get_op_params_f32(scl, 1) != 0.0f) {
        return false;
    }

    // pusty tensor dalby siatke o zerowej liczbie blokow
    if (ggml_nelements(src) <= 0) {
        return false;
    }

    // dst czytane i pisane pod tym samym indeksem, wiec dokladny alias jest bezpieczny;
    // czesciowe nachodzenie juz nie
    if (silu->data != src->data && ggml_cuda_mix_overlap(silu, src)) {
        return false;
    }

    const enum ggml_op ops[]     = { GGML_OP_SCALE, GGML_OP_UNARY };
    const int          idxs[]    = { node_idx, node_idx + 1 };
    const int          outputs[] = { node_idx + 1 };

    return ggml_can_fuse_subgraph_ext(cgraph, idxs, 2, ops, outputs, 1);
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * scl  = cgraph->nodes[node_idx + 0];
    const ggml_tensor * silu = cgraph->nodes[node_idx + 1];
    const ggml_tensor * src  = scl->src[0];

    const float scale = ggml_get_op_params_f32(scl, 0);
    const float bias  = ggml_get_op_params_f32(scl, 1);

    const int64_t nelements = ggml_nelements(src);

    if (getenv("GGML_CUDA_DUMP_DISPATCH")) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[johnv8] scale_silu fused: nelements=%d (2 wezly -> 1 jadro)\n", (int) nelements);
        }
    }

    // taka sama siatka jak scale_f32_cuda, zeby kolejnosc elementow byla ta sama
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3((unsigned int) MIN((int64_t) 0x7FFFFFFF, num_blocks), 1, 1),
            dim3(CUDA_SCALE_BLOCK_SIZE, 1, 1),
            (size_t) 0,
            ctx.stream());

    ggml_cuda_kernel_launch(scale_silu_f32, launch_params,
            (const float *) src->data,
            (float *) silu->data,
            scale, bias, nelements);
}
