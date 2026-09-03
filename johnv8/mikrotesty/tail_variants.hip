#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
// stock: 3 jadra
__global__ void k_sig(const float * g, float * s, int nt) { int t = blockIdx.x*blockDim.x + threadIdx.x; if (t < nt) s[t] = 1.0f / (1.0f + expf(-g[t])); }
__global__ void k_mul(const float * a, const float * s, float * m, int n, int nt) { int i = blockIdx.x*blockDim.x + threadIdx.x; int t = blockIdx.y; if (i < n) m[i + n*t] = a[i + n*t] * s[t]; }
__global__ void k_add(const float * a, const float * b, float * d, int n) { int i = blockIdx.x*blockDim.x + threadIdx.x; int t = blockIdx.y; if (i < n) d[i + n*t] = a[i + n*t] + b[i + n*t]; }
// V1: jak shexp-tail.cu (pragma off + rn)
__global__ void v1(const float * g, const float * sh, const float * mo, float * d, int n) {
#pragma clang fp contract(off)
    int i = blockIdx.x*blockDim.x + threadIdx.x; int t = blockIdx.y; if (i >= n) return;
    float gg = g[t]; __asm__("" : "+v"(gg)); const float sg = 1.0f / (1.0f + expf(-gg));
    const float m = __fmul_rn(sh[i + n*t], sg); d[i + n*t] = __fadd_rn(mo[i + n*t], m); }
// V2: pragma off, zwykle operatory
__global__ void v2(const float * g, const float * sh, const float * mo, float * d, int n) {
#pragma clang fp contract(off)
    int i = blockIdx.x*blockDim.x + threadIdx.x; int t = blockIdx.y; if (i >= n) return;
    float gg = g[t]; __asm__("" : "+v"(gg)); const float sg = 1.0f / (1.0f + expf(-gg));
    const float m = sh[i + n*t] * sg; d[i + n*t] = mo[i + n*t] + m; }
// V3: bez pragmy, zwykle operatory (mozliwa kontrakcja)
__global__ void v3(const float * g, const float * sh, const float * mo, float * d, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; int t = blockIdx.y; if (i >= n) return;
    float gg = g[t]; __asm__("" : "+v"(gg)); const float sg = 1.0f / (1.0f + expf(-gg));
    const float m = sh[i + n*t] * sg; d[i + n*t] = mo[i + n*t] + m; }
// V4: sigmoid z pamieci (jak stock: policzony osobno), potem rn
__global__ void v4(const float * s, const float * sh, const float * mo, float * d, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; int t = blockIdx.y; if (i >= n) return;
    const float sg = s[t]; const float m = __fmul_rn(sh[i + n*t], sg); d[i + n*t] = __fadd_rn(mo[i + n*t], m); }
// V5: sigmoid bez bariery asm, rn
__global__ void v5(const float * g, const float * sh, const float * mo, float * d, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; int t = blockIdx.y; if (i >= n) return;
    const float sg = 1.0f / (1.0f + expf(-g[t]));
    const float m = __fmul_rn(sh[i + n*t], sg); d[i + n*t] = __fadd_rn(mo[i + n*t], m); }
int main() {
    const int n = 2560, nt = 64; if (hipSetDevice(0) != hipSuccess) return 1;
    float *g,*sh,*mo,*s,*m,*d0,*d1,*d2,*d3,*d4,*d5; size_t N = (size_t) n*nt;
    hipMallocManaged(&g, nt*4); hipMallocManaged(&sh, N*4); hipMallocManaged(&mo, N*4); hipMallocManaged(&s, nt*4); hipMallocManaged(&m, N*4);
    hipMallocManaged(&d0, N*4); hipMallocManaged(&d1, N*4); hipMallocManaged(&d2, N*4); hipMallocManaged(&d3, N*4); hipMallocManaged(&d4, N*4); hipMallocManaged(&d5, N*4);
    srand(5); for (int t = 0; t < nt; t++) g[t] = (rand()/(float)RAND_MAX - 0.5f)*12; for (size_t i = 0; i < N; i++) { sh[i] = (rand()/(float)RAND_MAX - 0.5f)*4; mo[i] = (rand()/(float)RAND_MAX - 0.5f)*4; }
    dim3 grid((n+255)/256, nt); k_sig<<<1,64>>>(g, s, nt); k_mul<<<grid,256>>>(sh, s, m, n, nt); k_add<<<grid,256>>>(mo, m, d0, n);
    v1<<<grid,256>>>(g, sh, mo, d1, n); v2<<<grid,256>>>(g, sh, mo, d2, n); v3<<<grid,256>>>(g, sh, mo, d3, n); v4<<<grid,256>>>(s, sh, mo, d4, n); v5<<<grid,256>>>(g, sh, mo, d5, n); hipDeviceSynchronize();
    auto cnt = [&](float * d) { int c = 0; for (size_t i = 0; i < N; i++) if (d[i] != d0[i]) c++; return c; };
    printf("rozne vs stock (3 jadra), N=%zu:  V1(pragma off+rn)=%d  V2(pragma off, zwykle)=%d  V3(bez pragmy)=%d  V4(sigmoid z pamieci+rn)=%d  V5(bez bariery asm+rn)=%d\n", N, cnt(d1), cnt(d2), cnt(d3), cnt(d4), cnt(d5));
    // czy sam sigmoid sie rozni?
    int ds = 0; for (int t = 0; t < nt; t++) { float gg = g[t]; float ref = 1.0f/(1.0f+expf(-gg)); if (ref != s[t]) ds++; } printf("sigmoid GPU vs CPU expf: %d/%d roznych\n", ds, nt);
    return 0; }
