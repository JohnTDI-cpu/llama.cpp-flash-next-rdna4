#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
// A: jak unary.cu (bez pragmy)
__global__ void sig_default(const float * x, float * y, int n) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i])); }
// B: jak shexp-tail.cu (pragma contract off w ciele jadra)
__global__ void sig_nocontract(const float * x, float * y, int n) {
#pragma clang fp contract(off)
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i])); }
// C: wartosc jednorodna w bloku (jak gate[t]) z barierą asm, pragma off
__global__ void sig_uniform_off(const float * x, float * y, int n) {
#pragma clang fp contract(off)
    float g = x[blockIdx.x];
    __asm__("" : "+v"(g));
    const float s = 1.0f / (1.0f + expf(-g));
    int i = threadIdx.x; if (i < n) y[blockIdx.x*n + i] = s; }
// D: jednorodna, bez pragmy
__global__ void sig_uniform_def(const float * x, float * y, int n) {
    float g = x[blockIdx.x];
    __asm__("" : "+v"(g));
    const float s = 1.0f / (1.0f + expf(-g));
    int i = threadIdx.x; if (i < n) y[blockIdx.x*n + i] = s; }
int main() {
    const int n = 1<<16; float *x, *ya, *yb, *yc, *yd;
    if (hipSetDevice(0) != hipSuccess) { printf("hipSetDevice padl\n"); return 1; }
    if (hipMallocManaged(&x, n*4) != hipSuccess || hipMallocManaged(&ya, n*4) != hipSuccess || hipMallocManaged(&yb, n*4) != hipSuccess || hipMallocManaged(&yc, n*4) != hipSuccess || hipMallocManaged(&yd, n*4) != hipSuccess) { printf("hipMallocManaged padl\n"); return 1; }
    srand(3); for (int i = 0; i < n; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f)*16;
    sig_default<<<n/256,256>>>(x, ya, n); sig_nocontract<<<n/256,256>>>(x, yb, n); hipDeviceSynchronize();
    int dab = 0; for (int i = 0; i < n; i++) if (ya[i] != yb[i]) dab++;
    // jednorodne: 1024 blokow po 64 wartosci
    sig_uniform_off<<<1024,64>>>(x, yc, 64); sig_uniform_def<<<1024,64>>>(x, yd, 64); hipDeviceSynchronize();
    int dac = 0, dad = 0; for (int b = 0; b < 1024; b++) { if (ya[b] != yc[b*64]) dac++; if (ya[b] != yd[b*64]) dad++; }
    printf("elementowo: default vs contract(off): %d/%d roznych\n", dab, n);
    printf("jednorodne: default vs uniform+off: %d/1024 roznych; default vs uniform+default: %d/1024 roznych\n", dac, dad);
    return 0; }
