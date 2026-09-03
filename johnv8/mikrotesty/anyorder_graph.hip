// Czy flaga hipExtAnyOrderLaunch przezywa capture do hipGraph i replay? Porownanie tempa 2000 jader ze spin=2000 w grafie.
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdio>
__global__ void tiny(float * p, int spin) { float v = p[threadIdx.x]; for (int i = 0; i < spin; ++i) v = v * 1.0001f + 0.5f; p[threadIdx.x] = v; }
int main() {
    if (hipSetDevice(0) != hipSuccess) return 1;
    float * d; hipMalloc(&d, 4096); hipMemset(d, 0, 4096); hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    const int N = 2000, spin = 2000;
    for (int mode = 0; mode < 2; ++mode) {
        hipGraph_t g; hipGraphExec_t ge;
        hipError_t e = hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal); if (e != hipSuccess) { printf("BeginCapture: %s\n", hipGetErrorString(e)); return 1; }
        for (int i = 0; i < N; ++i) { if (mode == 0) tiny<<<1, 64, 0, s>>>(d, spin); else hipExtLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, nullptr, nullptr, hipExtAnyOrderLaunch, d, spin); }
        e = hipStreamEndCapture(s, &g); if (e != hipSuccess) { printf("EndCapture (%s): %s\n", mode ? "anyorder" : "plain", hipGetErrorString(e)); continue; }
        size_t nn = 0; hipGraphGetNodes(g, nullptr, &nn);
        e = hipGraphInstantiate(&ge, g, nullptr, nullptr, 0); if (e != hipSuccess) { printf("Instantiate: %s\n", hipGetErrorString(e)); continue; }
        hipGraphLaunch(ge, s); hipStreamSynchronize(s);   // rozgrzewka
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0, s); for (int r = 0; r < 3; ++r) hipGraphLaunch(ge, s); hipEventRecord(e1, s); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        printf("graf %-9s: %zu wezlow, replay: %.2f us/jadro (spin=%d; serializacja ~17 us, any-order ~2 us)\n", mode ? "anyorder" : "plain", nn, ms * 1000.0 / (3.0 * N), spin);
        hipGraphExecDestroy(ge); hipGraphDestroy(g);
    }
    // i bez grafu, dla odniesienia w tym samym procesie
    for (int mode = 0; mode < 2; ++mode) { hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0, s);
        for (int i = 0; i < N; ++i) { if (mode == 0) tiny<<<1, 64, 0, s>>>(d, spin); else hipExtLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, nullptr, nullptr, hipExtAnyOrderLaunch, d, spin); }
        hipEventRecord(e1, s); hipEventSynchronize(e1); float ms = 0; hipEventElapsedTime(&ms, e0, e1); printf("strumien %-9s: %.2f us/jadro\n", mode ? "anyorder" : "plain", ms * 1000.0 / N); }
    return 0;
}
