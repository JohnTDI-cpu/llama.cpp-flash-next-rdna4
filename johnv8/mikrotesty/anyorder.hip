#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdio>
#include <chrono>
__global__ void tiny(float * p, int spin) { float v = p[threadIdx.x]; for (int i = 0; i < spin; ++i) v = v * 1.0001f + 0.5f; p[threadIdx.x] = v; }
int main() {
    if (hipSetDevice(0) != hipSuccess) return 1;
    float * d; hipMalloc(&d, 4096); hipMemset(d, 0, 4096); hipStream_t s; hipStreamCreate(&s);
    for (int spin : {0, 2000, 8000}) {
        for (int mode = 0; mode < 2; ++mode) {
            // rozgrzewka
            for (int i = 0; i < 200; ++i) { if (mode == 0) tiny<<<1, 64, 0, s>>>(d, spin); else hipExtLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, nullptr, nullptr, hipExtAnyOrderLaunch, d, spin); }
            hipStreamSynchronize(s);
            hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
            auto t0 = std::chrono::steady_clock::now(); hipEventRecord(e0, s);
            const int N = 4000;
            for (int i = 0; i < N; ++i) { if (mode == 0) tiny<<<1, 64, 0, s>>>(d, spin); else hipExtLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, nullptr, nullptr, hipExtAnyOrderLaunch, d, spin); }
            hipEventRecord(e1, s); auto t1 = std::chrono::steady_clock::now(); hipEventSynchronize(e1); auto t2 = std::chrono::steady_clock::now();
            float ms = 0; hipEventElapsedTime(&ms, e0, e1);
            double host_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N, wall_us = std::chrono::duration<double, std::micro>(t2 - t0).count() / N;
            printf("spin=%5d %-22s GPU %.2f us/jadro (event), host enqueue %.2f us/jadro, wall %.2f us/jadro\n", spin, mode == 0 ? "<<<>>> (bariera)" : "hipExtAnyOrderLaunch", ms * 1000.0 / N, host_us, wall_us);
        }
    }
    return 0;
}
