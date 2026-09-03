// E0f: host-side enqueue cost of the kernel launch APIs.
//
// A kernel with 12 arguments (pointers, ints, floats, long long, unsigned, double) that writes one
// int (and optionally spins ~20 us on the wall clock). For N launches (default 4000, argv[1]
// overrides) we measure the host wall time of the launch loop only (no GPU wait inside), then
// hipStreamSynchronize and report the GPU-side batch time (hipEvent before the loop -> after a
// plain join kernel that closes the batch). Modes:
//   a  kernel<<<grid, block, 0, stream>>>(...)
//   b  hipExtLaunchKernelGGL(..., flags = 0, ...)
//   c  hipExtLaunchKernelGGL(..., flags = hipExtAnyOrderLaunch, ...)
//   d  hipLaunchKernel(func, grid, block, void** args, 0, stream)
// Each batch runs 3 times; the median host cost per launch (us) is reported with the batch times
// of that run. The spin variant (4000 x ~20 us = 80 ms of serialized GPU time) shows whether the
// host loop returns early (async) or blocks when the queue fills; an extra 4N-launch spin round
// for modes a and c probes deeper queue depths.
//
// compile: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -Wall -o e0f_launchcost e0f_launchcost.hip
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>

#define CHECK(x) do { hipError_t err_ = (x); if (err_ != hipSuccess) { \
    fprintf(stderr, "HIP error at %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(err_)); \
    exit(2); } } while (0)

__global__ void k12(int* out, const float* a, const float* b, const int* c,
                    int n0, int n1, float f0, float f1, long long l0, unsigned u0, double d0, int spin_ticks) {
    if (spin_ticks > 0) {
        const long long t0 = wall_clock64();
        while (wall_clock64() - t0 < spin_ticks) {}
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float acc = a[0] + b[0] + (float)c[0] + (float)n0 + (float)n1 + f0 + f1 + (float)l0 + (float)u0 + (float)d0;
        out[0] = (int)acc;
    }
}

__global__ void join_kernel() {}

struct Meas { double host_us_per_launch; double host_loop_ms; double gpu_ms; double wall_ms; };

static const char* mode_name(int m) {
    switch (m) {
        case 0: return "a <<<grid,block,0,stream>>>          ";
        case 1: return "b hipExtLaunchKernelGGL flags=0      ";
        case 2: return "c hipExtLaunchKernelGGL AnyOrder     ";
        case 3: return "d hipLaunchKernel(void** args)       ";
        default: return "?";
    }
}

struct Args {
    int* out; const float* a; const float* b; const int* c;
    int n0; int n1; float f0; float f1; long long l0; unsigned u0; double d0; int spin;
};

static Meas run_batch(int mode, int N, Args g, hipStream_t s, hipEvent_t e0, hipEvent_t e1) {
    const dim3 grid(1), block(64);
    // local copies with the exact formal types (needed for the void** form)
    int* out = g.out; const float* a = g.a; const float* b = g.b; const int* c = g.c;
    int n0 = g.n0, n1 = g.n1; float f0 = g.f0, f1 = g.f1; long long l0 = g.l0; unsigned u0 = g.u0; double d0 = g.d0; int spin = g.spin;
    void* argv[12] = { &out, &a, &b, &c, &n0, &n1, &f0, &f1, &l0, &u0, &d0, &spin };
    const void* fn = reinterpret_cast<const void*>(&k12);

    CHECK(hipEventRecord(e0, s));
    const auto t0 = std::chrono::steady_clock::now();
    switch (mode) {
        case 0:
            for (int i = 0; i < N; ++i) k12<<<grid, block, 0, s>>>(out, a, b, c, n0, n1, f0, f1, l0, u0, d0, spin);
            break;
        case 1:
            for (int i = 0; i < N; ++i) hipExtLaunchKernelGGL(k12, grid, block, 0, s, nullptr, nullptr, 0u, out, a, b, c, n0, n1, f0, f1, l0, u0, d0, spin);
            break;
        case 2:
            for (int i = 0; i < N; ++i) hipExtLaunchKernelGGL(k12, grid, block, 0, s, nullptr, nullptr, (std::uint32_t)hipExtAnyOrderLaunch, out, a, b, c, n0, n1, f0, f1, l0, u0, d0, spin);
            break;
        case 3:
            for (int i = 0; i < N; ++i) CHECK(hipLaunchKernel(fn, grid, block, argv, 0, s));
            break;
        default: break;
    }
    const auto t1 = std::chrono::steady_clock::now();
    join_kernel<<<1, 1, 0, s>>>();        // plain launch: barrier bit -> e1 fires after ALL N kernels (also for any-order)
    CHECK(hipEventRecord(e1, s));
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(s));
    const auto t2 = std::chrono::steady_clock::now();
    Meas m;
    m.host_loop_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    m.host_us_per_launch = m.host_loop_ms * 1000.0 / N;
    float ms = 0.f; CHECK(hipEventElapsedTime(&ms, e0, e1));
    m.gpu_ms = ms;
    m.wall_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
    return m;
}

static Meas median3(Meas r[3]) {
    std::sort(r, r + 3, [](const Meas& x, const Meas& y) { return x.host_us_per_launch < y.host_us_per_launch; });
    return r[1];
}

static void run_round(const char* label, int N, Args g, hipStream_t s, hipEvent_t e0, hipEvent_t e1, bool only_a_c) {
    printf("E0f --- %s: N=%d launches per batch, %s ---\n", label, N, g.spin > 0 ? "kernel spins ~20 us" : "empty kernel");
    for (int mode = 0; mode < 4; ++mode) {
        if (only_a_c && mode != 0 && mode != 2) continue;
        Meas r[3];
        for (int rep = 0; rep < 3; ++rep) r[rep] = run_batch(mode, N, g, s, e0, e1);
        const Meas m = median3(r);
        printf("E0f %s host %7.3f us/launch (loop %8.2f ms) | GPU batch %8.2f ms (%6.2f us/kernel) | wall-to-sync %8.2f ms",
               mode_name(mode), m.host_us_per_launch, m.host_loop_ms, m.gpu_ms, m.gpu_ms * 1000.0 / N, m.wall_ms);
        if (g.spin > 0) {
            const bool async = m.host_loop_ms < 0.25 * m.gpu_ms;
            printf(" | HOST ENQUEUE: %s", async ? "async (returned early)" : "BLOCKS (host loop ~ GPU time)");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int N = 4000;
    if (argc > 1) N = atoi(argv[1]);
    if (N < 1) N = 4000;
    int dev = 0;
    CHECK(hipSetDevice(dev));
    CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    int wall_khz = 0;
    CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
    if (wall_khz <= 0) wall_khz = 100000;
    const int spin20us = (int)((long long)wall_khz * 20 / 1000);   // ticks for 20 us
    printf("E0f device=%d %s (%s) wallclock=%d kHz, 20 us = %d ticks\n", dev, prop.name, prop.gcnArchName, wall_khz, spin20us);

    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    int* out = nullptr; float *fa = nullptr, *fb = nullptr; int* ic = nullptr;
    CHECK(hipMalloc(&out, 64)); CHECK(hipMalloc(&fa, 64)); CHECK(hipMalloc(&fb, 64)); CHECK(hipMalloc(&ic, 64));
    CHECK(hipMemset(out, 0, 64)); CHECK(hipMemset(fa, 0, 64)); CHECK(hipMemset(fb, 0, 64)); CHECK(hipMemset(ic, 0, 64));
    CHECK(hipDeviceSynchronize());

    Args g; g.out = out; g.a = fa; g.b = fb; g.c = ic; g.n0 = 3; g.n1 = 5; g.f0 = 1.5f; g.f1 = 2.5f; g.l0 = 7; g.u0 = 9u; g.d0 = 0.25; g.spin = 0;

    // warm-up all four paths
    for (int mode = 0; mode < 4; ++mode) (void)run_batch(mode, 256, g, s, e0, e1);
    g.spin = spin20us;
    for (int mode = 0; mode < 4; ++mode) (void)run_batch(mode, 64, g, s, e0, e1);
    g.spin = 0;

    run_round("empty kernel", N, g, s, e0, e1, false);
    g.spin = spin20us;
    run_round("spin kernel", N, g, s, e0, e1, false);
    run_round("spin kernel, deeper queue", 4 * N, g, s, e0, e1, true);

    // sanity: last kernel wrote something plausible
    int hval = -1;
    CHECK(hipMemcpy(&hval, out, sizeof(int), hipMemcpyDeviceToHost));
    const int expect = (int)(0.f + 0.f + 0.f + 3.f + 5.f + 1.5f + 2.5f + 7.f + 9.f + 0.25f);
    const bool ok = (hval == expect);
    printf("E0f sanity out[0]=%d expected %d %s\n", hval, expect, ok ? "PASS" : "FAIL");
    printf("E0f RESULT: %s\n", ok ? "PASS" : "FAIL");
    CHECK(hipFree(out)); CHECK(hipFree(fa)); CHECK(hipFree(fb)); CHECK(hipFree(ic));
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    CHECK(hipStreamDestroy(s));
    return ok ? 0 : 1;
}
