// E0b: event / sync semantics after any-order kernels (hipExtAnyOrderLaunch).
//
// 64 any-order kernels are launched on ONE non-blocking stream. Kernel k (k = 0..63) spins
// (64-k)*2000 iterations of a wall_clock64()-based busy loop, so the LAST launched kernel finishes
// FIRST, then writes marker (k+1) into every int of region k (64 regions x 4096 bytes).
// After a host-side wait we immediately read all regions (hipMemcpy D2H) and count regions whose
// markers are missing or partial. Variants:
//   A  : hipEventRecord(ev, s1) + hipEventSynchronize(ev)              (default event, timing on)
//   A2 : same as A with an event created with hipEventDisableTiming
//   B  : hipStreamSynchronize(s1)
//   C  : one empty any-order kernel, then hipEventRecord + hipEventSynchronize
//   D  : hipEventRecord(ev, s1) + hipStreamWaitEvent(s2, ev) + plain copy kernel on s2 that copies
//        all 64 regions into a second buffer + hipStreamSynchronize(s2), then check the copy
//   D2 : same as D with an event created with hipEventDisableTiming
// A variant FAILs when any trial has a region with missing/partial markers, i.e. the host wait
// returned while any-order kernels were still in flight. If that happens, a "join" (plain launch
// with the AQL barrier bit) is required before every hipEventRecord / hipStreamSynchronize.
//
// compile: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -Wall -o e0b_eventsync e0b_eventsync.hip
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { hipError_t err_ = (x); if (err_ != hipSuccess) { \
    fprintf(stderr, "HIP error at %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(err_)); \
    exit(2); } } while (0)

static constexpr int NREG      = 64;
static constexpr int REG_BYTES = 4096;
static constexpr int REG_INTS  = REG_BYTES / (int)sizeof(int);
static constexpr int BUF_BYTES = NREG * REG_BYTES;
static constexpr int ITER_UNIT = 2000;     // kernel k spins (NREG-k)*ITER_UNIT iterations
static constexpr int TRIALS    = 200;
static constexpr int BLOCK     = 256;
static constexpr int NS_PER_ITER_MIN = 20; // each iteration waits >= 20 ns of wall clock

// One iteration = wait until the constant-rate wall clock advanced by >= ticks_per_iter ticks.
__device__ __forceinline__ void spin_iters(int iters, int ticks_per_iter) {
    long long t = wall_clock64();
    for (int i = 0; i < iters; ++i) {
        const long long target = t + ticks_per_iter;
        long long now;
        do { now = wall_clock64(); } while (now < target);
        t = now;
    }
}

__global__ void spin_mark(int* buf, int k, int iters, int ticks_per_iter) {
    spin_iters(iters, ticks_per_iter);
    int* reg = buf + (size_t)k * REG_INTS;
    for (int i = threadIdx.x; i < REG_INTS; i += blockDim.x) reg[i] = k + 1;
}

__global__ void empty_kernel(int dummy) { (void)dummy; }

// plain launch (barrier bit set) -> waits for every earlier packet in the queue
__global__ void join_kernel() {}

// one block per region, one thread per int
__global__ void copy_regions(int* dst, const int* src) {
    const size_t i = (size_t)blockIdx.x * REG_INTS + threadIdx.x;
    dst[i] = src[i];
}

static int count_bad_regions(const int* h, int* first_bad, int* last_bad) {
    int bad = 0; *first_bad = -1; *last_bad = -1;
    for (int k = 0; k < NREG; ++k) {
        const int* reg = h + (size_t)k * REG_INTS;
        bool ok = true;
        for (int i = 0; i < REG_INTS; ++i) if (reg[i] != k + 1) { ok = false; break; }
        if (!ok) { ++bad; if (*first_bad < 0) *first_bad = k; *last_bad = k; }
    }
    return bad;
}

static void launch_batch_anyorder(int* buf, hipStream_t s, int ticks) {
    for (int k = 0; k < NREG; ++k)
        hipExtLaunchKernelGGL(spin_mark, dim3(1), dim3(BLOCK), 0, s, nullptr, nullptr,
                              hipExtAnyOrderLaunch, buf, k, (NREG - k) * ITER_UNIT, ticks);
}

static void launch_batch_plain(int* buf, hipStream_t s, int ticks) {
    for (int k = 0; k < NREG; ++k)
        spin_mark<<<1, BLOCK, 0, s>>>(buf, k, (NREG - k) * ITER_UNIT, ticks);
}

static float elapsed_ms(hipEvent_t e0, hipEvent_t e1) { float ms = 0.f; CHECK(hipEventElapsedTime(&ms, e0, e1)); return ms; }

enum Variant { VA, VA2, VB, VC, VD, VD2, NVAR };
static const char* variant_name(int v) {
    switch (v) {
        case VA:  return "A  hipEventRecord+hipEventSynchronize (timing event)   ";
        case VA2: return "A2 hipEventRecord+hipEventSynchronize (DisableTiming)  ";
        case VB:  return "B  hipStreamSynchronize                                ";
        case VC:  return "C  empty any-order kernel + EventRecord+EventSync      ";
        case VD:  return "D  EventRecord+StreamWaitEvent(s2)+plain copy on s2    ";
        case VD2: return "D2 same as D with DisableTiming event                  ";
        default:  return "?";
    }
}

int main() {
    int dev = 0;
    CHECK(hipSetDevice(dev));
    CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    int wall_khz = 0;
    CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
    if (wall_khz <= 0) wall_khz = 100000;  // 100 MHz is the usual steady counter rate
    int ticks = (int)((long long)wall_khz * NS_PER_ITER_MIN / 1000000LL);
    if (ticks < 1) ticks = 1;
    const double ns_per_iter = 1e6 * (double)ticks / (double)wall_khz;
    const double nominal_k0_ms = ns_per_iter * (double)NREG * ITER_UNIT * 1e-6;
    const double nominal_k63_ms = ns_per_iter * (double)ITER_UNIT * 1e-6;
    double nominal_sum_ms = 0.0;
    for (int k = 0; k < NREG; ++k) nominal_sum_ms += ns_per_iter * (double)(NREG - k) * ITER_UNIT * 1e-6;
    printf("E0b device=%d %s (%s) wallclock=%d kHz ticks/iter=%d -> %.1f ns/iter; nominal kernel0=%.3f ms, kernel63=%.3f ms, serialized sum=%.1f ms\n",
           dev, prop.name, prop.gcnArchName, wall_khz, ticks, ns_per_iter, nominal_k0_ms, nominal_k63_ms, nominal_sum_ms);

    hipStream_t s1, s2;
    CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    int *buf = nullptr, *buf2 = nullptr;
    CHECK(hipMalloc(&buf, BUF_BYTES));
    CHECK(hipMalloc(&buf2, BUF_BYTES));
    CHECK(hipMemset(buf, 0, BUF_BYTES));
    CHECK(hipMemset(buf2, 0, BUF_BYTES));
    hipEvent_t ev_t, ev_nt, e0, e1;
    CHECK(hipEventCreate(&ev_t));
    CHECK(hipEventCreateWithFlags(&ev_nt, hipEventDisableTiming));
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    int* h = nullptr;
    CHECK(hipHostMalloc(&h, BUF_BYTES));   // pinned: D2H goes through the copy engine, no staging blit kernel

    // ---- warm-up (loads code objects) ----
    spin_mark<<<1, BLOCK, 0, s1>>>(buf, 0, 10, ticks);
    empty_kernel<<<1, 1, 0, s1>>>(0);
    join_kernel<<<1, 1, 0, s1>>>();
    copy_regions<<<NREG, REG_INTS, 0, s2>>>(buf2, buf);
    hipExtLaunchKernelGGL(spin_mark, dim3(1), dim3(BLOCK), 0, s1, nullptr, nullptr, hipExtAnyOrderLaunch, buf, 0, 10, ticks);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    // ---- calibration: measured durations ----
    CHECK(hipEventRecord(e0, s1));
    spin_mark<<<1, BLOCK, 0, s1>>>(buf, 0, NREG * ITER_UNIT, ticks);
    CHECK(hipEventRecord(e1, s1));
    CHECK(hipStreamSynchronize(s1));
    const float k0_ms = elapsed_ms(e0, e1);

    CHECK(hipEventRecord(e0, s1));
    spin_mark<<<1, BLOCK, 0, s1>>>(buf, NREG - 1, ITER_UNIT, ticks);
    CHECK(hipEventRecord(e1, s1));
    CHECK(hipStreamSynchronize(s1));
    const float k63_ms = elapsed_ms(e0, e1);

    // any-order batch, closed by a plain join kernel so the stop event is reliable
    CHECK(hipEventRecord(e0, s1));
    launch_batch_anyorder(buf, s1, ticks);
    join_kernel<<<1, 1, 0, s1>>>();
    CHECK(hipEventRecord(e1, s1));
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    const float batch_ao_ms = elapsed_ms(e0, e1);

    // plain (serialized) batch for reference
    CHECK(hipEventRecord(e0, s1));
    launch_batch_plain(buf, s1, ticks);
    CHECK(hipEventRecord(e1, s1));
    CHECK(hipDeviceSynchronize());
    const float batch_plain_ms = elapsed_ms(e0, e1);

    printf("E0b calib: kernel0 alone=%.3f ms (longest), kernel63 alone=%.3f ms (shortest), any-order batch+join=%.3f ms, plain serialized batch=%.1f ms\n",
           k0_ms, k63_ms, batch_ao_ms, batch_plain_ms);
    if (k0_ms < 2.0f) printf("E0b calib WARNING: longest kernel < 2 ms, increase NS_PER_ITER_MIN\n");
    printf("E0b calib: any-order overlap %s (batch %.1fx of kernel0; serialized would be %.1fx)\n",
           batch_ao_ms < 0.5f * batch_plain_ms ? "OBSERVED" : "NOT observed", batch_ao_ms / k0_ms, batch_plain_ms / k0_ms);

    // ---- the experiment ----
    bool all_pass = true;
    int sanity_fail = 0;
    for (int v = 0; v < NVAR; ++v) {
        int bad_trials = 0, max_missing = 0, worst_first = -1, worst_last = -1;
        for (int trial = 0; trial < TRIALS; ++trial) {
            CHECK(hipMemsetAsync(buf, 0, BUF_BYTES, s1));
            CHECK(hipMemsetAsync(buf2, 0, BUF_BYTES, s1));
            CHECK(hipStreamSynchronize(s1));

            launch_batch_anyorder(buf, s1, ticks);

            const int* read_from = buf;
            switch (v) {
                case VA:
                    CHECK(hipEventRecord(ev_t, s1));
                    CHECK(hipEventSynchronize(ev_t));
                    break;
                case VA2:
                    CHECK(hipEventRecord(ev_nt, s1));
                    CHECK(hipEventSynchronize(ev_nt));
                    break;
                case VB:
                    CHECK(hipStreamSynchronize(s1));
                    break;
                case VC:
                    hipExtLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, s1, nullptr, nullptr, hipExtAnyOrderLaunch, 0);
                    CHECK(hipEventRecord(ev_t, s1));
                    CHECK(hipEventSynchronize(ev_t));
                    break;
                case VD:
                case VD2: {
                    hipEvent_t ev = (v == VD) ? ev_t : ev_nt;
                    CHECK(hipEventRecord(ev, s1));
                    CHECK(hipStreamWaitEvent(s2, ev, 0));
                    copy_regions<<<NREG, REG_INTS, 0, s2>>>(buf2, buf);
                    CHECK(hipStreamSynchronize(s2));
                    read_from = buf2;
                    break;
                }
                default: break;
            }
            // immediately read the regions (blocking hipMemcpy on the null stream; s1/s2 are non-blocking so no implicit sync)
            CHECK(hipMemcpy(h, read_from, BUF_BYTES, hipMemcpyDeviceToHost));
            int fb, lb;
            const int bad = count_bad_regions(h, &fb, &lb);
            if (bad > 0) {
                ++bad_trials;
                if (bad > max_missing) { max_missing = bad; worst_first = fb; worst_last = lb; }
            }

            // drain: plain join kernel (barrier bit) + device sync, then sanity check that all markers exist now
            join_kernel<<<1, 1, 0, s1>>>();
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(h, buf, BUF_BYTES, hipMemcpyDeviceToHost));
            if (count_bad_regions(h, &fb, &lb) != 0) ++sanity_fail;
        }
        const bool pass = (bad_trials == 0);
        all_pass = all_pass && pass;
        printf("E0b %s: trials=%d bad_trials=%d max_missing_regions=%d", variant_name(v), TRIALS, bad_trials, max_missing);
        if (max_missing > 0) printf(" (worst trial: missing k=%d..%d)", worst_first, worst_last);
        printf(" %s\n", pass ? "PASS" : "FAIL");
    }
    printf("E0b sanity (all markers present after join kernel + hipDeviceSynchronize): failures=%d %s\n",
           sanity_fail, sanity_fail == 0 ? "PASS" : "FAIL");
    all_pass = all_pass && (sanity_fail == 0);
    printf("E0b RESULT: %s\n", all_pass ? "PASS" : "FAIL");

    CHECK(hipEventDestroy(ev_t)); CHECK(hipEventDestroy(ev_nt)); CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    CHECK(hipHostFree(h));
    CHECK(hipFree(buf)); CHECK(hipFree(buf2));
    CHECK(hipStreamDestroy(s1)); CHECK(hipStreamDestroy(s2));
    return all_pass ? 0 : 1;
}
