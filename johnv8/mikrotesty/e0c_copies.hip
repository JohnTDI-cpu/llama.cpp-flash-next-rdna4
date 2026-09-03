// E0c: copies vs any-order kernels (hipExtAnyOrderLaunch).
//
// Test 1 : 64 slow any-order kernels write markers (k+1) into 64 regions (last launched finishes
//          first), then hipMemcpyAsync D2D of the whole buffer into a second device buffer on the
//          SAME stream, hipStreamSynchronize, D2H, verify all markers. Checked twice per trial:
//            (i)  right after hipStreamSynchronize   -> what a user sees in practice
//            (ii) after a plain join kernel + hipDeviceSynchronize -> the copy is definitely complete,
//                 so remaining missing markers prove the copy ran BEFORE the kernels (copy not ordered)
// Test 1b: same, but hipMemcpyPeerAsync from the current device to another visible device
//          (skipped unless >= 2 devices are visible).
// Test 2 : the reverse. A 512 MB hipMemcpyAsync D2D (source filled with a per-trial pattern by a
//          plain kernel + sync) followed IMMEDIATELY by an any-order kernel that reads the LAST 1 MB
//          of the destination and counts ints that do not match. Informational: prints
//          "OVERTAKE: yes/no" and PASSes either way (plus a plain-launch control).
//
// compile: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -Wall -o e0c_copies e0c_copies.hip
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
static constexpr int ITER_UNIT = 2000;
static constexpr int TRIALS    = 200;
static constexpr int CONTROL_TRIALS = 20;
static constexpr int BLOCK     = 256;
static constexpr int NS_PER_ITER_MIN = 20;

static constexpr size_t BIG_BYTES  = 512ull * 1024 * 1024;
static constexpr size_t BIG_INTS   = BIG_BYTES / sizeof(int);
static constexpr size_t TAIL_BYTES = 1024 * 1024;
static constexpr int    TAIL_INTS  = (int)(TAIL_BYTES / sizeof(int));

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

__global__ void join_kernel() {}

__global__ void fill_pattern(int* p, int seed, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = seed + (int)i;
}

// tail[i] must equal seed + (base_index + i); count mismatches into *result
__global__ void check_tail(const int* tail, int seed, int base_index, int ntail, int* result) {
    __shared__ int sh;
    if (threadIdx.x == 0) sh = 0;
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ntail && tail[i] != seed + (base_index + i)) atomicAdd(&sh, 1);
    __syncthreads();
    if (threadIdx.x == 0 && sh != 0) atomicAdd(result, sh);
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

struct T1Stats { int bad_after_sync = 0, max_after_sync = 0, bad_after_join = 0, max_after_join = 0, sanity_fail = 0; };

// One test-1 style run: any-order kernels -> copy (issued by `issue_copy`) -> sync -> check dst.
// dst lives on device dst_dev (== dev for the D2D test, the peer device for T1b); memset and readback
// of dst are done with dst_dev current so CLR handles the buffer on its own device.
template <typename IssueCopy>
static T1Stats run_test1(int* src, int* dst, int dev, int dst_dev, hipStream_t s, int ticks, int trials, int* h, IssueCopy issue_copy) {
    T1Stats st;
    auto clear_dst = [&]() {
        if (dst_dev == dev) { CHECK(hipMemsetAsync(dst, 0, BUF_BYTES, s)); }
        else { CHECK(hipSetDevice(dst_dev)); CHECK(hipMemset(dst, 0, BUF_BYTES)); CHECK(hipDeviceSynchronize()); CHECK(hipSetDevice(dev)); }
    };
    auto read_dst = [&]() {
        if (dst_dev != dev) CHECK(hipSetDevice(dst_dev));
        CHECK(hipMemcpy(h, dst, BUF_BYTES, hipMemcpyDeviceToHost));
        if (dst_dev != dev) CHECK(hipSetDevice(dev));
    };
    for (int trial = 0; trial < trials; ++trial) {
        CHECK(hipMemsetAsync(src, 0, BUF_BYTES, s));
        clear_dst();
        CHECK(hipStreamSynchronize(s));

        launch_batch_anyorder(src, s, ticks);
        issue_copy();
        CHECK(hipStreamSynchronize(s));
        read_dst();                                   // (i) what a user sees right after the stream sync
        int fb, lb;
        int bad = count_bad_regions(h, &fb, &lb);
        if (bad > 0) { ++st.bad_after_sync; if (bad > st.max_after_sync) st.max_after_sync = bad; }

        join_kernel<<<1, 1, 0, s>>>();                // plain launch: barrier bit -> waits for every earlier packet
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        if (dst_dev != dev) { CHECK(hipSetDevice(dst_dev)); CHECK(hipDeviceSynchronize()); CHECK(hipSetDevice(dev)); }
        read_dst();                                   // (ii) copy is definitely complete now
        bad = count_bad_regions(h, &fb, &lb);
        if (bad > 0) { ++st.bad_after_join; if (bad > st.max_after_join) st.max_after_join = bad; }
        // sanity: source itself must be complete now
        CHECK(hipMemcpy(h, src, BUF_BYTES, hipMemcpyDeviceToHost));
        if (count_bad_regions(h, &fb, &lb) != 0) ++st.sanity_fail;
    }
    return st;
}

static bool report_test1(const char* name, const T1Stats& st, int trials) {
    const bool p_sync = (st.bad_after_sync == 0);
    const bool p_join = (st.bad_after_join == 0);
    const bool p_san  = (st.sanity_fail == 0);
    printf("E0c %s checked after hipStreamSynchronize          : trials=%d bad_trials=%d max_missing_regions=%d %s\n",
           name, trials, st.bad_after_sync, st.max_after_sync, p_sync ? "PASS" : "FAIL");
    printf("E0c %s checked after join+hipDeviceSynchronize     : trials=%d bad_trials=%d max_missing_regions=%d %s\n",
           name, trials, st.bad_after_join, st.max_after_join, p_join ? "PASS" : "FAIL");
    printf("E0c %s sanity (source complete after join)         : failures=%d %s\n", name, st.sanity_fail, p_san ? "PASS" : "FAIL");
    printf("E0c %s COPY ORDERED AFTER ANY-ORDER KERNELS: %s%s\n", name, p_join ? "yes" : "NO",
           (p_join && !p_sync) ? " (but hipStreamSynchronize returned before the copy finished -> see E0b)" : "");
    return p_sync && p_join && p_san;
}

int main() {
    int dev = 0, ndev = 0;
    CHECK(hipGetDeviceCount(&ndev));
    CHECK(hipSetDevice(dev));
    CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    int wall_khz = 0;
    CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
    if (wall_khz <= 0) wall_khz = 100000;
    int ticks = (int)((long long)wall_khz * NS_PER_ITER_MIN / 1000000LL);
    if (ticks < 1) ticks = 1;
    printf("E0c device=%d/%d %s (%s) wallclock=%d kHz ticks/iter=%d (kernel0 nominal %.2f ms)\n", dev, ndev, prop.name, prop.gcnArchName,
           wall_khz, ticks, 1e6 * ticks / wall_khz * NREG * ITER_UNIT * 1e-6);

    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    int *src = nullptr, *dst = nullptr;
    CHECK(hipMalloc(&src, BUF_BYTES));
    CHECK(hipMalloc(&dst, BUF_BYTES));
    int* h = nullptr;
    CHECK(hipHostMalloc(&h, BUF_BYTES));   // pinned: D2H goes through the copy engine, no staging blit kernel
    bool all_pass = true;

    // warm-up
    spin_mark<<<1, BLOCK, 0, s>>>(src, 0, 10, ticks);
    join_kernel<<<1, 1, 0, s>>>();
    hipExtLaunchKernelGGL(spin_mark, dim3(1), dim3(BLOCK), 0, s, nullptr, nullptr, hipExtAnyOrderLaunch, src, 0, 10, ticks);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    // ---------------- Test 1: D2D copy after any-order kernels ----------------
    {
        T1Stats st = run_test1(src, dst, dev, dev, s, ticks, TRIALS, h, [&]() {
            CHECK(hipMemcpyAsync(dst, src, BUF_BYTES, hipMemcpyDeviceToDevice, s));
        });
        all_pass = report_test1("T1  D2D hipMemcpyAsync after 64 any-order kernels ", st, TRIALS) && all_pass;
    }

    // ---------------- Test 1b: peer copy to another device ----------------
    if (ndev >= 2) {
        const int other = (dev + 1) % ndev;
        int can = 0;
        CHECK(hipDeviceCanAccessPeer(&can, dev, other));
        hipError_t pe = hipDeviceEnablePeerAccess(other, 0);
        if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) printf("E0c T1b hipDeviceEnablePeerAccess(%d): %s (continuing)\n", other, hipGetErrorString(pe));
        (void)hipGetLastError();
        int* peer_dst = nullptr;
        CHECK(hipSetDevice(other));
        CHECK(hipMalloc(&peer_dst, BUF_BYTES));
        CHECK(hipMemset(peer_dst, 0, BUF_BYTES));
        CHECK(hipDeviceSynchronize());
        CHECK(hipSetDevice(dev));
        printf("E0c T1b peer copy device %d -> device %d (canAccessPeer=%d)\n", dev, other, can);
        T1Stats st = run_test1(src, peer_dst, dev, other, s, ticks, TRIALS, h, [&]() {
            CHECK(hipMemcpyPeerAsync(peer_dst, other, src, dev, BUF_BYTES, s));
        });
        all_pass = report_test1("T1b hipMemcpyPeerAsync after 64 any-order kernels ", st, TRIALS) && all_pass;
        CHECK(hipSetDevice(other));
        CHECK(hipFree(peer_dst));
        CHECK(hipSetDevice(dev));
    } else {
        printf("E0c T1b hipMemcpyPeerAsync: SKIPPED (only %d device visible)\n", ndev);
    }

    // ---------------- Test 2: any-order kernel right after a 512 MB D2D copy ----------------
    {
        int *big_src = nullptr, *big_dst = nullptr, *result = nullptr;
        CHECK(hipMalloc(&big_src, BIG_BYTES));
        CHECK(hipMalloc(&big_dst, BIG_BYTES));
        CHECK(hipMalloc(&result, 2 * sizeof(int)));
        CHECK(hipMemset(big_dst, 0, BIG_BYTES));
        CHECK(hipDeviceSynchronize());
        const int base_index = (int)(BIG_INTS - TAIL_INTS);
        const int* tail = big_dst + (BIG_INTS - TAIL_INTS);
        const dim3 tail_grid((TAIL_INTS + BLOCK - 1) / BLOCK), tail_block(BLOCK);
        hipEvent_t e0, e1;
        CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

        // measure the copy alone once
        fill_pattern<<<4096, BLOCK, 0, s>>>(big_src, 1, BIG_INTS);
        CHECK(hipStreamSynchronize(s));
        CHECK(hipEventRecord(e0, s));
        CHECK(hipMemcpyAsync(big_dst, big_src, BIG_BYTES, hipMemcpyDeviceToDevice, s));
        CHECK(hipEventRecord(e1, s));
        CHECK(hipStreamSynchronize(s));
        float copy_ms = 0.f; CHECK(hipEventElapsedTime(&copy_ms, e0, e1));
        printf("E0c T2  512 MB D2D copy alone: %.2f ms (%.0f GB/s)\n", copy_ms, BIG_BYTES / (copy_ms * 1e-3) / 1e9);

        for (int mode = 0; mode < 2; ++mode) {   // mode 0: any-order kernel (the experiment), mode 1: plain kernel (control)
            const int trials = (mode == 0) ? TRIALS : CONTROL_TRIALS;
            int stale_trials = 0, min_cnt = TAIL_INTS + 1, max_cnt = 0;
            long long sum_cnt = 0;
            for (int trial = 0; trial < trials; ++trial) {
                const int seed = 1000 + 7 * trial + 3 + mode * 100000;
                CHECK(hipMemsetAsync(result, 0, 2 * sizeof(int), s));
                fill_pattern<<<4096, BLOCK, 0, s>>>(big_src, seed, BIG_INTS);   // plain launch
                CHECK(hipStreamSynchronize(s));

                CHECK(hipMemcpyAsync(big_dst, big_src, BIG_BYTES, hipMemcpyDeviceToDevice, s));
                if (mode == 0)
                    hipExtLaunchKernelGGL(check_tail, tail_grid, tail_block, 0, s, nullptr, nullptr, hipExtAnyOrderLaunch,
                                          tail, seed, base_index, TAIL_INTS, result);
                else
                    check_tail<<<tail_grid, tail_block, 0, s>>>(tail, seed, base_index, TAIL_INTS, result);
                CHECK(hipGetLastError());
                CHECK(hipStreamSynchronize(s));
                join_kernel<<<1, 1, 0, s>>>();
                CHECK(hipDeviceSynchronize());
                int cnt = 0;
                CHECK(hipMemcpy(&cnt, result, sizeof(int), hipMemcpyDeviceToHost));
                if (cnt > 0) { ++stale_trials; sum_cnt += cnt; if (cnt < min_cnt) min_cnt = cnt; if (cnt > max_cnt) max_cnt = cnt; }
            }
            if (mode == 0) {
                printf("E0c T2  any-order kernel reading last 1 MB right after 512 MB D2D copy: trials=%d stale_trials=%d", trials, stale_trials);
                if (stale_trials > 0) printf(" (mismatched ints per stale trial: min=%d max=%d avg=%lld of %d)", min_cnt, max_cnt, sum_cnt / stale_trials, TAIL_INTS);
                printf(" PASS\n");
                printf("E0c T2  OVERTAKE: %s\n", stale_trials > 0 ? "yes" : "no");
            } else {
                const bool ok = (stale_trials == 0);
                printf("E0c T2c control: plain kernel after the same copy: trials=%d stale_trials=%d %s\n", trials, stale_trials, ok ? "PASS" : "FAIL");
                all_pass = all_pass && ok;
            }
        }
        CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
        CHECK(hipFree(big_src)); CHECK(hipFree(big_dst)); CHECK(hipFree(result));
    }

    printf("E0c RESULT: %s\n", all_pass ? "PASS" : "FAIL");
    CHECK(hipHostFree(h));
    CHECK(hipFree(src)); CHECK(hipFree(dst));
    CHECK(hipStreamDestroy(s));
    return all_pass ? 0 : 1;
}
