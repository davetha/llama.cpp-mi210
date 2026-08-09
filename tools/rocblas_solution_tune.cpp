// Enumerate and time every Tensile solution rocBLAS can use for the dense FP16
// GEMM shapes llama.cpp issues on gfx90a, and report which one wins.
//
// WHY THIS EXISTS. Profiling put the rocBLAS dense path at ~23% of prefill GPU
// time on 2x MI210, running at 44-56% of the card's 181 TFLOPS FP16 peak.
// rocBLAS picks a Tensile kernel per problem from its shipped library; the pick
// is a heuristic, and ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH exists precisely
// because it is sometimes wrong.
//
// rocblas-bench's --solution_index cannot be used to search for a better pick:
// with the default --algo 0 the index is accepted and silently IGNORED (passing
// 99999 or -1 changes nothing), and with --algo 1 every small index returns
// rocblas_status_invalid_value. Valid indices are not 1..N -- they are specific
// opaque values that must be obtained from rocblas_gemm_ex_get_solutions.
//
// So this queries the real list, then times each candidate against the default.
//
// Build (inside the bench image):
//   hipcc -O2 -o rocblas_solution_tune rocblas_solution_tune.cpp -lrocblas
//
// Reading the output: a winner is only interesting if it beats the default by
// more than run-to-run noise, which on this box is about +/-0.5%. Anything
// under ~2% is not worth acting on.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
// rocblas_gemm_ex_get_solutions is gated behind the beta-features macro; without
// it the declaration is not visible and the call fails to compile.
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at line %d\n", (int)e, __LINE__); exit(1); } } while (0)

struct Shape { int m, n, k; const char* label; };

// The four dense FP16 GEMMs captured from a real prefill with ROCBLAS_LAYER=2.
// transA=T, transB=N; n scales with the micro-batch, so these are at -ub 2048.
static const Shape SHAPES[] = {
    {4096, 2048, 8192, "4096x2048x8192"},
    {4096, 2048, 5376, "4096x2048x5376"},
    {5376, 2048, 4096, "5376x2048x4096"},
    { 512, 2048, 4096, "512x2048x4096"},
};

// Time one gemm_ex call. solution_index < 0 means "let rocBLAS choose"
// (algo_standard), which is the behaviour we are trying to beat.
static double time_gemm(rocblas_handle handle, const Shape& s,
                        const void* dA, const void* dB, void* dC, void* dD,
                        int solution_index, int iters, bool* ok) {
    const float alpha = 1.0f, beta = 0.0f;
    const rocblas_operation tA = rocblas_operation_transpose;
    const rocblas_operation tB = rocblas_operation_none;
    const int lda = s.k, ldb = s.k, ldc = s.m, ldd = s.m;

    const rocblas_gemm_algo algo = solution_index < 0
        ? rocblas_gemm_algo_standard : rocblas_gemm_algo_solution_index;
    const int32_t idx = solution_index < 0 ? 0 : solution_index;

    // One untimed call: it both validates the index and absorbs any lazy
    // per-kernel setup that would otherwise land in the first timed iteration.
    rocblas_status st = rocblas_gemm_ex(handle, tA, tB, s.m, s.n, s.k, &alpha,
        dA, rocblas_datatype_f16_r, lda, dB, rocblas_datatype_f16_r, ldb, &beta,
        dC, rocblas_datatype_f32_r, ldc, dD, rocblas_datatype_f32_r, ldd,
        rocblas_datatype_f32_r, algo, idx, 0);
    if (st != rocblas_status_success) { *ok = false; return 0.0; }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    CHECK_HIP(hipEventCreate(&e0));
    CHECK_HIP(hipEventCreate(&e1));
    CHECK_HIP(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) {
        rocblas_gemm_ex(handle, tA, tB, s.m, s.n, s.k, &alpha,
            dA, rocblas_datatype_f16_r, lda, dB, rocblas_datatype_f16_r, ldb, &beta,
            dC, rocblas_datatype_f32_r, ldc, dD, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, algo, idx, 0);
    }
    CHECK_HIP(hipEventRecord(e1));
    CHECK_HIP(hipEventSynchronize(e1));
    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
    CHECK_HIP(hipEventDestroy(e0));
    CHECK_HIP(hipEventDestroy(e1));
    *ok = true;
    return (double)ms * 1000.0 / iters;   // microseconds per call
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 20;

    rocblas_handle handle;
    if (rocblas_create_handle(&handle) != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle failed\n");
        return 1;
    }

    for (const Shape& s : SHAPES) {
        // Allocate once per shape; contents do not affect kernel selection or
        // timing for these types, only correctness, which is not under test here.
        void *dA, *dB, *dC, *dD;
        CHECK_HIP(hipMalloc(&dA, (size_t)s.m * s.k * 2));
        CHECK_HIP(hipMalloc(&dB, (size_t)s.k * s.n * 2));
        CHECK_HIP(hipMalloc(&dC, (size_t)s.m * s.n * 4));
        CHECK_HIP(hipMalloc(&dD, (size_t)s.m * s.n * 4));
        CHECK_HIP(hipMemset(dA, 0, (size_t)s.m * s.k * 2));
        CHECK_HIP(hipMemset(dB, 0, (size_t)s.k * s.n * 2));

        const float alpha = 1.0f, beta = 0.0f;
        const int lda = s.k, ldb = s.k, ldc = s.m, ldd = s.m;

        // Two-call protocol: NULL list_array asks for the count, then fill it.
        rocblas_int n_sol = 0;
        rocblas_status st = rocblas_gemm_ex_get_solutions(handle,
            rocblas_operation_transpose, rocblas_operation_none, s.m, s.n, s.k, &alpha,
            dA, rocblas_datatype_f16_r, lda, dB, rocblas_datatype_f16_r, ldb, &beta,
            dC, rocblas_datatype_f32_r, ldc, dD, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, nullptr, &n_sol);
        if (st != rocblas_status_success) {
            printf("%-16s get_solutions(count) failed: status %d\n", s.label, (int)st);
            continue;
        }

        std::vector<rocblas_int> sols(n_sol > 0 ? n_sol : 1);
        if (n_sol > 0) {
            st = rocblas_gemm_ex_get_solutions(handle,
                rocblas_operation_transpose, rocblas_operation_none, s.m, s.n, s.k, &alpha,
                dA, rocblas_datatype_f16_r, lda, dB, rocblas_datatype_f16_r, ldb, &beta,
                dC, rocblas_datatype_f32_r, ldc, dD, rocblas_datatype_f32_r, ldd,
                rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, sols.data(), &n_sol);
            if (st != rocblas_status_success) {
                printf("%-16s get_solutions(fill) failed: status %d\n", s.label, (int)st);
                continue;
            }
        }

        bool ok = false;
        const double t_def = time_gemm(handle, s, dA, dB, dC, dD, -1, iters, &ok);
        const double gflop = 2.0 * s.m * s.n * s.k / 1e9;

        printf("\n=== %s : %d candidate solutions ===\n", s.label, (int)n_sol);
        printf("  default (algo_standard): %8.1f us  %7.1f TFLOPS\n",
               t_def, gflop / t_def * 1e6 / 1e3);

        double best_t = t_def;
        int best_idx = -1;
        int tried = 0;
        for (rocblas_int idx : sols) {
            bool sok = false;
            const double t = time_gemm(handle, s, dA, dB, dC, dD, idx, iters, &sok);
            if (!sok) continue;
            tried++;
            if (t < best_t) { best_t = t; best_idx = idx; }
        }

        if (best_idx >= 0) {
            printf("  BEST solution_index %d: %8.1f us  %7.1f TFLOPS  (%.1f%% faster than default)\n",
                   best_idx, best_t, gflop / best_t * 1e6 / 1e3,
                   (t_def / best_t - 1.0) * 100.0);
        } else {
            printf("  no candidate beat the default (%d of %d ran)\n", tried, (int)n_sol);
        }

        hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dD);
    }

    rocblas_destroy_handle(handle);
    return 0;
}
