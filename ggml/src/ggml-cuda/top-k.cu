#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE


// MI210_TOPK_GPU: multi-pass block reduction so TOP_K stays on the GPU above
// ncols 1024. See ggml_cuda_op_top_k below for why the CPU fallback is costly
// out of proportion to its arithmetic.
#define TOP_K_CHUNK 1024

// One block reduces one (row, chunk) pair: sorts up to TOP_K_CHUNK values
// descending and emits that chunk's top k_keep as (value, index) pairs.
// idx_in is null on the first pass, where indices are just global positions.
static __global__ void k_top_k_reduce(
        const float * __restrict__ vals,
        const int   * __restrict__ idx_in,
        float       * __restrict__ vals_out,
        int         * __restrict__ idx_out,
        const int n_in, const int k_keep, const int n_out) {

    const int tid   = threadIdx.x;
    const int chunk = blockIdx.x;
    const int row   = blockIdx.y;

    __shared__ float sval[TOP_K_CHUNK];
    __shared__ int   sidx[TOP_K_CHUNK];

    const int g = chunk * TOP_K_CHUNK + tid;

    if (g < n_in) {
        sval[tid] = vals[(size_t) row * n_in + g];
        sidx[tid] = idx_in ? idx_in[(size_t) row * n_in + g] : g;
    } else {
        // padding loses every comparison and can never reach the output while
        // n >= k, which top-k requires anyway
        sval[tid] = -INFINITY;
        sidx[tid] = -1;
    }
    __syncthreads();

    // bitonic sort, descending
    for (int kk = 2; kk <= TOP_K_CHUNK; kk *= 2) {
        for (int j = kk / 2; j > 0; j /= 2) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const bool up = (tid & kk) == 0;
                if (up ? (sval[tid] < sval[ixj]) : (sval[tid] > sval[ixj])) {
                    const float tv = sval[tid]; sval[tid] = sval[ixj]; sval[ixj] = tv;
                    const int   ti = sidx[tid]; sidx[tid] = sidx[ixj]; sidx[ixj] = ti;
                }
            }
            __syncthreads();
        }
    }

    if (tid < k_keep) {
        const int o = chunk * k_keep + tid;
        if (o < n_out) {
            if (vals_out) {
                vals_out[(size_t) row * n_out + o] = sval[tid];
            }
            idx_out[(size_t) row * n_out + o] = sidx[tid];
        }
    }
}

static void top_k_multi_pass(ggml_cuda_pool & pool,
                             const float * src, int * dst,
                             const int ncols, const int nrows, const int k,
                             cudaStream_t stream) {
    const int nchunks0 = (ncols + TOP_K_CHUNK - 1) / TOP_K_CHUNK;
    const size_t cap   = (size_t) nrows * nchunks0 * k;

    ggml_cuda_pool_alloc<float> va(pool, cap), vb(pool, cap);
    ggml_cuda_pool_alloc<int>   ia(pool, cap), ib(pool, cap);

    const float * cur_v = src;
    const int   * cur_i = nullptr;
    int           cur_n = ncols;
    int           parity = 0;

    // reduce until a single chunk remains
    while (cur_n > TOP_K_CHUNK) {
        const int nch   = (cur_n + TOP_K_CHUNK - 1) / TOP_K_CHUNK;
        const int n_out = nch * k;

        float * ov = parity ? vb.get() : va.get();
        int   * oi = parity ? ib.get() : ia.get();

        const dim3 grid(nch, nrows);
        k_top_k_reduce<<<grid, TOP_K_CHUNK, 0, stream>>>(cur_v, cur_i, ov, oi, cur_n, k, n_out);

        cur_v  = ov;
        cur_i  = oi;
        cur_n  = n_out;
        parity ^= 1;
    }

    // final pass writes indices straight into dst
    const dim3 grid(1, nrows);
    k_top_k_reduce<<<grid, TOP_K_CHUNK, 0, stream>>>(cur_v, cur_i, nullptr, dst, cur_n, k, k);
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    if (ncols > TOP_K_CHUNK) {
        // MI210_TOPK_GPU: above the bitonic block limit, reduce chunk-wise on
        // the GPU rather than letting the whole node fall back to the CPU.
        GGML_ASSERT(2*k <= TOP_K_CHUNK);
        top_k_multi_pass(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    }
#endif
}
