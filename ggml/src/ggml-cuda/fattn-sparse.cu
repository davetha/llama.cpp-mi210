// DSV4-Flash gather-sparse flash attention (CSA layers).
//
// Each query attends a dense prefix [0, n_dense) (the raw/SWA window) followed by
// its own top-k gathered compressed keys (index + n_dense into K). This skips the
// masked-but-computed compressed keys that the dense flash-attn kernel wastes on
// ratio-4 CSA layers (n_csa >> top_k at long context).
//
// Specialized for the DSV4 latent-MLA/MQA case: single KV head, f16 K/V where V
// ALIASES K (same data, DV==DK==512), f32 Q, f16 mask, f32 sinks, i32 top-k indices,
// f32 output. Anything else -> supported() rejects and the op falls back to CPU.
//
// Layout: one block handles NWAVES query heads of ONE query token (one 32-lane wave
// per head). top-k / mask are per query token (identical across the block's heads),
// so gathered K rows are staged into shared memory ONCE and reused by all heads.
// Keys are processed in tiles of TK so the block barriers once per tile instead of
// once per key -- the kernel is latency-bound (serial online softmax + barriers),
// not compute-bound, so cutting barriers and gather traffic is what matters. V is
// read straight from the staged K tile (MLA alias), halving the gather. Online
// softmax (f32) matches the CPU reference in ggml-cpu/ops.cpp exactly.

#include "common.cuh"
#include "fattn-sparse.cuh"
#include <type_traits>

template <int D, int DPT, int NWAVES, int TK, bool KV_Q8>
static __global__ void __launch_bounds__(NWAVES*WARP_SIZE)
flash_attn_sparse_kernel(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ maskp,
        const char * __restrict__ sinksp,
        const int32_t * __restrict__ top_k,
        float * __restrict__ dst,
        const float scale,
        const int   diag,
        const int   n_kv,
        const int   n_top_k,
        const int   n_dense,
        const int   n_head,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk1, const int64_t nbk3,
        const int64_t nbm1, const int64_t nbm3,
        const int64_t nbtk1, const int64_t nbtk3,
        const int64_t nbd1, const int64_t nbd2, const int64_t nbd3) {
    const int iq1  = blockIdx.x;                 // query token
    const int seq  = blockIdx.z;                 // stream
    const int wid  = threadIdx.x / WARP_SIZE;    // wave = head offset within block
    const int lane = threadIdx.x % WARP_SIZE;
    const int head = blockIdx.y * NWAVES + wid;  // query head
    const bool head_active = head < n_head;
    constexpr int nthreads = NWAVES*WARP_SIZE;

    __shared__ half  ksh[TK][D];
    __shared__ int   ic_sh[TK];
    __shared__ float mv_sh[TK];

    const half    * m_row  = maskp ? (const half *)(maskp + iq1*nbm1 + seq*nbm3) : nullptr;
    const int32_t * tk_row = (const int32_t *)((const char *) top_k + iq1*nbtk1 + seq*nbtk3);

    float qreg[DPT];
    if (head_active) {
        const float * q_row = (const float *)(Q + iq1*nbq1 + head*nbq2 + seq*nbq3);
#pragma unroll
        for (int i = 0; i < DPT; ++i) {
            qreg[i] = q_row[lane + i*WARP_SIZE];
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) {
        acc[i] = 0.0f;
    }

    const int total = n_dense + n_top_k;
    for (int t0 = 0; t0 < total; t0 += TK) {
        const int nt = min(TK, total - t0);

        // resolve gathered index + mask for each key in the tile (per query, uniform)
        if ((int) threadIdx.x < nt) {
            const int kk = t0 + threadIdx.x;
            int ic;
            if (kk < n_dense) {
                ic = kk;
            } else {
                const int csa = tk_row[kk - n_dense];
                ic = (csa >= 0 && csa < n_kv - n_dense) ? n_dense + csa : -1;
            }
            float mv = (ic < 0) ? -INFINITY : (m_row ? __half2float(m_row[ic]) : 0.0f);
            ic_sh[threadIdx.x] = (mv == -INFINITY) ? -1 : ic; // invalid/masked -> -1
            mv_sh[threadIdx.x] = mv;
        }
        __syncthreads();

        // stage the tile's K rows into shared memory as 128-bit (int4 = 8 half) loads:
        // fewer global transactions on the scattered gather, which dominates the kernel.
        if constexpr (KV_Q8) {
            // q8_0 KV: dequantize into the SAME f16 shared tile, so every downstream
            // consumer (QK dot and the PV accumulate that aliases V onto K) is
            // unchanged. Global traffic drops from 2 B/elem to ~1.06 B/elem, and this
            // kernel is gather/latency bound, so the smaller gather is the point.
            static_assert(D % QK8_0 == 0, "D must be a multiple of QK8_0");
            constexpr int NB = D / QK8_0;
            for (int idx = threadIdx.x; idx < nt*NB; idx += nthreads) {
                const int kloc = idx / NB;
                const int b    = idx - kloc*NB;
                const int ic   = ic_sh[kloc];
                if (ic >= 0) {
                    const block_q8_0 * blk =
                        (const block_q8_0 *)((const char *) K + ic*nbk1 + seq*nbk3) + b;
                    const float d = __half2float(blk->d);
#pragma unroll
                    for (int j = 0; j < QK8_0; ++j) {
                        ksh[kloc][b*QK8_0 + j] = __float2half(d * (float) blk->qs[j]);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < QK8_0; ++j) {
                        ksh[kloc][b*QK8_0 + j] = __float2half(0.0f);
                    }
                }
            }
        } else {
            static_assert(D % 8 == 0, "D must be a multiple of 8 for int4 staging");
            constexpr int D8 = D / 8;
            for (int idx = threadIdx.x; idx < nt*D8; idx += nthreads) {
                const int kloc = idx / D8;
                const int d8   = idx - kloc*D8;
                const int ic   = ic_sh[kloc];
                const int4 v = ic >= 0 ? *(const int4 *)((const char *)(K + ic*nbk1 + seq*nbk3) + d8*16)
                                       : make_int4(0, 0, 0, 0);
                *(int4 *)&ksh[kloc][d8*8] = v;
            }
        }
        __syncthreads();

        if (head_active) {
            for (int kloc = 0; kloc < nt; ++kloc) {
                if (ic_sh[kloc] < 0) {
                    continue;
                }
                const float mv = mv_sh[kloc];

                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < DPT; ++i) {
                    partial += qreg[i] * __half2float(ksh[kloc][lane + i*WARP_SIZE]);
                }
                float s = (diag == 1 ? partial : warp_reduce_sum(partial))*scale + mv; // diag1: skip reduce (timing only)

                const float m_new = fmaxf(m, s);
                const float corr  = expf(m - m_new);
                const float p     = expf(s - m_new);
                l = l*corr + p;
                if (diag != 2) { // diag2: skip PV accumulate (timing only)
#pragma unroll
                for (int i = 0; i < DPT; ++i) {
                    acc[i] = acc[i]*corr + p*__half2float(ksh[kloc][lane + i*WARP_SIZE]); // V aliases K
                }
                }
                m = m_new;
            }
        }
        __syncthreads(); // before the next tile overwrites the LDS
    }

    if (!head_active) {
        return;
    }

    if (sinksp) {
        const float sink = ((const float *) sinksp)[head];
        const float m_new = fmaxf(m, sink);
        const float corr  = expf(m - m_new);
        const float p     = expf(sink - m_new);
        l = l*corr + p;
#pragma unroll
        for (int i = 0; i < DPT; ++i) {
            acc[i] *= corr;
        }
    }

    const float inv = l > 0.0f ? 1.0f/l : 0.0f;
    float * o = (float *)((char *) dst + head*nbd1 + iq1*nbd2 + seq*nbd3);
#pragma unroll
    for (int i = 0; i < DPT; ++i) {
        o[lane + i*WARP_SIZE] = acc[i]*inv;
    }
}

void ggml_cuda_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * top_k = dst->src[5];

    GGML_ASSERT(top_k && top_k->type == GGML_TYPE_I32);
    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT((K->type == GGML_TYPE_F16  && V->type == GGML_TYPE_F16) ||
                (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0));
    GGML_ASSERT(V->data == K->data && "sparse kernel requires MLA V aliasing K");
    const bool kv_q8 = (K->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int D = Q->ne[0];
    GGML_ASSERT(D == K->ne[0] && D == V->ne[0] && D == dst->ne[0]);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int n_dense = ((const int32_t *) dst->op_params)[4];
    const int diag = []{ const char * e = getenv("GGML_DSV4_DIAG"); return e ? atoi(e) : 0; }();

    const int n_q      = Q->ne[1];
    const int n_head   = Q->ne[2];
    const int n_stream = Q->ne[3];
    const int n_kv     = K->ne[1];
    const int n_top_k  = top_k->ne[0];

    int nwaves = 1;
    // decode (n_q==1) wants LOW nwaves => more blocks for the single query; prefill wants
    // high nwaves (max gather amortization across heads). n_q-aware.
    int nwaves_cap = 16;
    if (n_q == 1) {
        nwaves_cap = 4;
        const char * ed = getenv("GGML_DSV4_SPARSE_DECODE_NWAVES"); if (ed) { int v = atoi(ed); if (v >= 1 && v <= 16) nwaves_cap = v; }
    }
    for (int g = nwaves_cap; g >= 1; --g) {
        if (n_head % g == 0) { nwaves = g; break; }
    }
    const dim3 grid(n_q, (n_head + nwaves - 1)/nwaves, n_stream);
    cudaStream_t stream = ctx.stream();

    const char * Qd = (const char *) Q->data;
    const char * Kd = (const char *) K->data;
    const char * Md = mask  ? (const char *) mask->data  : nullptr;
    const char * Sd = sinks ? (const char *) sinks->data : nullptr;
    const int32_t * Td = (const int32_t *) top_k->data;
    float * Dd = (float *) dst->data;

    constexpr int TK = 16; // keys staged per barrier round

    auto launch = [&](auto d_tag, auto w_tag) {
        constexpr int DD = decltype(d_tag)::value;
        constexpr int WW = decltype(w_tag)::value;
        constexpr int DPT = DD / WARP_SIZE;
        if (kv_q8) {
            flash_attn_sparse_kernel<DD, DPT, WW, TK, true><<<grid, WW*WARP_SIZE, 0, stream>>>(
                Qd, Kd, Md, Sd, Td, Dd, scale, diag, n_kv, n_top_k, n_dense, n_head,
                Q->nb[1], Q->nb[2], Q->nb[3],
                K->nb[1], K->nb[3],
                mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0,
                top_k->nb[1], top_k->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            return;
        }
        flash_attn_sparse_kernel<DD, DPT, WW, TK, false><<<grid, WW*WARP_SIZE, 0, stream>>>(
            Qd, Kd, Md, Sd, Td, Dd, scale, diag, n_kv, n_top_k, n_dense, n_head,
            Q->nb[1], Q->nb[2], Q->nb[3],
            K->nb[1], K->nb[3],
            mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0,
            top_k->nb[1], top_k->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
    };

    auto launch_d = [&](auto d_tag) {
        switch (nwaves) {
            case 16: launch(d_tag, std::integral_constant<int,16>{}); break;
            case 8:  launch(d_tag, std::integral_constant<int, 8>{}); break;
            case 4:  launch(d_tag, std::integral_constant<int, 4>{}); break;
            case 2:  launch(d_tag, std::integral_constant<int, 2>{}); break;
            default: launch(d_tag, std::integral_constant<int, 1>{}); break;
        }
    };

    switch (D) {
        case 512: launch_d(std::integral_constant<int, 512>{}); break;
        case 576: launch_d(std::integral_constant<int, 576>{}); break;
        default: GGML_ABORT("flash_attn_sparse: unsupported head dim %d", D);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_flash_attn_ext_sparse_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * top_k = dst->src[5];
    if (!top_k || top_k->type != GGML_TYPE_I32) return false;
    if (Q->type != GGML_TYPE_F32) return false;
    const bool kv_f16 = (K->type == GGML_TYPE_F16  && V->type == GGML_TYPE_F16);
    const bool kv_q8  = (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0);
    if (!kv_f16 && !kv_q8) return false;
    if (V->data != K->data) return false;      // require MLA V aliasing K
    if (dst->type != GGML_TYPE_F32) return false;
    const int D = Q->ne[0];
    if (D != V->ne[0] || D != K->ne[0]) return false;
    if (D % WARP_SIZE != 0) return false;
    if (D != 512 && D != 576) return false;
    return true;
}
