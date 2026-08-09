# llama.cpp prefill optimisation for CDNA2 (MI210 / gfx90a)

Patches that raise prompt-processing throughput on 2× AMD Instinct MI210 by
**~23%**, measured on NVIDIA-Nemotron-3-Super-120B-A12B (hybrid Mamba-2 +
attention MoE) at `i1-Q4_K_M`.

Base commit: `67b9b0e` (`llama-arch: fix DeepSeek4 APE tensor op (#25945)`,
2026-07-22). Branch: `ssd-cdna2`.

## Results

`llama-bench`, 2× MI210, `-b 4096 -ub 2048 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 -t 24 -r 2`

| build | pp4096 (t/s) | pp16384 (t/s) | vs base |
|---|---:|---:|---:|
| base `67b9b0e` | 1366 | 1775 | — |
| + chunked SSD on CDNA | 1624.07 ± 0.81 | 2114.21 ± 0.73 | +19.1% |
| + stream-k disabled | 1683.57 ± 0.40 | **2192.26 ± 0.84** | **+23.5%** |

Reference point: vLLM with the AITER fast paths reaches 4,070 t/s at 16k on the
same hardware with an AWQ-INT4 Nemotron. The gap narrows from 2.29× to 1.86×.
These patches do not close it; see "What is left" below.

## Patch 1 — enable the chunked SSD Mamba-2 path on CDNA

**Files:** `ggml/src/ggml-cuda/ssm-scan.cu` (cherry-pick of upstream `b62b350`
plus a guard change)

Upstream PR #22675 (merged 2026-07-28, six days after this tree's base commit)
replaces the sequential SSM scan with a chunked State-Space-Duality
formulation. Per chunk, the intra-chunk output and the chunk-final state become
batched GEMMs; only a short scan over `n_tok / 256` chunk boundaries remains
sequential. The heavy GEMMs run FP16-in / FP32-accumulate through
`cublasGemmStridedBatchedEx`.

It was gated off for HIP:

```c
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)   // kernels + dispatch
...
const bool use_ssd = ... && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
```

The PR author states the change "does not affect ... HIP" — a scoping decision,
not a finding that it cannot work on AMD. Nobody had tried it.

**Why it should work on CDNA2.** Profiling pp4096 with `rocprofv3` put
`ssm_scan_f32_group` at **22.8% of GPU time** (1759 ms of 7703 ms, 160 calls,
11 ms each) — the single largest kernel, scalar FP32, with zero matrix-core
use. The SSD path converts that work into FP16 GEMMs, which on gfx90a is the
181 TFLOPS `v_mfma_f32_16x16x16f16` path rather than the 22.6 TFLOPS vector
path. Every cuBLAS symbol involved already has a hipBLAS alias in
`ggml-cuda/vendors/hip.h`, and ggml exercises those same aliases today in
`ggml_cuda_mul_mat_batched_cublas`.

**The change** admits HIP to both preprocessor guards and adds CDNA to the
runtime capability test, leaving NVIDIA's condition untouched:

```c
&& ((GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING)
    || GGML_CUDA_CC_IS_CDNA(cc))
```

Scoped to CDNA, **not** blanket AMD: RDNA's matrix cores use WMMA with different
tile shapes and are entirely unvalidated here.

CUB is deliberately *not* enabled. `USE_CUB` stays HIP-excluded and the file's
shared-memory sequential-scan fallback is used instead. Enabling hipCUB pulls in
a header collision with ggml's own `__trap` macro — the reason upstream PR
#26388 stalled. Note the `USE_CUB` line on line 1 *contains* the same guard text
as the two we edit, so any patch tooling must anchor to end-of-line or it will
silently enable hipCUB as well.

**Verification.** `test-backend-ops -o SSM_SCAN`: 7/7 on both MI210s, including
the multi-chunk shapes (`n_seq_tokens` = 256, 512, 300) that actually exercise
the SSD path past its 128-token threshold. `rocprofv3` after the change shows
`ssm_ssd_pre_matmul_kernel<256, __half>` dispatching and new
`Cijk_..._MI16x16x16x1_...` kernels — i.e. the FP16 matrix cores are in use, as
intended.

Effect on the profile:

| | before | after |
|---|---:|---:|
| total GPU kernel time | 7703 ms | 6538 ms |
| SSM scan | 22.8% (1759 ms) | 1.0% (67 ms) |
| quantized GEMM (MMQ) | 45.2% | 53.3% |

Total SSM cost including the new GEMM and mask kernels is ~540 ms, down from
1759 ms — a 3.2× reduction on that component.

## Patch 2 — disable stream-k for K-quants in the CDNA MMQ config

**File:** `ggml/src/ggml-cuda/mmq-config-cdna.cuh`

Every real `CASE` entry in this file sets `stream_k = true` (8th positional
argument); only the unreachable `GGML_TYPE_COUNT` sentinel is false. A single
`mmq-config-cdna.cuh` covers all CDNA generations, so gfx90a inherits whatever
was tuned elsewhere.

Upstream PR #26199, which retuned the RDNA configs, reports: *"I have also found
that stream_k true helps a lot for Dense models and hurts MoE models."* This
workload is MoE, and after patch 1 MMQ is 53.3% of prefill GPU time.

Stream-k splits the K dimension across more workgroups than there are output
tiles, then reconciles partial sums in a second pass
(`mul_mat_q_stream_k_fixup`). It exists to fill a GPU that has too few output
tiles to saturate it. In an MoE prefill each expert already produces many tiles,
so the fixup pass and the extra global-memory traffic for partial accumulators
are overhead.

Scoped to the K-quants (Q2_K–Q6_K, 35 entries) because that is what an
`i1-Q4_K_M` model dispatches; leaving other types alone keeps the result
attributable.

**Caveat, stated plainly:** the stream-k/MoE evidence is from RDNA3.5/RDNA4, not
CDNA2. No CDNA stream-k benchmark exists upstream. This was an experiment that
happened to pay off (+3.7%), not a transfer of a known result.

**Verification.** `test-backend-ops -o MUL_MAT_ID`: 790/790 on both MI210s.

## Verifying correctness, not just throughput

Every change here was gated on reading generated tokens, not only on the
benchmark number. This is not ceremony: an earlier experiment in this project
produced a 3,820 t/s benchmark from a kernel that was emitting garbage, and the
number was published before anyone read the output.

The specific risk with patch 1 is that the SSD path chains batched GEMMs with
`beta=1` accumulation for inter-chunk state propagation and materialises a
causal decay mask in a helper kernel. A wrong transpose flag, stride, or
alpha/beta does not crash — it propagates a subtly wrong SSM state and yields
fluent, confident, wrong text.

Procedure used: `llama-server`, `temperature 0`, a prompt of 240 tokens (past
the 128-token SSD threshold), and read the answer for technical correctness —
not merely for well-formed prose. The model was asked to explain why prefill is
compute-bound and decode is bandwidth-bound; the answer had to use the figures
given in the prompt correctly, which a corrupted SSM state would not do.

Because of the multi-GPU fault documented below, the multi-request correctness
runs were done on a single card (`ROCR_VISIBLE_DEVICES=0`, `-ngl 62`, remaining
layers on CPU). Three sequential requests returned correct prose and correct
Python. Single-request verification was additionally done on both cards.

## Known pre-existing issue: multi-GPU fault on sequential requests

**Not caused by these patches**, but you will hit it, so it is documented here.

Running `llama-server` across both MI210s (`-sm layer`), the *second* sequential
request faults:

```
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x... Reason: Unknown.
```

The first request completes and returns correct output; the next one launches
(`slot launch_slot_: id 3 | task 263`) and the GPU faults. The server process
survives and keeps answering `/health` with `ok`, so it looks alive while being
unable to serve.

Isolated by bisecting configuration rather than assuming:

| build | GPUs | result |
|---|---|---|
| base `67b9b0e`, unpatched | 2 | **faults on request 2** |
| + SSD | 2 | faults on request 2 |
| + SSD + stream-k off | 2 | faults on request 2 |
| + SSD | 1 (`ROCR_VISIBLE_DEVICES=0`, `-ngl 62`) | 3 sequential requests clean |

Since the unpatched baseline faults identically, this is pre-existing on this
tree and unrelated to the SSD or stream-k changes. It is specific to the
multi-GPU split: the same build on a single card handles repeated requests
without incident.

`llama-bench` does **not** surface it — it ran many prefills across both cards
without a fault — which suggests the problem is in state reuse between requests
rather than in any prefill kernel. That also means benchmark results here are
unaffected, and it is why correctness was additionally verified on a single
card.

Not yet root-caused. Anyone picking this up should start by bisecting llama.cpp
between `67b9b0e` and current master with a two-request `llama-server` script on
two cards.

## Tested and rejected

Recorded so they are not re-attempted.

| change | result |
|---|---|
| Extend CDNA3's rocBLAS carve-out to CDNA2 in `mmq.cu` (`ggml_cuda_should_use_mmq` returning true for Q4_K/Q5_K at any `ne11`) | **6.5% slower** (1277 vs 1366 t/s at pp4096). rocBLAS genuinely beats MMQ at `ne11=2048` on gfx90a. |
| `ROCBLAS_USE_HIPBLASLT=1` | no-op |
| `-sm row` | unsupported on ROCm |
| Crossing the attention MFMA gate by batching (`fattn.cu` requires `Q->ne[1] * gqa_ratio > 16`; Nemotron's gqa_ratio is exactly 16 at batch 1) | no gain; flash attention is only 0.9% of prefill |
| `-ub 4096` | slower than 2048 |
| `-ub 1024` | better at pp4096, worse at pp16384 |
| W4A8 weights | dead end on CDNA2 for two independent reasons: `mfma.i32.16x16x32.i8` (K=32) is MI300-only and fails to select on gfx90a, and CDNA2 gives INT8 and BF16 the *same* 181 TOPS peak, so there is no throughput to win |

Also confirmed already optimal at base, so not worth revisiting: MMQ already
uses int8 MFMA (`mfma_i32_16x16x16i8` via `AMD_MFMA_AVAILABLE`), CUDA graphs are
active (188 graph reuses per request), `GGML_HIP_MMQ_MFMA=ON`.

## What is left

MMQ is now 53.3% of prefill and is the obvious next target. Untested ideas, in
rough order of expected value:

1. **Retune `SSM_SSD_CHUNK_SIZE`** (currently 256, a compile-time `#define`).
   It was tuned against cuBLAS on NVIDIA; rocBLAS's kernel-selection sweet spots
   on gfx90a differ. Each value costs a rebuild.
2. **Extend the stream-k experiment** beyond K-quants, and sweep the other MMQ
   config fields (`nthreads`, `occupancy`, `I`/`J` tile shape) for CDNA2
   specifically — the single shared CDNA config file is unlikely to be optimal
   for all three CDNA generations.
3. **`mm_ids_helper`** was 4.5% at base; re-measure its share now.
4. `ggml-cuda.cu` `[TAG_MUL_MAT_ID_CUDA_GRAPHS]` disables CUDA graphs for MoE
   `MUL_MAT_ID` above `mmvq_mmid_max`. At prefill batch sizes this op falls out
   of graph capture; this is inherent rather than a regression, but worth
   confirming the cost.

An `ssm_scan_f32_group` micro-optimisation was investigated and abandoned: the
kernel indexes lanes with the hardcoded `WARP_SIZE` of 32 while gfx90a's
wavefront is 64, so `warp_reduce_sum` takes a slow path once per token. Making
it wavefront-aware requires changing `c_factor` — which serves double duty as
both warps-per-block and state-elements-per-lane — along with the grid
dimensions, or the kernel reads `state[]` out of bounds. Patch 1 makes this moot
for prefill: the scalar kernel now handles only sequences under 128 tokens.

## Upstream

Patch 1 is worth sending upstream — it enables an existing, NVIDIA-validated
feature on hardware whose vendor simply was not in scope for the original PR.
Before submitting, it would need testing on CDNA3 (MI300) as well, since the
gate as written admits all of CDNA and only CDNA2 has been measured here.
