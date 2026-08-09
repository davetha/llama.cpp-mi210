# llama.cpp-mi210

> **Fork of [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)** (itself a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)) with three change sets optimized for **AMD MI210 (gfx90a / CDNA2)** inference.

This repo does **not** contain the full llama.cpp tree (too large to mirror here). Instead it ships:

```
patches/          — apply these on top of the upstream fork
modified-files/   — the exact modified files (drop-in replacements)
tools/            — patch/revert scripts and the rocprofv3 trace analyser
tests/            — KIVI2 correctness tests
BUILD.md          — how to build for gfx90a in Docker
```

---

## What changed (three logical change sets)

### 1. Per-layer KV cache types (`-ctk-cpu` / `-ctv-cpu`)  → [`patches/01-per-layer-kv-types.patch`](patches/01-per-layer-kv-types.patch)

The headline feature for **CPU-hybrid MoE** models (e.g. a 230B MoE where 25/48 expert layers are pinned to CPU via `-ot`). Previously, `-ctk` / `-ctv` applied uniformly to *every* layer, on every device. Now you can choose independently:

| Flag | Scope | Example use |
|------|-------|-------------|
| `-ctk f16 -ctv f16` | GPU layers | maximum quality where bandwidth is free |
| `-ctk-cpu turbo3 -ctv-cpu turbo3` | CPU-pinned layers | 5× compressed → 5× less DDR4 traffic |

The CPU TurboQuant path is **proven numerically correct** (cosine > 0.98 round-trip); only the GPU path is broken on gfx90a (see change set 3). So compressing *only* the CPU layers gives you the bandwidth win without touching the broken GPU kernels.

**Usage:**
```bash
llama-server -m model.gguf -ngl 999 \
  -ot "blk\.([0-9]|1[0-9]|2[0-4])\.ffn.*exps=CPU,blk\.(2[5-9]|3[0-6])\.ffn.*exps=ROCm0,blk\.(3[7-9]|4[0-8])\.ffn.*exps=ROCm1" \
  -ctk f16 -ctv f16 -ctk-cpu turbo3 -ctv-cpu turbo3 -fa on
```

**Backward compatible:** when `-ctk-cpu` / `-ctv-cpu` are unset they default to `GGML_TYPE_COUNT` (a sentinel meaning "inherit `-ctk` / `-ctv`"), so existing configs are unchanged.

**Files (12):** `common/{arg.cpp,common.cpp,common.h}`, `include/llama.h`, `src/{llama-context,llama-kv-cache,llama-kv-cache-dsa,llama-kv-cache-iswa,llama-memory-hybrid,llama-model}.cpp`, `src/{llama-kv-cache,llama-memory}.h`.

**Rotation matrix fix:** the TurboQuant rotation tensors are now created when turbo is set on *either* the GPU type *or* the CPU type, so `-ctk f16 -ctk-cpu turbo3` correctly sets up the WHT rotation.

---

### 2. KIVI 2-bit KV cache (`GGML_TYPE_KIVI2`)  → [`patches/02-kivi2-quant-type.patch`](patches/02-kivi2-quant-type.patch)

A new, hardware-agnostic 2-bit quantization type based on the [KIVI paper (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750):

- **Algorithm:** per-group (group_size = 32) min-max asymmetric quantization, 4 levels `{0,1,2,3}`, `scale = (max-min)/3`.
- **Block layout:** `block_kivi2` = 2-byte fp16 scale `d` + 2-byte fp16 min `m` + 8 bytes of packed 2-bit indices (4 per byte) = **12 bytes / 32 values = 3.0 bits/value** → 5.3× compression vs fp16.
- **Correctness:** all 4 unit tests **PASS** (exact 4-level round-trip, endpoints, constant block, random invariance). Pure scalar C — no wave-level intrinsics, so it is correct on **every** architecture including gfx90a wave64.
- **Usage:** `-ctk kivi2 -ctv kivi2` (or `-ctk-cpu kivi2` for the per-layer split above).

**Files (9):** `ggml/include/ggml.h` (enum), `ggml/src/ggml-common.h` (`block_kivi2`), `ggml/src/ggml-quants.{c,h}` (quantize/dequantize/quantize_kivi2), `ggml/src/ggml.c` (type traits), `ggml/src/ggml-cpu/ggml-cpu.c` (CPU traits + vec_dot), `ggml/src/ggml-cpu/quants.{c,h}` (dispatch), `common/arg.cpp` (enum registration).

---

### 3. TurboQuant wave64 fixes  → [`patches/03-turboquant-wave64-fixes.patch`](patches/03-turboquant-wave64-fixes.patch)

Four categories of fixes to the CUDA/HIP kernels for the 64-lane wavefronts on gfx90a (the TurboQuant kernels were written/validated for 32-lane warps):

| Fix | File | What |
|-----|------|------|
| Ballot macro | `vendors/hip.h:59` | `__ballot_sync` truncated to `uint32_t` → now `uint64_t` (root cause: lanes 32–63 silently dropped) |
| shfl width ×5 | `set-rows.cu` | 5 plain `__shfl_sync` calls missing explicit `WARP_SIZE` width → cross-block contamination when 2 blocks share a wavefront |
| turbo3 signs ballot | `set-rows.cu:381` | `uint32_t ballot` → `uint64_t` + physical-lane byte indexing |
| turbo2 signs ballot | `set-rows.cu:510` | same `uint64_t` ballot fix on the TURBO2 path |

**Status:** these are genuine, correct fixes (each changed the corruption pattern, improving output from total garbage → semi-coherent). However the corruption is **pervasive** across the TurboQuant codebase (also in `mmvq-tq.cu`, `convert.cu`, the fattn path), so a full wave64 port is still needed for GPU-correct TurboQuant. The **CPU path is fully correct** — which is exactly why change set 1 (per-layer KV types) pins turbo only to CPU layers.

The recommended platform-agnostic alternative is the **Triton** implementation in [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd) (GEMM-based WHT, zero wave64 issues).

---

### 4. Chunked SSD Mamba-2 prefill on CDNA  → [`patches/04-ssd-mamba2-prefill-cdna.patch`](patches/04-ssd-mamba2-prefill-cdna.patch)

**+19% prompt processing** on hybrid Mamba-2 models. Measured on
NVIDIA-Nemotron-3-Super-120B-A12B (`i1-Q4_K_M`, 80 GiB) across 2× MI210.

Upstream [PR #22675](https://github.com/ggml-org/llama.cpp/pull/22675) (merged
2026-07-28) replaces the sequential SSM scan with a chunked **State-Space
Duality** formulation: per chunk, the intra-chunk output and chunk-final state
become batched GEMMs (FP16 in, FP32 accumulate), leaving only a short scan over
`n_tok / 256` chunk boundaries. It is gated off for HIP:

```c
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)   // kernels + dispatch
const bool use_ssd = ... && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
```

The PR author states the change "does not affect ... HIP" — a **scoping
decision, not a technical limitation**. Nobody had tried it on AMD.

**Why CDNA2 wants this.** `rocprofv3` on a 4096-token prefill put
`ssm_scan_f32_group` at **22.8% of all GPU time** (1759 ms of 7703 ms, 160 calls
at 11 ms each) — the single largest kernel, scalar FP32, with zero matrix-core
use. It is not occupancy-limited (2048 blocks over 104 CUs), so the cost is the
work itself. The SSD path converts that into FP16 GEMMs, which on gfx90a is the
181 TFLOPS `v_mfma_f32_16x16x16f16` path rather than the 22.6 TFLOPS vector
path. Every cuBLAS symbol involved already has a hipBLAS alias in
`ggml-cuda/vendors/hip.h`, exercised today by `ggml_cuda_mul_mat_batched_cublas`
— so **the kernels compiled for HIP with no changes to the GEMM calls at all**.

The patch admits HIP to both preprocessor guards and adds CDNA to the runtime
capability test, leaving NVIDIA's condition byte-for-byte unchanged:

```c
&& ((GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING)
    || GGML_CUDA_CC_IS_CDNA(cc))
```

Scoped to CDNA, **not** blanket AMD: RDNA uses WMMA with different tile shapes
and is unvalidated here.

`USE_CUB` is deliberately left HIP-excluded; the file's shared-memory sequential
scan fallback is used instead, because hipCUB collides with ggml's own `__trap`
macro (the reason upstream PR #26388 stalled). **Watch out:** line 1 of
`ssm-scan.cu` *contains* the same guard text as the two that need editing, so
patch tooling must anchor to end-of-line or it will silently enable hipCUB too.

| | before | after |
|---|---:|---:|
| pp4096 | 1366 t/s | **1624 t/s** (+18.9%) |
| pp16384 | 1775 t/s | **2114 t/s** (+19.1%) |
| total GPU kernel time | 7703 ms | 6538 ms |
| SSM scan share | 22.8% (1759 ms) | 1.0% (67 ms) |

Total SSM cost including the new GEMM and mask kernels is ~540 ms, down from
1759 ms — a **3.2× reduction** on that component. `rocprofv3` after the change
confirms `ssm_ssd_pre_matmul_kernel<256, __half>` dispatching alongside new
`Cijk_*_MI16x16x16x1_*` kernels, i.e. the FP16 matrix cores really are in use.

**Correctness:** `test-backend-ops -o SSM_SCAN` 7/7 on both MI210s, including
the multi-chunk shapes (`n_seq_tokens` = 256, 512, 300) that exercise this path
past its 128-token threshold. Additionally verified by reading generated tokens
at `temperature 0` — see "Verifying correctness" below.

**Tunables** (compile-time `#define`s in `ssm-scan.cu`, no CLI flag):
`SSM_SSD_MIN_TOKENS` (128) and `SSM_SSD_CHUNK_SIZE` (256). The chunk size was
tuned against cuBLAS on NVIDIA; retuning it for rocBLAS on gfx90a is an obvious
next experiment, at one rebuild per value.

**Files (2):** `ggml/src/ggml-cuda/ssm-scan.cu`, `tests/test-backend-ops.cpp`.

---

### 5. Disable stream-k for K-quants on CDNA  → [`patches/05-mmq-cdna-no-streamk.patch`](patches/05-mmq-cdna-no-streamk.patch)

**+3.7% prompt processing** on MoE models.

Every real `CASE` entry in `mmq-config-cdna.cuh` sets `stream_k = true` (8th
positional argument); only the unreachable `GGML_TYPE_COUNT` sentinel is false.
A single `mmq-config-cdna.cuh` covers all CDNA generations, so gfx90a inherits
tuning done elsewhere.

Upstream [PR #26199](https://github.com/ggml-org/llama.cpp/pull/26199), which
retuned the RDNA configs, reports: *"I have also found that stream_k true helps
a lot for Dense models and hurts MoE models."* This workload is MoE, and after
change set 4 the MMQ kernels are **53.3%** of prefill GPU time.

Stream-k splits the K dimension across more workgroups than there are output
tiles, then reconciles partial sums in `mul_mat_q_stream_k_fixup`. It exists to
fill a GPU that has too few output tiles to saturate it. In an MoE prefill each
expert already produces many tiles, so the fixup pass and the extra
global-memory traffic for partial accumulators are pure overhead.

Originally scoped to the K-quants, on the assumption that an `i1-Q4_K_M` model
dispatches nothing else. **The profile disproved that.** The three hottest MMQ
kernels are types 12, 6 and 8 — `Q4_K`, **`Q5_0` and `Q8_0`** — and the latter
two are 27.4% of prefill GPU time on their own.

`Q5_0`/`Q8_0` appear in a `Q4_K_M` model because `llama-quant.cpp` silently
downgrades any tensor whose column count is not divisible by `QK_K`=256:
`Q4_K → Q5_0`, `Q6_K → Q8_0`. Nemotron-3-Super's Mamba-2 projection widths
frequently are not, so a large slice of the model is not K-quantised at all.

The hypothesis is about workload shape (MoE, many tiles per expert), not about
any property of a quant, so this covers **every** type in the table.

| | K-quants only | all types |
|---|---:|---:|
| pp4096 | 1684 t/s (+3.7%) | **1834 t/s (+13.0%)** |
| pp16384 | 2192 t/s (+3.7%) | **2374 t/s (+12.3%)** |

**Stated plainly:** the stream-k/MoE evidence is from RDNA3.5/RDNA4, *not*
CDNA2, and no CDNA stream-k benchmark exists upstream. This was an experiment
that happened to pay off, not the transfer of a known result. An earlier
experiment on the same theory — extending CDNA3's rocBLAS carve-out to CDNA2 —
came back **6.5% slower** and was discarded.

**Correctness:** `test-backend-ops -o MUL_MAT_ID` 790/790 on both MI210s.

**Files (1):** `ggml/src/ggml-cuda/mmq-config-cdna.cuh`.

---

### 6. Retune the CDNA MMQ tile: `nthreads=256`, `I=64`  → [`patches/06-mmq-cdna-tile-retune.patch`](patches/06-mmq-cdna-tile-retune.patch)

**+6.5% prompt processing.**

`mmq-config-cdna.cuh` has **never been tuned on CDNA hardware.** PR
[#24127](https://github.com/ggml-org/llama.cpp/pull/24127) refactored the MMQ
configuration into per-arch tables and transcribed the CDNA values from
pre-refactor blanket-AMD constants:

| field | value | where it came from |
|---|---|---|
| `nthreads=512` | 8 waves × 64 | `mmq_get_nwarps_host`: `amd_mfma_available(cc) ? 8 : …` — tuned for gfx942/MI300 |
| `I=128` | | `get_mmq_y_host`: `GGML_CUDA_CC_IS_AMD(cc) ? (RDNA1 ? 64 : 128)` — a **catch-all AMD value** spanning GCN through CDNA4 |
| `occupancy=1` | | changed from the pre-refactor `2`; CDNA is the only arch whose value moved |

The only AMD GPUs benchmarked in that PR were MI100, RX 6800 and a Radeon
8060S. No MI210, MI250 or MI300 numbers appear anywhere in it, and the reviewer
noted it was "pretty easy for this to have some regressions on some arch by
accident."

On gfx90a the smaller tile is simply better:

| | pp4096 | pp16384 |
|---|---:|---:|
| `nthreads=512, I=128` (upstream) | 1834 t/s | 2374 t/s |
| **`nthreads=256, I=64`** | **1962 t/s** | **2529 t/s** |
| `nthreads=128, I=32` | 1835 t/s | 2370 t/s |

It also halves LDS per workgroup — 48.2 KiB → 28.2 KiB for the Q8_1 layout
against a 64 KiB budget — which is what makes two workgroups per CU feasible at
all, and what makes change set 7's larger `J` fit if anyone revisits it.

#### Two invariants that make this dangerous to tune

**`I` is not an independent knob.** `mmq-vec-dot.cuh` hardcodes
`rows_per_warp = 16` on the MFMA path and has no loop over the row index, so the
I-extent is covered exactly by the block's warps:

> **On CDNA, `I == (nthreads/64) * 16 == nthreads/4`.**

Changing one without the other does not error. It silently computes a partial
tile — and it benchmarks *faster*, because it is doing less work:

| config | pp4096 | MUL_MAT fails | MUL_MAT_ID fails |
|---|---:|---:|---:|
| `I=64` with `nthreads=512` | 2374 | **362** | **637** |
| `nthreads=256` with `I=128` | 2268 | **11** | **595** |
| `nthreads=256, I=64` (paired) | 1962 | 0 | 0 |

The two mismatched configs were the fastest numbers measured in the entire
sweep. Both were wrong.

**`I` must also divide 128.** `mmq.cuh` selects the out-of-bounds fallback from
a hardcoded `args.nrows_x % 128 == 0` rather than from `config.I`. So
`nthreads=384 / I=96` satisfies `I == nthreads/4` and still fails MUL_MAT on
every quant type, with errors of 0.09–0.44 against a 5e-4 tolerance,
reproducibly. **Valid `I` values are 32, 64 and 128.**

#### Knobs that turned out to be dead or harmful

- **`occupancy` does nothing at `nthreads=512`.** It feeds the second
  `__launch_bounds__` argument, which on HIP is `MIN_WARPS_PER_EXECUTION_UNIT`
  (not CUDA's `minBlocksPerMultiprocessor`). LLVM clamps a request below the
  default derived from the workgroup size, so `1` and `2` compile identically.
  Measured: 1836 vs 1834 t/s. It only becomes live below 512 threads, and at
  `nthreads=256` it still measured flat (1962 vs 1966).
- **`J=128` is 26% slower, `J=96` is 22% slower** — both numerically correct.
  CDNA's table stops at `J=64` while Ampere/Blackwell/RDNA4 reach 128; that cap
  turns out to be right for gfx90a, corroborating aviallon's MI210 measurements
  in PR [#21849](https://github.com/ggml-org/llama.cpp/pull/21849). LDS was
  never the constraint — `J=128` fits in 37.5 KiB.
- **`MMQ_ITER_K` 256 → 512** benchmarks +25% at pp16384 (3238 t/s) and fails
  263 MUL_MAT / 541 MUL_MAT_ID tests. Another fast-and-wrong.

---

### 7. SSD chunk size 256 → 128 for CDNA  → [`patches/07-ssd-chunk-size-cdna.patch`](patches/07-ssd-chunk-size-cdna.patch)

**+2.5% prompt processing.**

`SSM_SSD_CHUNK_SIZE` is a compile-time `#define` in `ssm-scan.cu`, set to 256
and tuned against cuBLAS on NVIDIA. rocBLAS on gfx90a prefers half that:

| chunk | pp4096 | pp16384 |
|---|---:|---:|
| 64 | 1932 | 2459 |
| **128** | **2013** | **2588** |
| 192 | 2005 | 2579 |
| 256 (upstream) | 1964 | 2529 |
| 512 | 1867 | 2406 |

All five are numerically correct; this is a pure throughput choice. The curve is
flat between 128 and 192 and falls off sharply either side.

---

### 8. Dispatch `mm_ids_helper`'s fast path for top-22 routing  → [`patches/08-mmid-generalize-neu-padded.patch`](patches/08-mmid-generalize-neu-padded.patch)

**+1.7% prompt processing.**

`ggml_cuda_launch_mm_ids_helper` only dispatched the specialised kernel for
`n_expert_used ∈ {2, 4, 6, 8, 16, 32}`. Nemotron-3-Super routes **top-22 of 512
experts**, so 22 missed the list and every MoE matmul ran the generic
`mm_ids_helper<0>` — 6.7% of prefill GPU time (359 ms over 320 calls).

The two paths differ structurally:

```c
generic:      for (int it = 0; it < n_tokens; ++it)                     // 1 token/iter
specialised:  for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded)
```

`it_compact` is loop-carried, so the generic version is a 2048-iteration
dependency chain of memory round trips with 22 of 64 lanes active. At 1.12 ms
per call that is ~547 ns per iteration — about one uncached round trip, i.e.
latency-bound rather than bandwidth-bound.

Only the padding logic blocked it:

```c
static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used;
```

`neu_padded` was hardcoded with a special case for 6, and the assert was written
against the **unpadded** count — but nothing requires the unpadded count to
divide anything. The shuffle scan steps by `neu_padded` and
`warp_reduce_any<neu_padded>` needs a power of two dividing `warp_size`. Both
are now generic: next power of two, asserted against the padded width. For 22
that gives `neu_padded = 32` → 2 tokens per iteration on a 64-wide wavefront.

**Backwards compatible**: `next_pow2` of every already-dispatched value is
itself (2, 4, 8, 16, 32), and 6 still pads to 8, so no supported configuration
changes behaviour. The kernel body was already padding-correct — the
`iex < n_expert_used` guard gives padded lanes `expert_used = INT_MAX`, which
never matches and contributes nothing to `nex_prev`.

| | before | after |
|---|---:|---:|
| `mm_ids_helper` | 359 ms (`<0>`) | **254 ms** (`<22>`) |
| pp4096 | 2011 t/s | **2045 t/s** |
| pp16384 | 2590 t/s | **2634 t/s** (+1.7%) |

Adding any other expert count whose next power of two divides the warp is one
more `case` line and one more template instantiation.

---

### 9. Materialise the Mamba-2 conv transpose  → [`patches/09-mamba-conv-concat-cont.patch`](patches/09-mamba-conv-concat-cont.patch)

**+0.2% — kept, but well below what the model predicted. Read the caveat.**

`mamba-base.cpp` builds the conv input as

```c
ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);
```

`ggml_transpose` only rewrites strides, so src1 is non-contiguous and
`concat_cuda` falls into the branch upstream itself labels
`// non-contiguous kernel (slow)`. There, consecutive threads read
`src1 + (i0-ne00)*nb10` at the original row stride — roughly one useful dword
per cache line. Measured: 171 ms over 160 calls (3.2%).

Wrapping the transpose in `ggml_cont` removes that kernel:

| | before | after |
|---|---:|---:|
| `concat_non_cont` | 171 ms | — |
| `cpy_scalar` | — | 122 ms |
| net | | **−49 ms** |

**Why it under-delivered.** The plan was for `ggml_cont` to hit
`cpy_scalar_transpose`, a 32×32 LDS-tiled copy. It doesn't. `can_be_transposed`
requires `nb01 == elt_size` **and** `nb02 == ne00*ne01*elt_size`, and `xBC` is a
strided view into `zxBCdt`, so the second condition fails and the copy lands on
the generic `cpy_scalar`. Reaching the tiled path would mean materialising `xBC`
first, which trades the saving straight back.

End to end this is 2634 → 2640 t/s, about +0.2% — near measurement noise, not
the ~2.6% a read-amplification model predicted. Kept because it is one line,
numerically neutral, and removes a kernel upstream calls slow; recorded honestly
because the estimate was wrong.

**Blast radius**: `mamba-base.cpp` is backend-agnostic graph construction, so
this adds an explicit materialisation for *every* backend, not just HIP. On a
backend whose concat already handles strided input well it is pure cost. Anyone
upstreaming this should gate on the backend or fix the concat kernel instead.

---

## Combined result (change sets 4–9)

`llama-bench`, 2× MI210, Nemotron-3-Super-120B-A12B `i1-Q4_K_M`,
`-b 4096 -ub 2048 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 -t 24`:

| build | pp4096 | pp16384 | vs base |
|---|---:|---:|---:|
| upstream `67b9b0e` | 1366 | 1775 | — |
| + SSD on CDNA (4) | 1624 | 2114 | +19.1% |
| + stream-k off, K-quants (5) | 1684 | 2192 | +23.5% |
| + stream-k off, all types (5) | 1834 | 2374 | +33.7% |
| + MMQ tile retune (6) | 1962 | 2529 | +42.5% |
| + SSD chunk 128 (7) | 2013 | 2590 | +45.9% |
| + `mm_ids_helper<22>` (8) | 2045 | 2634 | +48.4% |
| **+ conv concat cont (9)** | **2048** | **2640** | **+48.7%** |

Total GPU kernel time at pp4096: **7703 ms → 5129 ms (−33%)**.

Gap to vLLM+AITER (4,070 t/s at 16k): **2.29× → 1.54×**. Still open.

### Profile at pp4096 after change sets 4–9

| kernel | share | ms |
|---|---:|---:|
| `mul_mat_q<Q4_K,64>` | 22.1% | 1132 |
| `mul_mat_q<Q5_0,64>` | 10.7% | 548 |
| rocBLAS `Cijk_..._MT256x160x64` | 9.4% | 483 |
| `mul_mat_q<Q8_0,64>` | 9.0% | 463 |
| rocBLAS `Cijk_..._MT160x128x64` | 6.8% | 350 |
| `mm_ids_helper<22>` | 5.0% | 254 |
| `ssm_ssd_pre_matmul` | 3.2% | 165 |
| `unary_op_kernel<relu_sqr>` | 2.8% | 143 |
| `convert_unary<f32,f16>` | 2.7% | 140 |
| `dequantize_block_q4_K<f16>` | 2.7% | 136 |
| `cpy_scalar` | 2.4% | 122 |

MMQ is ~42%; the rocBLAS dense path plus the conversions feeding it is ~23%.

---

## Use AMD's rocBLAS, not Ubuntu's (+2.7%, no rebuild)

**Two different rocBLAS builds are installed on this box, and llama.cpp links the
weaker one.**

| library | built with hipBLASLt | linked by default |
|---|---|---|
| `/usr/lib/x86_64-linux-gnu/librocblas.so.5` (Ubuntu `librocblas5` 7.1.0) | **no** (0 references) | ✅ yes |
| `/opt/rocm/lib/librocblas.so.5.5` (AMD ROCm 7.14) | **yes** (158 references) | ❌ no |

`libhipblaslt.so.1.4` is present at `/opt/rocm/core-7.14/lib/`. Pointing the
loader at AMD's build is a runtime change — no recompilation:

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib:$LD_LIBRARY_PATH
```

Confirm it took effect with `ldd ./build/bin/llama-bench | grep -E 'rocblas|hipblaslt'`;
you should see `/opt/rocm/lib/librocblas.so.5` **and** `libhipblaslt.so.1`.

| build | pp4096 | pp16384 |
|---|---:|---:|
| Ubuntu rocBLAS (default) | 2046 | 2631 |
| **AMD rocBLAS** | **2097** | **2696** |
| AMD rocBLAS + `ROCBLAS_USE_HIPBLASLT=1` | 2101 | 2701 |

**The gain is AMD's Tensile library, not hipBLASLt.** Explicitly enabling the Lt
backend adds only a further +0.2%, inside run-to-run noise. What this does
explain is why `ROCBLAS_USE_HIPBLASLT=1` was previously recorded as a no-op:
the backend was absent from the linked library, so the variable had nothing to
switch on. That earlier entry in "tested and rejected" was correct in outcome
but wrong in cause.

Verified: `test-backend-ops` MUL_MAT clean, plus two sequential requests read at
`temperature 0` (technical explanation and step-by-step arithmetic).

For a permanent fix, link with
`-Wl,-rpath,/opt/rocm/lib:/opt/rocm/core-7.14/lib` instead of relying on the
environment.

### An unresolved intermittent

Three times during this work, `test-backend-ops -o MUL_MAT` reported exactly 3
lines matching `FAIL`. Each time it was a bare count — the failing test names
were never captured. Across 37 subsequent controlled runs (12 + 25) it did not
recur, and a paired 25-run sample scored **0/25 on both libraries**, so it
cannot be attributed to either build. Every generated-token check in this
project has been correct.

Recorded rather than explained. Do not treat the throughput numbers here as
affected, but if you see it, capture the actual `FAIL` lines — a count alone has
now been ambiguous three times.

---

## Scope: what is left, and what it would cost

Prefill is now **2697 t/s at pp16384** against 1775 upstream (**+51.9%**). The
remaining profile at pp4096 (total ~5100 ms):

| block | share | what it is |
|---|---:|---|
| MMQ expert matmuls | ~42% | already tuned; three config knobs swept |
| rocBLAS dense GEMMs | ~19% | attention QKV/out, SSM out-proj, MoE router |
| dequant + convert | ~5% | exists **only** to feed rocBLAS |
| `mm_ids_helper<22>` | ~5% | fixed in change set 8 |
| everything else | ~29% | long tail, nothing over 3% |

The dense block is the only remaining concentration. It is **not** avoidable:
those four shapes are the attention projections, the SSM output projection and
the 512-way MoE router — all essential model computation.

### Closed: route the dense GEMMs to MMQ

Tested **three times**, at three different MMQ configurations, as the premise
changed each time:

| when | MMQ config | result |
|---|---|---|
| before stream-k fix | 512/I128, stream_k on | −6.5% |
| after stream-k fix | 512/I128, stream_k off | −10.5% |
| after tile retune | 256/I64, stream_k off | **−9.2%** |

rocBLAS wins these shapes regardless of how MMQ is tuned. This option is closed.

### Closed: per-shape Tensile solution override

908 candidate kernels timed across the four shapes; none beat rocBLAS's default
pick. See the section above.

### Option A — fused dequantise + FP16 MFMA GEMM

**The strongest remaining idea, and the only one that attacks two blocks at once.**

Today the dense path does: dequantise Q4_K weights to FP16 into a scratch
buffer, convert activations F32→F16, then `cublasGemmEx`. For the
4096×8192 weight that is ~67 MB written and ~67 MB read back **per call, per
ubatch**, plus 276 ms of kernel time that produces no arithmetic.

A GEMM that dequantises inline from LDS would delete the conversion kernels and
their HBM round-trip outright.

- **Ceiling**: 276 ms of conversions (5.4%) plus whatever the improved locality
  buys in the GEMM itself. Optimistically 8–12% end to end.
- **Effort**: 4–8 weeks of specialist work — MFMA intrinsics, LDS
  double-buffering, prefetch pipelining, split-k, per-shape tiling.
- **Risk**: high. rocBLAS currently achieves 50–70% of the 181 TFLOPS peak on
  these shapes (90.5 / 94.5 / 127.2 TFLOPS). Tensile has person-years of tuning
  behind it; hand-written first attempts typically land at 40–60% of rocBLAS.
  The dequant saving is real and certain, but it has to more than pay for
  whatever the hand-written GEMM gives up.
- **Note**: this is essentially what MMQ does, except MMQ uses int8. On CDNA2
  int8 and FP16 MFMA have the *same* 181 TOPS peak, so MMQ pays quantisation
  overhead for no throughput advantage — which is exactly why it loses here. An
  FP16 variant does not have that handicap.

### Option B — hipBLASLt directly

llama.cpp has a hipBLASLt batched-GEMM path (PR #16457) gated to CDNA3.
Extending it to CDNA2 would bypass rocBLAS's dispatch shim.

- **Ceiling**: low. Routing through the shim measured **+0.2%**, and hipBLASLt
  shares Tensile's kernel library.
- **Effort**: 1–2 weeks including validation.
- **Verdict**: poor ratio. Not recommended before Option A.

### Option C — hand-written FP16 MFMA GEMM, no dequant fusion

Option A minus its best justification. Same effort, smaller prize (~19% block
rather than ~24%), same risk of losing to Tensile. **Not recommended** —
if this work is done at all, do Option A.

### Cheaper things worth doing first

1. **Land the AMD rocBLAS switch** above — +2.7% for an environment variable.
2. **Upstream PR [#25952](https://github.com/ggml-org/llama.cpp/pull/25952)**
   (fused MoE expert reduction, +3.6–7.1% measured on CUDA) currently handles
   `k = 2..15`; this model's `top_k = 22` exceeds the cap and falls back.
   Extending the cap is far less work than any GEMM project.
3. **Upstream PR [#26621](https://github.com/ggml-org/llama.cpp/pull/26621)**
   (L2 cache-set aliasing, up to 20% on RDNA3.5, untested on CDNA2). The
   dequantised FP16 buffers feeding rocBLAS alias whenever `ne00 % 1024 == 0`,
   which is common here — and Option A would delete those buffers entirely, so
   test this *before* committing to a GEMM rewrite.

**Recommendation**: do the three cheap items. Treat Option A as a genuine
project to be scheduled deliberately, not as a next step — and only after
measuring #26621, which targets the same buffers Option A would remove and
could change the arithmetic.

---

## A note on measuring these changes

Four separate configurations in this work benchmarked **faster while computing
wrong results**, three of them by a wide margin:

| config | apparent gain | reality |
|---|---|---|
| `I=64` alone | +29% | 362 + 637 test failures |
| `nthreads=256` alone | +23% | 11 + 595 test failures |
| `nthreads=384 / I=96` | +2% | fails every quant type |
| `MMQ_ITER_K=512` | +25% | 263 + 541 test failures |

The fastest number measured in the entire session was wrong. Anything that
breaks a tiling invariant does less work, and doing less work looks exactly like
an optimisation on a throughput chart.

Two practical consequences:

1. **`test-backend-ops -o MUL_MAT` / `MUL_MAT_ID` / `SSM_SCAN` before any
   benchmark is believed**, then generated tokens at `temperature 0` on top.
2. **Count failures with a plain `grep FAIL`.** The harness prints them as
   `[MUL_MAT] ERR = 0.128 > 0.0005   MUL_MAT(...): FAIL` — a pattern anchored to
   leading whitespace matches nothing and reports a clean run for a broken
   build. That happened here and briefly cleared a config that was in fact fine,
   but the same mistake in the other direction is what ships corruption.

---

## Tested and rejected (change sets 6–7)

| change | result |
|---|---|
| Force MMQ for the dense FP16 GEMMs — tested **three times** as the premise changed (before the stream-k fix, after it, and again after the tile retune made MMQ ~7% faster) | −6.5%, −10.5%, then −9.2%. rocBLAS wins these shapes regardless of how MMQ is tuned. Closed. |
| `J=128` / `J=96` CDNA entries | −26% / −22%, both correct |
| `occupancy` 2 or 4 | flat at 512 threads (LLVM clamps it); `occupancy=4` is −50% |
| `nthreads=384 / I=96` | numerically wrong |
| `MMQ_ITER_K=512` | numerically wrong |
| `-ub` 512 / 1024 / 4096 | 1871 / 2335 / 2388 vs 2564 at 2048 — 2048 still optimal after all kernel changes |
| `-b` 2048 / 4096 / 8192 | 2574 / 2567 / 2570 — no effect |
| `-ctk f16 -ctv f16` instead of `q8_0` | 2584 vs 2590 — no prefill difference, so `q8_0` stays for the memory saving at long context |
| `stream_k=true` **re-tested** after the `I=128→64` retune doubled the tile count | 1485 / 1947 vs 2011 / 2590 — still far worse, so the retune does not change that conclusion |
| `J=96` at `I=32` (keeps 2 workgroups/CU where `I=64` drops to 1) | 1566 / 2041 — still worse; the mean expert width of 88 does not rescue it |
| uneven `-ts` split across the two cards | model fails to load; the even layer split is effectively forced |

---

## Base commit

Change sets **1-3** are generated against the `TheTom/llama-cpp-turboquant`
fork at:

```
c26cbdffcf6fc9b7430cd6b117757e9a3f70b7ea  Merge pull request #225 from TheTom/fix-ui-assets-partial-dist
```

Change sets **4-9** are generated against **upstream `ggml-org/llama.cpp`** at:

```
67b9b0e7f6ce45d929a4411907d3c48ec719e81c  llama-arch: fix DeepSeek4 APE tensor op (#25945)
```

These are different bases. Change sets 4-9 were developed and measured on the
upstream tree, **not** on the TurboQuant fork, and have not been tested there —
`ssm-scan.cu` in particular changed substantially upstream in the interim, so
expect `patches/04-*` to need rebasing before it applies to the fork. The two
groups touch disjoint files, so there is no conflict between them in principle.

## How to apply

See [`BUILD.md`](BUILD.md) for the full Docker build procedure with ccache. Short version:

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git checkout c26cbdffcf6fc9b7430cd6b117757e9a3f70b7ea
git apply 01-per-layer-kv-types.patch
git apply 02-kivi2-quant-type.patch
git apply 03-turboquant-wave64-fixes.patch
# build for gfx90a (see BUILD.md)
```

Change sets 4-9 target upstream llama.cpp instead (see "Base commit" above):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 67b9b0e7f6ce45d929a4411907d3c48ec719e81c
git apply patches/04-ssd-mamba2-prefill-cdna.patch
git apply patches/05-mmq-cdna-no-streamk.patch
git apply patches/06-mmq-cdna-tile-retune.patch
git apply patches/07-ssd-chunk-size-cdna.patch
git apply patches/08-mmid-generalize-neu-padded.patch
git apply patches/09-mamba-conv-concat-cont.patch
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a -DGGML_HIP_MMQ_MFMA=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench llama-server test-backend-ops -j
```

`patches/04-*` bundles the upstream SSD kernels together with the CDNA
enablement, so it applies to a bare `67b9b0e` checkout with no cherry-pick
first — verified with `git apply --check`, and the resulting files are
byte-identical to the `modified-files/` copies. If you would rather keep the
upstream work as its own commit, `git cherry-pick b62b350` instead and then
apply only the guard changes via `tools/patch_ssm_ssd_cdna.py`.

Verify before trusting the build:

```bash
./build/bin/test-backend-ops -o SSM_SCAN      # expect 7/7 per device
./build/bin/test-backend-ops -o MUL_MAT_ID    # expect 790/790 per device
```

`tools/patch_ssm_ssd_cdna.py` and `tools/patch_mmq_cdna_no_streamk.py` apply and
revert the same changes against a clean tree, with `--check` / `--revert`, and
refuse to apply if their anchors have moved upstream.

The [`modified-files/`](modified-files/) directory contains the final state of every changed file if you prefer drop-in replacement over `git apply`.

---

## Related

- **Hub repo:** [`davetha/mi210-llm-stack`](https://github.com/davetha/mi210-llm-stack) — full optimization write-up, architecture, and all guides.
- **Triton TurboQuant:** [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd) — the wave64-safe GEMM-based WHT alternative.

## License

MIT. The underlying llama.cpp is MIT-licensed.
