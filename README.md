# llama.cpp-mi210

> **Fork of [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)** (itself a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)) with three change sets optimized for **AMD MI210 (gfx90a / CDNA2)** inference.

This repo does **not** contain the full llama.cpp tree (too large to mirror here). Instead it ships:

```
patches/          — apply these on top of the upstream fork
modified-files/   — the exact modified files (drop-in replacements)
tools/            — patch/revert scripts and the rocprofv3 trace analyser
tools/rejected/   — patches that were tried and lost, kept with their verdicts
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

### 10. Choose the micro-batch size per decode call  → [`patches/10-adaptive-ubatch.patch`](patches/10-adaptive-ubatch.patch)

**+12.4% on 2k prompts, +8.8% on 3k.** A latency win for interactive workloads.

With `-sm layer` the two devices form a pipeline: GPU0 holds the first half of
the layers, GPU1 the second. A call that produces a **single** micro-batch runs
them strictly sequentially -- one idles while the other works. Only two or more
micro-batches overlap them. Measured aggregate utilisation:

| prompt | micro-batches | aggregate GPU utilisation |
|---|---:|---:|
| pp4096 | 2 | 62.0% |
| pp16384 | 8 | 85.6% |

A larger ubatch buys kernel efficiency; a smaller one buys pipeline depth. `-ub`
is fixed at startup, so a context sized for long prompts pays for it on every
short request. This picks the largest power-of-two ubatch that still yields at
least two per call:

```c
ubatch = clamp(pow2_floor(n_tokens_all / 2), 256, cparams.n_ubatch)
```

| prompt | fixed `-ub 2048` | adaptive | delta |
|---|---:|---:|---:|
| pp1024 | 1346 | **1377** | +2.3% |
| pp2048 | 1589 | **1786** | **+12.4%** |
| pp3072 | 1841 | **2003** | **+8.8%** |
| pp4096 | 2099 | 2093 | −0.3% |
| pp6144 | 2336 | 2257 | **−3.4%** |
| pp8192 | 2483 | 2475 | −0.3% |
| pp16384 | 2696 | 2681 | −0.6% |

**The signal is the size of this call, not the prompt length.** A first attempt
keyed on prompt length regressed 16k by 10%: llama.cpp already caps a call at
`n_batch`, so a 16k prompt arrives as four 4096-token calls and a
prompt-length test fires on every one.

**The pp6144 regression is real and not fixable here.** That prompt splits into a
4096 call plus a 2048 tail; the tail does not need extra depth because the
preceding call already filled the pipeline, but a *standalone* 2048 does — and
the two are indistinguishable without tracking continuation state. The trade was
taken deliberately: +12.4% and +8.8% in the common chat range against −3.4% in a
narrow band around 6k.

Only ever shrinks `cparams.n_ubatch`, so compute buffers sized at context
creation stay valid. Gated on `causal_attn`, because non-causal attention asserts
`n_ubatch >= n_tokens` a few lines above.

**Files (1):** `src/llama-context.cpp`.

---

## Measured against vLLM: what the remaining gap actually is

The 4,070 t/s vLLM figure quoted in earlier revisions of this document was
inherited, not measured here. Measuring both engines on the same box, same
model architecture (`NemotronHForCausalLM`, 512 experts, top-22, hidden 4096),
at matched prompt lengths:

| prefill | llama.cpp | vLLM TP=2 | gap |
|---|---:|---:|---:|
| ~4k | 2102 | 3635 | **1.73x** |
| ~16k | 2708 | 3563 | **1.32x** |

vLLM was run from `local/vllm-mi210:dsa7-aiterint8` on
`/mnt/llm-storage/nemotron3-120b-awq` with `--tensor-parallel-size 2`. Note that
image bakes in `HIP_VISIBLE_DEVICES=0`, so it must be overridden to see both
cards. vLLM's numbers include HTTP and tokenisation, so they are marginally
pessimistic; llama.cpp's are pure prefill from `llama-bench`.

### The difference is tensor parallelism

Sampling `/sys/class/drm/card*/device/gpu_busy_percent` during a 16k prefill:

| engine | gpu0 mean | gpu1 mean | both >50% simultaneously |
|---|---:|---:|---:|
| vLLM TP=2 | 99.4% | 99.6% | **99% of samples** |
| llama.cpp `-sm layer` | — | — | 73.5% of window (pp16384), 30.5% (pp4096) |

vLLM runs both cards on **every layer**. llama.cpp splits layers across cards and
relies on multiple micro-batches to overlap them, which works well at long
context (85.6% aggregate) and poorly at short (62%). That is exactly why the gap
is worse at 4k than at 16k.

### llama.cpp cannot do tensor parallelism on ROCm

`-sm row` is the tensor-parallel mode, and it fails with

```
device ROCm0 does not support split buffers
```

This is not a configuration problem. `ggml_backend_split_buffer_type` is
implemented by **only one backend in the entire tree, SYCL** — confirmed on both
this branch and upstream `master`. The CUDA/HIP backend exposes no such function,
so `-sm row` cannot work here at all. Closing that part of the gap would mean
implementing split buffers for the CUDA/HIP backend: a substantial upstream
project, not a patch.

### What this means for the remaining ~1.3x

Roughly, at 16k: ~1.16x of it is the parallelism difference (85.6% vs ~99%
utilisation) and the rest is kernel efficiency — consistent with the earlier
finding that the MoE expert path runs at 0.99 FLOP/time against 2.42 for the
dense path.

So the two remaining levers are unchanged in kind, but their sizes are now known:

1. **Grouped/fused MoE GEMM** — the kernel-efficiency half. Still the highest
   yield available without a backend-level project.
2. **Split buffers for CUDA/HIP** — the parallelism half. Larger, upstream, and
   would benefit every multi-GPU ROCm user, not just this box.

Neither is a tuning knob. The tuning knobs are exhausted: `nthreads`,
`occupancy`, `I`, `J`, `stream_k`, `K_vram`, SSD chunk size, `-b`, `-ub`,
`-ctk`/`-ctv`, `-ts`, all seven `GGML_CUDA_*` environment variables, per-shape
Tensile solution override (908 kernels timed), and the MMQ carve-out (three
separate tests).

---

### 11. Do not divide by `n_gqa` when a layer has no attention heads  → [`patches/11-tp-ngqa-div0.patch`](patches/11-tp-ngqa-div0.patch)

An upstream bug, found while investigating tensor parallelism. `get_split_granularity`
in `llama-model.cpp` computes

```c
const int64_t granularity_kv = granularity_q / n_gqa;
```

unconditionally in its regular-attention branch, **before** the tensor is known
to be a KV tensor. In a hybrid model a pure FFN/MoE block has no attention
heads, so `hparams.n_gqa(il)` is 0 and any FFN tensor in such a layer divides by
zero.

Reproduced with `LLAMA_SPLIT_MODE_TENSOR` on `nemotron_h_moe`: SIGFPE inside
`llama_meta_device_get_split_state` while allocating
`blk.1.ffn_down_exps.weight`.

Unreachable today because every hybrid architecture is on the
`llm_arch_supports_sm_tensor` exclusion list, but the expression is wrong
regardless of which architectures are enabled, and it blocks anyone extending
that list. No behaviour change for architectures that already support
`-sm tensor` — their layers all have `n_gqa != 0`. **Worth sending upstream on
its own.**

---

## Tensor parallelism: it exists, and here is exactly what is missing

An earlier revision of this document concluded that llama.cpp cannot do tensor
parallelism on ROCm, because `-sm row` fails with `does not support split
buffers` and only the SYCL backend implements `ggml_backend_split_buffer_type`.
**That conclusion was right about `-sm row` and wrong about the capability.**

`-sm row` was deliberately removed in
[74976e1ae / PR #24216](https://github.com/ggml-org/llama.cpp/pull/24216)
("CUDA: remove -sm row, refactor cuBLAS", 2026-07-06, 16 days before this
tree's base commit). The stated reason:

> This PR removes CUDA backend support for split buffers (`--split-mode row`) —
> by now `-sm tensor` has all of the necessary features to make it obsolete.

So the replacement is **`-sm tensor`** (`LLAMA_SPLIT_MODE_TENSOR`), and it is
present in this tree.

### How it works

Not via split buffers. `llama_prepare_model_devices` builds a **meta device**
(`ggml_backend_meta_device`) wrapping the N real GPUs, with a callback,
`llama_meta_device_get_split_state`, that decides how each tensor is sharded.

That callback is **generic and pattern-based**: it matches tensor names by regex
and assigns a split axis — column-parallel (`AXIS_1`) paired with row-parallel
(`AXIS_0`), the classic TP arrangement, with `MIRRORED` for replicated tensors
and `PARTIAL` for a few special cases. Unmatched tensors default to `MIRRORED`,
which is safe but redundant.

It already understands SSM tensors (`ssm_out`, `ssm_conv1d`, `ssm_dt`, `ssm_a`,
`ssm_alpha`/`beta`, `cache_r`, `cache_s`) **and** MoE experts (`ffn_up_exps`,
`ffn_gate_exps`, `ffn_down_exps`). It also carries a worked example of a
fused-projection architecture: Qwen3Next / Qwen3.5 get bespoke segmentation for
their fused QKV and their `n_v_heads > n_k_heads` broadcasting.

### Why this model is excluded

`llm_arch_supports_sm_tensor` is an **exclusion list** — TP is on by default and
these opt out. Every SSM/hybrid architecture is on it: `MAMBA`, `MAMBA2`,
`JAMBA`, `FALCON_H1`, `NEMOTRON_H`, `NEMOTRON_H_MOE`, `GRANITE_HYBRID`, `LFM2`,
`LFM2MOE`, `KIMI_LINEAR`, plus `DEEPSEEK2/32/4`, `GROK`, `T5` and others.

Removing `NEMOTRON_H_MOE` from that list and building gets:

1. **SIGFPE** in the split callback — the `n_gqa == 0` division above. Fixed in
   change set 11.
2. **`GGML_ASSERT(offset + ... <= ggml_nbytes(tensor) && "tensor write out of
   bounds")`** in `ggml-backend.cpp:371` — the sharding arithmetic does not
   handle this architecture's tensor shapes.

So the exclusion is not arbitrary, but it is also not fundamental. The blocker
is a set of missing split rules, not a missing mechanism.

### What implementing it would take

Nemotron-3-Super's tensors, and what each needs:

| tensor | shape | status |
|---|---|---|
| `attn_q/k/v/output` | 2D | already matched |
| `ffn_up_exps` / `ffn_down_exps` | **3D** `[2688, 1024, 512]` | matched by pattern, but the shard arithmetic overruns — 3D expert tensors need handling |
| `ssm_in` | `[4096, 18560]` | **unmatched** — fused `z\|x\|B\|C\|dt` (8192+8192+1024+1024+128), needs multi-segment splitting exactly like Qwen3Next's fused QKV |
| `ssm_out` | `[8192, 4096]` | already matched (`AXIS_0`) |
| `ffn_latent_down` / `ffn_latent_up` | `[4096,1024]` / `[1024,4096]` | **unmatched** — the LatentMoE wrapper around the experts |
| `ssm_norm`, `exp_probs_b`, `ffn_gate_inp` | small | default `MIRRORED` is probably correct |

The `ssm_in` case is the substantial one: splitting a Mamba-2 fused projection
means splitting `z` and `x` by inner dimension, `B` and `C` by group, and `dt`
by head, keeping all four consistent with how `ssm_out` and the SSM state caches
are split. The Qwen3Next branch in `get_split_segments` is a direct template for
this.

**Estimate (superseded -- see the verdict section below, which supersedes this): days, not weeks, and confined to one file** (`src/llama-model.cpp`),
with `test-backend-ops` plus generated-token checks as the gate. That is far
cheaper than the "implement split buffers for the CUDA/HIP backend" framing in
the previous revision, which was based on the wrong mechanism.

### What it would be worth

Measured on this box, vLLM with `--tensor-parallel-size 2` keeps both cards at
99.4%/99.6% simultaneously; llama.cpp's layer split reaches 85.6% aggregate at
16k and 62% at 4k. Closing that would be worth roughly **1.16x at 16k and more
at short context**, where the pipeline bubble is worst — which is exactly where
latency matters most.

It would also benefit every multi-GPU ROCm *and* CUDA user running a hybrid
model, not just this box, since the exclusion list is backend-agnostic.

---

## Tensor parallelism for Mamba-2 hybrids: why it is a framework redesign, not a patch

Three options were considered for making `-sm tensor` work on Mamba-2 hybrid
architectures. All three were investigated to the point of a definite answer,
and **none is a viable fork-level change.** The blocker is a design limitation
in ggml's split-state model, not anything specific to Nemotron.

### What was built and how far it got

Preserved as [`patches/wip-tp-nemotron-h.patch`](patches/wip-tp-nemotron-h.patch),
generated by [`tools/patch_tp_nemotron_h.py`](tools/patch_tp_nemotron_h.py) and
[`tools/patch_tp_ssm_scan_handler.py`](tools/patch_tp_ssm_scan_handler.py). It is
**not applied** to this fork and should not be — it reaches the wall below and
stops. Each fix exposed the next layer:

| step | blocker | resolution |
|---|---|---|
| 1 | SIGFPE, `granularity_q / n_gqa` with `n_gqa == 0` | fixed, **landed on main** as change set 11 |
| 2 | tensor write out of bounds | full Nemotron-H tensor split table |
| 3 | ADD mismatch on `ssm_conv1d.bias` | bias needs the conv weight's segmentation |
| 4 | `SSM_SCAN` refused any dimensional split | wrote `handle_ssm_scan`; the op now passes |
| 5 | ADD mismatch on `mamba2_y_add_d` | **the wall** |

### The wall, stated precisely

`ggml_ssm_scan` returns **one** 1-D tensor holding two concatenated regions:

```c
result = ggml_new_tensor_1d(ctx, F32,
             ggml_nelements(x) + s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);
       =  y( head_dim * n_head * n_seq_tokens * n_seqs )
       ++ states( d_state * head_dim * n_head * n_seqs )
```

Both regions are linear in `n_head`, so a single-segment proportional split is
**numerically exact in size** — device *j* gets `n_head_j / n_head` of the
buffer. SSM_SCAN itself passes with that description.

It is nonetheless **wrong in layout**. Device *j* owns its heads' slice of `y`
*and* its heads' slice of the states — two disjoint regions of the logical
tensor, not a contiguous prefix. `handle_reshape` then sees a 1-D source split
on its last axis, applies the (correct in general) rule "flat split maps to the
view's outermost dim", and hands the `y` view an axis that disagrees with the
`x*D` operand it is added to:

```
[TPOP] UNKNOWN op=ADD name=mamba2_y_add_d-0
    src0 node_35 (view)  axis=2   <- y viewed out of the fused result
    src1 node_45         axis=1   <- x*D, split by head
```

The two-segment description does carry the missing information. It is rejected
before any view sees it: the post-pass that reconciles an op output against its
sources asserts `n_segments == 1`.

### Why each option fails as a fork-level change

**Option 1 — let op outputs carry multi-segment split state.** Not confined to
the post-pass. `n_segments == 1` is asserted or required at **11 sites**
spanning `handle_reshape`, `handle_permute`, `handle_transpose`, the post-pass,
the split-state cache, and all four data-movement paths
(`buffer_set_tensor`, `buffer_get_tensor`, `set_tensor_async`,
`get_tensor_async`). That is the core of the meta backend, and it is code that
currently works for every supported architecture — so the regression risk lands
on Qwen3Next, DeepSeek and the rest, with silent numerical wrongness as the
failure mode.

**Option 2 — return `y` and the states as separate tensors.** Changes a public
ggml op contract. Seven backends implement SSM_SCAN and each writes the fused
buffer at computed offsets (CUDA, CPU, SYCL, Vulkan, WebGPU, ET, meta), plus
four callers, the tests, and `ggml.h`. In a fork this also means permanent
divergence on a core op, so every future rebase fights it.

**Option 3 — special-case the view of an SSM_SCAN result.** Cannot work. Even if
the view recovered the head axis for `y`, the state region at `s_off` is still
described by the same false "contiguous prefix" premise. The information is
missing from the split state itself, not merely mis-propagated by the view.

### Conclusion — CORRECTED, the earlier one was wrong

An earlier revision of this section concluded that "ggml ops with fused
multi-region outputs cannot be tensor-split under the current single-segment
model, which is exactly why every SSM/hybrid architecture sits on the
`llm_arch_supports_sm_tensor` exclusion list."

**Both halves of that are false**, and the refutation is in the same file the
analysis was done in.

`ggml_gated_delta_net` returns a fused output-plus-recurrent-state buffer — the
same problem class — and `handle_gated_delta_net` splits it with a plain
single-segment rule:

```c
return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};   // n_segments == 1
```

Qwen3-Next uses that op, `llm_arch_is_hybrid()` returns true for it, and it is
**not** on the exclusion list. So a fused-state hybrid already runs tensor-
parallel here. The exclusion list is also not "every SSM/hybrid" — it carries
GROK, MPT, DeepSeek2/3.2/4, T5, BitNet and others that have no SSM at all.

The real difference is the **declared output shape**:

| op | output shape | head axis |
|---|---|---|
| `ggml_gated_delta_net` | `[S_v*H, n_tokens*n_seqs + K*S_v*n_seqs]` | axis 0 ✓ |
| `ggml_rwkv_wkv7` | `[S*H, n_tokens + S*n_seqs]` | axis 0 ✓ |
| `ggml_ssm_scan` | `[nelements(x) + state_size]` | **1-D flat — destroyed** |

The two siblings keep the head count as a real axis shared by both regions.
`ggml_ssm_scan` throws it away, so a proportional split of it is exact in size
and wrong in layout, and every view downstream inherits that. The blocker was
the shape declaration, not the framework.

### The fix, and what it costs

Give `ggml_ssm_scan` the same discipline:

```
[d_inner, n_seq_tokens*n_seqs + d_state*n_seqs]      d_inner = head_dim*n_head
```

Total size is unchanged and the `y` region's bytes are unchanged. Only the state
region reorders, from `{d_state, head_dim, n_head, n_seqs}` to
`{head_dim, n_head, d_state, n_seqs}`, so its rows are `d_inner` wide too. Then
one AXIS_0 split at a multiple of `head_dim` is head-aligned in **both** regions
and **no framework change is needed at all** — neither the 11-site multi-segment
change nor the 7-backend op-signature change considered earlier.

It is not free. `d_state` was the fastest-moving axis of the state, and that is
exactly what the sequential kernels vectorise over — the CPU path loads
`s0 + i + ii*nc` along `d_state` in three SIMD variants, and the CUDA group
kernel reads `s0_warp[WARP_SIZE*j + lane]` the same way. Making `d_state` the
slowest axis makes both strided.

That cost lands on **decode and CPU inference**, not on the SSD chunked prefill
path: there the state lives in a cuBLAS scratch buffer and the transpose is an
operand swap (`m=head_dim, n=d_state, ldc=head_dim`), same FLOPs, no extra
kernel. Prefill is what tensor parallelism is being chased for, so the trade
points the right way — but it is a real regression and has to be measured. The
CPU loop can be re-vectorised over `head_dim` by swapping the `i`/`i1` loops.

**Status: incomplete.** `ggml.c` and the CPU kernel are done on the
`ssm-scan-2d` branch. `ggml-cuda/ssm-scan.cu` (two sequential kernels, plus
`init_state`, `scale_state`, the 3e state GEMM and the final copy in the SSD
path) and the `get_ssm_rows` reshape at the three call sites are not. Nothing is
proven until it runs and the generated tokens match `-sm layer` at temperature 0.

The one piece that was independently correct — the `n_gqa == 0` divide-by-zero —
is already on `main` and is worth sending upstream on its own.

### What this leaves on the table

The measured prize was ~1.16x at 16k and more at short context, from closing the
utilisation gap (llama.cpp 85.6% aggregate at 16k and 62% at 4k, against vLLM
TP=2 holding both cards at 99.4%/99.6% simultaneously). That remains unclaimed,
and on this codebase it is not claimable without upstream work.

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

Gap to vLLM at 16k: **2.01× → 1.35×**. Still open. That is against the
**3,563 t/s measured on this box**, not the 4,070 t/s figure earlier revisions
quoted second-hand — see [Measured against vLLM](#measured-against-vllm-what-the-remaining-gap-actually-is).

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

## Scope: where the remaining headroom actually is

**Correction to an earlier version of this document.** A previous revision named
a fused dequantise + FP16 GEMM as the strongest remaining project, on the
grounds that the rocBLAS dense path was the largest single block of time. That
ranking was wrong: it compared blocks by *time* without checking how much
arithmetic each was doing. Cross-referencing the profile against FLOP counts
derived from the GGUF tensor shapes reverses the conclusion.

### The two paths, measured

Weight-matmul FLOPs per token, computed from the actual tensor shapes:

| block | MACs/token | share |
|---|---:|---:|
| routed experts | 4.844 G | 41.4% |
| SSM projections (`ssm_in`, `ssm_out`) | 4.383 G | 37.5% |
| shared expert | 1.762 G | 15.1% |
| latent up/down | 0.336 G | 2.9% |
| attention | 0.285 G | 2.4% |
| router | 0.084 G | 0.7% |
| **total** | **11.694 G** | (matches the A12B label) |

Against measured GPU-time shares at pp4096:

| path | FLOP share | time share | FLOP per unit time |
|---|---:|---:|---:|
| routed experts — MMQ | 41.4% | 41.8% | **0.99** |
| everything dense — rocBLAS + its conversions | 58.6% | 24.2% | **2.42** |

**The rocBLAS dense path is ~2.4x more FLOP-efficient than the MMQ expert path.**
The dense GEMMs run at 50-70% of the 181 TFLOPS peak when benchmarked in
isolation (90.5 / 94.5 / 127.2 TFLOPS). Overall the model achieves 13.6% of
peak. The deficit is not in the dense block.

### Why the experts are slow, and why that is not the kernel's fault

This is a **LatentMoE**: experts operate in a 1024 -> 2688 -> 1024 latent space
rather than at model width. With 512 experts, top-22 routing, and a 2048-token
ubatch, each expert receives on average

    2048 tokens * 22 / 512 experts = 88 tokens

so every routed-expert GEMM is `m=2688, n=88, k=1024`. That is extremely skinny,
against a tile whose J is 64. No kernel reaches dense-GEMM efficiency at n=88 --
this is geometry, not implementation quality. Any project here is about
*mitigating* the shape, not about writing a faster inner loop.

### Ceilings

| project | recoverable GPU time | throughput |
|---|---:|---:|
| fused dequant + FP16 GEMM on the dense path | 9.2% | ~+10% |
| MoE expert path 30% faster | 12.5% | ~+14% |
| MoE expert path 50% faster | 20.9% | ~+26% |

### Option 1 — grouped / fused MoE GEMM  *(highest yield)*

One kernel covering all experts with a shared tiling strategy, rather than the
current per-expert dispatch. This is what vLLM's AITER path does, and is the
most likely single explanation for its measured 3,563 t/s on the same hardware
(alongside tensor parallelism, which accounts for a separate ~1.16×).

Concretely it attacks three things at once: the launch and indexing overhead
(`mm_ids_helper` is still 4.9% even after change set 8), the poor tile
utilisation at n=88, and the intermediate traffic in the combine tail.

- **Ceiling**: +14% at a 30% improvement, +26% at 50%.
- **Effort**: large. This is a new kernel plus a graph-level change to the
  MUL_MAT_ID path, not a config edit.
- **Risk**: high, but *lower than a dense-GEMM rewrite* -- here you are
  competing against a path running at 0.99 FLOP/time, not against Tensile at
  50-70% of peak. There is real slack to recover.

### Option 2 — upstream PR [#25952](https://github.com/ggml-org/llama.cpp/pull/25952), fused MoE combine

Fuses the post-`MUL_MAT_ID` scale-and-sum into a single reduction, removing the
intermediate `[n_embd, k, n_tokens]` tensor.

Measured cost of that tail here: one `k_bin_bcast<op_mul>` (76.6 ms) plus four
`k_bin_bcast<op_add>` kernels (44.5 + 25.1 + 18.4 + 9.1 ms) = **173.7 ms, 3.4%**
of GPU time.

- **Ceiling here: ~2%**, not the +3.6-7.1% in the PR's own numbers -- those were
  measured on models whose combine tail is a larger share.
- **Effort**: back-porting an open PR across 243 commits, *plus* raising its
  `k <= 15` cap. That cap comes from `ggml_can_fuse_subgraph`'s 31-node limit
  applied to the long form (`2k + 1 <= 31`); this model's `top_k = 22` needs 45
  nodes on that form. The short form (`experts * router_weight`, no per-expert
  scale) needs roughly `k + 1` nodes, so 22 would fit -- but that has to be
  confirmed against which form this graph actually produces.
- **Verdict**: poor ratio at ~2%. Reconsider if it merges upstream and the
  rebase cost disappears.

### Option 3 — more tokens per expert

The shape problem would soften with a larger ubatch: `-ub 4096` would give 176
tokens per expert instead of 88. **Already measured and rejected** -- `-ub 4096`
is 2388 t/s against 2564 at `-ub 2048`. Whatever the better GEMM shape buys is
more than lost elsewhere.

### Demoted: fused dequantise + FP16 MFMA GEMM on the dense path

Still a real ~+10%, and the dequant elimination (276 ms, 5.4%) is certain rather
than speculative. But it targets the path already at 2.42 FLOP/time and requires
beating Tensile at 50-70% of peak, where hand-written first attempts typically
land at 40-60% of rocBLAS. **Do Option 1 first.**

### Closed, with evidence

| idea | outcome |
|---|---|
| route dense GEMMs to MMQ ([`tools/rejected/patch_mmq_cdna2_large_ne11.py`](tools/rejected/patch_mmq_cdna2_large_ne11.py)) | tested 3x at 3 different MMQ configs: -6.5%, -10.5%, -9.2% |
| per-shape Tensile solution override ([`tools/rocblas_solution_tune.cpp`](tools/rocblas_solution_tune.cpp)) | 908 candidate kernels timed, none beat the default |
| `J=96` / `J=128` tiles ([`tools/patch_mmq_cdna_add_j128.py`](tools/patch_mmq_cdna_add_j128.py)) | `J=96` matches the 88-token mean expert width and still lost: -22% at `I=64`, worse at `I=32` where occupancy is preserved |
| larger ubatch for better expert shape | `-ub 4096` is 7% slower |
| hipBLASLt | present in AMD's rocBLAS; adds +0.2% over it |
| crossing the attention MFMA gate at batch=2 ([`tools/bench_batch_mfma.py`](tools/bench_batch_mfma.py)) | gate crossed, no gain: 53.86 → 52.01 t/s aggregate. The kernel was not the constraint |
| wave64-aware sequential SSM scan ([`tools/rejected/patch_ssm_scan_wave64.py`](tools/rejected/patch_ssm_scan_wave64.py)) | never applied — the premise was wrong (`c_factor` is warps-per-block *and* state-per-lane). The scan's 22.8% was removed by change set 4 instead |

---

## A note on measuring changes here

Two failure modes have each cost real time in this project. Both are cheap to
guard against and expensive to miss.

**1. A stale binary after reverting the source.** Patch scripts edit source;
`cmake --build` is a separate step. Reverting a patch at the end of a sweep
script without rebuilding leaves the *previous* binary in place, and everything
measured afterwards silently belongs to the wrong build. This happened twice:
once producing a profile whose kernel mix made no sense, and once leaving a
force-MMQ binary in place for several subsequent measurements. **Always rebuild
after `--revert`, and check `git status` plus a known-good throughput number
before trusting a profile.**

**2. A benchmark that is faster because it is wrong.** Four configurations in
this work benchmarked faster while failing correctness:

| config | apparent gain | reality |
|---|---|---|
| `I=64` alone | +29% | 362 + 637 test failures |
| `MMQ_ITER_K=512` | +25% | 263 + 541 failures |
| `nthreads=256` alone | +23% | 11 + 595 failures |
| `nthreads=384 / I=96` | +2% | fails every quant type |

The fastest number measured in the entire project was wrong. Anything that
breaks a tiling invariant does less work, and doing less work looks exactly like
an optimisation.

Count harness failures with a plain `grep FAIL`: they print as
`[MUL_MAT] ERR = 0.128 > 0.0005   MUL_MAT(...): FAIL`, so a pattern anchored to
leading whitespace matches nothing and reports a clean run for a broken build.

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

Change sets **4-10** are generated against **upstream `ggml-org/llama.cpp`** at:

```
67b9b0e7f6ce45d929a4411907d3c48ec719e81c  llama-arch: fix DeepSeek4 APE tensor op (#25945)
```

These are different bases. Change sets 4-10 were developed and measured on the
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

Change sets 4-10 target upstream llama.cpp instead (see "Base commit" above):

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
git apply patches/10-adaptive-ubatch.patch
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
