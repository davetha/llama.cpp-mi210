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

Scoped to the K-quants (Q2_K–Q6_K, 35 entries), which is what an `i1-Q4_K_M`
model dispatches; leaving other types alone keeps the result attributable.

| | before | after |
|---|---:|---:|
| pp4096 | 1624 t/s | **1684 t/s** (+3.7%) |
| pp16384 | 2114 t/s | **2192 t/s** (+3.7%) |

**Stated plainly:** the stream-k/MoE evidence is from RDNA3.5/RDNA4, *not*
CDNA2, and no CDNA stream-k benchmark exists upstream. This was an experiment
that happened to pay off, not the transfer of a known result. An earlier
experiment on the same theory — extending CDNA3's rocBLAS carve-out to CDNA2 —
came back **6.5% slower** and was discarded.

**Correctness:** `test-backend-ops -o MUL_MAT_ID` 790/790 on both MI210s.

**Files (1):** `ggml/src/ggml-cuda/mmq-config-cdna.cuh`.

---

## Combined result (change sets 4 + 5)

`llama-bench`, 2× MI210, Nemotron-3-Super-120B-A12B `i1-Q4_K_M`,
`-b 4096 -ub 2048 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 -t 24 -r 2`:

| build | pp4096 (t/s) | pp16384 (t/s) | vs base |
|---|---:|---:|---:|
| upstream `67b9b0e` | 1366 | 1775 | — |
| + SSD on CDNA | 1624.07 ± 0.81 | 2114.21 ± 0.73 | +19.1% |
| + stream-k off | 1683.57 ± 0.40 | **2192.26 ± 0.84** | **+23.5%** |

For reference, vLLM with its AITER fast paths reaches 4,070 t/s at 16k on the
same hardware with an AWQ-INT4 Nemotron. These changes narrow the gap from
2.29× to 1.86×; they do not close it.

---

## Verifying correctness

Both change sets were gated on **reading generated tokens**, not only on the
benchmark number. This is not ceremony: a fast kernel emitting garbage already
cost this project one published benchmark.

The specific risk in change set 4 is that the SSD path chains batched GEMMs with
`beta=1` accumulation for inter-chunk state propagation and materialises a
causal decay mask in a helper kernel. A wrong transpose flag, stride, or
alpha/beta does not crash — it propagates a subtly wrong SSM state and yields
fluent, confident, **wrong** text.

Procedure: `llama-server` at `temperature 0` with a 240-token prompt (past the
128-token SSD threshold), asking the model to explain why prefill is
compute-bound and decode is bandwidth-bound. The answer had to use the figures
supplied in the prompt correctly — something a corrupted SSM state would not do.
Multi-request runs were done on a single card because of the pre-existing fault
below; single-request verification was done on both cards.

---

## Known pre-existing issue: multi-GPU fault on sequential requests

**Not caused by change sets 4 or 5**, but you will hit it, so it is recorded here.

Running `llama-server` across both MI210s (`-sm layer`), the *second* sequential
request faults:

```
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x... Reason: Unknown.
```

The first request returns correct output; the next one launches
(`slot launch_slot_: id 3 | task 263`) and the GPU faults. The process survives
and keeps answering `/health` with `ok`, so it looks alive while being unable to
serve — which makes it easy to misdiagnose.

Isolated by bisecting configuration rather than assuming:

| build | GPUs | result |
|---|---|---|
| upstream `67b9b0e`, unpatched | 2 | **faults on request 2** |
| + SSD | 2 | faults on request 2 |
| + SSD + stream-k off | 2 | faults on request 2 |
| + SSD | 1 (`ROCR_VISIBLE_DEVICES=0`, `-ngl 62`) | 3 sequential requests clean |

Since the **unpatched baseline faults identically**, this is pre-existing on
this tree and unrelated to either change set. It is specific to the multi-GPU
split; the same build on a single card handles repeated requests without
incident.

`llama-bench` does not surface it across many prefills, which points at state
reuse between requests rather than any prefill kernel — and means the throughput
numbers above are unaffected.

Not root-caused. A good starting point is bisecting llama.cpp between `67b9b0e`
and current master with a two-request `llama-server` script on two cards.

---

## Tested and rejected (change sets 4–5)

Recorded so they are not re-attempted.

| change | result |
|---|---|
| Extend CDNA3's rocBLAS carve-out to CDNA2 in `mmq.cu` (`ggml_cuda_should_use_mmq` true for Q4_K/Q5_K at any `ne11`) | **6.5% slower** (1277 vs 1366 t/s at pp4096). rocBLAS genuinely beats MMQ at `ne11=2048` on gfx90a. |
| `ROCBLAS_USE_HIPBLASLT=1` | no-op |
| `-sm row` | unsupported on ROCm |
| Crossing the attention MFMA gate by batching (`fattn.cu` needs `Q->ne[1] * gqa_ratio > 16`; Nemotron's ratio is exactly 16 at batch 1) | no gain; flash attention is only 0.9% of prefill |
| `-ub 4096` | slower than 2048 |
| `-ub 1024` | better at pp4096, worse at pp16384 |
| W4A8 weights | dead end on CDNA2 twice over: `mfma.i32.16x16x32.i8` (K=32) is MI300-only and fails to select on gfx90a, and CDNA2 gives INT8 and BF16 the *same* 181 TOPS peak, so there is no throughput to win |
| Making `ssm_scan_f32_group` wavefront-aware (it indexes lanes with the hardcoded `WARP_SIZE` 32 while gfx90a's wavefront is 64, so `warp_reduce_sum` takes a slow path once per token) | abandoned: `c_factor` serves double duty as warps-per-block *and* state-elements-per-lane, so changing it requires reworking the grid too or `state[]` reads out of bounds. Change set 4 makes it moot for prefill — the scalar kernel now only handles sequences under 128 tokens. |

Confirmed already optimal at base, so not worth revisiting: MMQ already uses
int8 MFMA (`mfma_i32_16x16x16i8` via `AMD_MFMA_AVAILABLE`), CUDA graphs are
active (188 graph reuses per request), `GGML_HIP_MMQ_MFMA=ON`.

---

## What is left

MMQ is now 53.3% of prefill and is the obvious next target:

1. **Retune `SSM_SSD_CHUNK_SIZE`** (256) for rocBLAS's kernel-selection sweet
   spots on gfx90a.
2. **Extend the stream-k experiment** beyond K-quants, and sweep the other MMQ
   config fields (`nthreads`, `occupancy`, `I`/`J` tile shape) for CDNA2 — one
   shared CDNA config file is unlikely to suit all three CDNA generations.
3. **`mm_ids_helper`** was 4.5% at base; re-measure its share now.
4. Root-cause the multi-GPU fault above.

---

## Base commit

Change sets **1-3** are generated against the `TheTom/llama-cpp-turboquant`
fork at:

```
c26cbdffcf6fc9b7430cd6b117757e9a3f70b7ea  Merge pull request #225 from TheTom/fix-ui-assets-partial-dist
```

Change sets **4-5** are generated against **upstream `ggml-org/llama.cpp`** at:

```
67b9b0e7f6ce45d929a4411907d3c48ec719e81c  llama-arch: fix DeepSeek4 APE tensor op (#25945)
```

These are different bases. Change sets 4-5 were developed and measured on the
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

Change sets 4-5 target upstream llama.cpp instead (see "Base commit" above):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 67b9b0e7f6ce45d929a4411907d3c48ec719e81c
git apply patches/04-ssd-mamba2-prefill-cdna.patch
git apply patches/05-mmq-cdna-no-streamk.patch
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
