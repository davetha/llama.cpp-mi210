# llama.cpp-mi210

> **Fork of [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)** (itself a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)) with three change sets optimized for **AMD MI210 (gfx90a / CDNA2)** inference.

This repo does **not** contain the full llama.cpp tree (too large to mirror here). Instead it ships:

```
patches/          — apply these on top of the upstream fork
modified-files/   — the exact modified files (drop-in replacements)
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

## Base commit

All patches are generated against the `TheTom/llama-cpp-turboquant` fork at:

```
c26cbdffcf6fc9b7430cd6b117757e9a3f70b7ea  Merge pull request #225 from TheTom/fix-ui-assets-partial-dist
```

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

The [`modified-files/`](modified-files/) directory contains the final state of every changed file if you prefer drop-in replacement over `git apply`.

---

## Related

- **Hub repo:** [`davetha/mi210-llm-stack`](https://github.com/davetha/mi210-llm-stack) — full optimization write-up, architecture, and all guides.
- **Triton TurboQuant:** [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd) — the wave64-safe GEMM-based WHT alternative.

## License

MIT. The underlying llama.cpp is MIT-licensed.
