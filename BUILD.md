# Building for gfx90a (AMD MI210)

> **Scope note.** This page documents the **turboquant lineage** — change sets
> 1–3 (`patches/01-*` .. `03-*`), which are cut against
> `llama-cpp-turboquant`, not upstream llama.cpp.
>
> For the **CDNA2 prefill work** (change sets 4–11, the SSD/MMQ/rocBLAS
> material that the README is mostly about) use the
> [`Dockerfile`](Dockerfile) in the repo root. It pins upstream
> `ggml-org/llama.cpp` at `67b9b0e`, applies patches 04–11, and builds with the
> flags those change sets assume — including `GGML_HIP_MMQ_MFMA=ON`, which the
> retuned tiles in change set 6 depend on, and the `LD_LIBRARY_PATH` that picks
> AMD's rocBLAS over Ubuntu's.


Build inside a **ROCm 7.14** Docker container. The host kernel is untouched;
all GPU work happens in containers with `/dev/kfd` and `/dev/dri` passed through.

## Prerequisites

- Docker image with ROCm 7.14 + cmake + hipcc + git. On the `big` host this is
  the prebuilt `llama-rocm714:latest` image.
- Both MI210s exposed to the container.

## 1. Clone + apply patches

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git checkout c26cbdffcf6fc9b7430cd6b117757e9a3f70b7ea

# Apply the three change sets from this repo:
git apply 01-per-layer-kv-types.patch
git apply 02-kivi2-quant-type.patch
git apply 03-turboquant-wave64-fixes.patch
```

> If you prefer drop-in replacement, the final state of every changed file is in
> [`tools/materialize_tree.sh --lineage turboquant`](tools/materialize_tree.sh) —
> clones this base commit and applies 01-03 for you.

## 2. CMake configure

```bash
cmake -B build \
  -DGPU_TARGETS=gfx90a \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_SERVER=ON
```

Key flags and **why**:

| Flag | Reason |
|------|--------|
| `-DGPU_TARGETS=gfx90a` | Compile only for MI210 (faster than autodetect). |
| `-DGGML_HIP_ROCWMMA_FATTN=OFF` | rocWMMA FlashAttention requires CDNA3+ matrix cores; MI210 (CDNA2) only has Vector Cores → must be **off**. |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | Build flash-attention instances for every KV quant type (needed for turbo/KIVI). |
| `-DLLAMA_CURL=OFF` | Avoids a curl dependency. |

> **Do not** build `llama-server` via the default target if you hit
> `llama-ui-assets` failures (the web UI needs npm). In that case use
> `--target llama-cli llama-server` explicitly — the patched fork above
> already fixes the partial-dist issue, but it's worth knowing.

## 3. Build (with ccache — see below)

```bash
cmake --build build --target ggml-hip llama-cli llama-server -- -j$(nproc)
```

Binaries land in `build/bin/`.

## 4. ccache for incremental rebuilds  → also see [`guides/setup-ccache-docker.md`](https://github.com/davetha/mi210-llm-stack/blob/main/guides/setup-ccache-docker.md) in the hub repo

GPU kernel objects are expensive to compile (the CK backend alone is ~2926
objects). ccache makes incremental rebuilds return from cache instantly.

```bash
# Install ccache inside the image (one-time):
apt-get update && apt-get install -y ccache

# Reconfigure with launchers:
cmake -B build \
  -DGPU_TARGETS=gfx90a -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache

# Mount a persistent cache dir when you run the container:
docker run ... -v /mnt/llm-storage/ccache:/root/.ccache ...
```

## 5. Incremental rebuild pattern (host mounts source into ROCm container)

The working pattern on `big` mounts the host source tree into the container at
exactly `/build/src` (the CMake cache has hardcoded paths expecting this):

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  -v /mnt/llm-storage/turbo-build/src:/build/src \
  -w /build/src/build \
  --entrypoint bash \
  llama-rocm714:latest \
  -c 'cmake --build . --target ggml-hip llama-cli -- -j$(nproc)'
```

## 6. Verify

```bash
# Quick CPU-correctness smoke (turbo3 on CPU is proven correct):
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  -v /path/to/models:/models -v $(pwd)/build/bin:/turbo-bin \
  -e LD_LIBRARY_PATH=/turbo-bin \
  --entrypoint /turbo-bin/llama-cli llama-rocm714:latest \
  -m /models/small-model.Q4_K_M.gguf \
  -p "The capital of France is" -n 10 -ngl 0 \
  -ctk turbo3 -ctv turbo3 -fa on --no-warmup --temp 0 --log-disable
```

## KIVI2 unit tests

See [`tests/`](tests/) for the KIVI2 round-trip correctness tests (exact 4-level,
endpoints, constant block, random invariance — all PASS).

---

## Docker group IDs

On `big` (Ubuntu), the render/video groups are exposed by **numeric GID**, not
name (`--group-add render` fails with "no matching entries"):

```bash
--group-add 44    # video
--group-add 991   # render
```
