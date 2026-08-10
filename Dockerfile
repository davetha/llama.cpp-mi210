# Reproducible gfx90a (MI210) build of the CDNA2 prefill work in this repo.
#
#   docker build -t llama-mi210:ssd .
#   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
#     --ipc=host --shm-size 16g -v /path/to/models:/models -p 8080:8080 \
#     llama-mi210:ssd -m /models/<model>.gguf -ngl 99 -sm layer -np 1 \
#     -c 32768 -b 4096 -ub 2048 -fa on -ctk q8_0 -ctv q8_0 --host 0.0.0.0
#
# Covers change sets 4-11 (patches/04-*.patch .. 11-*.patch). The turboquant
# work in patches 01-03 targets a different base tree -- see BUILD.md.

ARG ROCM_TAG=7.1.4-complete
FROM rocm/dev-ubuntu-24.04:${ROCM_TAG}

# Base commit these patches are cut against.
ARG LLAMA_REF=030ebb558

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential python3 libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp . \
    && git checkout ${LLAMA_REF}

COPY patches/ /patches/
# 04-13 only: 01-03 belong to the turboquant lineage and do not apply here.
# Note 13 is self-contained -- it bundles upstream PR #26001 (unmerged, pinned
# at 1e1885f3d) with the CDNA fixes that make it run, so it applies to a bare
# ${LLAMA_REF} checkout like 04 does. If that PR merges or is force-pushed
# upstream, 13 must be re-cut; see the README.
RUN set -eux; \
    for p in /patches/0[4-9]-*.patch /patches/1[0-3]-*.patch; do \
      echo "applying $(basename "$p")"; git apply --3way "$p"; \
    done

# GPU_TARGETS=gfx90a only -- MI210 is CDNA2. HIP_MMQ_MFMA is what the retuned
# MMQ tiles in change set 6 depend on. RPC is ON so a two-process-per-card setup
# can also be built from this image, though it is no longer needed for
# stability: change set 12 + GGML_CUDA_REGISTER_HOST=1 makes the in-process
# 2-card path safe (see USAGE.md and the README's fault section).
RUN cmake -B build \
      -G Ninja \
      -DGPU_TARGETS=gfx90a \
      -DGGML_HIP=ON \
      -DGGML_HIP_MMQ_MFMA=ON \
      -DGGML_HIP_GRAPHS=ON \
      -DGGML_HIP_NO_VMM=ON \
      -DGGML_HIP_ROCWMMA_FATTN=OFF \
      -DGGML_RPC=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_CURL=OFF \
    && cmake --build build --config Release -j"$(nproc)"

# AMD's rocBLAS, not Ubuntu's. Worth +2.7% and silent if wrong -- nothing errors,
# it just links the slower library. See change set 11 in the README.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib
ENV PATH=/src/build/bin:${PATH}

EXPOSE 8080
ENTRYPOINT ["/src/build/bin/llama-server"]
