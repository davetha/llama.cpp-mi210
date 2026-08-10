# Usage guide

Serving **NVIDIA-Nemotron-3-Super-120B-A12B** (and other large MoE / hybrid-Mamba
models) on **2× AMD MI210** with llama.cpp, using this repo's patches. What you
get over stock llama.cpp on this hardware:

| | stock | this fork |
|---|---:|---:|
| prefill, 2k prompt | ~1050 t/s | **~1850 t/s** |
| prefill, 16k prompt | ~1100 t/s | **~2770 t/s** |
| decode | ~54 t/s | ~55 t/s |
| 2-GPU prompt cache | **crashes** (ROCm bug) | **stable** (change set 12) |

Everything below was measured on 2× MI210 (gfx90a), ROCm 7.14, and is the
configuration actually serving on that machine. The [README](README.md) is the
engineering narrative — why each change works; this page is just how to run it.

---

## 1. Build the image

```bash
git clone https://github.com/davetha/llama.cpp-mi210.git
cd llama.cpp-mi210
docker build -t llama-mi210 .
```

The Dockerfile clones upstream `ggml-org/llama.cpp` at the pinned base commit,
applies `patches/04-*` … `12-*`, and builds for `gfx90a` only with the flags
the patches assume (`GGML_HIP_MMQ_MFMA=ON` is required — change set 6's retuned
MMQ tiles depend on it). Patches 01–03 belong to a different lineage
([BUILD.md](BUILD.md)) and are not part of this image.

## 2. Get the model (and requantize it — this matters)

Stock `Q4_K_M` GGUFs of Nemotron are silently mis-quantized: the expert FFN
width (2688) is not divisible by 256, so `llama-quantize` upgrades all 40
`ffn_down_exps` tensors to Q8_0/Q5_0 — 46.8% of the weights at ~2× the intended
bpw. Force them to IQ4_NL:

```bash
docker run --rm -v /path/to/models:/models --entrypoint /src/build/bin/llama-quantize \
  llama-mi210 --allow-requantize --tensor-type ffn_down_exps=iq4_nl \
  /models/nemotron-120b-q4km.gguf /models/nemotron-120b-iq4nl.gguf Q4_K_M
```

| | PPL | size | prefill | decode |
|---|---|---:|---:|---:|
| stock Q4_K_M | 3.5272 ± 0.045 | 80.1 GiB | 1794 | 53.6 |
| **IQ4_NL down_exps** | **3.5192 ± 0.044** | **63.7 GiB** | **1844** | **55.0** |

Better on every axis — 16 GiB smaller, slightly faster, no quality loss.

## 3. Run it

The canonical 2-card serving command:

```bash
docker run -d --name nemotron --restart unless-stopped \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -v /path/to/models:/models -p 8038:8038 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib \
  -e GGML_CUDA_REGISTER_HOST=1 \
  --entrypoint /src/build/bin/llama-server llama-mi210 \
  -m /models/nemotron-120b-iq4nl.gguf \
  -ngl 99 -sm layer -np 4 -c 1048576 \
  -b 4096 -ub 2048 -fa on -ctk q8_0 -ctv q8_0 \
  -t 24 -cram 16384 \
  --host 0.0.0.0 --port 8038 -a nemotron
```

### The flags that carry weight

| flag | why |
|---|---|
| `-e GGML_CUDA_REGISTER_HOST=1` | **Required for a stable prompt cache on 2 GPUs.** Activates change set 12's pinned state restore, working around an open ROCm runtime bug ([ROCm/rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817)). Without it (or on stock builds), set `-cram 0` instead — or the server will eventually hit a GPU memory fault and hang. |
| `-e LD_LIBRARY_PATH=...` | Selects AMD's rocBLAS over Ubuntu's. Worth +2.7%, and omitting it fails **silently**. |
| `-c 1048576 -np 4` | Total KV split across slots: 4 slots × **256k tokens each**. Nemotron is natively 1M-context (no RoPE scaling in the GGUF), and the hybrid-Mamba design makes long context cheap — attention KV is only ~10 KiB/token. Measured VRAM per card: ~34 GiB at 64k total, ~40 GiB at 1M total. |
| `-ub 2048` | The prefill sweet spot. `-ub 4096` measured 7% *slower*. Change set 10 shrinks it adaptively for short prompts. |
| `-ctk q8_0 -ctv q8_0` | Halves KV memory; no measurable quality or speed cost here. |
| `-cram 16384` | Prompt cache (MiB, in host RAM). Multi-turn conversations and agent loops skip re-prefilling their shared prefix. A ~72k-token agent system prompt restores in ~1 s instead of re-prefilling for ~30 s. Size it to your RAM; entries are ~0.5 GiB + ~10 KiB/token. |
| `-fa on` | Flash attention. Required for the q8_0 KV cache. |

### Verify it's healthy

```bash
curl -s localhost:8038/health          # {"status":"ok"}
curl -s localhost:8038/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":8,"temperature":0}'
```

Expect ~1850 t/s prefill on 2k prompts, rising to ~2770 t/s at 16k, and ~54 t/s
decode, roughly flat with context length.

---

## Serving stack integration

### litellm

```yaml
model_list:
  - model_name: nemotron
    litellm_params:
      model: openai/nemotron
      api_base: http://YOUR_HOST:8038/v1
      api_key: "none"
      timeout: 3600
```

### open-webui

Point an OpenAI connection at litellm (or directly at `:8038/v1`). Nothing else
needed — the model list follows the endpoint.

### opencode

```jsonc
"provider": {
  "mi210": {
    "npm": "@ai-sdk/openai-compatible",
    "name": "big (litellm)",
    "options": { "baseURL": "http://YOUR_HOST:4000/v1", "apiKey": "sk-..." },
    "models": {
      "nemotron": {
        "name": "Nemotron-3-Super-120B (256K)",
        "limit": { "context": 262144, "output": 32768 }
      }
    }
  }
}
```

Agent CLIs are the best case for the prompt cache: their huge fixed system
prompt is prefilled once and restored from cache on every subsequent call.

---

## Troubleshooting

**GPU memory fault, then the server hangs but `/health` still says ok.**
```
Memory access fault by GPU node-1 (...) on address 0x7...
```
This is the ROCm multi-GPU pageable-copy bug (README: ["The 2-card concurrency
fault"](README.md#the-2-card-concurrency-fault-resolved)). You are running 2+
GPUs with the prompt cache or context checkpoints enabled, without the fix.
Either add `GGML_CUDA_REGISTER_HOST=1` (this fork, change set 12) or disable
the cache (`-cram 0`, and `--ctx-checkpoints 0` if you use checkpoints). Note
the hung process **passes health checks** — restart policies won't save you.

**Prefill much slower than the numbers above.** Check `LD_LIBRARY_PATH` (Ubuntu
rocBLAS costs ~3%, silently) and confirm the image was built with
`GGML_HIP_MMQ_MFMA=ON` (the Dockerfile does this; a stock build loses the MMQ
retune).

**Model loads but answers are garbage at long context.** Check the GGUF's
`context_length` metadata before raising `-c` — this fork's numbers are for
Nemotron's native 1M. Other models need RoPE scaling flags past their trained
length.

**Out of VRAM.** Drop `-c` first (KV is the only thing that scales with it),
then `-np`. The model itself needs ~32 GiB per card at IQ4_NL.
