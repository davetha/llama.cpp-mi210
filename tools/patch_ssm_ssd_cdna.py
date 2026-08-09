#!/usr/bin/env python3
"""Enable llama.cpp's chunked-SSD Mamba-2 prefill path on CDNA (gfx90a / MI210).

BACKGROUND. Upstream commit b62b350 (PR #22675, merged 2026-07-28) replaces the
sequential SSM scan with a chunked State-Space-Duality formulation: per-chunk
intra-chunk output and chunk-final-state become batched GEMMs, leaving only a
short scan over n_tok/256 chunk boundaries. The heavy GEMMs run FP16-in /
FP32-accumulate via cublasGemmStridedBatchedEx.

It is gated off for HIP by `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`
plus a runtime `GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING`. The PR
author states the change "does not affect ... HIP" -- a scoping decision, not a
finding that it cannot work on AMD.

WHY IT SHOULD WORK ON CDNA2. Profiling pp4096 of Nemotron-3-Super-120B (Q4_K_M)
on 2x MI210 puts ssm_scan_f32_group at 22.8% of GPU time (1759 ms / 7703 ms), and
that kernel is scalar FP32 with zero matrix-core use. The SSD path converts that
work into FP16 GEMMs, which is gfx90a's 181 TFLOPS v_mfma_f32_16x16x16f16 path
rather than its 22.6 TFLOPS vector path. Every cuBLAS symbol the kernel uses
(cublasGemmStridedBatchedEx, cublasSgemm, cublasSetStream, CUBLAS_OP_*,
CUDA_R_16F/32F, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) already has a hipBLAS
alias in ggml-cuda/vendors/hip.h, and ggml exercises those same aliases today in
ggml_cuda_mul_mat_batched_cublas. CUB is not required: USE_CUB stays undefined on
HIP and the file ships a working shared-memory sequential-scan fallback.

SCOPE. Gated to CDNA specifically, not blanket AMD. RDNA's matrix cores behave
differently (WMMA, different tile shapes) and are entirely unvalidated here;
opening this up for all AMD would be shipping an untested path to other people's
hardware.

CORRECTNESS IS THE WHOLE RISK. The SSD path chains batched GEMMs with beta=1
accumulation for inter-chunk state propagation and materializes a causal decay
mask in a helper kernel. A wrong transpose flag, stride, or alpha/beta does not
crash -- it propagates a subtly wrong SSM state and yields fluent, confident,
WRONG text. Gate on `test-backend-ops -o SSM_SCAN` (which carries the
Nemotron-9B-shaped multi-chunk cases added by the same commit) BEFORE trusting
any throughput number, then read real generated tokens.

    python patch_ssm_ssd_cdna.py [--check] [--revert]
"""
import sys

TARGET = "ggml/src/ggml-cuda/ssm-scan.cu"

# The two `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)` guards that
# wrap (a) the SSD kernel definitions and (b) the dispatch site. MUSA stays
# excluded -- it is untested here and not ours to enable.
#
# The trailing newline is load-bearing. Line 1 of the file is
#   #if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
# which *contains* this text as a substring and controls USE_CUB. Matching it
# would enable hipCUB, whose headers collide with ggml's own __trap macro (the
# reason upstream PR #26388 stalled). Anchoring to end-of-line excludes it.
GUARD_ORIG = "#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)\n"
GUARD_NEW = (
    "// SSD_CDNA: HIP admitted here; the runtime `use_ssd` test below still\n"
    "// restricts this to CDNA. MUSA remains excluded (untested).\n"
    "#if !defined(GGML_USE_MUSA)\n"
)

# Matching #endif trailer comments, so the file still reads correctly. Same
# substring hazard as above -- the USE_CUB #endif shares this prefix.
ENDIF_ORIG = "#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)\n"
ENDIF_NEW = "#endif // SSD_CDNA: !defined(GGML_USE_MUSA)\n"

# The runtime capability test. Keep NVIDIA's condition exactly as upstream has
# it and add CDNA alongside, so this cannot change behaviour on NVIDIA.
COND_ORIG = """                      && GGML_CUDA_CC_IS_NVIDIA(cc)
                      && cc >= GGML_CUDA_CC_TURING
                      && nr % 8 == 0;  // cuBLAS requires 8-element (16-byte) alignment"""

COND_NEW = """                      && ((GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING)
                          // SSD_CDNA: CDNA has the FP16 matrix cores this path wants
                          // (v_mfma_f32_16x16x16f16) and full hipBLAS aliases for the
                          // batched GEMMs. Deliberately NOT all of AMD: RDNA's WMMA
                          // path is unvalidated here.
                          || GGML_CUDA_CC_IS_CDNA(cc))
                      && nr % 8 == 0;  // cuBLAS requires 8-element (16-byte) alignment"""

EDITS = [
    ("HIP guards", GUARD_ORIG, GUARD_NEW, 2),
    # Only the kernel block's #endif carries a trailer comment; the dispatch
    # block closes with a bare `#endif`, which stays balanced and needs no edit.
    ("endif trailers", ENDIF_ORIG, ENDIF_NEW, 1),
    ("use_ssd capability test", COND_ORIG, COND_NEW, 1),
]
MARKER = "SSD_CDNA"


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv

    with open(TARGET) as f:
        src = f.read()
    patched = MARKER in src

    if check:
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        for _, orig, new, _ in EDITS:
            src = src.replace(new, orig)
        with open(TARGET, "w") as f:
            f.write(src)
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.", file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    # Verify every anchor's occurrence count up front. A half-applied set of
    # preprocessor guards produces an unbalanced #if/#endif and a wall of
    # confusing compiler errors far from the real cause.
    for name, orig, _, want in EDITS:
        n = src.count(orig)
        if n != want:
            print(f"ERROR: anchor '{name}' matched {n} times, expected {want}. "
                  "Upstream moved; re-derive rather than forcing.", file=sys.stderr)
            return 1
    for name, orig, new, _ in EDITS:
        src = src.replace(orig, new)
    with open(TARGET, "w") as f:
        f.write(src)
    print("patched: chunked-SSD Mamba-2 prefill now enabled on CDNA")
    print("NEXT: test-backend-ops -o SSM_SCAN must pass before any benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
