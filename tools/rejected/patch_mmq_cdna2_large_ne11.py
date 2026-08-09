#!/usr/bin/env python3
"""REJECTED -- MEASURED SLOWER, THREE TIMES. Read the verdict before the pitch.

VERDICT. Applied and benchmarked at three different MMQ configurations. Every
one was a regression on pp16384: **-6.5%, -10.5%, -9.2%**. The hypothesis below
-- that CDNA2 was left on rocBLAS by omission rather than by measurement -- is
reasonable and turns out to be false for gfx90a at ne11 = 2048. rocBLAS/Tensile
wins the dense Q4_K/Q5_K GEMMs on this architecture, and it is not close.

Two later results reinforce it rather than undercut it:
  - AMD's own rocBLAS (not Ubuntu's) is worth +2.7% on its own -- the dense path
    was slow because the wrong library was linked, not because it was rocBLAS.
    See change set 11 / the LD_LIBRARY_PATH note in the README.
  - 908 candidate Tensile solutions were timed per shape and none beat the
    default, so the shipped kernels are already the right ones.

The dense GEMM path is closed. The remaining headroom is the MoE expert path
(m=2688, n=88, k=1024 per expert), which runs at roughly 0.4x the FLOP
efficiency of the dense path -- see "grouped / fused MoE GEMM" in the README.

Kept because "we tried the obvious thing and it lost by 10%" is worth exactly as
much as a patch that won. The original text follows unedited.

--- ORIGINAL (REJECTED) ---

Let CDNA2 (gfx90a) use MMQ for large-batch Q4_K/Q5_K instead of rocBLAS.

WHY. Profiling a 4096-token prefill of Nemotron-3-Super-120B (Q4_K_M) on 2x MI210
with rocprofv3 put ~13% of all GPU kernel time in rocBLAS/Tensile `Cijk_*`
kernels (995 ms of 7703 ms). Those are the DENSE matmuls: ggml_cuda_should_use_mmq
only returns true for Q4_K on CDNA when `ne11 <= 256`, and prefill runs at
ne11 = ubatch = 2048, so they fall through to rocBLAS. The MoE expert matmuls are
unaffected -- they hit the earlier `n_experts > 64` branch (Nemotron has 256).

The upstream code already carves CDNA3 out of rocBLAS entirely:

    // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt
    // performs better but is currently suffering from a crash on this architecture.
    if (GGML_CUDA_CC_IS_CDNA3(cc)) {
        return true;
    }

CDNA2 never got that treatment -- not because it was measured and found fine, but
because nobody measured it. This patch extends the same carve-out to CDNA2 so the
question can be answered by benchmark rather than assumption.

This is an EXPERIMENT, not a known win. rocBLAS may well be faster than MMQ at
ne11=2048 on gfx90a, in which case this makes prefill slower and should be
reverted. A/B it against the unpatched binary on the same model and flags, and
read the generated tokens both times -- a wrong-but-fast kernel has already cost
this project one published benchmark.

    python patch_mmq_cdna2_large_ne11.py [--check] [--revert]
"""
import sys

TARGET = "ggml/src/ggml-cuda/mmq.cu"

ORIG = """        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }"""

PATCHED = """        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        // MI210_MMQ_CDNA2: CDNA2 gets the same carve-out as CDNA3 above. Measured
        // on gfx90a: rocBLAS/Tensile Cijk_* kernels were ~13% of prefill GPU time
        // on a Q4_K 120B at ne11=2048, where MMQ was never even considered.
        if (GGML_CUDA_CC_IS_CDNA2(cc) && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }"""


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv

    with open(TARGET) as f:
        src = f.read()
    patched = "MI210_MMQ_CDNA2" in src

    if check:
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        with open(TARGET, "w") as f:
            f.write(src.replace(PATCHED, ORIG, 1))
        print("reverted")
        return 0

    if patched:
        print("already patched")
        return 0

    n = src.count(ORIG)
    if n != 1:
        print(f"ERROR: anchor matched {n} times, expected 1 -- upstream moved; "
              "re-derive rather than forcing.", file=sys.stderr)
        return 1

    with open(TARGET, "w") as f:
        f.write(src.replace(ORIG, PATCHED, 1))
    print("patched: CDNA2 now uses MMQ for Q4_K/Q5_K at any ne11")
    return 0


if __name__ == "__main__":
    sys.exit(main())
