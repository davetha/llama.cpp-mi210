#!/usr/bin/env python3
"""REJECTED -- NO EFFECT, and that is the useful part. pp2048 -0.2%, pp16384 -0.3%.

VERDICT. Correctness-clean (0 failures on MUL_MAT_ID and MUL_MAT) and completely
flat on throughput -- both deltas are inside the error bars.

That result is worth more than the patch. It was built as a cheap discriminator
for a much larger question: the grid is provisioned from ncols_max, so ~94% of
MoE blocks own no columns, and a known-wrong probe that shrank the grid ran
+23.1% / +23.5%. The open question was whether that ceiling came from empty-block
overhead (fixable with a grid-stride loop over jt, a real kernel restructure) or
from the probe simply skipping real work (not fixable at all).

If empty blocks cost 23%, letting them retire before a J-int shared load and a
__syncthreads() -- roughly 10x cheaper per block -- would have moved the number.
It moved nothing. So empty blocks were already nearly free, the +23% was mostly
work the wrong build never did, and THE GRID-STRIDE RESTRUCTURE WOULD NOT PAY.
Four lines answered a question that would otherwise have cost a kernel rewrite.

The methodological error to avoid repeating: an upper-bound probe that also
removes real work bounds nothing useful unless you know how much real work it
removed. The +23% was quoted as a ceiling on empty-block waste when it was only
a ceiling on "skip the work entirely", which was never on offer.

The original text follows unedited.

--- ORIGINAL (REJECTED) ---

Let empty MoE tiles retire before the shared-memory load and the syncthreads.

WHY. ggml_cuda_mul_mat_id launches grid (nty, ntx, n_experts) where
ntx = ceil(ncols_max / J) and ncols_max is the worst case of every token routing
to one expert. For ffn_down_exps [2688,1024,512] at ubatch 2048 with I=64, J=64
that is (16, 32, 512) = 262,144 blocks, while each expert holds about 88 columns
and so needs 2 of the 32 jt slices. Roughly 94% of blocks have nothing to do.

Measured ceiling on that waste: forcing ntx small (a deliberately INCORRECT
build, 158 MUL_MAT_ID failures, see tools/rejected/patch_mmq_mmid_ncols_max.py)
runs pp2048 +23.1% and pp16384 +23.5%. That is an upper bound -- it also skips
real work for experts above 2*J tokens -- but most of it is empty blocks.

WHAT THIS DOES. It does not change the block count. It only lets a block that
owns no columns exit immediately, instead of first loading J ints into
ids_dst_shared and passing a __syncthreads(). If the cost is that prologue, this
captures most of the ceiling for four lines. If the cost is raw dispatch, this
captures little -- and that is itself the answer, and tells us whether the much
larger grid-stride restructure is worth building.

WHY IT IS SAFE. The condition is block-uniform: jt comes from blockIdx, and
col_low/col_high are read identically by every thread. So the whole block
returns together and no thread is left waiting at a __syncthreads() its peers
skipped. It is also semantically a no-op -- an empty tile has
j_max = col_diff - jt*J - 1 < 0, so the existing inner loop already returns on
its first iteration without writing anything. This only makes that cheaper.

GATE. test-backend-ops MUL_MAT_ID counted with a plain `grep FAIL`. Doing less
work is what a wrong build looks like here, and this patch's whole purpose is to
do less work, so the gate is not optional.

    python patch_mmq_mmid_early_out.py [--check] [--revert]
"""
import sys

TARGET = "ggml/src/ggml-cuda/mmq.cuh"
MARKER = "MI210_MMID_EARLY_OUT"

ORIG = """    const int col_diff = col_high - col_low;

    for (int j = threadIdx.y*warp_size + threadIdx.x; j < J; j += nwarps*warp_size) {"""

NEW = """    const int col_diff = col_high - col_low;

    // MI210_MMID_EARLY_OUT: this block owns no columns of this expert. The grid
    // is sized from ncols_max (every token to one expert), so most blocks land
    // here. Block-uniform, and the tile would have written nothing anyway --
    // j_max below would be negative on the first iteration.
    if (jt*J >= col_diff) {
        return;
    }

    for (int j = threadIdx.y*warp_size + threadIdx.x; j < J; j += nwarps*warp_size) {"""


def main() -> int:
    src = open(TARGET).read()
    patched = MARKER in src

    if "--check" in sys.argv:
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}")
        return 0 if patched else 1

    if "--revert" in sys.argv:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        open(TARGET, "w").write(src.replace(NEW, ORIG))
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt.",
              file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    if src.count(ORIG) != 1:
        print(f"ERROR: anchor matched {src.count(ORIG)} times, expected 1.", file=sys.stderr)
        return 1

    open(TARGET, "w").write(src.replace(ORIG, NEW, 1))
    print("patched: empty MoE tiles now retire before the prologue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
