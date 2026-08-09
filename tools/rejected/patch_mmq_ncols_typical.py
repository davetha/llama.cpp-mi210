#!/usr/bin/env python3
"""REJECTED -- CORRECT BUT SLOWER. pp2048 -0.9%, pp16384 -3.3%.

VERDICT. This is the survivor of tools/rejected/patch_mmq_mmid_ncols_max.py, and
unlike that one it is correctness-clean: test-backend-ops MUL_MAT_ID and MUL_MAT
both report 0 failures, because the grid still spans ncols_max and only the tile
width changes. The design was right. The performance was not:

    pp2048    1794.4 +/- 2.2  ->  1778.9 +/- 0.6   (-0.9%)
    pp16384   2704.4 +/- 4.4  ->  2615.7 +/- 4.6   (-3.3%)

Picking J=48 for the 88-column typical width costs ceil(2048/48) = 43 blocks
against ceil(2048/64) = 32. That ~34% increase in blocks and per-block setup
outweighs the column waste it reclaims -- the MFMA path amortises a partly-empty
tile better than the scheduler amortises extra blocks.

This closes the tile-width question from both directions. J=96 and J=128 lost
going wider (-22% at I=64); J=48 loses going narrower. The shipped J=64 is
genuinely the right width for this workload, and the "31% waste at n=88" in the
profile write-up is real but is not a bottleneck worth trading blocks for.

The original text follows unedited; its statement that the sign was unknown was
the honest position, and the answer turned out to be negative.

--- ORIGINAL (REJECTED) ---

Choose the MMQ tile width from the TYPICAL expert width, keep the grid at the bound.

BACKGROUND. mmq_args.ncols_max is used for two different things:

    mmq.cuh:1308   const int ntx      = (args.ncols_max + config.J - 1) / config.J;   // GRID
    mmq.cuh:1391   const int ntiles_x = (args.ncols_max + config.J - 1) / config.J;   // J CHOICE

The grid use genuinely needs a worst-case bound -- every token could route to one
expert -- which is why ggml_cuda_mul_mat_id passes ne12, the batch token count.
An earlier patch (tools/rejected/patch_mmq_mmid_ncols_max.py) lowered ncols_max
to the routing average and produced 158 MUL_MAT_ID failures, because it
under-sized the grid. That patch was wrong.

WHAT SURVIVED IT. The J choice does not want a bound, it wants a typical width.
With top-k routing each expert receives about n_tokens*n_expert_used/n_experts
columns -- 2048*22/512 = 88 on Nemotron-3-Super at ubatch 2048 -- but the chooser
is handed 2048 and so minimises tiles for a width no expert ever sees:

    told 2048 -> J=16:128 tiles  J=32:64  J=48:43  J=64:32  -> J=64
    told 88   -> J=16:6   tiles  J=32:3   J=48:2   J=64:2   -> J=48

At 88 real columns J=64 spends 2 tiles * 64 = 128 column slots; J=48 spends
2 * 48 = 96. The MFMA path does the full tile's arithmetic either way.

THIS PATCH splits the two uses. ncols_max keeps its meaning and keeps sizing the
grid; a new ncols_typical feeds the chooser only. The grid still covers the full
bound at whatever J is picked -- ceil(2048/48) = 43 blocks instead of 32 -- so
every column is still covered and correctness is unchanged. For the dense path
ncols_typical == ncols_max, so nothing there moves.

THE SIGN IS GENUINELY UNKNOWN. 43 blocks instead of 32 is ~34% more blocks and
more per-block setup, traded against less wasted column work inside each. Which
dominates is an empirical question and this patch exists to answer it, not to
assert it. Three predictions in this project were wrong today -- 4x on a
requant speedup, the direction on sparse-MoE decode bandwidth, and the claim
that the rejected patch above was correctness-neutral.

GATE. test-backend-ops MUL_MAT_ID with a plain `grep FAIL` (the previous attempt
scored 158, and an under-covered grid benchmarks FASTER because it does less
work), then pp2048/pp16384 against the unpatched build.

    python patch_mmq_ncols_typical.py [--check] [--revert]
"""
import sys

MARKER = "MI210_NCOLS_TYPICAL"
CUH = "ggml/src/ggml-cuda/mmq.cuh"
CU  = "ggml/src/ggml-cuda/mmq.cu"

STRUCT_ORIG = """    int64_t ncols_max;
};"""
STRUCT_NEW = """    int64_t ncols_max;
    int64_t ncols_typical;   // MI210_NCOLS_TYPICAL: tile-width selection only, never grid sizing
};"""

# J selection (NOT the grid at :1308) reads the typical width.
SWITCH_ORIG = """        const int ntiles_x = (args.ncols_max + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {"""
SWITCH_NEW = """        // MI210_NCOLS_TYPICAL: pick the tile shape for the width an expert actually
        // sees. The grid at launch_mul_mat_q still spans ncols_max, so a narrower
        // J only means more blocks, never uncovered columns.
        const int ntiles_x = (args.ncols_typical + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {"""

# Dense path: typical == max, no behaviour change.
DENSE_ORIG = """            ne03, ne13, s03, s13, s3,
            ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;"""
DENSE_NEW = """            ne03, ne13, s03, s13, s3,
            ne1, ne1};   // MI210_NCOLS_TYPICAL: dense path, typical == max
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;"""

# MoE path: grid keeps ne12; the chooser gets the routing average.
MOE_ORIG = """    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12};"""
MOE_NEW = """    // MI210_NCOLS_TYPICAL: ne12 remains ncols_max because the grid must cover the
    // worst case (every token to one expert). The tile-width chooser instead gets
    // the width an expert typically sees under top-k routing.
    const int64_t ncols_typical = std::max<int64_t>(1, (ne12*n_expert_used) / ne02);

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_typical};"""

EDITS = [
    (CUH, "mmq_args field",   STRUCT_ORIG, STRUCT_NEW),
    (CUH, "J selection",      SWITCH_ORIG, SWITCH_NEW),
    (CU,  "dense args",       DENSE_ORIG,  DENSE_NEW),
    (CU,  "MoE args",         MOE_ORIG,    MOE_NEW),
]


def main() -> int:
    check  = "--check"  in sys.argv
    revert = "--revert" in sys.argv

    files = sorted({f for f, _, _, _ in EDITS})
    srcs = {f: open(f).read() for f in files}
    patched = any(MARKER in s for s in srcs.values())

    if check:
        for f in files:
            print(f"{'PATCHED' if MARKER in srcs[f] else 'not patched'}  {f}")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        for f, _, o, n in EDITS:
            srcs[f] = srcs[f].replace(n, o)
        for f in files:
            open(f, "w").write(srcs[f])
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.",
              file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    for f, name, o, _ in EDITS:
        if srcs[f].count(o) != 1:
            print(f"ERROR: anchor '{name}' matched {srcs[f].count(o)} times, expected 1.",
                  file=sys.stderr)
            return 1
    for f, _, o, n in EDITS:
        srcs[f] = srcs[f].replace(o, n, 1)
    for f in files:
        open(f, "w").write(srcs[f])
    print("patched: tile width chosen from ncols_typical, grid still spans ncols_max")
    return 0


if __name__ == "__main__":
    sys.exit(main())
