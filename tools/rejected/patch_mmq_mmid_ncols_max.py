#!/usr/bin/env python3
"""REJECTED -- 158 MUL_MAT_ID FAILURES. The safety argument below is wrong.

VERDICT. Applied, built, and gated on test-backend-ops before benchmarking:
**158 MUL_MAT_ID failures**. The claim further down that this "cannot change
results -- only tile shape" is false, and it is false for a reason worth
remembering.

ncols_max has a SECOND use that this analysis missed entirely. It does not only
feed the tile-width chooser; it sizes the launch grid:

    // launch_mul_mat_q, mmq.cuh:1308
    const int ntx = (args.ncols_max + config.J - 1) / config.J;

So it is a genuine MAXIMUM -- the worst case in which every token routes to a
single expert -- and the field name means exactly what it says. Replacing it
with the routing average under-sizes the grid, and the columns beyond the
average are simply never computed. Upstream is right; this reading was wrong.

Note the failure mode this avoided. An under-covered grid does less work, so it
would have benchmarked FASTER while producing wrong output. The correctness gate
ran before the benchmark, so the fast-and-wrong number was never recorded as a
win. That ordering is the whole point of it.

WHAT SURVIVES. One observation here does hold up: ncols_max serves two different
purposes through one value. Grid sizing genuinely needs the worst-case bound;
tile-width selection would be better served by the typical per-expert width.
Splitting them is safe in a way this patch was not -- keep ncols_max for the
grid, and give mul_mat_q_switch_J a separate value, so the grid still covers the
full bound at whatever J is chosen (ceil(2048/48) = 43 blocks instead of 32).

Whether that is FASTER is unknown and the sign is genuinely unclear: 43 blocks
instead of 32 is more launch and per-block overhead, against less wasted column
work per expert. It is a cheap experiment, not a lead.

The original text follows unedited.

--- ORIGINAL (REJECTED) ---

Tell the MMQ tile-width chooser how many columns an EXPERT gets, not the batch.

WHAT IS WRONG. mul_mat_q_switch_J (mmq.cuh) picks the tile width J at runtime by
minimising ceil(args.ncols_max / J) over the widths the arch's config table
offers. On CDNA that table offers J = 16, 32, 48 and 64.

For the MoE path, ggml_cuda_mul_mat_id builds mmq_args with `ne12` as the final
field -- that is ncols_max -- and ne12 is the TOKEN COUNT of the batch:

    const mmq_args args = {
        ..., ne03, ne13, s03, s13, s3,
        ne12};                              <- ncols_max = n_tokens

But no expert sees n_tokens columns. With top-k routing each expert receives
about n_tokens * n_expert_used / n_experts of them. On Nemotron-3-Super at
ubatch 2048 that is 2048 * 22 / 512 = 88, not 2048.

WHAT IT COSTS. Working the chooser by hand for Q4_K on CDNA:

    told 2048  ->  J=16:128 tiles  J=32:64  J=48:43  J=64:32   -> picks J=64
    told 88    ->  J=16:6    tiles  J=32:3   J=48:2   J=64:2    -> picks J=48

Both end up at 2 tiles for the real 88 columns, but the tiles are different
widths: 2*64 = 128 column slots for 88 real columns (31% of the column
dimension wasted) versus 2*48 = 96 slots (8% wasted). The MFMA path does the
full tile's arithmetic either way; only the stores are predicated. That waste is
the "poor tile utilisation at n=88" noted in the profile write-up, and it is not
a missing tile width -- J=48 and J=32 already exist on CDNA. It is the wrong
number reaching the chooser.

WHY THIS IS SAFE. ncols_max feeds tile-width selection only. What each tile
actually processes is bounded by expert_bounds, so a smaller J yields more,
narrower tiles and identical results. The src1_q8_1 padding term uses
ggml_cuda_mmq_get_J_max(..., ne11), not the selected J, so the allocation does
not change; and a narrower J can only shrink any partial-tile overread, never
grow it. This is a heuristic knob, not a correctness one.

ESTIMATE, NOT ORACLE. The true per-expert maximum is computed on device by
mm_ids_helper, and reading it host-side would need a device->host sync -- the
very thing [TAG_MUL_MAT_ID_CUDA_GRAPHS] disables CUDA graphs to avoid. So this
uses the routing average instead. If routing is lopsided and some expert draws
far more than the average, the chooser simply picks a narrower tile than ideal
and that expert runs more tiles. Correct, marginally less efficient -- the
failure mode is a slower kernel, not a wrong one.

MEASURE IT. Two things in this project have benchmarked faster while being
wrong, and two size estimates today were off by 4x. Gate on test-backend-ops
MUL_MAT_ID (count with a plain `grep FAIL`) and on pp2048/pp16384 against the
unpatched build, and read the generated tokens.

    python patch_mmq_mmid_ncols_max.py [--check] [--revert]
"""
import sys

TARGET = "ggml/src/ggml-cuda/mmq.cu"
MARKER = "MI210_MMID_NCOLS_MAX"

ORIG = """    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12};"""

NEW = """    // MI210_MMID_NCOLS_MAX: ncols_max only selects the MMQ tile width J. Passing
    // ne12 (the whole batch's token count) makes the chooser size tiles for
    // columns no single expert ever sees: with top-k routing each expert gets
    // roughly ne12*n_expert_used/ne02 of them. Passing the batch picks J=64 and
    // wastes ~31% of the column dimension at 88 real columns; passing the
    // routing average picks J=48 and wastes ~8%. Work is bounded by
    // expert_bounds either way, so this cannot change results -- only tile shape.
    const int64_t ncols_max_per_expert = std::max<int64_t>(1, (ne12*n_expert_used) / ne02);

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ncols_max_per_expert};"""


def main() -> int:
    check  = "--check"  in sys.argv
    revert = "--revert" in sys.argv

    src = open(TARGET).read()
    patched = MARKER in src

    if check:
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        open(TARGET, "w").write(src.replace(NEW, ORIG))
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.",
              file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    if src.count(ORIG) != 1:
        print(f"ERROR: anchor matched {src.count(ORIG)} times, expected 1.", file=sys.stderr)
        return 1

    open(TARGET, "w").write(src.replace(ORIG, NEW, 1))
    print("patched: MUL_MAT_ID now sizes MMQ tiles for per-expert columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
