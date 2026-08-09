#!/usr/bin/env python3
"""Add J=128 MMQ tile entries to the CDNA config table.

WHY. `mul_mat_q_switch_J` (mmq.cuh) picks the tile width J at runtime:

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT) continue;      // no entry -> skip
        if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;
        ...keep the largest J that strictly reduces ceil(ncols_max/J)
    }

so J is capped by whatever the arch table happens to contain. Per-arch maxima:

    ampere    128        blackwell 128        rdna4  128
    cdna       64        pascal     64        rdna2   64

CDNA stops at 64 not because 128 was measured and rejected, but because the
pre-refactor code returned 64 for any HIP target lacking Turing-style MMA
(`get_mmq_x_max_device`), and PR #24127 transcribed that constant into the new
table. CDNA was excluded from the 128 branch for being MFMA rather than WMMA --
an instruction-family test, not a capacity one.

At prefill with -ub 2048 the J switch saturates at the table maximum, so on CDNA
every large-batch matmul runs J=64 tiles when it could run J=128, halving the
tokens covered per tile and doubling the tile count.

LDS BUDGET. mmq_get_nbytes_shared is
    J*4 + I*sram_stride*4 + pad(J*sizeof(block_q8_1_mmq), nthreads*4)
with sizeof(block_q8_1_mmq)=144 and sram_stride 76 (Q8_0/Q8_1/Q6_K), 84 (Q3_K),
100 (Q2_K). Against gfx90a's 64 KiB limit, at the current I=64 / nthreads=256:

    J=64,  Q8_1  28928 B (28.2 KiB)      J=128, Q8_1  38400 B (37.5 KiB)
    J=64,  Q2_K  35072 B (34.2 KiB)      J=128, Q2_K  44544 B (43.5 KiB)

so J=128 fits with room to spare. Note this would NOT have fit as comfortably at
the old I=128/nthreads=512 (Q2_K would have been 70912 B, over the limit), which
is a reason the two changes belong together.

CAVEAT, STATED UP FRONT. aviallon measured mmq_x=128 as catastrophic on MI210
pre-refactor (-60% at n=128 for IQ3_XXS) due to 412 B of scratch spill at
min_blocks=2. Whether that survives the refactor's SRAM-layout changes is
unknown. This is an experiment; the switch will simply ignore any entry that
does not fit, but it will happily select one that fits and is slower. A/B it.

Entries are cloned from each type's existing J=64 line so sram_layout, K_vram,
stream_k and fallback stay whatever that type already uses.

    python patch_mmq_cdna_add_j128.py [--check] [--revert] [--j N]
"""
import re
import sys

TARGET = "ggml/src/ggml-cuda/mmq-config-cdna.cuh"
MARKER = "MI210_ADD_J"

CASE_RE = re.compile(
    r"^(?P<indent>\s*)CASE\("
    r"(?P<type>GGML_TYPE_[A-Z0-9_]+),\s*"
    r"(?P<nthreads>\d+),\s*(?P<occupancy>\d+),\s*(?P<I>\d+),\s*(?P<J>\d+),\s*"
    r"(?P<sram>[A-Z0-9_]+),\s*(?P<kvram>[A-Z0-9_]+),\s*"
    r"(?P<streamk>true|false),\s*(?P<fallback>true|false)\s*\);\s*(?P<trail>//.*)?$"
)


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv
    new_j = 128
    if "--j" in sys.argv:
        new_j = int(sys.argv[sys.argv.index("--j") + 1])

    with open(TARGET) as f:
        lines = f.readlines()
    patched = any(MARKER in ln for ln in lines)

    if check:
        n = sum(1 for ln in lines if MARKER in ln)
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}  ({n} added entries)")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        out = [ln for ln in lines if MARKER not in ln]
        with open(TARGET, "w") as f:
            f.writelines(out)
        print(f"reverted ({len(lines) - len(out)} entries removed)")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.", file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    if new_j % 8:
        print("ERROR: J must be a multiple of 8", file=sys.stderr)
        return 1

    # Clone each J=64 line into a new J=<new_j> line placed directly after it,
    # so each type keeps its own layout/K_vram/stream_k/fallback settings and the
    # table stays grouped by type.
    out, n = [], 0
    for ln in lines:
        out.append(ln)
        m = CASE_RE.match(ln.rstrip("\n"))
        if not m or int(m["J"]) != 64:
            continue
        g = m.groupdict()
        out.append(
            f"{g['indent']}CASE({g['type']}, {g['nthreads']}, {g['occupancy']}, "
            f"{g['I']}, {new_j}, {g['sram']}, {g['kvram']}, "
            f"{g['streamk']}, {g['fallback']});  // {MARKER}\n"
        )
        n += 1

    if n == 0:
        print("ERROR: found no J=64 entries to clone -- table layout changed; "
              "re-derive rather than forcing.", file=sys.stderr)
        return 1

    with open(TARGET, "w") as f:
        f.writelines(out)
    print(f"added {n} J={new_j} entries (cloned from each type's J=64 config)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
