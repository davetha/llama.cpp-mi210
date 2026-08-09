#!/usr/bin/env python3
"""Disable stream-k decomposition for K-quants in llama.cpp's CDNA MMQ config.

WHY. Every real CASE entry in ggml-cuda/mmq-config-cdna.cuh sets stream_k=true
(the 8th positional argument); only the unreachable GGML_TYPE_COUNT sentinel is
false. There is a single mmq-config-cdna.cuh covering all CDNA generations, so
gfx90a inherits whatever was tuned elsewhere.

Upstream PR #26199 (merged 2026-07-29), which retuned the RDNA configs, reports
directly: "I have also found that stream_k true helps a lot for Dense models and
hurts MoE models." Our workload is MoE -- Nemotron-3-Super-120B-A12B -- and MMQ
is 45.2% of prefill GPU time in the rocprofv3 profile (of 7703 ms total). That
makes this the largest single lever identified, ahead of the 22.8% SSM scan.

Stream-k splits the K dimension across more workgroups than there are output
tiles, then fixes up partial sums in a second pass (mul_mat_q_stream_k_fixup).
It exists to fill the GPU when there are too few output tiles to saturate it. In
an MoE prefill each expert already produces many tiles, so the machine is
saturated without it and the fixup pass plus the extra global-memory traffic for
partial accumulators is pure overhead.

IMPORTANT -- THIS IS AN EXPERIMENT, NOT A KNOWN WIN. The stream-k/MoE evidence is
from RDNA3.5/RDNA4, NOT CDNA2. No CDNA stream-k benchmark exists upstream. It may
well be slower on gfx90a, exactly as the earlier CDNA2 rocBLAS carve-out patch
turned out 6.5% slower than the assumption behind it. A/B against the same
binary, and read generated tokens both ways.

Scoped to the K-quants (Q2_K..Q6_K) because that is what an i1-Q4_K_M model
actually dispatches; leaving the other types alone keeps the experiment narrow
and the result attributable.

    python patch_mmq_cdna_no_streamk.py [--check] [--revert]
"""
import re
import sys

TARGET = "ggml/src/ggml-cuda/mmq-config-cdna.cuh"
TYPES = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K")
MARKER = "MI210_NO_STREAMK"

# CASE(type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback)
# Rewrite only the 8th argument, and only on lines for the listed types. Anchor
# on the two trailing args so a config whose shape changes upstream fails to
# match instead of silently corrupting a different field.
LINE_RE = re.compile(
    r"^(\s*CASE\(GGML_TYPE_(?:" + "|".join(TYPES) + r"),[^;]*?,\s*)true(\s*,\s*(?:true|false)\s*\);)$"
)


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv

    with open(TARGET) as f:
        lines = f.readlines()
    patched = any(MARKER in ln for ln in lines)

    if check:
        n = sum(1 for ln in lines if MARKER in ln)
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}  ({n} entries)")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        out = []
        for ln in lines:
            if MARKER in ln:
                ln = ln.split("  // " + MARKER)[0] + "\n"
                ln = re.sub(r",\s*false(\s*,\s*(?:true|false)\s*\);)$", r", true\1", ln)
            out.append(ln)
        with open(TARGET, "w") as f:
            f.writelines(out)
        print("reverted")
        return 0

    if patched:
        print("already patched")
        return 0

    out, n = [], 0
    for ln in lines:
        m = LINE_RE.match(ln.rstrip("\n"))
        if m:
            out.append(f"{m.group(1)}false{m.group(2)}  // {MARKER}\n")
            n += 1
        else:
            out.append(ln)

    if n == 0:
        print("ERROR: no CASE lines matched -- the config layout changed upstream; "
              "re-derive rather than forcing.", file=sys.stderr)
        return 1

    with open(TARGET, "w") as f:
        f.writelines(out)
    print(f"patched: stream_k disabled on {n} CDNA K-quant MMQ configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
