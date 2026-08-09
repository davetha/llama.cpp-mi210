#!/usr/bin/env python3
"""Rewrite the CDNA MMQ config fields so they can be swept empirically.

The table in ggml-cuda/mmq-config-cdna.cuh is a list of

    CASE(type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback)

and mmq.cuh states outright that these "should not affect results, only
speed/register pressure/shared memory use". Every CDNA entry currently uses
nthreads=512, occupancy=1, I=128, and one file covers CDNA1/2/3, so nothing here
was necessarily tuned on gfx90a.

`occupancy` is the second argument of __launch_bounds__. That parameter means
different things on the two platforms: CUDA reads it as minBlocksPerMultiprocessor,
HIP reads it as MIN_WARPS_PER_EXECUTION_UNIT. So on gfx90a `occupancy=1` asks the
compiler for as little as one wave per SIMD, which lets it spend registers freely
and hide very little latency. Raising it forces the register budget down in
exchange for more resident waves -- worth measuring rather than assuming.

Usage:
    python tune_mmq_cdna.py --nthreads 256 --occupancy 2 --tile-i 64
    python tune_mmq_cdna.py --revert

Only the fields named on the command line are touched; the rest keep their
upstream values. Restricted to entries whose J matches --only-j when given, so a
sweep can target just the hot configs instead of the whole table.
"""
import argparse
import re
import sys

TARGET = "ggml/src/ggml-cuda/mmq-config-cdna.cuh"
BACKUP = TARGET + ".tune-orig"

# CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)
CASE_RE = re.compile(
    r"^(?P<indent>\s*)CASE\("
    r"(?P<type>GGML_TYPE_[A-Z0-9_]+),\s*"
    r"(?P<nthreads>\d+),\s*"
    r"(?P<occupancy>\d+),\s*"
    r"(?P<I>\d+),\s*"
    r"(?P<J>\d+),\s*"
    r"(?P<sram>[A-Z0-9_]+),\s*"
    r"(?P<kvram>[A-Z0-9_]+),\s*"
    r"(?P<streamk>true|false),\s*"
    r"(?P<fallback>true|false)\s*\);\s*(?P<trail>//.*)?$"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nthreads", type=int)
    ap.add_argument("--occupancy", type=int)
    ap.add_argument("--tile-i", type=int, dest="tile_i")
    ap.add_argument("--stream-k", choices=("true", "false"))
    ap.add_argument("--only-j", type=int,
                    help="restrict to entries with this J (e.g. 64 for the hot configs)")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    if a.revert:
        import os
        if not os.path.exists(BACKUP):
            print("no backup; nothing to revert")
            return 0
        with open(BACKUP) as f:
            src = f.read()
        with open(TARGET, "w") as f:
            f.write(src)
        os.remove(BACKUP)
        print("reverted to pre-tuning config")
        return 0

    with open(TARGET) as f:
        lines = f.readlines()

    # Keep one pristine copy so a sweep always starts from the same baseline
    # rather than compounding edits from the previous iteration.
    import os
    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.writelines(lines)

    if a.show:
        seen = {}
        for ln in lines:
            m = CASE_RE.match(ln.rstrip("\n"))
            if m:
                k = (m["nthreads"], m["occupancy"], m["I"], m["J"], m["streamk"])
                seen[k] = seen.get(k, 0) + 1
        print(f"{'nthr':>5} {'occ':>4} {'I':>4} {'J':>4} {'stream_k':>9}  count")
        for k, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"{k[0]:>5} {k[1]:>4} {k[2]:>4} {k[3]:>4} {k[4]:>9}  {n}")
        return 0

    out, n = [], 0
    for ln in lines:
        m = CASE_RE.match(ln.rstrip("\n"))
        if not m:
            out.append(ln)
            continue
        if a.only_j is not None and int(m["J"]) != a.only_j:
            out.append(ln)
            continue
        g = m.groupdict()
        if a.nthreads is not None:
            g["nthreads"] = str(a.nthreads)
        if a.occupancy is not None:
            g["occupancy"] = str(a.occupancy)
        if a.tile_i is not None:
            g["I"] = str(a.tile_i)
        if a.stream_k is not None:
            g["streamk"] = a.stream_k
        trail = ("  " + g["trail"]) if g.get("trail") else ""
        out.append(
            f"{g['indent']}CASE({g['type']}, {g['nthreads']}, {g['occupancy']}, "
            f"{g['I']}, {g['J']}, {g['sram']}, {g['kvram']}, "
            f"{g['streamk']}, {g['fallback']});{trail}\n"
        )
        n += 1

    if n == 0:
        print("ERROR: no CASE lines matched -- layout changed upstream, re-derive.",
              file=sys.stderr)
        return 1

    # The static_asserts in the CASE macro are the real guard rails; mirror the
    # cheap ones here so a bad sweep value fails before a 10-minute build.
    if a.nthreads is not None and (a.nthreads % 32 or a.nthreads > 512):
        print("ERROR: nthreads must be a multiple of 32 and <= 512", file=sys.stderr)
        return 1
    if a.occupancy is not None and a.occupancy > 8:
        print("ERROR: occupancy must be <= 8", file=sys.stderr)
        return 1
    if a.tile_i is not None and a.tile_i % 32:
        print("ERROR: I must be a multiple of 32", file=sys.stderr)
        return 1

    with open(TARGET, "w") as f:
        f.writelines(out)
    print(f"rewrote {n} CASE entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
