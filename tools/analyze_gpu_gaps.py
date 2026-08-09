#!/usr/bin/env python3
"""Measure GPU idle time between kernels in a rocprofv3 trace.

Kernel-time breakdowns answer "which kernel is slow". They cannot answer "is the
GPU waiting", because a trace that sums to 5000 ms of kernel time says nothing
about how much wall clock elapsed around those kernels.

That distinction decides what to optimise. If the device is ~100% busy, the only
way forward is making kernels cheaper. If there is meaningful idle time, the
work is in launch overhead, host-side scheduling, or synchronisation -- CUDA
graphs, fewer dispatches, better overlap -- and no amount of kernel tuning will
recover it.

Gaps are computed per device: with `-sm layer` the two GPUs run different layers
and interleave, so pooling their dispatches would hide real stalls on one behind
activity on the other. Overlapping kernels on the same device are merged rather
than double-counted, so "busy" means "at least one kernel resident".
"""
import sqlite3
import sys
from collections import defaultdict

BIG_GAP_NS = 100_000  # 100 us: comfortably above launch overhead, so these are stalls


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "/pf/pf9_results.db"
    con = sqlite3.connect(db)

    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    disp = next((t for t in tables if "kernel_dispatch" in t), None)
    if not disp:
        print("no kernel_dispatch table in", db)
        return 1

    cols = [r[1] for r in con.execute("PRAGMA table_info(%s)" % disp).fetchall()]
    agent = next((c for c in ("agent_id", "agent_abs_index", "device_id") if c in cols), None)

    if agent:
        rows = con.execute(
            "SELECT %s, start, end FROM %s" % (agent, disp)).fetchall()
    else:
        rows = [(0, s, e) for s, e in con.execute(
            "SELECT start, end FROM %s" % disp).fetchall()]

    per = defaultdict(list)
    for dev, s, e in rows:
        per[dev].append((s, e))

    print("%-8s %8s %10s %10s %8s %10s %8s %7s %10s" %
          ("device", "kernels", "span_ms", "busy_ms", "busy%", "idle_ms", "idle%", "gaps", ">100us_ms"))
    for dev, iv in sorted(per.items()):
        iv.sort()
        span = iv[-1][1] - iv[0][0]
        if span <= 0:
            continue
        gap = 0
        ngap = 0
        big = 0
        cur_end = iv[0][0]
        for s, e in iv:
            if s > cur_end:
                g = s - cur_end
                gap += g
                ngap += 1
                if g > BIG_GAP_NS:
                    big += g
                cur_end = e
            else:
                cur_end = max(cur_end, e)
        busy = span - gap
        print("%-8s %8d %10.1f %10.1f %7.1f%% %10.1f %7.1f%% %7d %10.1f" %
              (str(dev), len(iv), span / 1e6, busy / 1e6, 100 * busy / span,
               gap / 1e6, 100 * gap / span, ngap, big / 1e6))

    print()
    print("Reading it: high busy%% means kernel cost is the only lever left.")
    print("Meaningful idle%% -- especially concentrated in >100us gaps -- means")
    print("launch overhead or host-side stalls, which kernel tuning cannot fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
