#!/usr/bin/env python3
"""Measure how much the two GPUs actually work at the same time.

With `-sm layer` llama.cpp splits layers across devices: GPU0 holds the first
half, GPU1 the second. Layer N+1 depends on layer N, so for a single sequence
the devices should run *sequentially*, not in parallel -- each idle while the
other works. Tensor parallelism (what vLLM does) instead has both devices work
on every layer together.

If that is what is happening, aggregate utilisation is capped near 50% no matter
how fast the kernels get, and it is a structural limit rather than a tuning
problem. This measures it directly instead of assuming it.

Method: isolate the busiest window (to exclude model load and inter-run pauses),
then compute each device's busy intervals inside it and the intersection of the
two. Overlap near zero confirms serialisation.
"""
import sqlite3
import sys
from collections import defaultdict


def merge(intervals):
    """Union of intervals -> disjoint, sorted."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    """Total length of the intersection of two disjoint-interval lists."""
    i = j = 0
    acc = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            acc += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return acc


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "/pf/pf9_results.db"
    # Gaps longer than this are treated as run boundaries (load, warmup pauses),
    # not as stalls inside a prefill.
    split_ns = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000_000  # 200 ms

    con = sqlite3.connect(db)
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    disp = next(t for t in tables if "kernel_dispatch" in t)
    rows = con.execute("SELECT agent_id, start, end FROM %s" % disp).fetchall()

    per = defaultdict(list)
    for d, s, e in rows:
        per[d].append((s, e))

    # Segment the whole trace on long global quiet periods, then keep the
    # segment with the most kernel activity: that is the measured prefill.
    allk = sorted((s, e) for v in per.values() for s, e in v)
    segs = []
    cur = [allk[0][0], allk[0][1]]
    for s, e in allk:
        if s - cur[1] > split_ns:
            segs.append(tuple(cur))
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    segs.append(tuple(cur))

    def busy_in(seg):
        lo, hi = seg
        return sum(total(merge([(max(s, lo), min(e, hi))
                                for s, e in v if e > lo and s < hi]))
                   for v in per.values())

    best = max(segs, key=busy_in)
    lo, hi = best
    print("trace has %d segments; analysing the busiest, %.1f ms wide" %
          (len(segs), (hi - lo) / 1e6))
    print()

    dev_iv = {}
    for d, v in sorted(per.items()):
        clipped = [(max(s, lo), min(e, hi)) for s, e in v if e > lo and s < hi]
        dev_iv[d] = merge(clipped)

    span = hi - lo
    for d, iv in dev_iv.items():
        print("  device %s busy %8.1f ms of %8.1f ms window  (%.1f%%)" %
              (d, total(iv) / 1e6, span / 1e6, 100 * total(iv) / span))

    devs = list(dev_iv)
    if len(devs) == 2:
        both = intersect(dev_iv[devs[0]], dev_iv[devs[1]])
        union = total(merge([tuple(x) for x in dev_iv[devs[0]] + dev_iv[devs[1]]]))
        print()
        print("  both devices busy simultaneously: %8.1f ms  (%.1f%% of window)" %
              (both / 1e6, 100 * both / span))
        print("  at least one device busy:         %8.1f ms  (%.1f%% of window)" %
              (union / 1e6, 100 * union / span))
        print()
        sum_busy = sum(total(v) for v in dev_iv.values())
        print("  aggregate utilisation (sum busy / 2 devices / window): %.1f%%" %
              (100 * sum_busy / (2 * span)))
        if both / span < 0.10:
            print()
            print("  => The devices are essentially SERIALISED. This is inherent to")
            print("     -sm layer on a single sequence: layer N+1 needs layer N, so the")
            print("     second GPU waits. Kernel tuning cannot recover this; only")
            print("     tensor-parallel execution or pipelining several micro-batches can.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
