#!/usr/bin/env python3
"""Test whether batching flips llama.cpp's attention onto the MFMA path.

llama.cpp picks its flash-attention kernel in fattn.cu:512-520. For head_dim<=128
the MFMA kernel requires `Q->ne[1] * gqa_ratio_eff > 16`. Nemotron-3-Super has
gqa_ratio = 32/2 = 16, so at batch=1 the product is exactly 16 -- one short of a
STRICTLY-greater test -- and it falls back to the generic tile kernel.

At batch=2 the product is 32 and MFMA engages. If that matters, per-stream decode
at batch=2 should hold up rather than halve.

Baseline for a purely memory-bound model with no kernel change: 2 concurrent
streams each run at ~1/2 the single-stream rate (aggregate flat). If aggregate
climbs well above 1.0x, something other than bandwidth improved -- i.e. the
kernel switch is real.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8038"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "nemotron-heretic"
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 96

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain what a B-tree is and when you would use one.",
    "Write a bash one-liner that finds the ten largest files under a path.",
    "Describe how a bloom filter works.",
]


def one(prompt, out, idx):
    """Stream a completion; record decode tok/s excluding prefill."""
    payload = {
        "model": MODEL, "prompt": prompt, "max_tokens": NTOK,
        "temperature": 0, "stream": True,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            s = line.decode().strip()
            if not s.startswith("data: ") or s.endswith("[DONE]"):
                continue
            if ttft is None:
                ttft = time.time() - t0
            n += 1
    dec = time.time() - t0 - (ttft or 0)
    out[idx] = ((n - 1) / dec if n > 1 and dec > 0 else 0.0, ttft or 0.0)


def run(batch, reps=2):
    """Return (mean per-stream tok/s, mean aggregate tok/s)."""
    per, agg = [], []
    for rep in range(reps):
        out = [None] * batch
        ts = [threading.Thread(target=one,
                               args=(PROMPTS[(rep * batch + i) % len(PROMPTS)], out, i))
              for i in range(batch)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        rates = [o[0] for o in out if o]
        if rates:
            per.append(statistics.mean(rates))
            agg.append(sum(rates))
        time.sleep(2)
    return (statistics.mean(per) if per else 0.0,
            statistics.mean(agg) if agg else 0.0)


print(f"{'batch':>6} {'per-stream t/s':>15} {'aggregate t/s':>14}  note")
print("-" * 58)
base_per = base_agg = None
for b in (1, 2):
    p, a = run(b)
    if base_per is None:
        base_per, base_agg = p, a
        note = "tile kernel (gqa*1 = 16, not > 16)"
    else:
        ratio = a / base_agg if base_agg else 0.0
        note = f"MFMA gate open; aggregate {ratio:.2f}x vs batch=1"
    print(f"{b:>6} {p:>15.2f} {a:>14.2f}  {note}")

print()
print("Reading it: a memory-bound model with no kernel change gives aggregate")
print("~1.0x and per-stream ~0.5x. Aggregate well above 1.0x means the MFMA")
print("switch bought real work, not just interleaving.")
