#!/usr/bin/env python3
"""Measure prefill throughput and per-GPU utilisation for an OpenAI-compatible server.

The question this answers: when a long prompt is being processed, are both GPUs
busy *at the same time* (tensor parallelism) or alternately (pipeline / layer
split)? That distinction sets the ceiling for any further optimisation, and it
cannot be read off a throughput number alone.

Sampling `/sys/class/drm/card*/device/gpu_busy_percent` is crude compared with a
kernel trace, but it needs no instrumentation of the server, works identically
across engines, and is more than accurate enough to tell 2-busy-at-once from
alternating.

Usage:
    prefill_util_probe.py <base_url> <model> [prompt_tokens] [label]
"""
import json
import sys
import threading
import time
import urllib.request
import glob


def sample_gpus(stop, out, period=0.02):
    """Poll every GPU's busy percentage until told to stop."""
    paths = sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"))
    while not stop.is_set():
        row = []
        for p in paths:
            try:
                with open(p) as f:
                    row.append(int(f.read().strip()))
            except Exception:
                row.append(-1)
        out.append(row)
        time.sleep(period)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8040"
    model = sys.argv[2] if len(sys.argv) > 2 else "nemo"
    want_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    label = sys.argv[4] if len(sys.argv) > 4 else base

    # A repeated-but-varied filler: token count only has to be roughly right, and
    # varying the numbers stops any prefix cache from short-circuiting the prefill.
    chunk = ("The quick brown fox jumps over the lazy dog while the system "
             "processes token number %d of the benchmark sequence. ")
    prompt = "".join(chunk % i for i in range(want_tok // 14 + 1))

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    samples = []
    stop = threading.Event()
    th = threading.Thread(target=sample_gpus, args=(stop, samples))
    th.start()
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
    finally:
        stop.set()
        th.join()
    dt = time.time() - t0

    ptok = r.get("usage", {}).get("prompt_tokens", 0)
    print("%s: %d prompt tokens in %.2f s = %.0f tok/s" % (label, ptok, dt, ptok / dt))

    if not samples:
        print("  (no GPU samples captured)")
        return 0

    ngpu = len(samples[0])
    # Ignore near-idle samples so load/teardown does not dilute the busy window.
    active = [row for row in samples if max(row) > 20]
    print("  samples: %d total, %d with any GPU >20%% busy" % (len(samples), len(active)))
    if not active:
        return 0

    for g in range(ngpu):
        vals = [row[g] for row in active]
        print("    gpu%d: mean %5.1f%%  max %3d%%" % (g, sum(vals) / len(vals), max(vals)))

    both = sum(1 for row in active if all(v > 50 for v in row))
    either = sum(1 for row in active if any(v > 50 for v in row))
    print("    both GPUs >50%% simultaneously: %d/%d active samples (%.0f%%)" %
          (both, len(active), 100.0 * both / max(either, 1)))
    print()
    print("  Reading it: near 100%% => tensor parallel (both cards on every layer).")
    print("  Near 0%% with each card individually busy => pipelined / layer split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
