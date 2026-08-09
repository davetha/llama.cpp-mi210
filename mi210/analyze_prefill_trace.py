#!/usr/bin/env python3
"""Break down GPU kernel time from a rocprofv3 kernel trace.

Answers the only question that matters for optimisation work: which kernels
actually consume the wall clock. Guessing at this is how you spend a week
speeding up something that was 3% of the runtime.
"""
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "/hosttmp/pf/pf_results.db"
con = sqlite3.connect(db)

tables = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
sym = next((t for t in tables if "kernel_symbol" in t), None)
disp = next((t for t in tables if "kernel_dispatch" in t), None)
if not sym or not disp:
    print("no kernel tables in", db)
    print("tables:", tables[:10])
    raise SystemExit(1)

rows = con.execute(f"""
    SELECT s.display_name, COUNT(*), SUM(d.end - d.start) / 1e6
    FROM {disp} d JOIN {sym} s ON d.kernel_id = s.id
    GROUP BY s.display_name
    ORDER BY 3 DESC
""").fetchall()

total = sum(r[2] for r in rows)
print(f"total GPU kernel time: {total:.0f} ms across {sum(r[1] for r in rows)} dispatches")
print()
print(f"{'%':>6} {'ms':>9} {'calls':>8}  kernel")
print("-" * 78)
for name, n, ms in rows[:18]:
    print(f"{ms/total*100:>5.1f}% {ms:>9.0f} {n:>8}  {name[:56]}")

# Group by what the kernel actually is, so the fix target is obvious.
print()
print("by category:")
cats = {
    "SSM / Mamba scan": ("ssm_scan", "ssm_conv", "selective"),
    "quantized GEMM (MMQ)": ("mul_mat_q", "mmq", "vec_dot"),
    "flash attention": ("flash_attn", "fattn"),
    "dequant / convert": ("dequantize", "cpy", "convert", "quantize"),
    "norms / activations": ("rms_norm", "norm", "silu", "glu", "soft_max"),
    "MoE routing": ("mul_mat_id", "argsort", "top_k", "moe"),
}
seen = set()
for label, keys in cats.items():
    ms = sum(r[2] for r in rows if any(k in r[0].lower() for k in keys))
    n = sum(r[1] for r in rows if any(k in r[0].lower() for k in keys))
    seen |= {r[0] for r in rows if any(k in r[0].lower() for k in keys)}
    if ms:
        print(f"  {label:<24} {ms/total*100:>5.1f}%  {ms:>8.0f} ms  {n:>7} calls")
other = sum(r[2] for r in rows if r[0] not in seen)
if other:
    print(f"  {'other':<24} {other/total*100:>5.1f}%  {other:>8.0f} ms")
