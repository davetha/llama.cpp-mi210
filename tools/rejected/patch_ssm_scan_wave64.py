#!/usr/bin/env python3
"""REJECTED -- NEVER APPLIED. The premise below is wrong; read the verdict first.

VERDICT. This was abandoned before it was ever run, on reading the kernel more
carefully. The transformation described here treats `c_factor` as if it only
selected the number of warps per block. It does not: `c_factor` is
simultaneously warps-per-block AND state-elements-per-lane, so rewriting the
lane mapping for a 64-wide wave changes how much state each lane holds as well
as which lanes cooperate. The "make it wave64-aware" edit is therefore not the
drop-in it is presented as, and applying it would have produced a silently wrong
scan -- the exact failure mode this file's own RISK section warns about.

WHAT ACTUALLY FIXED THE COST. The 22.8% figure quoted below was real, but the
sequential scan was not made faster -- it was removed from the prefill path
entirely by change set 4 (chunked State-Space Duality), which converts the scan
into batched FP16 GEMMs. See patches/04-ssd-mamba2-prefill-cdna.patch.

Kept as a record of a plausible-looking analysis that did not survive contact
with the source. The original text follows unedited.

--- ORIGINAL (WRONG) ---

Make llama.cpp's ssm_scan_f32_group wavefront-aware on CDNA (gfx90a).

FINDING. ggml-cuda hardcodes `#define WARP_SIZE 32`, and ssm_scan_f32_group is
written entirely against it:

    const int warp = threadIdx.x / WARP_SIZE;      // /32
    const int lane = threadIdx.x % WARP_SIZE;      // %32  -> lanes 0..31
    state[j] = s0_warp[WARP_SIZE * j + lane];      // stride 32
    state_sum = warp_reduce_sum(state_sum);        // width defaults to WARP_SIZE=32

On gfx90a the physical wavefront is 64, not 32 (ggml_cuda_get_physical_warp_size()
returns 64 for __GFX9__). Two consequences, both in the per-token inner loop:

  1. warp_reduce_sum<32> on a 64-wide wave takes the SLOW branch --
     `width != ggml_cuda_get_physical_warp_size()` -- so it emits 5 rounds of
     __shfl_xor_sync instead of using the hardware's full-wave reduction. This
     runs ONCE PER TOKEN, and a 4096-token prefill executes it 4096 times per
     head per layer.
  2. The block is launched with 128 threads = 2 physical wavefronts but is
     treated as 4 logical warps, so each physical wave straddles two logical
     warps and half of every reduction's lanes are wasted.

Profiled cost of this kernel on a pp4096 run of Nemotron-3-Super-120B (Q4_K_M,
2x MI210): 1759 ms of 7703 ms total GPU time = 22.8%, the single largest kernel,
at 11 ms per call. It is NOT occupancy-limited (grid 262144 / wg 128 = 2048
blocks over 104 CUs, ~20x oversubscribed), so the cost is in the work itself.

THIS PATCH switches the CDNA instantiation to c_factor = d_state/64 with a
64-wide lane mapping, so `lane` spans a full wavefront, the state stride matches,
and warp_reduce_sum hits the single-instruction path. The arithmetic is
unchanged: the same d_state elements are covered, just partitioned across 64
lanes x (d_state/64) registers instead of 32 x (d_state/32).

RISK. Getting the lane mapping wrong does not crash -- it silently computes a
wrong scan, and a wrong SSM state produces fluent, confident, WRONG text rather
than an obvious error. ALWAYS read generated tokens after applying this, never
just the throughput number.

    python patch_ssm_scan_wave64.py [--check] [--revert]
"""
import sys

TARGET = "/src/ggml/src/ggml-cuda/ssm-scan.cu"

# The kernel indexes lanes/state with the compile-time WARP_SIZE (32). Introduce
# a per-instantiation width so CDNA can use its real 64-wide wavefront while
# every other backend keeps the existing 32-wide behaviour bit-for-bit.
ORIG_BODY = """    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x  * c_factor + warp;"""

PATCHED_BODY = """    // SSM_WAVE64: on CDNA the physical wavefront is 64, but WARP_SIZE is 32.
    // Using the physical width makes warp_reduce_sum below take the single
    // hardware-reduction path instead of 5 rounds of __shfl_xor_sync, and stops
    // each physical wave from straddling two logical warps.
    constexpr int ssm_warp = ggml_cuda_get_physical_warp_size();
    const int warp     = threadIdx.x / ssm_warp;
    const int lane     = threadIdx.x % ssm_warp;
    const int warp_idx = blockIdx.x  * c_factor + warp;"""

ORIG_STATE_LOAD = """    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }"""

PATCHED_STATE_LOAD = """    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[ssm_warp * j + lane];  // SSM_WAVE64
    }"""

ORIG_INNER = """            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];"""

PATCHED_INNER = """            const float B_val = B_warp[i * stride_B + ssm_warp * j + lane];  // SSM_WAVE64
            const float C_val = C_warp[i * stride_C + ssm_warp * j + lane];  // SSM_WAVE64"""

ORIG_REDUCE = """        state_sum = warp_reduce_sum(state_sum);"""
PATCHED_REDUCE = """        state_sum = warp_reduce_sum<ggml_cuda_get_physical_warp_size()>(state_sum);  // SSM_WAVE64"""

ORIG_WRITEBACK = """        s_warp[WARP_SIZE * j + lane] = state[j];"""
PATCHED_WRITEBACK = """        s_warp[ssm_warp * j + lane] = state[j];  // SSM_WAVE64"""

EDITS = [
    ("lane mapping", ORIG_BODY, PATCHED_BODY),
    ("state load", ORIG_STATE_LOAD, PATCHED_STATE_LOAD),
    ("inner loop B/C", ORIG_INNER, PATCHED_INNER),
    ("warp reduce", ORIG_REDUCE, PATCHED_REDUCE),
    ("state writeback", ORIG_WRITEBACK, PATCHED_WRITEBACK),
]


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv

    with open(TARGET) as f:
        src = f.read()
    patched = "SSM_WAVE64" in src

    if check:
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        for _, orig, new in EDITS:
            src = src.replace(new, orig)
        with open(TARGET, "w") as f:
            f.write(src)
        print("reverted")
        return 0

    if patched:
        print("already patched")
        return 0

    # Assert every anchor matches exactly once before writing anything: a
    # partially-applied kernel patch is far worse than none.
    for name, orig, _ in EDITS:
        n = src.count(orig)
        if n != 1:
            print(f"ERROR: anchor '{name}' matched {n} times, expected 1. "
                  "Upstream moved; re-derive rather than forcing.", file=sys.stderr)
            return 1
    for name, orig, new in EDITS:
        src = src.replace(orig, new, 1)
    with open(TARGET, "w") as f:
        f.write(src)
    print(f"patched {len(EDITS)} sites: ssm_scan now uses the physical wavefront width")
    return 0


if __name__ == "__main__":
    sys.exit(main())
