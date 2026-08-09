#!/usr/bin/env python3
"""Let mm_ids_helper use its fast path for any n_expert_used, not just a fixed list.

WHY. ggml_cuda_launch_mm_ids_helper dispatches the specialised kernel only for
n_expert_used in {2, 4, 6, 8, 16, 32}; anything else falls to the generic
template-0 path. Nemotron-3-Super routes top-22 of 512 experts, so 22 misses the
list and every MoE matmul runs the generic kernel -- visible in the profile as
mm_ids_helper<0> at 6.7% of prefill GPU time (359 ms over 320 calls).

The two paths differ structurally:

  generic:      for (int it = 0; it < n_tokens; ++it)          // 1 token/iter
  specialised:  for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded)

`it_compact` is loop-carried, so the generic loop is a dependency chain of
n_tokens iterations -- 2048 of them at ub=2048, each costing a memory round
trip, with only 22 of 64 lanes doing useful work. At 1.12 ms/call that is ~547 ns
per iteration, which is about one uncached round trip: latency-bound, not
bandwidth-bound.

With neu_padded = 32 the specialised path covers warp_size/neu_padded = 2 tokens
per iteration on a 64-wide wavefront, halving the number of round trips.

WHAT BLOCKED IT. Only two lines, both assuming the padded width equals the
unpadded count except for the hand-coded 6 -> 8 case:

    static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, ...);
    const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used;

The kernel body is already padding-correct: the `iex < n_expert_used` guard
gives padded lanes expert_used = INT_MAX, which contributes nothing to
`nex_prev += expert_used < expert` and never matches `expert_used == expert`.

BACKWARDS COMPATIBLE. next_pow2 of each currently dispatched value is itself
(2, 4, 8, 16, 32) or the existing special case (6 -> 8), so no already-supported
configuration changes behaviour. The real constraint is that neu_padded must be
a power of two that divides warp_size, which the new static_assert states
directly instead of implying it.

This is a shared file -- the change affects every backend, so it is written to
be a no-op everywhere except for newly-admitted expert counts.

    python patch_mmid_generalize_neu.py [--check] [--revert]
"""
import sys

TARGET = "ggml/src/ggml-cuda/mmid.cu"
MARKER = "MI210_NEU_PADDED"

ORIG_PAD = """        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2."""

NEW_PAD = """        // MI210_NEU_PADDED: pad to the next power of 2 generically instead of
        // special-casing 6. The scan below steps by neu_padded and
        // warp_reduce_any<neu_padded> needs a power of 2 dividing warp_size, so
        // assert on the PADDED width -- the unpadded count never had to divide
        // anything. Values already dispatched here are unchanged: next_pow2 of
        // 2/4/8/16/32 is itself, and 6 still pads to 8.
        constexpr int neu_padded = mm_ids_next_pow2(n_expert_used_template);
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0,
                      "n_expert_used pads to a width that does not divide the warp");"""

# The padded width must be a compile-time constant: it is a template argument to
# warp_reduce_any and an unroll bound.
ORIG_HELPER = """// Helper function for mul_mat_id, converts ids to a more convenient format."""
NEW_HELPER = """// MI210_NEU_PADDED: smallest power of 2 >= n (n >= 1), evaluated at compile time.
static constexpr int mm_ids_next_pow2(int n) {
    int p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}

// Helper function for mul_mat_id, converts ids to a more convenient format."""

# Dispatch entry for this model's routing width. Adding a case is additive: it
# only redirects a value that previously fell through to the generic path.
ORIG_CASE = """        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:"""

NEW_CASE = """        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        // MI210_NEU_PADDED: top-22 routing (Nemotron-3-Super: 22 of 512 experts)
        // pads to 32 and so can use the fast path. Any other count whose next
        // power of 2 divides the warp can be added the same way, at the cost of
        // one more template instantiation.
        case 22:
            launch_mm_ids_helper<22>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:"""

EDITS = [
    ("next_pow2 helper", ORIG_HELPER, NEW_HELPER),
    ("padding + assert", ORIG_PAD, NEW_PAD),
    ("dispatch case 22", ORIG_CASE, NEW_CASE),
]


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv

    with open(TARGET) as f:
        src = f.read()
    patched = MARKER in src

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

    for name, orig, _ in EDITS:
        n = src.count(orig)
        if n != 1:
            print(f"ERROR: anchor '{name}' matched {n} times, expected 1. "
                  "Upstream moved; re-derive rather than forcing.", file=sys.stderr)
            return 1
    for _, orig, new in EDITS:
        src = src.replace(orig, new, 1)
    with open(TARGET, "w") as f:
        f.write(src)
    print("patched: mm_ids_helper fast path generalised, case 22 dispatched")
    print("NEXT: test-backend-ops -o MUL_MAT_ID must pass before any benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
