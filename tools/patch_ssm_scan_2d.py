#!/usr/bin/env python3
"""Give SSM_SCAN an output shape whose head axis survives, so -sm tensor works.

WHY. ggml_ssm_scan declares its result as a flat 1-D buffer holding y followed
by the updated states. That throws away the head axis, and the meta backend can
then only describe a split of it as "a proportional slice of a flat buffer",
which is exact in SIZE and wrong in LAYOUT -- device j's contiguous byte range
is not device j's heads. Every downstream view inherits the false premise, and
the ADD on mamba2_y_add_d fails to resolve.

Its two sibling recurrent ops do not have this problem, because they keep the
head count as a real axis shared by both regions:

    ggml_gated_delta_net -> [S_v*H, n_tokens*n_seqs + K*S_v*n_seqs]
    ggml_rwkv_wkv7       -> [S*H,   n_tokens       + S*n_seqs]
    ggml_ssm_scan        -> [nelements(x) + state_size]          <- flat

and handle_gated_delta_net splits the first of those with a plain
single-segment AXIS_0 rule. So the framework can already do this; SSM_SCAN just
does not give it anything to work with.

THIS PATCH gives SSM_SCAN the same discipline:

    [d_inner, n_seq_tokens*n_seqs + d_state*n_seqs]      d_inner = head_dim*n_head

Total element count is unchanged. The y region's bytes are unchanged -- it was
already [head_dim, n_head, n_seq_tokens, n_seqs], i.e. rows of d_inner indexed
by token. Only the state region is reordered, from

    {d_state, head_dim, n_head, n_seqs}   ->   {head_dim, n_head, d_state, n_seqs}

so that its rows are also d_inner wide and also indexed head-major. With that,
one AXIS_0 split at a multiple of head_dim is head-aligned in BOTH regions, and
no framework change is needed at all.

COST, STATED UP FRONT. d_state was the fastest-moving axis of the state, and
that is exactly what the sequential kernels vectorise over: the CPU path loads
`s0 + i + ii*nc` along d_state in three separate SIMD variants, and the CUDA
group kernel reads `s0_warp[WARP_SIZE*j + lane]` the same way. Making d_state
the SLOWEST axis makes those accesses strided. This patch therefore drops the
CPU SIMD blocks to the scalar tail, which is correct but slower.

That cost lands on decode and on CPU inference. It does NOT land on the SSD
chunked prefill path -- that keeps a scratch state and updates it with cuBLAS,
where the transpose is just an operand swap (m=head_dim, n=d_state, ldc=head_dim
instead of m=d_state, n=head_dim, ldc=d_state), same FLOPs, no extra kernel.
Prefill is the thing tensor parallelism is being chased for, so the trade is in
the right direction, but it is a real regression and must be measured, not
assumed.

The CPU hot loop can be re-vectorised later by swapping the i/i1 loops so the
inner loop walks head_dim contiguously instead of d_state. Deliberately not
done here: correctness and the actual TP gain come first, and there is no point
optimising a path that may not survive measurement.

CORRECTNESS RISK. A wrong index here does not crash. It propagates a wrong SSM
state and produces fluent, confident, wrong text. Gate on BOTH:
  * test-backend-ops SSM_SCAN (count with a plain `grep FAIL`) -- catches CUDA
    and CPU disagreeing, i.e. an asymmetric mistake
  * generated tokens at temperature 0 against the pre-change build -- catches a
    mistake made symmetrically in both, which test-backend-ops cannot see

    python patch_ssm_scan_2d.py [--check] [--revert]
"""
import sys

MARKER = "MI210_SSM2D"

# ---------------------------------------------------------------- ggml.c ----
GGML_C = "ggml/src/ggml.c"

GGML_SHAPE_ORIG = """        const int64_t d_state      = s->ne[0];
        const int64_t head_dim     = x->ne[0];
        const int64_t n_head       = x->ne[1];"""
GGML_SHAPE_NEW = """        // MI210_SSM2D: state is {head_dim, n_head, d_state, n_seqs}
        const int64_t d_state      = s->ne[2];
        const int64_t head_dim     = x->ne[0];
        const int64_t n_head       = x->ne[1];"""

GGML_ASSERT_ORIG = """        GGML_ASSERT(s->ne[1] == head_dim);
        GGML_ASSERT(s->ne[2] == n_head);"""
GGML_ASSERT_NEW = """        GGML_ASSERT(s->ne[0] == head_dim);   // MI210_SSM2D
        GGML_ASSERT(s->ne[1] == n_head);"""

GGML_RESULT_ORIG = """    // concatenated y + ssm_states
    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);"""
GGML_RESULT_NEW = """    // MI210_SSM2D: concatenated y + ssm_states, declared 2-D as
    //   [d_inner, n_seq_tokens*n_seqs + d_state*n_seqs]
    // so that the head axis is axis 0 in BOTH regions and a single-segment
    // axis-0 split is head-aligned throughout. Same total size as the old flat
    // 1-D shape; the y region's bytes are identical.
    const int64_t d_inner_2d = x->ne[0]*x->ne[1];
    const int64_t rows_2d    = x->ne[2]*x->ne[3] + s->ne[2]*ids->ne[0];
    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner_2d, rows_2d);"""

# ------------------------------------------------------------- ggml-cpu -----
CPU = "ggml/src/ggml-cpu/ops.cpp"

# Row stride of the state region under the new layout: consecutive d_state
# indices are d_inner apart.
CPU_STRIDE_ORIG = """    const int32_t * ids = (const int32_t *) src6->data;"""
CPU_STRIDE_NEW = """    const int32_t * ids = (const int32_t *) src6->data;

    // MI210_SSM2D: state is {head_dim, n_head, d_state, n_seqs}, so element
    // (i1, h, i0) lives at (i1 + h*nr) + i0*sd. d_state is now the SLOWEST axis.
    const int sd = (int) (nr*nh);"""

# Both scalar state accesses (Mamba-2 branch and Mamba-1 branch).
CPU_SCALAR_ORIG = """                            const int i = i0 + ii*nc;"""
CPU_SCALAR_NEW = """                            const int i = ii + i0*sd;   // MI210_SSM2D"""

# The SIMD blocks all walk d_state contiguously, which the new layout breaks.
# Fall through to the scalar tail instead. See the cost note in the docstring.
CPU_SVE_NP_ORIG = """                        const int np = (nc & ~(ggml_f32_step - 1));"""
CPU_SVE_NP_NEW = """                        const int np = 0;   // MI210_SSM2D: d_state is no longer contiguous"""

CPU_SIMD_NP_ORIG = """                        const int np = (nc & ~(GGML_F32_STEP - 1));"""
CPU_SIMD_NP_NEW = """                        const int np = 0;   // MI210_SSM2D: d_state is no longer contiguous"""

EDITS = [
    (GGML_C, "ggml.c state shape",   GGML_SHAPE_ORIG,  GGML_SHAPE_NEW,  1),
    (GGML_C, "ggml.c assertions",    GGML_ASSERT_ORIG, GGML_ASSERT_NEW, 1),
    (GGML_C, "ggml.c result shape",  GGML_RESULT_ORIG, GGML_RESULT_NEW, 1),
    (CPU,    "cpu state stride",     CPU_STRIDE_ORIG,  CPU_STRIDE_NEW,  1),
    (CPU,    "cpu scalar index",     CPU_SCALAR_ORIG,  CPU_SCALAR_NEW,  2),
    (CPU,    "cpu SVE np",           CPU_SVE_NP_ORIG,  CPU_SVE_NP_NEW,  1),
    (CPU,    "cpu SIMD np",          CPU_SIMD_NP_ORIG, CPU_SIMD_NP_NEW, 1),
]


def main() -> int:
    check  = "--check"  in sys.argv
    revert = "--revert" in sys.argv

    files = sorted({f for f, _, _, _, _ in EDITS})
    srcs  = {f: open(f).read() for f in files}
    patched = any(MARKER in s for s in srcs.values())

    if check:
        for f in files:
            print(f"{'PATCHED' if MARKER in srcs[f] else 'not patched'}  {f}")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        for f, _, o, n, cnt in EDITS:
            srcs[f] = srcs[f].replace(n, o)
        for f in files:
            open(f, "w").write(srcs[f])
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.",
              file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    for f, name, o, _, cnt in EDITS:
        got = srcs[f].count(o)
        if got != cnt:
            print(f"ERROR: anchor '{name}' matched {got} times, expected {cnt}.",
                  file=sys.stderr)
            return 1
    for f, _, o, n, cnt in EDITS:
        srcs[f] = srcs[f].replace(o, n, cnt)
    for f in files:
        open(f, "w").write(srcs[f])
    print("patched: ggml.c shape + CPU state layout")
    print("NOTE: ggml-cuda/ssm-scan.cu and the call sites are separate patches; "
          "the tree does not build correctly until all are applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
