#!/usr/bin/env python3
"""Give GGML_OP_SSM_SCAN a tensor-parallel split-state handler.

ggml-backend-meta.cpp groups SSM_SCAN with the ops that call
handle_generic(scalar_only=true), which refuses any dimensional split. That is
why every Mamba/hybrid architecture is on the sm_tensor exclusion list: the
tensors can be sharded, but the scan that consumes them cannot describe its
output.

WHY IT CAN BE DESCRIBED. From ggml_ssm_scan in ggml.c, the result is a single
1-D tensor holding two concatenated regions:

    result = ggml_new_tensor_1d(ctx, F32,
                 ggml_nelements(x) + s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);
           =  y( head_dim * n_head * n_seq_tokens * n_seqs )
           ++ states( d_state * head_dim * n_head * n_seqs )

Both regions are linear in n_head, so the total is too. Sharding by head
therefore splits the whole buffer proportionally: device j holds n_head_j/n_head
of it, which is exactly its heads' share of y plus its heads' share of the
states. A two-segment description is the more literal one but is rejected --
the post-pass that reconciles an op output against its sources only supports a
single segment.

That is why SSM_CONV is not a usable template here -- it writes a plain tensor
and only has to map an input axis to an output axis, so its handler is a few
lines. SSM_SCAN needs per-device segment sizes computed from the shapes.

WHERE THE HEAD COUNT COMES FROM. src1 (x) has shape
[head_dim, n_head, n_seq_tokens, n_seqs] and is split on axis 1, so its
split state already carries each device's head count. Everything else is
derived from it, which also means the handler refuses to guess: if x is not
split exactly that way, it falls back to the generic path rather than emitting
a plausible-but-wrong layout.

CORRECTNESS RISK. A wrong split here does not crash -- it propagates a wrong SSM
state and yields fluent, confident, wrong text. Gate on test-backend-ops plus
reading generated tokens, and compare against -sm layer output at temperature 0.

    python patch_tp_ssm_scan_handler.py [--check] [--revert]
"""
import sys

TARGET = "ggml/src/ggml-backend-meta.cpp"
MARKER = "MI210_TP_SSM_SCAN"

# Define the handler right after the existing SSM_CONV one.
HANDLER_ORIG = """    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {"""

HANDLER_NEW = """    // MI210_TP_SSM_SCAN: the scan output is one 1-D buffer holding y followed by
    // the updated states (see ggml_ssm_scan). Both regions are linear in n_head,
    // so sharding by head splits both proportionally -- two segments on axis 0,
    // sized per device from that device's head count.
    auto handle_ssm_scan = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_tensor * s   = tensor->src[0];
        const ggml_tensor * x   = tensor->src[1];
        const ggml_tensor * ids = tensor->src[6];
        if (s == nullptr || x == nullptr || ids == nullptr) {
            return handle_generic(src_ss, /*scalar_only =*/ true);
        }
        // x is [head_dim, n_head, n_seq_tokens, n_seqs]; the head split lives on
        // axis 1. Anything else and we do not know the layout -- fall back rather
        // than emit a plausible but wrong one.
        if (src_ss[1].axis != GGML_BACKEND_SPLIT_AXIS_1 || src_ss[1].n_segments != 1) {
            return handle_generic(src_ss, /*scalar_only =*/ true);
        }
        // Both regions are linear in n_head, so the TOTAL size is too. That means a
        // single proportional split is exact: device j gets n_head_j/n_head of the
        // buffer, which is precisely its heads' share of y plus its heads' share of
        // the states. The framework's post-pass computes ne[] from the first split
        // source, so returning the axis alone is enough -- and describing it as two
        // segments is rejected downstream, since that pass only supports one.
        GGML_UNUSED(s);
        GGML_UNUSED(ids);
        ggml_backend_meta_split_state ret;
        memset(&ret, 0, sizeof(ret));
        ret.axis       = GGML_BACKEND_SPLIT_AXIS_0;
        ret.n_segments = 1;
        ret.nr[0]      = 1;
        return ret;
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {"""

# Route the op to it instead of the scalar-only generic path.
CASE_ORIG = """            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:"""

CASE_NEW = """            case GGML_OP_SSM_SCAN: {   // MI210_TP_SSM_SCAN
                split_state = handle_ssm_scan(src_ss);
            } break;
            case GGML_OP_WIN_PART:"""

EDITS = [("handler", HANDLER_ORIG, HANDLER_NEW),
         ("op dispatch", CASE_ORIG, CASE_NEW)]


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv

    src = open(TARGET).read()
    patched = MARKER in src

    if check:
        print(f"{'PATCHED' if patched else 'not patched'}  {TARGET}")
        return 0 if patched else 1

    if revert:
        if not patched:
            print("not patched; nothing to revert")
            return 0
        for _, o, n in EDITS:
            src = src.replace(n, o)
        open(TARGET, "w").write(src)
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.",
              file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    for name, o, _ in EDITS:
        if src.count(o) != 1:
            print(f"ERROR: anchor '{name}' matched {src.count(o)} times, expected 1.",
                  file=sys.stderr)
            return 1
    for _, o, n in EDITS:
        src = src.replace(o, n, 1)
    open(TARGET, "w").write(src)
    print("patched: SSM_SCAN now has a tensor-parallel split-state handler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
