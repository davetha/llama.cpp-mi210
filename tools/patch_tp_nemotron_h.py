#!/usr/bin/env python3
"""Teach the tensor-parallel splitter about Nemotron-H / Nemotron-H-MoE.

`-sm tensor` (LLAMA_SPLIT_MODE_TENSOR) shards a model across GPUs via a meta
device plus llama_meta_device_get_split_state, which assigns each tensor a split
axis by regex on its name. Every SSM/hybrid architecture is on the
llm_arch_supports_sm_tensor exclusion list because that table does not describe
their tensors. This adds the missing rules for Nemotron-H.

WHAT WAS WRONG, from instrumenting the splitter on this model:

  blk.0.ssm_a       ne=[1,128] axis=0 -> [0 1]   splits the size-1 axis; device 0
                                                 gets an empty slice
  blk.0.ssm_conv1d  ne=[4,10240] axis=1 -> [5120 5120]
                                                 one flat segment, ignoring the
                                                 x|B|C boundaries inside it
  blk.N.ssm_in                             absent -- unmatched, so MIRRORED, while
                                                 ssm_out is split: inconsistent

MAMBA-2 LAYOUT (confirmed in src/models/mamba-base.cpp, and against the GGUF):

    ssm_in  = [n_embd, 2*d_inner + 2*n_group*d_state + n_head]
            =  z(d_inner) | x(d_inner) | B(g*s) | C(g*s) | dt(n_head)
    xBC     =  x(d_inner) | B(g*s) | C(g*s)          <- what conv1d operates on

For this model: d_inner 8192, n_group 8, d_state 128, n_head (= ssm_dt_rank) 128,
head_dim = d_inner/n_head = 64. So ssm_in is 8192+8192+1024+1024+128 = 18560 and
conv1d is 8192+1024+1024 = 10240.

THE SPLIT. Shard by head: each device takes a contiguous range of heads, and the
corresponding ranges of z, x, dt, plus the matching groups of B and C. ssm_out is
already split on AXIS_0 by d_inner, which is consistent with that. Concretely on
2 devices: 64 heads, 4096 of d_inner, 4 groups, 512 of g*s each.

Splitting a flat 10240 in half would put 5120 on device 0 -- numerically the same
count, but the wrong *elements*: it would take x[0:5120] rather than
x[0:4096]+B[0:512]+C[0:512]. Segmenting is what makes the boundaries line up.

Granularity keeps each device's slice on a head boundary for z/x (head_dim) and a
group boundary for B/C (d_state), so no head or group is ever cut in half.

    python patch_tp_nemotron_h.py [--check] [--revert]
"""
import sys

TARGET = "src/llama-model.cpp"
ARCH   = "src/llama-arch.cpp"
MARKER = "MI210_TP_NEMOTRON_H"

# ---- 1. new tensor-name patterns -------------------------------------------
PAT_ORIG = """    static const std::regex pattern_ssm_out_weight  ("blk\\\\.\\\\d*\\\\.ssm_out.weight");
"""
PAT_NEW = """    static const std::regex pattern_ssm_out_weight  ("blk\\\\.\\\\d*\\\\.ssm_out.weight");
    // MI210_TP_NEMOTRON_H: Mamba-2 tensors the upstream table does not describe.
    static const std::regex pattern_ssm_in_weight   ("blk\\\\.\\\\d*\\\\.ssm_in.weight");
    static const std::regex pattern_ssm_d           ("blk\\\\.\\\\d*\\\\.ssm_d");
    static const std::regex pattern_ssm_norm        ("blk\\\\.\\\\d*\\\\.ssm_norm.weight");
    static const std::regex pattern_ssm_conv1d_bias ("blk\\\\.\\\\d*\\\\.ssm_conv1d.bias");
"""

# ---- 2. axis assignment ----------------------------------------------------
# ssm_a is matched upstream with AXIS_0, which is correct only when the head
# dimension is dim 0. Nemotron-H stores ssm_a and ssm_d as [1, n_head], so the
# head dimension is dim 1 and AXIS_0 splits a length-1 axis into [0, 1].
CFG_ORIG = """        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
"""
CFG_NEW = """        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                std::regex_match(tensor_name, pattern_ssm_d)) {
            // MI210_TP_NEMOTRON_H: these are per-head parameters, but the head
            // dimension is not always dim 0 -- Nemotron-H stores ssm_a/ssm_d as
            // [1, n_head]. Splitting dim 0 there yields an empty slice on one device.
            const auto axis = (tensor->ne[0] == 1 && tensor->ne[1] > 1) ?
                GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
            return get_tensor_config_impl(axis, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_in_weight)) {
            // MI210_TP_NEMOTRON_H: fused z|x|B|C|dt projection, column-parallel.
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_norm)) {
            // MI210_TP_NEMOTRON_H: [d_inner/n_group, n_group] -- split the groups.
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d_bias)) {
            // MI210_TP_NEMOTRON_H: the conv bias is 1-D over the same x|B|C run as
            // the conv weight, so it must be segmented identically or the ADD that
            // applies it mixes a mirrored tensor with a split one.
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
"""

# ---- 3. segmentation -------------------------------------------------------
SEG_ORIG = """        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_embd;
"""
SEG_NEW = """        // MI210_TP_NEMOTRON_H: the fused Mamba-2 projections must be split on their
        // internal boundaries, not as one flat run, or a device receives the wrong
        // elements even when it receives the right number of them.
        if (ud->model->arch == LLM_ARCH_NEMOTRON_H || ud->model->arch == LLM_ARCH_NEMOTRON_H_MOE) {
            const int64_t d_inner = hparams.ssm_d_inner;
            const int64_t n_head  = hparams.ssm_dt_rank;
            const int64_t gs      = (int64_t) hparams.ssm_n_group * hparams.ssm_d_state;
            if (std::regex_match(tensor_name, pattern_ssm_in_weight)) {
                GGML_ASSERT(tensor->ne[axis] == 2*d_inner + 2*gs + n_head);
                return {{d_inner, 2}, {gs, 2}, {n_head, 1}};   // z,x | B,C | dt
            }
            if (std::regex_match(tensor_name, pattern_ssm_conv1d) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d_bias)) {
                GGML_ASSERT(tensor->ne[axis] == d_inner + 2*gs);
                return {{d_inner, 1}, {gs, 2}};                // x | B,C
            }
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_embd;
"""

# ---- 4. granularity --------------------------------------------------------
# Keep every device's slice on a head boundary for z/x and a group boundary for
# B/C, so a head or group is never cut across devices.
GRAN_ORIG = """        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
"""
GRAN_NEW = """        // MI210_TP_NEMOTRON_H: align slices to whole heads (z, x) and whole groups
        // (B, C); dt is already one element per head.
        if (ud->model->arch == LLM_ARCH_NEMOTRON_H || ud->model->arch == LLM_ARCH_NEMOTRON_H_MOE) {
            const int64_t head_dim = hparams.ssm_d_inner / std::max<int64_t>(hparams.ssm_dt_rank, 1);
            const int64_t d_state  = hparams.ssm_d_state;
            if (std::regex_match(tensor_name, pattern_ssm_in_weight)) {
                GGML_ASSERT(segments.size() == 3);
                return {std::lcm(head_dim, blck_size), std::lcm(d_state, blck_size), 1};
            }
            if (std::regex_match(tensor_name, pattern_ssm_conv1d) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d_bias)) {
                GGML_ASSERT(segments.size() == 2);
                return {std::lcm(head_dim, blck_size), std::lcm(d_state, blck_size)};
            }
            if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(head_dim, blck_size)};
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_d) || std::regex_match(tensor_name, pattern_ssm_norm)) {
                GGML_ASSERT(segments.size() == 1);
                return {1};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
"""

EDITS = [("patterns", PAT_ORIG, PAT_NEW),
         ("axis assignment", CFG_ORIG, CFG_NEW),
         ("segmentation", SEG_ORIG, SEG_NEW),
         ("granularity", GRAN_ORIG, GRAN_NEW)]

# ---- 5. take the architecture off the exclusion list ------------------------
ARCH_ORIG = "        case LLM_ARCH_NEMOTRON_H_MOE:\n"


def main() -> int:
    check = "--check" in sys.argv
    revert = "--revert" in sys.argv

    src = open(TARGET).read()
    arch = open(ARCH).read()
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
        # restore the exclusion entry
        i = arch.index("bool llm_arch_supports_sm_tensor")
        j = arch.index("\n}\n", i)
        body = arch[i:j]
        if "        case LLM_ARCH_NEMOTRON_H:\n" in body and ARCH_ORIG not in body:
            body = body.replace("        case LLM_ARCH_NEMOTRON_H:\n",
                                "        case LLM_ARCH_NEMOTRON_H:\n" + ARCH_ORIG, 1)
            open(ARCH, "w").write(arch[:i] + body + arch[j:])
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.", file=sys.stderr)
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

    # drop NEMOTRON_H_MOE from the exclusion switch only (the symbol also appears
    # in the arch name table)
    i = arch.index("bool llm_arch_supports_sm_tensor")
    j = arch.index("\n}\n", i)
    body = arch[i:j]
    if ARCH_ORIG in body:
        open(ARCH, "w").write(arch[:i] + body.replace(ARCH_ORIG, "", 1) + arch[j:])

    print("patched: Nemotron-H tensor-parallel split rules added, arch enabled")
    print("NEXT: test-backend-ops plus generated tokens; a wrong split is silent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
