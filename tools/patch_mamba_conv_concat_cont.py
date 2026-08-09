#!/usr/bin/env python3
"""Materialise the Mamba-2 conv transpose so concat takes its coalesced path.

WHY. src/models/mamba-base.cpp builds the conv input as

    ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);

ggml_transpose only rewrites strides, so src1 is not contiguous. concat_cuda
gates on that:

    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1))
        ... fast path ...
    // non-contiguous kernel (slow)

so this lands in the kernel upstream itself labels slow. In the dim == 0 branch
consecutive threads read `src1 + (i0-ne00)*nb10` where nb10 is the ORIGINAL row
stride, i.e. one useful dword per cache line -- 16-32x read amplification. Writes
are coalesced; only the reads are pathological.

Measured on 2x MI210 at pp4096: concat_non_cont was 171 ms over 160 calls (3.2%
of prefill GPU time), about 1.07 ms per call. Only d_conv-1 = 3 of ~2051 columns
come from `conv`, so essentially all of that is the transpose being read badly.

THE FIX. ggml_cont on a transposed view is exactly the shape ggml's tiled copy
handles: cpy.cu selects cpy_scalar_transpose (a 32x32 LDS tile) when
`nb01 == ggml_element_size(src0) && ne[3] == 1 && ...`, which a transposed view
satisfies. So this trades one badly-coalesced kernel for two well-coalesced ones.

TRADE-OFF, STATED PLAINLY. mamba-base.cpp is backend-agnostic graph
construction, so this adds an explicit materialisation for every backend, not
just CUDA/HIP. On a backend whose concat already handles strided input well, the
extra copy is pure cost. It is numerically neutral either way -- ggml_cont
changes layout, not values. Scoped here to a fork targeting gfx90a; anyone
upstreaming it should gate on the backend or fix the concat kernel instead.

    python patch_mamba_conv_concat_cont.py [--check] [--revert]
"""
import sys

TARGET = "src/models/mamba-base.cpp"
MARKER = "MI210_CONV_CONT"

ORIG = """        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);"""

NEW = """        // MI210_CONV_CONT: materialise the transpose so concat can take its
        // contiguous path. A bare transposed view is only a stride change, which
        // sends concat into its own "non-contiguous kernel (slow)" branch where
        // reads hit one useful dword per cache line.
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_cont(ctx0, ggml_transpose(ctx0, xBC)), 0);"""


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
        with open(TARGET, "w") as f:
            f.write(src.replace(NEW, ORIG, 1))
        print("reverted")
        print("REBUILD REQUIRED: source reverted but the binary was not rebuilt; "
              "it still contains the previous state and will be measured instead.", file=sys.stderr)
        return 0

    if patched:
        print("already patched")
        return 0

    n = src.count(ORIG)
    if n != 1:
        print(f"ERROR: anchor matched {n} times, expected 1 -- upstream moved; "
              "re-derive rather than forcing.", file=sys.stderr)
        return 1

    with open(TARGET, "w") as f:
        f.write(src.replace(ORIG, NEW, 1))
    print("patched: Mamba-2 conv transpose materialised before concat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
