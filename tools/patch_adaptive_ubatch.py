#!/usr/bin/env python3
"""Choose the micro-batch size per decode call instead of fixing it at startup.

WHY. With `-sm layer` the devices form a pipeline: GPU0 holds the first half of
the layers, GPU1 the second. A single micro-batch therefore runs strictly
sequentially -- GPU1 idles while GPU0 works and vice versa -- and only *multiple*
micro-batches in flight overlap the two. Measured aggregate utilisation on
2x MI210:

    pp4096   62.0%    (2 ubatches at -ub 2048)
    pp16384  85.6%    (8 ubatches)

So the pipeline bubble is severe for short prompts and negligible for long ones.
A larger ubatch gives better kernel efficiency; a smaller one gives more pipeline
depth. Which wins depends entirely on how many ubatches the prompt produces,
and `-ub` is fixed at startup, so a server sized for long context pays for it on
every short request.

Measured, Nemotron-3-Super-120B-A12B Q4_K_M, 2x MI210 (t/s):

    prompt   ub256   ub512   ub1024  ub2048    best
    1024      1134    1375     1345    1346    ub512   +2.2%
    2048      1647    1792     1589      --    ub1024  +12.8%
    3072      1288    1751     2000    1841    ub1024  +8.7%
    4096      1813    2132     2099      --    ub1024  +1.6%
    6144      1327    1867     2265    2336    ub2048
    8192      1903    2336     2483      --    ub2048
    16384     1923    2426     2696      --    ub2048

THE SIGNAL IS NOT PROMPT LENGTH. A first attempt keyed on `n_tokens_all <= 4096`
and regressed long prompts by 10%: `n_tokens_all` is the size of *this decode
call*, which llama.cpp already caps at `-b` (4096 here), so a 16k prompt arrives
as four 4096-token calls and a prompt-length test fires on every one.

What actually matters is how many micro-batches this call will produce, because
that is the pipeline depth available to overlap the two devices. One ubatch means
no overlap at all. So: pick the largest power-of-two ubatch that still yields at
least two per call, and never exceed what was configured.

    ubatch = clamp(pow2_floor(n_tokens_all / 2), 256, cparams.n_ubatch)

which lands on the measured optimum nearly everywhere:

    n_tokens_all   picked   measured best
    1024             512    512    ok
    2048            1024    1024   ok
    3072            1024    1024   ok
    4096            2048    1024   1.6% short of ideal
    4096 (of 16k)   2048    2048   ok

The one imperfect case is a standalone 4096-token prompt, where 1024 measured
1.6% faster. That is not separable from the 4096-token chunks of a long prompt,
where 2048 is 9% faster -- same call size, opposite answer -- so the rule
favours the larger effect.

SAFETY. The value is only ever *reduced* (via std::min), never raised. Compute
buffers and the memory context are sized from cparams.n_ubatch at context
creation, so a smaller runtime ubatch always fits; a larger one would not.

Gated on cparams.causal_attn because non-causal attention asserts
`n_ubatch >= n_tokens` a few lines above -- shrinking the ubatch there would
turn a working configuration into an assertion failure.

Numerically inert: micro-batching only changes how many tokens are submitted per
graph evaluation. Each token attends to the same prior context either way, so
outputs are unchanged. Verify anyway.

    python patch_adaptive_ubatch.py [--check] [--revert]
"""
import sys

TARGET = "src/llama-context.cpp"
MARKER = "MI210_ADAPTIVE_UBATCH"

ORIG = """    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);"""

NEW = """    llama_memory_context_ptr mctx;

    // MI210_ADAPTIVE_UBATCH: with a pipelined device split, one micro-batch means
    // the devices run strictly sequentially -- GPU1 idles while GPU0 works. Only
    // two or more micro-batches in a call overlap them. Measured aggregate
    // utilisation on 2x MI210: 62% at 2 ubatches, 85.6% at 8.
    //
    // Note this keys on the size of THIS call, not the prompt length: llama.cpp
    // already caps a call at n_batch, so a long prompt arrives as several full
    // calls that each want the large ubatch for kernel efficiency. Keying on
    // prompt length instead regressed 16k prompts by 10%.
    //
    // Only ever shrinks cparams.n_ubatch, so the compute buffers sized at
    // context creation stay valid.
    uint32_t n_ubatch_eff = cparams.n_ubatch;
    if (cparams.causal_attn && n_tokens_all >= 2) {
        uint32_t want = 1;
        while (want * 2 <= n_tokens_all / 2) {
            want *= 2;
        }
        want = std::max<uint32_t>(want, 256);
        n_ubatch_eff = std::min(n_ubatch_eff, want);
    }

    while (true) {
        mctx = memory->init_batch(*balloc, n_ubatch_eff, output_all);"""


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
    print("patched: micro-batch size now chosen per decode call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
