# Drop-in modified files

**Patches are canonical.** Everything here is derived from `patches/` and exists
only for when a patch will not apply cleanly. If the two ever disagree, the
patch is right.

## These target two different base trees

That is why this directory is split. Mixing them gives a tree that matches
neither:

| directory | change sets | base |
|---|---|---|
| [`cdna2/`](cdna2/) | 4–11 — the SSD / MMQ / rocBLAS prefill work | `ggml-org/llama.cpp` @ `67b9b0e` |
| [`turboquant/`](turboquant/) | 1–3 — per-layer KV types, KIVI2, wave64 fixes | `llama-cpp-turboquant` @ `c26cbdff` |

`src/llama-context.cpp` is touched by **both** lineages, so a single flat
directory could only ever hold one version of it — which is what this repo had
before the split.

## The footgun

These are drop-ins **for their exact base commit**, not for a newer tree.
Copying `cdna2/src/llama-context.cpp` onto a llama.cpp newer than `67b9b0e`
silently reverts every upstream change to that file since then. No error, no
conflict — just quietly stale code, and it gets worse the longer the fork sits.

Prefer `git apply --3way patches/NN-*.patch`, which will at least tell you where
it conflicts.

## Keeping this honest

`cdna2/` is generated, and the generator doubles as a checker:

```bash
tools/sync_modified_files.sh           # regenerate from patches 04-11
tools/sync_modified_files.sh --check   # verify; non-zero exit on drift
```

`--check` reports `MISSING`, `DRIFTED` and `STALE` per file, so "patches and
drop-ins agree" becomes something CI asserts rather than something a human
remembers. `turboquant/` is not generated — its base tree is a different repo.
