#!/usr/bin/env bash
# Materialise a patched llama.cpp tree from patches/. Nothing derived is stored
# in this repo -- patches are the only representation of the changes.
#
#   tools/materialize_tree.sh                      # cdna2 -> ./patched-tree
#   tools/materialize_tree.sh --lineage turboquant --out /tmp/tq
#
# WHY NOT CHECKED-IN COPIES. This repo used to carry modified-files/, whole
# copies of every touched file. Two problems, both real rather than theoretical:
#
#   * It targeted TWO different base trees at once. Change sets 4-11 are cut
#     against ggml-org/llama.cpp @ 67b9b0e, 1-3 against
#     llama-cpp-turboquant @ c26cbdff, and both touch src/llama-context.cpp --
#     so one flat directory could only ever hold one version of it.
#   * Nothing enforced that the copies matched the patches. They drifted:
#     tests/test-backend-ops.cpp was touched by patches 04-11 and simply absent.
#
# Generating on demand removes both. If you want the files, run this.
set -euo pipefail

LINEAGE=cdna2
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --lineage) LINEAGE=$2; shift 2 ;;
        --out)     OUT=$2; shift 2 ;;
        *) echo "usage: $0 [--lineage cdna2|turboquant] [--out DIR]" >&2; exit 2 ;;
    esac
done

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

case "$LINEAGE" in
    cdna2)
        UPSTREAM=${UPSTREAM:-https://github.com/ggml-org/llama.cpp}
        BASE_REF=${BASE_REF:-030ebb558}
        PATCHES=$(ls "$REPO_ROOT"/patches/0[4-9]-*.patch "$REPO_ROOT"/patches/1[0-4]-*.patch 2>/dev/null | sort)
        ;;
    turboquant)
        UPSTREAM=${UPSTREAM:-https://github.com/TheTom/llama-cpp-turboquant.git}
        BASE_REF=${BASE_REF:-c26cbdffcf6fc9b7430cd6b117757e9a3f70b7ea}
        PATCHES=$(ls "$REPO_ROOT"/patches/0[1-3]-*.patch 2>/dev/null | sort)
        ;;
    *) echo "unknown lineage: $LINEAGE" >&2; exit 2 ;;
esac

[ -n "$PATCHES" ] || { echo "no patches found for lineage $LINEAGE" >&2; exit 1; }
[ -n "$OUT" ] || OUT="$REPO_ROOT/patched-tree-$LINEAGE"

rm -rf "$OUT"
echo "cloning $UPSTREAM at $BASE_REF ..."
git clone --quiet --filter=blob:none --no-checkout "$UPSTREAM" "$OUT"
git -C "$OUT" fetch --quiet --depth 1 origin "$BASE_REF" 2>/dev/null || true
git -C "$OUT" checkout --quiet "$BASE_REF"

for p in $PATCHES; do
    echo "applying $(basename "$p")"
    git -C "$OUT" apply --3way "$p"
done

echo
echo "patched tree ready: $OUT"
echo "files touched by lineage '$LINEAGE':"
grep -hoE '^\+\+\+ b/.*' $PATCHES | sed 's|^+++ b/|  |' | sort -u
