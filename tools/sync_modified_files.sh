#!/usr/bin/env bash
# Generate or verify modified-files/cdna2/ from patches 04-11.
#
#   tools/sync_modified_files.sh            regenerate modified-files/cdna2/
#   tools/sync_modified_files.sh --check    verify it matches; exit 1 if not
#
# WHY THIS EXISTS. patches/ and modified-files/ encode the same change twice,
# with nothing enforcing that they agree. They happened to agree when this was
# written, by hand. That is not a property, it is an accident waiting to lapse --
# and BUILD.md in this repo drifted to describing an entirely different base
# repo, so the hazard is not hypothetical.
#
# Patches are canonical. This script makes modified-files/cdna2/ a derived
# artifact, and --check turns "they agree" into something CI can assert.
#
# Set LLAMA_SRC to an existing clean checkout at BASE_REF to skip the clone.
set -euo pipefail

BASE_REF=${BASE_REF:-67b9b0e}
UPSTREAM=${UPSTREAM:-https://github.com/ggml-org/llama.cpp}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$REPO_ROOT/modified-files/cdna2"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

PATCHES=$(ls "$REPO_ROOT"/patches/0[4-9]-*.patch "$REPO_ROOT"/patches/1[01]-*.patch 2>/dev/null | sort)
[ -n "$PATCHES" ] || { echo "no patches 04-11 found" >&2; exit 1; }

# Files these patches touch -- the exact set modified-files/cdna2 should contain.
FILES=$(grep -hoE '^\+\+\+ b/.*' $PATCHES | sed 's|^+++ b/||' | sort -u)

# Always patch a THROWAWAY tree. Patching LLAMA_SRC in place would mutate the
# caller's checkout and make a second run fail on an already-patched tree --
# which is exactly what happened the first time this script was tested.
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT

if [ -n "${LLAMA_SRC:-}" ]; then
    echo "exporting $BASE_REF from $LLAMA_SRC ..."
    git -C "$LLAMA_SRC" archive "$BASE_REF" | tar -x -C "$SRC"
else
    echo "cloning $UPSTREAM at $BASE_REF ..."
    git clone --quiet --no-checkout "$UPSTREAM" "$SRC"
    git -C "$SRC" checkout --quiet "$BASE_REF"
fi
# git apply needs a repo for --3way; a bare export has no index.
if [ ! -d "$SRC/.git" ]; then
    git -C "$SRC" init --quiet
    git -C "$SRC" add -A
    git -C "$SRC" -c user.email=s@l -c user.name=sync commit --quiet -m base
fi

for p in $PATCHES; do
    echo "applying $(basename "$p")"
    git -C "$SRC" apply --3way "$p"
done

rc=0
if [ "$CHECK" = 1 ]; then
    for f in $FILES; do
        if [ ! -f "$OUT/$f" ]; then
            echo "MISSING  $f"; rc=1; continue
        fi
        if ! diff -q "$SRC/$f" "$OUT/$f" >/dev/null; then
            echo "DRIFTED  $f"; rc=1
        fi
    done
    # anything present that the patches do not produce is stale
    if [ -d "$OUT" ]; then
        while IFS= read -r extra; do
            echo "$FILES" | grep -qxF "$extra" || { echo "STALE    $extra"; rc=1; }
        done < <(cd "$OUT" && find . -type f ! -name README.md | sed 's|^\./||')
    fi
    [ "$rc" = 0 ] && echo "modified-files/cdna2 matches patches 04-11"
else
    rm -rf "$OUT"
    for f in $FILES; do
        mkdir -p "$OUT/$(dirname "$f")"
        cp "$SRC/$f" "$OUT/$f"
    done
    echo "regenerated modified-files/cdna2 ($(echo "$FILES" | wc -l) files)"
fi
exit $rc
