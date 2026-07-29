#!/usr/bin/env bash
# Cut CHANGELOG.md's [Unreleased] section into a released one.
#
#   scripts/cut-changelog.sh 0.1.0            # perform the cut (idempotent:
#                                            # re-running never moves the date)
#   scripts/cut-changelog.sh --check 0.1.0    # verify the cut is already done
#   scripts/cut-changelog.sh 0.1.0 --date 2026-07-29   # set the date, and
#                                            # RE-date an already-cut section
#
# A thin wrapper so the release procedure has a stable command name; the logic
# (and its edge cases) live in scripts/release_tools.py, which the tests
# exercise directly.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

check=0
args=()
for arg in "$@"; do
    if [ "$arg" = "--check" ]; then
        check=1
    else
        args+=("$arg")
    fi
done

if [ "${#args[@]}" -eq 0 ]; then
    echo "usage: $(basename "$0") [--check] <version> [--date YYYY-MM-DD]" >&2
    exit 2
fi

cd "$root"
if [ "$check" -eq 1 ]; then
    exec python3 -m scripts.release_tools cut "${args[@]}" --check
fi
exec python3 -m scripts.release_tools cut "${args[@]}"
