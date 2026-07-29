#!/usr/bin/env bash
# Print exactly one version's CHANGELOG.md section body - the text the GitHub
# Release page shows.
#
#   scripts/release-notes.sh 0.1.0
#   scripts/release-notes.sh v0.1.0     # a tag is accepted; the v is stripped
#
# Exits non-zero (printing why) when that version has no section, has no date,
# or is empty - so a release pipeline cannot publish a page with empty notes.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -ne 1 ]; then
    echo "usage: $(basename "$0") <version>" >&2
    exit 2
fi

cd "$root"
exec python3 -m scripts.release_tools notes "$1"
