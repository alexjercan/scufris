#!/usr/bin/env bash
# Everything that must be true before a tag may become a release.
#
#   scripts/check-release-ready.sh            # check the tree as it stands
#   scripts/check-release-ready.sh v0.1.0     # ...and that it matches this tag
#
# Run it locally before tagging; the release workflow runs the SAME script as
# its guard job, so a release cannot pass a check the operator did not.
#
# Every check prints what it verified, and the script exits non-zero on the
# first failure with a message that says how to fix it. It never prints a
# reassuring line it has not earned.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

tag="${1:-}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

ok() {
    echo "ok: $*"
}

# 1. The version sources agree: pyproject.toml, the changelog's top released
#    section, and (when given) the tag. This is the check the epic's DoD names.
if [ -n "$tag" ]; then
    python3 -m scripts.release_tools check "$tag" >/dev/null ||
        fail "version sources disagree"
else
    python3 -m scripts.release_tools check >/dev/null ||
        fail "version sources disagree"
fi
version="$(python3 -m scripts.release_tools version)"
ok "version sources agree on ${version}"

# 2. That version has real release notes. `notes` refuses a missing section, an
#    undated one, and an empty one, so this doubles as the "the release page
#    will not be blank" check.
notes="$(python3 -m scripts.release_tools notes "$version")" ||
    fail "no usable CHANGELOG.md section for ${version}"
[ -n "${notes//[[:space:]]/}" ] || fail "CHANGELOG.md section for ${version} is blank"
ok "CHANGELOG.md has release notes for ${version}"

# 3. Task records and the lessons ledger are clean. tatr lives in the dev shell
#    (and is a pinned flake input), so a missing binary means this was run from
#    the wrong environment - say so rather than skipping the check.
if ! command -v tatr >/dev/null 2>&1; then
    fail "tatr not on PATH - run this inside 'nix develop' (it is a flake input)"
fi
tatr check --ledger LESSONS.md || fail "task records or LESSONS.md fail tatr check"
ok "task records and LESSONS.md are clean"

# 4. No uncompiled ephemeral scratch. Per AGENTS.md, per-task material lives in
#    tasks/<id>/ and durable lessons in LESSONS.md, while docs/ holds long-form
#    DURABLE material - so the check must not object to a legitimate design doc
#    living there. The scratch drawer is the explicitly-named docs/scratch/,
#    which /lessons compiles into the ledger and then empties. A release must
#    not ship a drawer someone meant to compile.
#
#    No `| head` here: under `set -o pipefail` a closed pipe kills the script
#    with a bare 141 BEFORE its own diagnostic prints, which is the repo's
#    "never let a pipe eat the exit code" rule biting from the other side.
if [ -d docs/scratch ]; then
    stray="$(find docs/scratch -type f ! -name README.md)"
    if [ -n "$stray" ]; then
        printf '%s\n' "$stray" >&2
        fail "docs/scratch/ holds uncompiled notes (see above) - run /lessons to fold them into LESSONS.md, then clear it"
    fi
    ok "docs/scratch/ is empty"
else
    ok "no docs/scratch/ drawer exists"
fi

# 5. The working tree is clean. This is a LOCAL check: it is what stops an
#    operator tagging a tree with uncommitted changes, so the tag describes a
#    commit that exists. On a runner a fresh checkout is always clean, which is
#    why the workflow additionally asserts that HEAD is the tagged commit - the
#    invariant that actually matters there.
if [ -n "$(git status --porcelain)" ]; then
    git status --short >&2
    fail "working tree is dirty (see above); commit or stash before releasing"
fi
ok "working tree is clean"

echo
echo "release-ready: ${version}${tag:+ (tag ${tag})}"
