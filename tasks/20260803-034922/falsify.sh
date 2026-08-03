#!/usr/bin/env bash
# Falsification harness for the two tests Round 2 of 20260801-100415 flagged as
# unfalsifiable. For each sabotage patch: apply it, require the named test to
# FAIL, revert it, require the same test to PASS. Any deviation - including a
# patch that no longer applies, or a revert that leaves the tree dirty - exits
# non-zero, because a proof that cannot be run is not a proof.
#
# The patches under tasks/ are proof artifacts: nothing else applies them, and
# the suites never see them.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/tasks/20260803-034922"
cd "$ROOT" || exit 1

failures=0

note() { printf '\n=== %s\n' "$*"; }

if [ ! -x "$ROOT/web/node_modules/.bin/vitest" ]; then
    echo "web/node_modules/.bin/vitest is missing - run 'cd web && npm ci' first"
    exit 1
fi

# run_case <label> <patch> <command...>
# The command must exit non-zero exactly when the guarded test fails.
run_case() {
    local label="$1" patch="$2"
    shift 2

    note "$label: applying $(basename "$patch")"
    if ! git apply --check "$patch" 2>&1; then
        echo "FAIL($label): patch no longer applies - the sabotage target moved"
        failures=$((failures + 1))
        return
    fi
    git apply "$patch" || {
        echo "FAIL($label): git apply failed after a clean --check"
        failures=$((failures + 1))
        return
    }

    note "$label: expecting RED under sabotage"
    if "$@"; then
        echo "FAIL($label): test PASSED with the behaviour sabotaged - it pins nothing"
        failures=$((failures + 1))
    else
        echo "ok($label): red under sabotage"
    fi

    git apply -R "$patch" || {
        echo "FAIL($label): could not revert $patch - tree left dirty"
        failures=$((failures + 1))
        return
    }

    note "$label: expecting GREEN restored"
    if "$@"; then
        echo "ok($label): green once restored"
    else
        echo "FAIL($label): test still fails with the behaviour restored"
        failures=$((failures + 1))
    fi
}

run_case R2.1 "$HERE/sabotage-r21.patch" \
    python -m pytest -q \
    "tests/test_app.py::test_disabled_agent_is_supported_not_unsupported"

# Driven through the local vitest binary rather than `npm run test --`, so the
# -t pattern survives as ONE argument instead of being re-split by npm.
vitest_case() {
    (cd "$ROOT/web" && ./node_modules/.bin/vitest run "$@")
}

run_case R2.2 "$HERE/sabotage-r22.patch" \
    vitest_case src/agent-view.test.ts \
    -t "renders the meter from a supported envelope's value"

note "summary"
if [ "$failures" -ne 0 ]; then
    echo "$failures falsification check(s) failed"
    exit 1
fi
echo "both sabotages falsify their test"
