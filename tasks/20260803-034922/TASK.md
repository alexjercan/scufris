# Pin the two legacy-diagnostics tests that cannot go red

- PRIORITY: 45
- TAGS: bug, v0.2.0, agents, tests
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145

## Story

As a maintainer of the legacy `/api/agent/*` delegation, I want the two tests
that Round 2 of 20260801-100415 flagged to fail when their behaviour is
reverted, so that DECISION-4 (a disabled agent is supported, not unsupported)
and the `Capability` unwrap are actually pinned.

## Context

Carried from 20260801-100415's REVIEW.md Round 2 (APPROVE with two open
non-blocking findings, R2.1 MINOR and R2.2 NIT). Both tests pass today with or
without the change they are meant to guard.

## Steps

- [x] Build the falsification harness FIRST, so it is red before any test
      changes. Add `tasks/20260803-034922/sabotage-r21.patch` (restores the
      `if not deps.settings.agent_enabled` short-circuit in `get_usage`,
      `get_memory` and the account quota of `scufris/api/legacy_agent.py:476-497`,
      as removed by `git show 6da0c50 -- scufris/app.py`),
      `tasks/20260803-034922/sabotage-r22.patch` (replaces
      `renderUsage(quota.value)` with `renderUsage(quota as unknown as
      UsageQuota)` at `web/src/agent-view.ts:154`), and
      `tasks/20260803-034922/falsify.sh`, which for each patch applies it,
      requires the named test to FAIL, reverts it, requires the test to PASS,
      and exits non-zero on any deviation (including a patch that no longer
      applies). Expect it red on the current tests: both stay green under
      sabotage.
- [x] R2.1 - rewrite `test_disabled_agent_is_supported_not_unsupported`
      (`tests/test_app.py:1855`) to use a POPULATED codex home. Keep
      `agent_enabled=False`, but replace `codex_home=tmp_path / "no-codex"` with
      a home seeded by `_write_session_rollout(home, "sess-d", cwd=os.getcwd(),
      used_percent=42.0)` (helper at `tests/test_app.py:175`, used the same way
      at `tests/test_app.py:1846`). Assert the DELEGATED reading, which the
      short-circuit could not produce: `/api/agent/usage` returns
      `supported: true` with `value["primary"]["used_percent"] == 42.0`,
      `/api/agent/memory` returns `session_count == 1`, and
      `/api/agent/account` reports `enabled: false` beside a populated
      `quota` - the disabled state lives on `enabled` alone. Update the
      docstring comment, which still says "(empty) home".
- [x] R2.2 - delete the "hides the meter when the backend cannot report usage"
      case (`web/src/agent-view.test.ts:386`) per DECISION.md: it cannot
      discriminate, because `renderUsage` (`web/src/chat-sidebar.ts:165`)
      calls `replaceChildren()` and sets `hidden` for both the null value and
      the primary-less envelope. Extend the comment on the surviving
      "renders the meter from a supported envelope's value"
      (`agent-view.test.ts:375`) to record that it is the unwrap's pin.
- [x] Run the harness and both suites, and record the red/green transcript for
      each of the two sabotages in `tasks/20260803-034922/RETRO.md`.

## Definition of Done

- Reverting the delegation short-circuit turns the python test red, and
  reverting the `quota.value` unwrap turns the frontend test red; both go green
  again when restored (cmd: `bash tasks/20260803-034922/falsify.sh`).
- The suites stay green with the harness patches unapplied
  (cmd: `python -m pytest && cd web && npm run ci`).

## Notes

- Base-branch state of the DoD proof: `falsify.sh` is red today. Confirmed in
  scratch for R2.2 - sabotaging the unwrap leaves "hides the meter..." green
  (1 failed, 6 passed; the failure is the neighbouring positive case, which is
  why DECISION.md deletes the negative one rather than restating it).
- The delegation landed in `6da0c50` and `scufris/app.py` was since split into
  routers by `5444fa1`, so R2.1's sabotage targets
  `scufris/api/legacy_agent.py`, NOT `git show master:scufris/app.py` as the
  original Step read.
- `/api/agent/account` currently asserts `quota == {"supported": True, "value":
  None}`; with a populated home that assertion must change too, or the rewritten
  test fails for the wrong reason.
- Cost: the test becomes coupled to `_write_session_rollout`'s fixture shape.
  Accepted - three neighbouring tests already are.
- No product code changes. The patches under `tasks/` are proof artifacts and
  are never applied by the suites.

## Close-out

**What and why.** Two tests flagged in Round 2 of 20260801-100415 passed with
or without the behaviour they named. `falsify.sh` plus two sabotage patches now
prove each pin mechanically; the python test was rewritten to assert a
delegated reading, and the frontend's vacuous negative case was deleted per
DECISION.md. No product code changed.

**Alternatives.** For R2.2, restating the negative case as
`{supported: true, value: {primary: null, ...}}` was rejected in DECISION.md -
`renderUsage` hides and empties the meter for every primary-less shape, so no
assertion over it can discriminate.

**Difficulties and diagnosis.**
- The Step's `git show master:scufris/app.py` anchor was stale (`5444fa1` split
  the app into routers). The sabotage was authored against
  `scufris/api/legacy_agent.py` instead, as TASK.md's Notes already flagged.
- `get_account` has no `settings`-shaped seam left, so the r21 sabotage rewrites
  the quota on the returned `AccountInfo` rather than restoring the original
  branch verbatim. Same observable envelope, which is what the test reads.
- First harness draft drove vitest through `npm run test --`; npm re-split the
  `-t` pattern into separate words and the sabotage looked red for the wrong
  reason (`vitest: command not found`). Fixed by calling
  `web/node_modules/.bin/vitest` directly, plus a preflight that fails loudly
  when `npm ci` has not run in the worktree.
- R2.2 was already falsifying once the harness pointed at the SURVIVING
  positive case. The defect there was the negative twin, not a missing pin.

**Evidence.** `bash tasks/20260803-034922/falsify.sh` -> exit 0 (red then green
for both sabotages; transcript in RETRO.md). `python -m pytest` -> exit 0.
`cd web && npm run ci` -> exit 0.

**Reflection.** Building the harness before touching the tests paid: it was red
on R2.1 and green on R2.2 from the start, which split a "two broken tests" brief
into one rewrite and one deletion instead of two speculative rewrites.
