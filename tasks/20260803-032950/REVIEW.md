# Review: Make the health session count follow the orchestrator backend

- TASK: 20260803-032950
- BRANCH: fix/health-session-count-backend

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) scufris/telegram/render.py:294 - the new
  `session_count is not None` branch ships untested.
  `tests/test_telegram_render.py:325` asserts only the numeric path
  (`sessions 3  last 2026-07-20`); no case covers `None`, so the `sessions None`
  line Step 5 exists to remove would pass the suite. This is the Story's
  user-visible deliverable on the Telegram surface. Add a `render_health` case
  built from the existing fake health with `session_count=None,
  last_session=None`, asserting `"sessions" not in body` while the head line and
  the check lines are still present.
  - Response: fixed in the round-1 commit - added
    `test_render_health_omits_the_session_line_without_a_reading` in
    `tests/test_telegram_render.py`, built from `_fake_health()` with
    `session_count=None, last_session=None`. Asserts `"sessions" not in body`
    with the head line and both check lines still present. Verified red by
    replacing the `session_count is not None` guard with `True`.

- [x] R1.2 (MINOR) web/src/settings-view.test.ts:337 - the new case asserts only
  absence (`not.toContain("session")`), and no test anywhere asserts that a
  non-null `session_count` still renders `N sessions` - the default fixture sets
  `session_count: 3` (`web/src/settings-view.test.ts:44`) but nothing reads it
  back. The bits line's delivery is guarded only indirectly by the neighbouring
  `claude cli` case. Add a sibling case
  `renderHealthCard(health({ session_count: 3 }))` asserting the note contains
  `3 sessions`, so the omission case is paired with a proof the bit renders at
  all.
  - Response: fixed in the round-1 commit - added
    "renders the session count when the backend took a reading" in
    `web/src/settings-view.test.ts`, asserting the note contains `3 sessions`.
    Verified red by replacing the interpolated bit with a bare `"sessions"`.

- [x] R1.3 (NIT) scufris/health.py:262 - `footprint.value.session_count if
  footprint.value else 0` narrows a pydantic model by truthiness where
  `is not None` is meant, and re-narrows the same value on two lines. Replace
  with `value = footprint.value` and a single `if value is not None:` block
  assigning both fields, else `0` / `None`.
  - Response: fixed in the round-1 commit -
    `scufris/health.py` now binds `value = footprint.value` and branches on
    `if value is not None:`, assigning both fields there and `0` in the else.

- [x] R1.4 (NIT) tests/test_health.py:246 - `_write_rollout(home, "sess-health",
  cwd=os.getcwd())` and the fixture's `"originator": "codex_exec"` are shaped for
  the removed cwd+originator-scoped reader; `read_memory_footprint`
  (`scufris/sessions/usage.py:102`) counts every `rollout-*.jsonl` under
  `codex_home` regardless. Keep them - they are what makes the test red on
  `master` - but add a one-line comment saying the cwd and originator are
  deliberately irrelevant to the new reader, so a cold reader does not infer the
  count is cwd-scoped.
  - Response: fixed in the round-1 commit - a four-line comment above
    `_write_rollout` in `tests/test_health.py` says the cwd and originator are
    irrelevant to `read_memory_footprint` and are kept only so the removed
    reader also saw the rollout, which is what makes the test red.

Process signal: Step 3 mandates a `value is None -> 0/None` branch that
DECISION.md's last alternative argues cannot occur for a supported reader; it
exists only to satisfy `Capability`'s optional value. The plan would read
cleaner saying so.

Process signal: the task has no `manual:` proofs - all five are `test`/`cmd`.
Nothing is pending on the user.

Verified in this recording pass, independently of the round-1 reviewer:
`ruff check . && ruff format --check . && mypy . && python -m pytest` exit 0
(228 files typed, full suite green); `cd web && npm run ci` exit 0 (25 files,
262 tests); `rg -n "resolve_codex_home|list_sessions" scufris/health.py` empty,
exit 1. Re-derived the load-bearing claim behind the CHANGELOG's scope note by
reading `scufris/sessions/usage.py:88-113`: the footprint counts every
`rollout-*.jsonl` under `codex_home` by `rglob` with no cwd or originator
filter, and dates it by file mtime rather than the rollout's recorded
timestamp - so the documented widening is real. Confirmed
`read_memory_footprint` is `Capability.unsupported()` on claude
(`scufris/backends/claude.py:497`), opencode (`:267`) and mock (`:67`), which is
what makes the new parametrized test discriminate. Doc sweep: no README or
`docs/` surface names `session_count` or `last_session`, so the CHANGELOG entry
is the whole owed surface.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

All four round-1 findings confirmed fixed at their stated sites: R1.1
`tests/test_telegram_render.py:334-344` (proven red by forcing the
`scufris/telegram/render.py:294` guard to `True`), R1.2
`web/src/settings-view.test.ts:337-341` (proven red by replacing the
interpolated bit with a bare `"sessions"`), R1.3 `scufris/health.py:261-267`
(one `value = footprint.value` bind, one `is not None` branch, no truthiness
narrowing left), R1.4 `tests/test_health.py:252-256`. No regression from the
fixes and no defect round 1 missed, so no new findings.

Process signal: the `else: session_count = 0` branch at
`scufris/health.py:266-267` is unreachable - `read_memory_footprint`
(`scufris/sessions/usage.py:113-116`) always returns a `MemoryFootprint` with
its `Capability.supported`. Step 3 mandates it to satisfy `Capability`'s
optional value, so it is a plan artifact, not a code defect; round 1 raised the
same point about the plan's wording. For the retro, not a change.

Process signal: no route-level test asserts `session_count: null` in the JSON of
`/api/agents/{id}/health` for a claude agent - coverage sits at the
`agent_health()` level. Acceptable, since both routes call that function and
`tests/test_app.py:2184-2195` already pins per-agent backend dispatch, but the
Story is phrased on the surface.

No `manual:` proofs on this task; nothing is pending on the user.

Verified in this recording pass, independently of the round-2 reviewer:
`ruff check . && ruff format --check . && mypy . && python -m pytest` exit 0
(228 files typed, full suite green) and `cd web && npm run ci` exit 0. Then
re-derived the claim the reviewer left open - the red state on `master` - by
checking out `master`'s `scufris/health.py` into the worktree and running
`tests/test_health.py::test_session_summary_follows_the_backend`: exit 1 with
the claude, opencode and mock parameters failing at `tests/test_health.py:264`
and codex passing, which is exactly the leak the Story names. Restored
`scufris/health.py` from HEAD; the tree carries only the ACTIVITY edit.
