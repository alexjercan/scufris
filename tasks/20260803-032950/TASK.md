# Make the health session count follow the orchestrator backend

- PRIORITY: 60
- TAGS: bug, v0.2.0, agents, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145

## Story

As an operator switching the orchestrator off Codex, I want the agent health
surface's session count to come from the orchestrator's own backend, so that a
claude or opencode orchestrator stops reporting a CODEX rollout count.

## Context

`scufris/health.py:258` calls `list_sessions(resolve_codex_home(settings), ...)`
unconditionally whenever `settings.agent_enabled`, so `session_count` and
`last_session` are codex readings regardless of the effective backend. Both the
legacy `/api/agent/health` and the scoped `/api/agents/{id}/health` carry it, so
the two agree and 20260801-100415's contract test passes - it is a residual
leak, not a regression.

Surfaced as a process signal in that task's round 1 review; explicitly out of
its scope (its DoD grep targets `scufris/app.py`, and `health.py` owns its own
probing).

## Steps

- [x] Add the failing test `test_session_summary_follows_the_backend` in
      `tests/test_health.py`: write one scufris-originated rollout for
      `os.getcwd()` into a `tmp_path` codex home (fixture shape as
      `tests/test_sessions.py:55-80`), then for each of codex, claude,
      opencode and mock call
      `agent_health(Settings(codex_home=..., agent_enabled=True), backend=...)`.
      Assert codex reports `session_count == 1` with a non-null `last_session`,
      and the other three report `session_count is None` and
      `last_session is None`. Red today: all four report 1 and the same
      timestamp (reproduced in scratch on `master`).
- [x] Make `AgentHealth.session_count` `int | None = None` in
      `scufris/health.py:44`, documenting `None` as "no reading was taken" and a
      number (0 included) as a real reading - see DECISION.md D3.
- [x] Replace the session summary at `scufris/health.py:253-262` with
      `get_backend(effective_backend).read_memory_footprint(settings)`:
      unsupported -> `None`/`None`; supported -> `value.session_count` and
      `value.newest` (a `None` value -> `0`/`None`). Keep the
      `settings.agent_enabled` gate (disabled -> `None`, no reading) and the
      never-raise `try`.
- [x] Drop the now-unused `os`, `list_sessions` and `resolve_codex_home`
      imports (`scufris/health.py:13,20`). Import direction is safe:
      `scufris/backends/__init__.py` does not import `health`, and
      `agent_diagnostics` already imports both - confirm with
      `python -c "import scufris.health"`.
- [x] Emit the session line in `scufris/telegram/render.py:293` only when
      `session_count is not None`, so a claude orchestrator's card drops the
      line instead of printing `sessions None`.
- [x] Widen `session_count` to `number | null` in
      `web/src/agent-types.ts:149` and push the `N session(s)` bit in
      `renderHealthCard` (`web/src/settings-view.ts:313`) only when it is not
      null; `last_session` is already conditional.
- [x] Add the frontend case
      "omits the session summary when the backend has no session reader" in
      `web/src/settings-view.test.ts`: `health({ backend: "claude",
      session_count: null })` renders a note with no `session` text.
- [x] Update `tests/test_health.py:204` (disabled agent) from
      `session_count == 0` to `session_count is None`, then run the full suite
      and fix any other fixture that asserts the old zero.

## Definition of Done

- A claude/opencode/mock orchestrator reports no session count while a codex
  one still counts its rollouts
  (test: `tests/test_health.py::test_session_summary_follows_the_backend`).
- `health.py` holds no codex-specific session reader
  (cmd: `rg -n "resolve_codex_home|list_sessions" scufris/health.py`, expected
  empty).
- The console omits the session bit when there is no reading
  (test: `web/src/settings-view.test.ts` "omits the session summary when the
  backend has no session reader").
- All Python checks pass
  (cmd: `ruff check . && ruff format --check . && mypy . && python -m pytest`).
- Frontend gate passes (cmd: `cd web && npm run ci`).

## Notes

- Epic: 20260729-102145. Sibling 20260801-100415 recorded this leak in its
  close-out as the last codex-shaped read outside `scufris/backends/`.
- Decisions in DECISION.md: reuse `read_memory_footprint` (D1), the accepted
  codex counting-scope change (D2), nullable int over a `Capability` envelope
  (D3).
- Behaviour change for codex: the count goes from cwd + originator scoped to
  every rollout under `codex_home`, which is what the Memory panel already
  shows for the same agent.
- Three renderers read these fields: the web Health card
  (`web/src/settings-view.ts:304`, shared with the per-agent settings page),
  Telegram `render_health` (`scufris/telegram/render.py:288`), and the raw JSON
  of `/api/agent/health` + `/api/agents/{id}/health`.
- Existing fixtures that set `session_count` to a number
  (`tests/test_telegram.py:127`, `tests/test_telegram_render.py:210`,
  `web/src/settings-view.test.ts:44`, `web/src/agent-settings-view.test.ts:61`)
  stay valid under the widened type.
- No new mode or gate is introduced, so no new-gate grep applies.

## Close-out

**What / why.** `scufris/health.py` now asks
`get_backend(effective_backend).read_memory_footprint(settings)` for the health
card's session summary instead of scanning codex rollouts unconditionally. A
supported reader gives `session_count` + `newest`; an unsupported one gives no
reading, so `AgentHealth.session_count` widened to `int | None` where `None`
means "no reading was taken" and a number (`0` included) is a real measurement.
The three renderers follow: Telegram `render_health` drops the whole session
line on `None`, the web Health card drops the `N session(s)` bit, and the JSON
carries `null`. `health.py` lost its `os` / `list_sessions` /
`resolve_codex_home` imports - the last codex-shaped reader outside
`scufris/backends/` (per 20260801-100415's close-out) is gone.

**Alternatives.** All weighed in DECISION.md and unchanged by the
implementation: a `list_sessions` protocol method (YAGNI - health needs a count
and a timestamp, not a list), a `backend == "codex"` branch in `health.py`
(keeps the codex reader outside the adapters), dropping the fields entirely
(the Story asks for a correct count, not none), counting the ownership registry
(`agent_health` has no registry handle, and `last_session` would cost a parse
per session), and a `Capability[MemoryFootprint]` field on `AgentHealth` (its
third state cannot occur here, so it buys three renderer unwraps for nothing).

**Difficulties / diagnosis.** None material. The import direction held as
planned (`python -c "import scufris.health"` clean; `scufris/backends/` does
not import `health`). The only fallout beyond the plan was the disabled-agent
assertion the plan already named, plus the sprout's `web/node_modules` needing
an `npm ci` before the frontend gate could run.

**Evidence.**

| Proof | Result |
|---|---|
| `pytest tests/test_health.py::test_session_summary_follows_the_backend` | pass (exit 0); red first with `assert 1 is None` for claude/opencode/mock |
| `rg -n "resolve_codex_home\|list_sessions" scufris/health.py` | empty (exit 1) |
| vitest "omits the session summary when the backend has no session reader" | pass |
| `ruff check . && ruff format --check . && mypy . && python -m pytest` | pass (exit 0, 228 files typed) |
| `cd web && npm run ci` | pass (exit 0) |
| round 1: `test_render_health_omits_the_session_line_without_a_reading` | pass; red with the `session_count is not None` guard forced to `True` |
| round 1: vitest "renders the session count when the backend took a reading" | pass; red with the interpolated bit replaced by a bare `"sessions"` |

**Round 1 fixes.** R1.1 and R1.2 added the two missing renderer tests above -
both omission cases now sit beside a proof the bit renders at all. R1.3
replaced the truthiness narrowing in `scufris/health.py` with a single
`value is not None` block. R1.4 documented why the test rollout keeps its cwd
and `codex_exec` originator. No production behaviour changed.

**Reflection.** The leak survived because both health routes shared the same
wrong reader, so their contract test agreed - two surfaces agreeing is not
evidence either is right. The test that catches it compares backends against
each other on one fixture rather than pinning one backend's number. Doc sweep
added a BREAKING CHANGELOG entry: `session_count` is nullable, a disabled agent
reports `null` instead of `0`, and the codex count's scope widened to every
rollout under `codex_home` (matching the Memory panel).
