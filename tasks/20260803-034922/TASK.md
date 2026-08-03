# Pin the two legacy-diagnostics tests that cannot go red

- PRIORITY: 45
- TAGS: bug,v0.2.0,agents,tests
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
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

- [ ] R2.1 - `test_disabled_agent_is_supported_not_unsupported`
      (`tests/test_app.py:1855`) uses an EMPTY `codex_home`, so the deleted
      `settings.agent_enabled` short-circuit and the delegated reader are
      observationally identical. Point it at a POPULATED home
      (`_write_session_rollout(...)`, as at `tests/test_app.py:1841`) with
      `agent_enabled=False` and assert usage and memory report the real reading
      beside `enabled: false`. Verify red by restoring the short-circuit from
      `git show master:scufris/app.py`.
- [ ] R2.2 - "hides the meter when the backend cannot report usage"
      (`web/src/agent-view.test.ts:386`) passes with or without the unwrap,
      because `renderUsage` (`web/src/chat-sidebar.ts:165`) reads
      `usage?.primary`, which the raw envelope also lacks. Assert something the
      envelope discriminates (e.g. the meter is empty as well as hidden), or
      drop the case as covered by the supported one.

## Definition of Done

- Both tests fail when their guarded change is reverted, and pass when it is
  restored (cmd: `python -m pytest && cd web && npm run ci`).
