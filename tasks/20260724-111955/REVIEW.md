# Review: Record session ownership at launch per backend (part 2)

- TASK: 20260724-111955
- BRANCH: fix/session-launch-handles

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (out-of-context reviewer, re-confirmed in-session): pytest 429
passed, ruff clean, mypy clean. All five named DoD tests pass; the codex
`clientInfo` grep matches `agent.py:515` (unchanged, not in the diff). A/B on the
`StreamDone` substitution: neutering `and not event.session_id` turns
`test_claude_stream_done_carries_minted_id` red, green when restored - the test
genuinely proves the substitution.

Independently re-derived in-session: resume ALWAYS wins over mint - `stream` sets
`new_session_id=None` whenever `resumable`, and `resumable` short-circuits on
`bool(session_id)` so `_find_claude_session(..., session_id or "")` is never
actually called with `""`; `_claude_stream_args` emits `--resume` xor
`--session-id`, never both, and only a fresh `uuid4()` ever reaches
`--session-id`. Holds at both layers.

- [x] R1.1 (NIT) scufris/backends.py:443 and :560 - the resumability disk scan
  (`_find_claude_session`) runs twice per fresh claude turn: once in `stream` to
  decide `new_session_id`, again inside `_claude_stream_args`. Harmless and
  consistent, but a redundant `rglob`. Suggest passing the already-computed
  decision into the arg builder instead of re-scanning.
  - Response: fixed - `_claude_stream_args` gained an optional
    `resumable: bool | None` override; `stream` computes it once and passes it,
    so the arg builder no longer re-scans on the turn path. The override defaults
    to None, so the pure-function unit tests (which pass no override) keep
    exercising the self-contained disk-check path unchanged. Verified in-session;
    ticked on that confirmation.

No BLOCKER/MAJOR/MINOR findings. No open `manual:` DoD items. APPROVE.
