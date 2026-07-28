# Review: Cancel in-flight chat runs (stop button + cancel_agent tool + CANCELLED)

- TASK: 20260728-134840
- BRANCH: feature/chat-cancel

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (findings), in-session (merge + re-verify)

Verification run in the worktree (out-of-context reviewer, re-confirmed
in-session): `ruff check .` clean, `mypy .` clean (62 files), `python -m pytest`
561 passed, web `vitest run` 189 passed (19 files), `webpack` build compiled,
`prettier --check` clean. Every DoD `test:`/`cmd:` proof ran and passed
individually; `grep -rn "cancel" CHANGELOG.md` returns the Added entry. The one
`manual:` DoD item (drive the running app: square stop button, cancel, orchestrator
"cancel agent X") remains PENDING for the user to accept.

Load-bearing claims re-derived in-session: the supervisor cancel test awaits a
`closed` event set only in the generator's `finally`, so it genuinely proves the
`_drain` aclose ran (delete the aclose -> the test times out); CANCELLED is keyed
off the explicit `run.cancelled` flag, not the error string, so app-shutdown
aborts (which do not set the flag) stay ERROR; and the orchestrator `forkTurn`
signature at agent-view.ts:247 was confirmed to drop the `signal` param (R1.1).

- [x] R1.1 (MINOR) web/src/agent-view.ts:247 - the orchestrator's `forkTurn`
  signature is `(index, text, handlers)` and drops the trailing `signal` param the
  config now supplies, so a cancel during an orchestrator edit-to-fork turn does
  not abort the local fetch (unlike the per-agent view's `forkTurn`, which threads
  `signal`). The backend cancel POST still closes the SSE server-side so it is not
  a leak and the partial is still kept, but the local read lingers until the server
  closes. Suggest accepting `signal` and passing it into the `fetch(...)` call for
  symmetry with the per-agent path.
  - Response: Fixed - `forkTurn` now accepts `signal` and passes it to the
    `/api/agent/session/fork` fetch, so an orchestrator fork turn is locally
    abortable like the per-agent path.
- [x] R1.2 (NIT) scufris/agent_store.py:752 - `preserve_signal` only fires for
  `state == DONE`, so if an agent emits `request_input` (WAITING) and the user
  cancels that same run, the unacknowledged WAITING signal is overwritten by
  CANCELLED. This is arguably correct (an explicit user stop supersedes the pending
  question) and is a rare race, but it is undocumented; consider a one-line comment
  noting CANCELLED intentionally clobbers a same-run signal.
  - Response: Fixed - added a comment at the `preserve_signal` guard noting that a
    non-DONE terminal (ERROR/CANCELLED) intentionally supersedes a same-run
    WAITING/REPORTED signal.
