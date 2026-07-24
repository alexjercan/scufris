# Review: Align orchestrator->sub-agent steer path to idle timeout

- TASK: 20260724-081804
- BRANCH: bug/steer-idle-timeout

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Out-of-context reviewer ran the suite (test_opencode_client.py +
test_mcp_server.py -> all pass) and independently verified: `httpx.Timeout(t,
read=None)` sets connect/write/pool=t, read=None; opencode `_request` threads
the optional timeout correctly (None -> client default, not "no timeout"); quick
calls keep their bounded read; `_CHAT_TIMEOUT` fully removed (0 references). It
confirmed the `read=None` backstops: `app.py:1207` runs `OpenCodeBackend.stream`
under `supervisor.start(heartbeat_seconds=600)`, and `send_message` sits before
the first `yield` so it is covered by the supervisor's `wait_for(anext(),
timeout=heartbeat)` (in-session pass re-derived this from supervisor._drain).
Both new tests assert real request-boundary behavior and fail pre-fix (read was
30/120). Goal delivered.

- [x] R1.1 (MINOR) scufris/cli.py:73 - the one-shot `scufris chat` path drives
  `backend.stream` OUTSIDE the supervisor, so with the opencode backend (no
  supervisor heartbeat, no internal idle guard) the new `read=None` had no
  backstop and could hang the CLI forever - a regression this branch introduces
  (pre-fix it capped at agent_timeout_seconds). Give the CLI path its own
  no-output backstop, or document the limitation.
  - Response: Fixed. `_chat_once` now iterates the stream under
    `asyncio.wait_for(anext(), timeout=agent_heartbeat_seconds)`, mirroring the
    supervisor's per-event heartbeat, so a stalled turn is bounded for ANY
    backend and `read=None` always has a backstop. Pinned by
    `test_chat_one_shot_stalled_turn_is_bounded` (a stalling backend raises
    within a 0.2s heartbeat).
- [ ] R1.2 (NIT) tasks/20260724-081804/TASK.md title still names the removed
  `_CHAT_TIMEOUT` constant.
  - Response: Declined - the task title is append-only history (flow guideline:
    the `tasks/` tree is not rewritten to match later renames). No live code or
    doc surface references the removed constant.

No open `manual:` items.
