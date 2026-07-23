# Review: BC5 end-to-end example + acceptance test

- TASK: 20260723-094318
- BRANCH: feat/bc5-comms-acceptance
- DATE: 20260723
- REVIEWER: out-of-context agent (round 1)
- VERDICT: APPROVE

## Round 1 - VERDICT: APPROVE

Reviewed against the real code; reviewer ran the gate (ruff, mypy -> 47 files
clean, `pytest -k stalled_merge` 2/2, `python examples/comms_loop.py` exit 0) and
re-ran the test 5x (10 executions) with zero failures.

### Findings

1. [verified-ok] The test proves the loop, not a trivial pass. `agent_runs[id]` is
   set (app.py:1201) before the stream runs and `request_input` reads that same id
   (app.py:1370), so WAITING is keyed to the in-flight run and `mark_finished`'s
   `preserve_waiting` (agent_store.py:578-585) fires. The `blocked_once`/`release`
   guard holds only the first sub-agent non-orch turn - never the wake turn
   (is_orch) or the resume turn (blocked_once already True). No deadlock
   (`release.set()` at step 3 and in `finally`).
2. [verified-ok] Race-free. Supervisor sets `run.state=DONE` (supervisor.py:263)
   before the outcome overwrite (supervisor.py:288-290), so "/status done" can
   precede the DONE outcome; but `acknowledge` neutralizes BOTH interleavings and
   the test correctly does NOT assert acknowledge's return. `pending == []` holds
   on a slow machine; all waits bounded (4s).
3. [verified-ok] Both wake paths distinguished and non-vacuous. auto_wake off makes
   an orchestrator turn structurally impossible (wake.py:75-76 returns early), so
   the "no wake" assertion is not racy.
4. [verified-ok] Example standalone (scufris + httpx only, no fixtures), exit 0;
   acknowledge assertion race-free (asserts `pending == []`, not the bool).
5. [verified-ok] Standing in the MCP tools with their HTTP endpoints is a fair
   acceptance - those endpoints are the exact contract the tools call, and the mock
   backend genuinely cannot run MCP tools. Docstrings do not over-claim.
6. [verified-ok] mypy clean (no regression), zero non-ASCII in all changed files,
   README/CHANGELOG accurate.

No substantive issues. Ship it.
