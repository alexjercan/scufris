# Review: surface backend StreamError detail via agent_status / pending_agents

## Round 1 (out-of-context reviewer, diff vs master)

Reviewer read only the branch diff against `master`. No BLOCKER/MAJOR findings.

- [x] R1.1 (MINOR) `scufris/app.py` persist - on a StreamDone-then-StreamError
      turn, the captured reply won over `run_state.error`, so an ERROR outcome
      could carry a success reply as its message, misleading the "report WHY"
      story. FIXED: for a failed run the error detail now WINS
      (`message = run_state.error or captured.get("message", "")`); clean DONE
      unchanged.
- [x] R1.2 (MINOR) tests - no test pinned the exception-path message improvement
      that DECISION.md claims (stall/budget/crash paths persisting their detail,
      vs master's empty message). FIXED: added
      `test_agent_run_exception_persists_error_with_detail` (a backend that RAISES
      -> RunPhase ERROR -> persist writes `str(exc)` as the outcome message,
      surfaced in `/api/agents/pending`). This is red on master's
      `message=captured.get("message","")`.
- [x] R1.3 (MINOR) tests - the mcp test seeds the ERROR outcome directly, so
      `agent_status` rendering is proven against a hand-seeded outcome rather than
      the full drain->persist->outcome pipeline. ACCEPTED as covered transitively:
      the supervisor test pins drain->run.error, the app tests pin
      drain/exception->persist->outcome->pending, and the mcp test pins
      outcome->agent_status. No single end-to-end mcp test added (the MCP tool
      runs in a separate process and reads the persisted store the app tests
      already populate); the seam is the persisted OutcomeStore, exercised on both
      sides.
- [x] R1.4 (NIT) `scufris/mcp_server.py` - the `error:` line rendered
      `outcome.message` raw. FIXED: flatten newlines + cap at 200 chars, matching
      the `pending_agents` row rendering.
- [x] R1.5 (NIT) `scufris/agent_store.py` - the orchestrator's own turns now
      persist an ERROR outcome on a StreamError, so it could self-appear in its
      own `pending_agents` poll. FIXED: `pending_outcomes` now excludes
      `ORCHESTRATOR_ID` (mirrors `list()` hiding it), pinned by
      `test_pending_outcomes_excludes_the_orchestrator`.

Reviewer verification notes (checked, no issue): `_drain` last-wins is correct
and RunPhase/RunState.error are independent axes; the `failed` condition is
correct for all four exception paths and the StreamError path with no clean-DONE
regression; `preserve_signal` (WAITING/REPORTED) stays intact because it gates on
DONE, and an ERROR correctly overwrites per the existing "an ERROR wins" rule; the
CancelledError path's `run.error or "cancelled"` preserves an earlier StreamError
detail; `pending_agents` already renders `o.message`; the cross-process `error:`
condition is sound; and `no_error_line_on_clean_run` guards against over-eager
surfacing.

All five findings addressed on the branch; full QA gate (`nix flake check`)
green after the fixes.

- VERDICT: APPROVE
</content>
