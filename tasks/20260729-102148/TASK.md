# Extract the backend-aware orchestrator diagnostics service

- PRIORITY: 75
- TAGS: bug, v0.2.0, agents, backend, telegram
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100413

## Story

As an operator, I want one backend-aware service that answers info, account,
usage, memory, health, and visible tools for any orchestrator, so that
Codex-specific readers stop being the implicit definition of what an agent can
report.

## Steps

- [x] Add `tests/test_agent_diagnostics.py` with the four failing proofs,
      driving the scoped `/api/agents/{id}/` account, usage, memory, health and
      tools surfaces over codex, claude, opencode and mock agents plus the
      orchestrator. Watch each fail before writing the service.
- [x] Add `Capability[T]` (`supported`, `value`) to `scufris/backends/base.py`
      and extend the `AgentBackend` protocol with `read_usage`,
      `read_memory_footprint` and a `has_scufris_mcp` flag. Implement in
      `backends/codex.py` (via `sessions.rollout.resolve_codex_home` and
      `sessions.usage`), and unsupported in `claude.py`, `opencode.py`,
      `mock.py`.
- [x] Add `scufris/agent_diagnostics.py`: a transport-independent
      `AgentDiagnostics` over `Settings` returning `AgentRecord`,
      `AccountInfo`, `Capability[UsageQuota]`, `Capability[MemoryFootprint]`,
      `AgentHealth` and `Capability[list[AgentTool]]`. It takes a resolved
      record and raises nothing HTTP-shaped; 404 stays in the route. Move
      `_tool_parameters`, `_as_agent_tool`, `_mcp_servers_for_audience`,
      `_tools_for_servers` and `_probe_servers` out of `scufris/app.py` into
      it.
- [x] Delete `_agent_is_codex` (`scufris/app.py:3252`) and
      `_agent_has_scufris_mcp` (`scufris/app.py:3360`); point the scoped
      `/api/agents/{id}/usage|memory|health|account|tools|mcp` routes at the
      service so capability comes from the backend, not a name comparison.
      Legacy `/api/agent/*` keeps its current behaviour by importing the moved
      helpers, except `/api/agent/account`, whose `quota` field follows
      `AccountInfo` from `UsageQuota | null` to the capability envelope; a
      disabled agent reports it as a supported-but-empty reading.
- [x] Change `AccountInfo.quota` to `Capability[UsageQuota]`, and resolve
      backend, model and auth mode from `agents.get(agent_id)` - including the
      orchestrator's settings-backed synthetic record - so a backend switch
      moves the whole capability set with it.
- [x] Update `web/src/agent-types.ts` and `web/src/agent-settings-view.ts`
      (lines ~516-528) to read the envelope, so the dashboard keeps rendering.
      The richer "not supported by this backend" presentation belongs to
      20260801-100419.
- [x] Document the cross-backend diagnostics contract and the new module in
      `scufris/README.md` (module map plus the HTTP surface section).
- [x] Keep `scufris/agent_diagnostics.py` under the 600-line source cap and
      `tests/test_agent_diagnostics.py` under the 900-line test cap; no new
      `scripts/check_file_size.py` allowlist entry.

## Definition of Done

- All four backends produce consistent backend/model/auth shapes on the scoped
  surface
  (test: `test_scoped_diagnostics_are_backend_consistent`).
- A non-Codex orchestrator reports no Codex quota or rollout counts even when a
  populated codex home is present
  (test: `test_non_codex_orchestrator_hides_codex_account_data`).
- Unsupported capabilities return `supported=false`, distinct from a
  supported-but-empty result
  (test: `test_unsupported_diagnostics_are_explicit`).
- The service resolves backend, model, auth mode and capability from the
  persisted orchestrator record across a backend switch, not from static
  settings
  (test: `test_diagnostics_follow_the_persisted_orchestrator_record`).
- No name-comparison backend branching is left in the app
  (cmd: `! grep -rq --exclude-dir=tasks --exclude-dir=.git --include='*.py' -E '^[^#]*(_agent_is_codex|_agent_has_scufris_mcp)' .`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).
- The frontend gate is green (cmd: `cd web && npm run ci`).

## Notes

- Epic: 20260729-102145. Load-bearing choices are in this task's DECISION.md.
- Depends on the state migration lane, so the persisted orchestrator record it
  reads is already on the transactional store.
- Base-branch probe (ca060ff): the scoped surface ALREADY resolves
  backend/model/auth_mode from the record - a `PATCH /api/agents/orchestrator`
  to claude flips `/account` to `claude_ai`/`claude-opus-4-8`. So the original
  "static settings reads that can drift" step has no bite on this surface; the
  proof was rephrased to pin what IS broken: capability does not follow the
  record because there is no capability, only a name comparison per call site.
- Base-branch probe: a claude orchestrator returns `memory.session_count: 0`
  and `usage: null` - identical to a codex agent with no rollouts yet. That
  ambiguity is the bug the envelope removes.
- `/api/agents/{id}/tools` has no web consumer today (the dashboard uses the
  legacy `/api/agent/tools`), so only usage, memory and account force a
  frontend change.
- `tests/test_app.py` and `scufris/app.py` are both over the size cap and
  allowlisted; the new code goes in new files rather than growing either.
- `tests/test_app.py::test_agent_fork_reverts_single_session` is a PRE-EXISTING
  flake on master (failed 1 of 3 full-suite runs at ca060ff, passes alone).
  Filed as 20260803-020100; a red `pytest` proof on this branch should be
  checked against it before being attributed here.
- Legacy-route delegation (20260801-100415) and Telegram/UI alignment
  (20260801-100419) are the two successor tasks.

## Close-out

### What and why

`Capability[T]` (`supported`, `value`) landed in `scufris/backends/base.py`, and
`AgentBackend` grew `read_usage`, `read_memory_footprint` and `has_scufris_mcp`.
Codex implements the two readers over `sessions.usage`; claude, opencode, mock
and the example's `RecordingBackend` answer unsupported. `scufris/agent_diagnostics.py`
is the transport-independent service over a resolved `AgentRecord`, and it owns
the MCP tool-listing helpers moved out of `app.py` (`tool_parameters`,
`_as_agent_tool`, `mcp_servers_for_audience`, `tools_for_servers`,
`probe_servers`) plus `AccountInfo`. Both name comparisons are gone; the scoped
`usage|memory|health|account|tools|mcp` routes are one-liners over the service.

### Alternatives

The decision record's four alternatives stood. One choice was made here rather
than there: `_ensure_den_path` stayed in `app.py` and the service's `health`/`mcp`
document the precondition, as both callers already bridge it - moving it would
have meant a new home in `mcp_health.py` for no behaviour change.

### Difficulties

- `PATCH /api/agents/orchestrator` with `{"backend": "opencode", "model": ...}`
  422s: `opencode_model` is not in `settings_store.WRITABLE_KEYS` (`claude_model`
  is). Pre-existing and out of scope; the proof switches backend only and reads
  the model that follows, which is what the DoD is actually about.
- `nix flake check` first failed mypy because the new module was untracked and
  the nix source filter only sees git-known files. `git add` fixed it - worth
  remembering before diagnosing a nix-only type error.

### Evidence

- `python -m pytest`: 977 passed. `ruff check .`, `mypy .`: clean.
- `cd web && npm run ci`: green (format, lint, vitest, webpack build).
- `nix flake check --keep-going`: only `checks.records` red, and it reproduces on
  a clean master - filed as 20260803-022018.
- `nix build .#scufris`: ok. `nix build .#scufris-web` fails on an npmDepsHash
  mismatch that also reproduces on master - filed as 20260803-022030.
- The no-name-comparison grep proof and `scripts/check_file_size.py` both pass;
  `agent_diagnostics.py` is 196 lines, `test_agent_diagnostics.py` 278.

### Reflection

Making the four proofs fail first paid off twice: the fourth one exposed that the
record-drift half of the task was already true (as the base-branch probe in the
plan said), so it was rewritten to pin what the envelope actually fixes - the
whole capability set moving with a backend switch. Wire-shape changes rippled to
six existing tests; each was updated to the new shape rather than relaxed.
