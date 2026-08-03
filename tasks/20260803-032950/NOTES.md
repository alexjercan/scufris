# Notes: Make the health session count follow the orchestrator backend

## What changes

Before: `GET /api/agent/health` and `GET /api/agents/{id}/health` always report
`session_count` / `last_session` read out of the codex rollout directory, for
every backend. A claude orchestrator with 40 old codex rollouts under
`~/.codex` reports `sessions 40 last 2026-07-20` on the settings page and in
`/health` on Telegram, while its own `memory` panel next to it says
`unsupported`.

After: both fields follow the effective backend. codex keeps a number; claude,
opencode and mock report no sessions, matching what `GET /api/agents/{id}/memory`
already says for the same agent. The rest of `AgentHealth` (checks, backend,
backend_version) is untouched.

## Surfaces

| File | Why |
|-|-|
| `scufris/health.py:253-270` | the codex-hardcoded block; the only change of substance |
| `scufris/health.py:1-22` | drops the `sessions.list_sessions` / `resolve_codex_home` / `os` imports, gains `backends.get_backend` |
| `tests/test_app.py` | the legacy `/api/agent/health` cases that assert a count |
| `tests/test_telegram.py:127` | builds an `AgentHealth` directly; unaffected unless the model changes |
| `web/src/settings-view.ts:313-316` | renders the two fields; unchanged if the model shape holds |
| `scufris/telegram/render.py:293-295` | same, for `/health` |
| `CHANGELOG.md` | operator-visible reading change |

`scufris/agent_diagnostics.py:167` already passes the agent's own backend into
`agent_health`, so no plumbing is needed - `effective_backend` is right at the
call site and simply is not consulted by the session block.

## Data and interfaces

No new protocol method. `AgentBackend` already carries the per-backend reader
this needs:

```python
def read_memory_footprint(self, settings: Settings) -> Capability[MemoryFootprint]
```

`MemoryFootprint` (`scufris/sessions/usage.py:74`) is
`session_count: int, total_bytes: int, oldest: datetime | None, newest: datetime | None`.
codex answers `Capability.read(read_rollout_footprint(resolve_codex_home(settings)))`;
claude, opencode and mock answer `Capability.unsupported()`.

`AgentHealth.session_count: int = 0` and `last_session: datetime | None = None`
keep their current types. Adding a third state to `AgentHealth` is possible but
not proposed - see the open question.

## Sketches

Illustrative, not a patch:

```python
# scufris/health.py - Session summary
    session_count = 0
    last_session: datetime | None = None
    if settings.agent_enabled:
-       try:
-           sessions = list_sessions(resolve_codex_home(settings), os.getcwd())
-           session_count = len(sessions)
-           last_session = sessions[0].updated_at if sessions else None
-       except Exception:  # noqa: BLE001 - diagnostics never raise
-           pass
+       try:
+           footprint = get_backend(effective_backend).read_memory_footprint(settings)
+           if footprint.value is not None:
+               session_count = footprint.value.session_count
+               last_session = footprint.value.newest
+       except Exception:  # noqa: BLE001 - diagnostics never raise
+           pass
```

## Shape

```
agent_health(settings, backend=agent.backend)
        |
        +-- effective_backend = canonical_backend(backend or settings.agent_backend)
        |         |
        |         +--> [today] list_sessions(resolve_codex_home(settings), cwd)   # codex, always
        |         +--> [after] get_backend(effective_backend).read_memory_footprint(settings)
        |                          codex     -> Capability.read(rollout footprint)
        |                          claude    -> unsupported
        |                          opencode  -> unsupported
        |                          mock      -> unsupported
        v
   AgentHealth(session_count, last_session, checks, backend, backend_version)
        |
        +--> web/src/settings-view.ts  "N sessions - last <date>"
        +--> telegram/render.py        "sessions N  last <date>"
```

## Consequences and open questions

- **The codex number changes meaning.** `list_sessions(home, os.getcwd())`
  filters rollouts to the server's cwd; `read_rollout_footprint(home)` counts
  every rollout under the home. On a codex box the health count will usually go
  UP. That is the same number the `memory` panel already shows for the same
  agent, so the two surfaces stop disagreeing - which is the epic's Done Means 2
  - but it is a visible change and needs a CHANGELOG line. The alternative is a
  new cwd-scoped backend method; that is a second reader for one caller, and
  YAGNI argues against it. **Assumption taken: the footprint count is the right
  number, and cwd-scoping is dropped.**
- **Zero is still ambiguous on the health surface.** `AgentHealth` has no
  capability envelope, so an unsupported backend and a codex home with no
  rollouts both render `0 sessions`. `/api/agents/{id}/memory` already answers
  this properly with `supported: false`, and 20260801-100419 established the
  three-state operator vocabulary. Open question for planning: does health widen
  to `session_count: int | None` (and the two renderers grow an "unsupported"
  string), or does it stay a plain zero and lean on the memory panel next to it?
  Widening touches `web/src/agent-types.ts:149`, `settings-view.ts`,
  `telegram/render.py` and their tests; the plain zero is a one-file change.
- `os` may become unused in `health.py` - check before deleting the import.
- The proof wants a claude/opencode agent with a POPULATED codex home reporting
  no sessions; an empty home makes the change observationally invisible, which
  is exactly the trap 20260803-034922 is written about.
