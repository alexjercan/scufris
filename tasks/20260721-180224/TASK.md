# B5e: retire the codex-exec runner + fix the settings-view backend picker

- STATUS: OPEN
- PRIORITY: 30
- TAGS: agents,backend,frontend

## Goal

Final cleanup of the B5 unification. Once the orchestrator no longer uses the
`Agent` protocol (B5b) and the chat converges (B5d), the codex-`exec` runners
(`_run_codex_exec`/`_stream_codex_exec`) have no remaining users - retire them.
Reconcile the last stale backend vocabulary in the frontend (the B1 carried-in
note): `settings-view.ts` still shows raw `app_server`/`mock` ids for the
process chat agent's `agent_backend` field.

## Coarse steps (/plan expands)

- [ ] Confirm nothing references `_run_codex_exec`/`_stream_codex_exec`/
      `CodexRunner`/`StreamRunner` (grep) once B5b landed; remove them + their
      direct tests (the app-server runner + backends are the surviving path).
- [ ] Remove any remaining `agent_backend` legacy plumbing that only existed for
      the Agent path (e.g. the exec coercion validator can stay as a load guard,
      or be dropped if agent_backend itself is retired in favor of the
      orchestrator record's backend).
- [ ] Frontend `settings-view.ts`: replace the raw BACKENDS `["app_server",
      "mock"]` picker with the friendly Codex/Claude surface (or fold it into
      the orchestrator's own settings modal, since the orchestrator IS the
      process chat agent now).
- [ ] Doc-surface sweep: remove any remaining `codex exec` / Agent-protocol
      mentions that no longer describe the code.

## Definition of Done

- No `_run_codex_exec`/`_stream_codex_exec` remain
  (cmd: `grep -rn "_run_codex_exec\|_stream_codex_exec" scufris/`).
- The settings backend picker shows Codex/Claude, not `app_server`/`mock`
  (cmd: `grep -n "app_server" web/src/settings-view.ts` -> gone).
- Full check suite green.

## Notes
- Depends on: B5b (20260721-180208), B5d (20260721-180222).
- Low-complexity but must come LAST - it deletes what B5b/B5d made unused.
