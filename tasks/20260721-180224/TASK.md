# B5e: retire the codex-exec runner + fix the settings-view backend picker

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: agents, backend, frontend

## Goal

Final cleanup of the B5 unification. Once the orchestrator no longer uses the
`Agent` protocol (B5b) and the chat converges (B5d), the codex-`exec` runners
(`_run_codex_exec`/`_stream_codex_exec`) have no remaining users - retire them.
Reconcile the last stale backend vocabulary in the frontend (the B1 carried-in
note): `settings-view.ts` still shows raw `app_server`/`mock` ids for the
process chat agent's `agent_backend` field.

## Coarse steps (/plan expands)

- [x] Confirm nothing references `_run_codex_exec`/`_stream_codex_exec`/
      `CodexRunner`/`StreamRunner` (grep) once B5b landed; remove them + their
      direct tests (the app-server runner + backends are the surviving path).
      Also removed the now-dead exec-only helpers (`TurnOutcome`, `_parse_events`,
      `_tool_call_from`, `_usage_from`, `_exec_args`); renamed `_exec_mode` ->
      `_turn_mode` (the app-server runner is its sole caller). Ported the
      exec-only tests: kept the shared-helper units (`_mcp_overrides`/`_steer`),
      re-pointed missing-binary + cwd coverage onto `_stream_app_server`, added an
      app-server image-attach test.
- [x] Remove any remaining `agent_backend` legacy plumbing. Done by WIDENING (not
      dropping): `agent_backend` is now the canonical `codex|claude|mock` (default
      codex), so the LANDING ORCHESTRATOR can run on Claude; the legacy
      `app_server|exec -> codex` coercion stays as a load guard for env/state while
      the API input model is strict. Health probes the SELECTED backend (codex vs
      claude), not always codex.
- [x] Frontend `settings-view.ts`: replaced the raw
      `["app_server","exec","mock"]` picker with a server-authoritative friendly
      picker driven by `/api/agents/backends` (Codex/Claude, + Mock only when the
      dev flag is on, per the user's note). `selectRow` now takes `{value,label}`
      options and patches `agent_backend` by id.
- [x] Doc-surface sweep: updated stale `codex exec` / non-streaming-exec mentions
      in `agent.py`/`config.py` to the app-server-only reality. No stale mentions
      remain in README/AGENTS.md/docs.

## Definition of Done

- No `_run_codex_exec`/`_stream_codex_exec` remain
  (cmd: `grep -rn "_run_codex_exec\|_stream_codex_exec" scufris/`). [MET - empty]
- The settings backend picker shows Codex/Claude, not `app_server`/`mock`
  (cmd: `grep -n "app_server" web/src/settings-view.ts` -> gone). [MET - empty]
- Full check suite green. [MET - backend pytest + web `npm run ci` both green]
- manual: on the settings page the backend picker shows Codex/Claude (+ Mock only
  in dev), and switching the orchestrator to Claude runs the landing chat on it.

## Notes
- Depends on: B5b (20260721-180208), B5d (20260721-180222).
- Scope grew from "low-complexity cleanup" to widening `agent_backend` to
  codex|claude|mock (user chose the fuller fix so the orchestrator can be Claude),
  which touched config schema + health alongside the retirement + picker.
