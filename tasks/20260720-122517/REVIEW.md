# Review: settings page -> operator console

- VERDICT: APPROVE
- ROUND: 1

## Summary

The read-only settings page becomes a debuggable operator console: a Health card
(scufris/codex versions, session count, and green/amber/red rows for agent / codex
cli / codex auth / mcp tools / web assets, each with a fix hint), env-var name
chips on the config rows, and richer tool cards (source server + arg names). New
`scufris/health.py` + `GET /api/agent/health`; `AgentTool` gains `server`/`args`.
135 pytest + 95 frontend green; verified live (all-green endpoint, and it correctly
flags a missing web/dist before a build).

## What is good

- Directly answers the round-2 gap ("is codex logged in? is the MCP server up?"):
  the health card surfaces exactly the things that silently break, with the fix
  command in the hint ("run codex login", "npm run build"). Grounded on probes
  confirmed feasible in the spike.
- Best-effort by construction: every probe degrades to a check status, the whole
  function never raises, and shell-outs are timeout-guarded (3s + kill), so opening
  the page is cheap even when codex is missing/slow. The mock backend downgrades a
  missing codex to warn (it does not need it) rather than crying error.
- Scope discipline: the interactive "try it" runner was deferred to its own task
  (it is a real capability with a consent surface, not read-only) - the spike made
  that call and this flow honored it.
- Safe rendering: statuses clamp to a known dot class (no injected class, no
  invisible dot), everything escaped, health is optional so a failed health fetch
  does not blank the page. Session summary folded into the health payload (no extra
  round trip).

## Findings

- MINOR (accepted) - the env-var name map lives in the frontend (`ENV_VARS`) and
  must track `config.py`. It is small and the names are stable; a backend-sourced
  name would couple the config model to its env keys, which is worse. Noted.
- MINOR (accepted) - the health GET runs two codex subprocesses (`--version`,
  `login status`) per request; worst case ~6s under the timeout. The page is opened
  rarely, so per-request (uncached) is the right simplicity/accuracy trade for now
  (the spike flagged caching as a later option if it feels slow).
- FIXED in-review - a stray `import os` inside the session-summary block moved to
  module top.

## Verdict

APPROVE. High-value, correctly best-effort, verified live end-to-end, and scoped
cleanly (the interactive runner deferred). Findings are accepted trade-offs.
