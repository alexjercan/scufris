# Retro: settings page -> operator console

- DATE: 20260720
- VERDICT: shipped

## What went well

- Spiking the scope first (tasks/20260720-134459) paid off: probing every signal
  live BEFORE committing (codex login status/version, importlib metadata version,
  tool inputSchema.properties, web_dist) meant the build had zero unknowns, and the
  one interactive item ("try it" runner) got deferred to its own task instead of
  bloating this flow.
- Best-effort health was designed in from the start (timeout-guarded shell-outs,
  every probe -> a check status, never raises), so the diagnostics page can never
  itself be the thing that breaks - exactly right for a "why won't it work?" page.
- The fake-`codex_bin` trick made the health tests deterministic and fast:
  `create_subprocess_exec` on a nonexistent path raises immediately, so no real
  codex runs, yet the MCP/web/agent checks stay real.

## What went wrong / friction

- A broken shell probe (nested quotes in an inline `python -c`) both failed and
  left a junk file in the worktree; caught it as an untracked entry before it could
  be committed and removed it. Lesson reinforced: run multi-line python from a
  file, not a quote-nested `-c`.
- The AgentTool type gained two fields, which vitest (esbuild, lenient) accepted
  but the webpack ts-loader (strict tsc) rejected in an unrelated test's `tool()`
  helper. `npm run ci` caught it; fixed the helper.

## Lesson

- `type-change-fails-strict-tsc-not-vitest` - adding required fields to a shared
  interface passes vitest (esbuild transpiles, does not type-check) but fails the
  webpack `ts-loader` build. Always run the full `npm run ci` (which includes the
  webpack build), not just vitest, after a shared-type change - the build is the
  real type gate. (Frontend; watch for recurrence before promoting.)

## Follow-ups

- Deferred: the interactive "try it" tool runner (task 20260720-134545).
- Possible later: cache the codex probes briefly if the health GET ever feels slow
  (two subprocesses per request today).
- Round-3 remaining: 122516 (attachments/previews), 122514 (den tools - awaiting
  the unified-CLI decision), 122518 (projects sub-spike), 122519 (nixos reconcile).
