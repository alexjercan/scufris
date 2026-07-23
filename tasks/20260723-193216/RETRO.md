# Retro: role-scoped per-agent tools endpoint + settings panel

- TASK: 20260723-193216
- DATE: 20260723
- OUTCOME: landed, 1 review round (APPROVE, all findings non-blocking)

## What we set out to do

Stop a codex sub-agent's UI showing the orchestrator's full 18-tool surface, and
give each agent's settings page a transparent view of its OWN tools.

## What went well

- Two out-of-context maps (backend machinery + frontend render path) BEFORE
  writing code. The frontend map was decisive: it proved no per-agent tools panel
  existed at all, so the reported "18 tools" was the orchestrator-scoped count -
  reframing the task from "fix a wrong number" to "add the correct scoped panel +
  fix the latent unscoped endpoint". Recorded that honestly in NOTES.md rather than
  faking a repro I couldn't pin in a headless env.
- Single source of truth for role scoping: extracted a PURE `role_tool_names` and
  had both `apply_role` (mutating, spawned server) and the new read-only endpoint
  use it, so the UI can never drift from what the spawned server actually
  advertises. The reviewer verified the set-algebra refactor is byte-identical.
- Got the SEMANTICS right, which was the subtle part: `/api/agent/tools` is the
  orchestrator's IN-PROCESS operator console (genuinely runs all ~18 locally), so
  it stays full; `/api/agents/{id}/tools` answers a different question (what THIS
  agent's turn advertises) and is role+backend scoped. Kept them as two endpoints
  instead of forcing one to lie.
- Backend-truthful, not just role-truthful: a mock/claude sub-agent returns `[]`
  (no scufris MCP wiring), named the gate `_agent_has_scufris_mcp` so the claude
  parity task (20260723-193218) flips ONE function and the panel updates itself.
- Read-only panel (no toggles/"try it") - correct because a sub-agent's tool set is
  fixed by role+backend, and "try it" runs in-process as the dashboard, not as the
  sub-agent. Reused the existing `panel()` idiom for the empty case.

## What went wrong / friction

- The worktree had no `web/node_modules`; `npm run test` failed "vitest: command
  not found" until `npm ci`. A fresh sprout needs `npm ci` before the web suite -
  worth remembering (the python venv is provided by the flake, the node deps are
  not).
- Prettier flagged the new test file after I wrote it; caught by `format:check`,
  fixed with `npm run format`. The format-before-the-check-gate lesson applies to
  the frontend too (run `npm run format` before `npm run ci`).

## Lessons (candidates for the ledger)

- `sprout-worktree-needs-npm-ci-for-the-web-suite`: a fresh sprout worktree has no
  `web/node_modules` (unlike the flake-provided python venv), so `npm run test` /
  `npm run ci` die "vitest: command not found" until you run `npm ci` in `web/`
  first. Do it once per worktree before touching the frontend.
- `two-endpoints-when-one-answer-would-lie`: when a single endpoint is asked to
  serve two genuinely different questions (here: "what can the operator run
  in-process" vs "what does THIS agent's turn advertise"), split it rather than
  role-scope the shared one - the operator console legitimately runs all tools, so
  scoping it would be the wrong fix. The bug was a MISSING scoped endpoint, not a
  wrong shared one.

## Deferred to Finish

- manual: open a codex sub-agent's settings page in the running app and confirm it
  shows its one tool (`request_input`), not the orchestrator's eighteen; open the
  orchestrator settings and confirm its console is unchanged. (Batched.)
