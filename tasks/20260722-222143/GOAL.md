# Goal: Telegram frontend for Scufris - orchestrator-as-the-whole-UI

- DATE: 20260722
- UMBRELLA TASK: 20260722-222143
- LANDING SCOPE: squash-merge each task to local `master` via `sprout land`; no push (user's call). Outward-facing bot pieces (Telegram token, network transport) are gated on the Finish manual checkpoint.

## Goal

Give Scufris a Telegram face, like the old `github.com/alexjercan/scufris-bot`,
so the box is drivable from a phone without the web dashboard. Telegram is a
single chat, so there is exactly ONE session and it IS the orchestrator agent.
The Telegram frontend is a second face on the orchestrator, not a reimplemented
dashboard: the orchestrator gains MCP control tools over the app's endpoints
(create/run/steer an agent, create/list projects, host introspection) so it can
DO what the dashboard does by being talked to. Those control tools are scoped to
the orchestrator ONLY - regular agents keep getting their tools from their own
`.config` / project `.skills` and cannot create agents or projects.

This run STARTS with a spike (the direction is fuzzy): the spike decides the v1
feature cut, the control-tool set, the orchestrator-only scoping mechanism, the
keep/drop call on today's 8 MCP tools, and the Telegram transport/auth/rendering
approach. The build criteria below are refined by the spike and by the plan
checkpoint before any bot code is written.

## Done means

1. `tasks/20260722-221359/SPIKE.md` decides all five questions; the direction is
   confirmed by the user. (manual: user confirms the spike direction)
2. The chosen decomposition is seeded as tatr tasks with Steps and priorities.
   (cmd: `tatr ls -f ':tags contains telegram' --sort priority`)
3. Control MCP tools exist over the chosen endpoints, curated/bounded, and are
   available ONLY to the orchestrator (a regular agent does not receive them).
   (test: orchestrator-scoping + control-tool tests)
4. A Telegram transport maps the single chat to the orchestrator session, behind
   an auth allowlist, with the token from pydantic-settings/`.env`.
   (test: integration test against a stubbed Telegram API + stubbed backend)
5. An `examples/` script boots the bot end to end against stubs. (cmd: run it)
6. The full QA gate is green. (cmd: `nix flake check`)

Overall: I can talk to the box from Telegram and, through the orchestrator, see
host stats and create/inspect an agent on a project - with the control tools
withheld from ordinary agents.

## Tasks

Updated as tasks land (one line per land, like a spike's Fix record).

- [ ] 20260722-221359 (p0, scufris) Spike: Telegram frontend - decide scope, tools, scoping, transport
  (plan phase breaks the seeded tasks into steps and appends them here)

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.

- (pending) 20260722-221359: confirm the spike direction (feature cut, tool set, scoping mechanism, transport)
