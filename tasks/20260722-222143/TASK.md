# Goal: Telegram frontend for Scufris - orchestrator-as-the-whole-UI

- STATUS: CLOSED
- PRIORITY: 0
- TAGS: goal

## Goal: Telegram frontend for Scufris - orchestrator-as-the-whole-UI

- DATE: 20260722
- UMBRELLA TASK: 20260722-222143
- LANDING SCOPE: squash-merge each task to local `master` via `sprout land`; no push (user's call). Outward-facing bot pieces (Telegram token, network transport) are gated on the Finish manual checkpoint.
- THIS FLOW RUN: T1-T3 only (the orchestrator-only MCP tool model). T4-T5 (the Telegram transport + rendering) are DEFERRED to a later /flow run; the umbrella stays OPEN until they land. Decided at the post-spike checkpoint 20260722.

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

- [x] 20260722-221359 (p0, scufris) Spike: Telegram frontend - decide scope, tools, scoping, transport
      RECOMMENDED; SPIKE.md written; seeded T1-T5 below.
- [x] 20260722-222717 (p36, scufris) T1 - orchestrator-only scufris MCP scoping [foundation]
      landed 6f712bf; 1 review round (APPROVE, out-of-context); scufris MCP + steering now orchestrator-gated.
- [x] 20260722-222722 (p35, scufris) T2 - control MCP tools over local HTTP API [dep: T1]
      landed 2b90f5a; 1 review round (APPROVE, out-of-context; 2 NITs fixed); 5 orchestrator control tools + id guard.
- [x] 20260722-222729 (p34, scufris) T3 - prune MCP surface (drop tatr_*; host tools orchestrator-scoped) [dep: T1]
      landed 9ebcbe6; 2 review rounds (R1 1 MINOR stale .env.example, R2 APPROVE); tatr_* MCP tools gone, steering trimmed.
- [x] 20260722-222734 (p33, scufris) T4 - Telegram transport (httpx long-poll, auth, session) [dep: T1]
      landed 936b6f7; in-process long-poll bot transport maps the single allowed chat to the orchestrator session.
- [x] 20260722-222739 (p32, scufris) T5 - reply rendering + e2e example [dep: T4]
      landed 729d04c; reply rendering (tool footer + typing action) + examples/telegram_bot.py e2e against stubs.
- [x] 20260722-232723 (p36, scufris) CRUD control tools (get/update/delete project; update/delete agent) [extra, user-requested]
      landed 776ff4a; 1 review round (APPROVE, no findings); completes CRUD over projects+agents, regular agents only.

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.

- (ACCEPTED 2026-07-22) 20260722-221359: spike direction confirmed at the post-spike
  checkpoint - "Foundation only (T1-T3)" + "proceed as recommended".
- (ACCEPTED 2026-07-26) goal-level manual "talk to the box from Telegram, see host
  stats, create an agent" - the bot (T4/T5) landed and later gained live turn
  streaming (20260726-201901) and markdown reply rendering (20260726-205809). The
  outward-facing live-bot exercise is the user's own confirmation; user directed
  the umbrella closed on 2026-07-26.

## Run status (2026-07-22): MILESTONE - T1-T3 delivered, umbrella stays OPEN

This /flow run built the orchestrator-only MCP foundation (done-definition items
1-3) and stopped there by the user's scoping decision. The umbrella is NOT closed:

- DONE (this run): (1) spike direction confirmed; (2) T1-T5 seeded + T1-T3 planned
  and landed; (3) control MCP tools exist and are orchestrator-only (test-backed).
- DEFERRED to a later /flow run (done-definition items 4-6): T4 Telegram transport
  (`20260722-222734`) and T5 reply rendering + e2e example (`20260722-222739`). Both
  remain OPEN in the backlog with Steps to be planned when picked up.
- OPEN QUESTION carried to T4/T5 (SPIKE Q4): dropping `tatr_new` means the
  orchestrator needs a write-capable permission mode to create tatr tasks via Bash -
  confirm the orchestrator's default mode when wiring the bot.

Reopen with `/flow 20260722-222734` (or `/work` it) to build the Telegram bot.

## Run status (2026-07-26): CLOSED - all done-definition items met

The deferred bot pieces landed since the 2026-07-22 milestone, so every
done-definition item is now satisfied and the umbrella is CLOSED:

- (1) spike direction confirmed (ACCEPTED 2026-07-22).
- (2) decomposition seeded as tatr tasks (T1-T5 + spike + CRUD extra).
- (3) control MCP tools exist and are orchestrator-only, test-backed (T1-T3).
- (4) Telegram transport maps the single allowed chat to the orchestrator session
  behind an auth allowlist, token from pydantic-settings/`.env` (T4, 936b6f7).
- (5) `examples/telegram_bot.py` boots the bot end to end against stubs (T5, 729d04c).
- (6) full QA gate green (`nix flake check`).

All seven child tasks are CLOSED and landed. Follow-on polish shipped after the
original T1-T5 seed and is NOT part of this umbrella's done-definition (recorded
here only as trail): live turn streaming (20260726-201901) and markdown->MarkdownV2
reply rendering (20260726-205809). The goal-level manual live-bot check is accepted
by the user's close directive (see Manual acceptance).
