# Goal: interactive operator settings console (writable config, richer panels, tool editing)

- DATE: 20260720
- UMBRELLA TASK: 20260720-183719
- LANDING SCOPE: squash-merge each task to `master` (local default), do NOT
  push (user's call). Standard flow landing.

## Goal

Turn the read-only Settings page into an interactive operator console for the
single homelab operator. Today `GET /api/agent/config` renders effective config
from env/`.env` with no way to change it. This goal makes settings
LIVE-WRITABLE and PERSISTED (a scufris state file layered over env defaults,
gated + confirmed), and enriches the page with informative panels about
sessions, usage/quota, context, memory, and the account. It adds config
PROFILES (named per-operator config sets - not multi-user auth; scufris stays
single-operator/local). It makes the tools surface editable: enable/disable
existing scufris tools and add/remove MCP servers from the page. It SPIKES (does
not build) the fuzziest ask - "multiple agents with workflows" - into a
direction + seeded tasks. Wiring settings into the projects concept
(tasks/20260720-182842 spike) is explicitly a FOLLOW-UP, not in this goal.

Scope decisions (user, 20260720):
- Writable config = live-writable, PERSISTED (state file overrides env), gated.
- Account/users = single operator, richer view; "per user configs" = named
  config profiles, NOT auth.
- Multi-agent/workflows = SPIKE only, don't build in this goal.
- Tools editing (enable/disable + add) = in scope now; projects integration =
  later.

## Done means

1. Config is live-writable and persisted: toggling a setting on the page
   (agent enabled, tools enabled, model, a tool's enabled state, an MCP server)
   survives a server restart, layered over env defaults, and env still seeds
   first-boot. (test: backend store round-trip test; manual: toggle in the UI,
   restart the server, confirm it stuck)
2. The write path is safe: mutations are gated behind a setting and require a
   confirm; a read-only-mode server refuses writes with a clear error.
   (test: endpoint returns 403/clear error when the gate is off)
3. The page has richer read-only panels: sessions summary, usage/quota,
   context, a "memory" panel (persistent agent state - definition pinned in
   planning), and an account panel (codex/ChatGPT account + quota).
   (manual: each panel shows real data on the running app)
4. Config profiles: the operator can save/name/switch named config profiles;
   the active profile drives the effective config. (test: profile
   save+switch round-trip; manual: switch a profile in the UI)
5. Tools are editable from the page: enable/disable an existing scufris tool
   and add/remove an MCP server, persisted per (1). (test: tool-enable state
   affects the tools list / MCP registration; manual: disable a tool, confirm
   the agent no longer offers it)
6. "Multiple agents with workflows" is spiked: a SPIKE.md with a recommended
   direction and seeded tatr tasks exists; nothing built. (cmd: the spike
   task and its SPIKE.md exist and lint clean)

Overall: the full check suite passes on master (cmd: `nix develop --command
bash -c "ruff check . && mypy . && pytest -q"` plus `npm run ci` in web/), and
`tatr check --ledger LESSONS.md` is clean.

## Tasks

Updated as tasks land (one line per land).

(planned in the /plan phase)

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.

(none yet)
