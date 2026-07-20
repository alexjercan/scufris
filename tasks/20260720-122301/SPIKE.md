# Spike: agent page future - tool-chip persistence, den/tool integrations, commands, attachments, projects, nixos

- DATE: 20260720-122301
- STATUS: RECOMMENDED
- TAGS: spike, agent, ui

## Question

The round-2 UX arc made the agent chat pleasant to use. Now: where should it go
to become genuinely MORE USEFUL for its owner (a technical homelab operator whose
life runs through a handful of local tools)? Concretely: (a) fix the tool-chip
regression the user hit (chips vanish when you re-open a session); (b) figure out
whether/how to add `/commands` and skills like Claude Code; (c) an honest review
of the new settings page; (d) ideate high-value additions - interactive skills,
file attachments, file paths, previews, "projects", and integrations with the
user's own tools (`sesh`, `today`, `daily` -> `the-den`); and (e) how this all
plugs into the user's NixOS/dotfiles. Output: a prioritized set of tatr tasks.

## Context

The agent page (`web/src/index.html`, `agent-view.ts`, `markdown.ts`) is a
two-pane chat driving `codex exec` / `codex app-server` with a scufris MCP server
(`scufris/mcp_server.py`: `host_stats`, `disk_usage`, `list_processes`,
`tatr_ls/show/new`). Sessions are codex rollouts on disk; `sessions.py` harvests
them (list, context, usage, transcript). A new read-only Settings page
(`settings-view.ts` + `GET /api/agent/config`) shows config + tool cards.

Grounding gathered this spike (all verified on the host, codex 0.142.2):

- **Tool-chip bug is real and the data is on disk.** `read_transcript` returns
  `TranscriptMessage{role,text,ts}` only; `switchSession` maps to `{role,text,ts}`
  with no `reply`, and `messageMeta` renders chips from `reply.tool_calls`, which
  is set ONLY on a live `onDone`. The rollout DOES record each call as an
  `event_msg` `mcp_tool_call_end` with `{call_id, invocation:{server,tool},
  duration, result}`, ordered between the user turn and the following
  `agent_message`. So the chips are recoverable by parsing the rollout.
- **codex natively attaches images:** `codex exec -i/--image <FILE>...`. So image
  attachments are a supported path; app-server `turn/start` input is an array that
  can carry non-text items.
- **codex skills exist but are unused:** `~/.codex/skills/` exists and is EMPTY;
  codex's base instructions have a "# Using skills" section (SKILL.md discovery).
  So skills are a real codex mechanism we could populate later.
- **codex has NO slash-command for exec/app-server:** custom `/prompts` are a TUI
  feature; `~/.codex/prompts` does not exist. So a web `/command` palette is OURS
  to build client-side (intercept in the composer, expand to a prompt or call an
  action). This is the honest answer to "does codex support /commands".
- **the-den + tools (the real prize):** `the-den` is a markdown journal at
  `/home/alex/personal/the-den` (Daily/, Notes/, Templates/, tasks/). `daily
  --json` returns `{date,file,title,habits,tasks,tomorrow,macros,weight}` and
  mutates the day non-interactively (`--toggle-habit`, `--task-entry/-remove`,
  `--toggle-task`, `--weight-entry`, `--macros-entry`, `--notes-entry`,
  `--task-tomorrow-*`, `--offset`, `--json`). `today -p` prints today's path.
  `sesh` (tmux-sessionizer) lists/creates projects under `~/personal`, `~/work`.
  `daily` being a clean JSON in/out CLI makes it a near-trivial MCP tool wrap.
- **NixOS/dotfiles (important nuance):** scufris is ALREADY integrated in
  `nix.dotfiles` (`home/alex/default.nix`) as `systemd.user.services.scufris` -
  but via the flake input `github:alexjercan/scufris-bot`, NOT the local
  `/home/alex/personal/scufris`, which exports only `packages`/`devShells` (no
  `homeManagerModules`/`nixosModules`). So the local repo we develop and the
  deployed flake have diverged; the packaged derivation must also include the
  built `web/dist` or the UI 404s.

## Options considered (candidate directions, then converged)

### The conversation loop / provenance

- **Persist tool calls across reload (the bug).** Harvest `mcp_tool_call_end` in
  `read_transcript`, correlate to the following `agent_message`, carry via
  `TranscriptMessage` -> frontend rebuilds `reply.tool_calls`. Also lets a chip be
  clicked later to show args/result/duration (the rollout has all of it). Small,
  high value, data already exists. **Clear win.**

### Making the agent DO more (the real value)

- **the-den MCP tools (today/daily).** Wrap `daily`/`today` as scufris MCP tools so
  "log 80kg and check off gym", "what are today's tasks", "add a note" work in
  chat. Highest personal value; `daily --json` makes it cheap and safe. Unknown:
  gating (den path from config; already `user.journal.den_path` in the dotfiles).
- **sesh / projects.** List/create projects; a "projects" concept (group sessions
  by cwd - codex records it). Valuable but the data model is undefined -> smells
  like its own sub-spike, not a ready task.
- **Codex SKILL.md skills.** Populate `~/.codex/skills` (or per-invocation) so the
  agent gains packaged capabilities (a "journaling" skill, a "homelab" skill).
  More powerful than MCP tools for multi-step know-how, but more experimental and
  overlaps MCP tools for the near term. Defer; note as a follow-up.

### Making the chat richer (interaction)

- **Slash-commands palette.** Client-side `/command` in the composer: `/new`,
  `/settings`, `/today`, `/tasks`, `/export`, `/help` - some expand to a prompt,
  some call an API directly, with an autocomplete menu. This is the feasible
  version of "commands like Claude Code". Low-medium effort, high discoverability.
- **File attachments + paths + previews.** Attach images (codex `-i`), reference
  file paths (agent reads via shell/MCP), and render richer previews: inline
  images, a real diff view for ```diff fences, file-path chips. A cluster; the
  cheap wins (image attach + diff rendering) can lead.

### The operator surface

- **Settings -> operator console.** From the review: show env-var NAMES beside
  values; live health (codex login? MCP server reachable? web/dist present?);
  version info; richer tool cards (server + arg schema, a "try it" runner); session
  count. Turns a static status page into a real debug console. Editable settings
  stay deferred (own spike).

### Deployment

- **Reconcile scufris <-> nix.dotfiles.** Export `homeManagerModules`/`nixosModules`
  from the LOCAL flake and point the dotfiles at `path:/home/alex/personal/scufris`
  (single source of truth), ensuring `web/dist` is in the derivation - OR keep
  GitHub canonical and document the push loop. Needs a user decision on which repo
  is canonical.

### Do nothing

The chat is already good; nothing here is load-bearing for basic use. But the
user explicitly wants the agent to touch their real tools (den/sesh) and hit a
concrete regression (chips), so "do nothing" only fits the fuzziest items
(projects, skills), which are deferred to their own follow-ups anyway.

## Recommendation

Ship in value order. Fix the regression, then make the agent act on the user's own
tools, then make the chat richer, then upgrade the operator surface; treat the two
fuzzy items (projects, dotfiles reconciliation) as coarse tasks that each start
with a decision/sub-spike.

1. **Persist tool calls (+ per-turn usage) across reload (P50, bug).** The
   concrete regression; data is on disk.
2. **the-den journal MCP tools (P40).** The highest-value "make it useful" item and
   the cheapest to build (clean JSON CLI). This is what turns it into *your* Jarvis.
3. **Slash-commands palette in the composer (P40).** The "/commands" ask, built the
   only way codex allows (client-side); high discoverability, mostly existing APIs.
4. **File attachments + path refs + previews (P30).** Lead with image attach (codex
   `-i`) + diff rendering; the rest can split at /plan.
5. **Settings -> operator console (P30).** Env-var names + health + richer tool
   cards; makes "why won't it do X" answerable.
6. **Projects: sesh + per-project workspaces (P20, needs sub-spike).** Coarse;
   define the "project" model before building.
7. **Reconcile scufris <-> nix.dotfiles (P20, needs user decision).** Single source
   of truth + module export + web assets in the derivation.

Do 1-3 first (regression + the two features that most change what the agent is
FOR). 4-5 are quality upgrades. 6-7 each open with a decision.

## Open questions

- **Which repo is canonical** - RESOLVED (user, 20260720): the LOCAL
  `/home/alex/personal/scufris` is canonical and will REPLACE `scufris-bot`. Not on
  a remote yet; future deploy from `/scufris` (no `-bot`). Task 7 updated: export
  the modules from the local flake now; flip the dotfiles input over last (use
  `path:` in the interim). No longer blocked.
- **the-den foundation** - RE-SCOPED (user, 20260720): before/under task 2, the
  user wants a NEW unified, agentic-friendly journal CLI built as its OWN
  `~/personal` project in Python (merging the two `today`/`daily` bash scripts into
  one command, replacing them in the nix config, targeting the-den). That CLI is
  SEPARATE work (its own repo + its own spike); scufris's MCP tools (task 2) wrap
  it. Task 2 updated with the prerequisite + sequencing. The MCP tools still read a
  `settings.den_path` and no-op safely when unset.
- **Projects data model** (task 6) - DEFERRED (user, 20260720): "look into it later,
  a spike is a good idea." Task 6 now explicitly opens with its own dedicated
  `/spike`; not a ready feature task.
- **Skills vs MCP tools vs slash-commands** - three overlapping "capability"
  mechanisms. Near-term recommendation: MCP tools for what the agent can DO,
  client slash-commands for quick triggers; revisit codex SKILL.md once there is a
  multi-step capability worth packaging.
- **App-server image input shape** - `codex exec -i` is confirmed; the app-server
  `turn/start` image item shape needs a probe before task 4 commits to a backend.

## Next steps

Direction-level tasks seeded (for `/plan`):

- tatr 20260720-122513 (P50, bug): persist tool-call chips (+ usage) across reload
- tatr 20260720-122514 (P40): the-den journal MCP tools (today/daily)
- tatr 20260720-122515 (P40): slash-command palette in the composer
- tatr 20260720-122516 (P30): file attachments, path refs, rich previews
- tatr 20260720-122517 (P30): settings page -> operator console
- tatr 20260720-122518 (P20, sub-spike first): projects + sesh integration
- tatr 20260720-122519 (P20, user decision first): reconcile scufris <-> nix.dotfiles

Suggested order: 122513, 122514, 122515 first (the regression + the two features
that most change what the agent is FOR), then 122516 + 122517 (quality upgrades),
then 122518 and 122519 (each opens with a decision/sub-spike).

## Fix record

(Appended by each implementing task as it lands.)
