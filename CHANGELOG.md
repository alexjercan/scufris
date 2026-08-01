# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING: every flake output is now namespaced with a `scufris` prefix**, so
  pinning this flake next to others cannot collide on a generic name. Rename map:

  | old | new |
  |-----|-----|
  | `packages.web` | `packages.scufris-web` |
  | `packages.vm-test` | `packages.scufris-vm-test` |
  | `packages.hostd-vm-test` | `packages.scufris-hostd-vm-test` |
  | `nixosModules.hostd` | `nixosModules.scufris-hostd` |
  | `nixosModules.default` | also available as `nixosModules.scufris` |
  | `homeManagerModules.default` | also available as `homeManagerModules.scufris` |

  The old names are GONE, not aliased: a config importing
  `nixosModules.hostd` or building `.#web` must be updated. The conventional
  `default` attributes (`packages.default`, `apps.default`,
  `nixosModules.default`, `homeManagerModules.default`) are kept, so `nix build .`,
  `nix run .` and a plain `.default` import still work. `packages.scufris`,
  `apps.scufris` and the option paths (`services.scufris`,
  `programs.scufris`, `services.scufris-hostd`) are unchanged; `checks.*` keep
  their short names, since those are only labels for `nix flake check`.

- **Documentation restructured into per-component READMEs.** The root
  `README.md` is now setup only: what Scufris is, how to run and deploy it, every
  environment variable with its default and what it decides, and how to enable
  each optional feature one at a time. Everything else moved to where the code
  is: `scufris/README.md` (the architecture - processes, trust boundaries, the
  approval contract, which agent holds which tools, the HTTP surface, the module
  map), `scufris/host/README.md` (the read-only inspection package),
  `scufris/hostd/README.md` (the root helper: nix options, the socket language,
  every verb and its arguments, the refusals, the audit log) and `web/README.md`
  (pages, build, the frontend gate). `AGENTS.md` now names that set as the live
  doc surface, and task records stay append-only history.

### Fixed

- Every `nix` invocation now carries `--extra-experimental-features
  "nix-command flakes"` explicitly. Whether those features are enabled is the
  operator's `nix.conf`, and without them `nix path-info` and `nix store gc`
  fail outright - measured in the hostd VM test, where a default configuration
  made a store validation fail for a reason that had nothing to do with the
  store.
- The thermal card and `host_thermal` counted core throttle events once per
  LOGICAL cpu, so hyperthread siblings - which share a physical core and report
  the same counter - doubled every figure (162 reported where the truth was 81).
  Counters are now deduplicated per physical core via `topology/core_id` plus
  `physical_package_id`, reduced with max rather than last-write-wins, and both
  surfaces say which unit each figure is counted in ("81 per-core events on 3 of
  16 physical cores, and 82 whole-package events").

### Added

- **A state database, created and kept up to date at startup.** Scufris now
  writes `scufris.db` (mode 0600, with its `-wal`/`-shm` siblings) into the state
  directory and brings its schema to head before anything reads it, on both the
  dashboard and the orchestrator MCP subprocess. Nothing has moved onto it yet:
  every store still reads and writes the JSON files it always did, and this
  release changes nothing an operator sees. It exists so the store migrations
  that follow add a schema revision rather than invent a migration mechanism.
  Before a future release changes the schema, the database is copied to
  `scufris.db.pre-<revision>.bak` first.

- **Scheduled host checks and a proactive digest.** Scufris now watches the machine
  without being asked. `watch` (every 15 minutes) messages only when a check is in a
  warn/crit state or something recovered; `daily` (08:00) always sends, even when it
  is one line - that line is the heartbeat that makes `watch`'s silence mean
  something. The checks are code with explicit thresholds (disk, failed units in both
  scopes, temperatures and the CPU's throttle counters, the store's dead paths, flake
  pin age, and Scufris's own health), a check that cannot read something says so
  rather than passing, and one that raises or hangs becomes a named failure inside the
  digest instead of a missing digest.

  A breach may PROPOSE a store collection into the ordinary approval queue - never
  apply it, never anything but the disposable-cleanup verbs, and off until switched
  on. Schedules, thresholds, the mute window and run-now are all editable at runtime
  and survive a restart; a missed window is recorded rather than replayed, so an app
  that was down does not deliver a backlog. The digests are readable on `/host/`,
  which is where "did it fire" is answered when the answer was silence.

- **Host approvals over Telegram.** A pending host action now reaches the operator
  where they already are: the allowlisted chats get the proposal with its risk
  class, every command, the preview and the undo line - the SAME text the dashboard
  and the requesting agent see - plus inline Approve/Deny buttons. Deny asks why and
  the reason reaches the agent; a one-way action's first tap only arms it, so
  approving something irrecoverable takes a second, differently-worded tap.
  `/approvals` lists the queue and `/deny <id> <reason>` is the typed form.

  It is not a second decision path: the bot hands the app a CHAT ID and the app
  derives `operator:telegram:<chat_id>`, re-checks the allowlist and calls the one
  approval service - so the acknowledgement rule, the expiry, the drift check and
  the cross-surface race all have a single implementation. Whichever surface decides
  first wins; the chat message is edited to say who decided, and a stale button is
  refused rather than re-run.

- **The host approval queue, in the dashboard.** A new `/host/` page is where a
  proposal is decided: every command it would run in order, the preview as the
  helper wrote it, who asked, the expiry counting down, and the undo sentence -
  with the risk class visually distinct, so a service restart and a system switch
  do not read alike. A one-way action has NO ordinary approve control; the only one
  that can approve it requires typing the action's name, which is the same rule the
  service enforces. The page also shows the recent decisions (live apply output, a
  stop control, and an undo offered exactly where the record says one exists) and
  the root helper's own audit log. It works at phone width, and nothing
  host-supplied reaches innerHTML - a systemd unit is named by a file, so every
  string on the page is attacker-influenceable.

- **A host agent, and one decision path for approving what it proposes.** The
  machine now has an agent of its own (`/agents/host`), bound to the box rather
  than to a project, and it is the only audience carrying the mutating host tools
  - the orchestrator keeps the read-only ones and delegates a change to it. A
  proposal leaves that agent `blocked`: visible to the orchestrator, refused if
  the orchestrator tries to answer it, and resumed by the OPERATOR's decision
  with the applied result or the denial and its reason, so a denied agent adapts
  instead of proposing the same thing again.

  Approving is now one service (`host_approvals.HostApprovalService`) that the
  web routes call and the Telegram surface will call, so "the same enforcement on
  both" is one implementation rather than two descriptions of one. It also
  computes what confirming an action REQUIRES: an action that destroys something
  irrecoverably (a store collection) is refused unless the approval carries an
  explicit acknowledgement, while a reversible one - including a service restart,
  whose "no undo" is normal rather than alarming - keeps the ordinary
  confirmation and shows its undo sentence as written.

  The approval queue also survives a restart. The app's registry is deliberately
  in-memory (the root helper owns every proposal), so the helper gained a
  read-only `list_pending` verb and the app rebuilds the queue from it at startup
  and while listing - which also means a proposal made by another client of the
  socket is not invisible to the operator who has to decide it.

- **NixOS configuration changes, from a reviewed commit.** Scufris can now
  activate a NixOS configuration and roll the system back, as the third risk
  class of the host action contract. What it deliberately does NOT do is edit the
  configuration: `~/personal/nix.dotfiles` is a project, so an agent changes it
  through the ordinary project flow (a worktree, a commit, a review) and only the
  switch comes through the privileged helper.
  - `POST /api/host/config/changes` resolves a ref to a commit, builds
    `nixosConfigurations.<host>` from THAT COMMIT as the operator - never as
    root, because a configuration evaluated as root could read a host key or an
    age key into a derivation output - and proposes activating the exact store
    path it built. A build failure ends there, with its log on the record and no
    proposal: a configuration that does not build has nothing an approval could
    act on.
  - The build addresses the repository as `git+file://...?rev=`, so the tree
    comes from the commit. Uncommitted edits are structurally not in the build
    and are reported as such, no `result` symlink or lock-file write can land in
    the repository, and no worktree is created.
  - `activate` is refused on the generic propose surface and by
    `propose_host_action`. Its argument is a store path, and a caller choosing
    that path would be choosing what the machine boots while the closure diff
    described their choice faithfully.
  - The preview is `nix store diff-closures` between the running system and the
    built one, with the measured trap handled: identical closures print nothing
    and exit 0, so "no closure change" is stated explicitly rather than rendered
    as an empty panel. It does NOT list the units that would restart - producing
    that list means running the proposed configuration's own
    `switch-to-configuration` as root, before anyone approved it.
  - Activation is two commands in order (point the system profile at the path,
    then switch to it), run in a transient systemd unit so a configuration that
    restarts scufris cannot kill its own activation. A failure after the first
    step is recorded as the split state it is: this boot runs the old
    configuration, the next boot would run the new one.
  - Rolling back names a generation NUMBER; the helper resolves which store path
    that is. An applied activation records the generation it replaced and offers
    exactly that rollback.
  - Two MCP tools (`propose_nixos_change`, `nixos_change_status`),
    `examples/nixos_change.py` for the whole flow against a faked build, and a
    REAL activation and rollback as root in `nix build .#hostd-vm-test`.
- **Privileged host actions, behind an operator approval.** Scufris can now
  restart a systemd unit and collect Nix garbage on the machine it runs on, and
  every such change goes through one contract: propose -> preview -> approve ->
  apply -> audit -> roll back. An agent may PROPOSE; only a human with a
  dashboard session may approve.
  - A new root helper, `scufris-hostd`, is the entire privileged surface: a
    NixOS **system** unit speaking a closed set of typed verbs over a unix
    socket. It builds every command itself, holds every proposal, and writes its
    own append-only audit log as root - so the record survives the app being the
    thing that misbehaved. There is no shell verb at any privilege under any
    approval, and no verb at all for the refused class (partitioning, users, key
    material, the firewall, scufris itself).
  - Enable it with the new `nixosModules.hostd`
    (`services.scufris-hostd.enable`), which is deliberately separate from the
    app module: gaining agency over the machine is its own diffable act, never
    something a scufris upgrade acquires. It needs `SCUFRIS_HOSTD_SECRET` in the
    same sops dotenv as the password hash, and refuses to start without it.
  - Every preview says what KIND of preview it is, and reports only what its
    own command does. Collecting the store is simulated, with the space it would
    actually free (summed per path rather than over overlapping closures);
    trimming generations names the exact generations it will delete and says
    plainly that it frees nothing by itself; a service restart cannot be
    simulated at all, so it shows current state and reverse dependencies
    labelled as blast radius rather than a prediction.
  - The two most recent system generations - the running one and its rollback
    target - are never collectable, and that floor is in the command rather
    than in the text beside it.
  - Approvals are single-use, expire after ten minutes, and are refused
    outright if the system moved between the preview and the approval.
    Cancelling mid-apply signals the whole process group and records that it
    happened, rather than leaving an unknown state.
  - `POST /api/host/actions` (propose), `/approve`, `/deny`, `/cancel`,
    `/revert` and `GET /api/host/audit`. The decision endpoints refuse the
    machine bearer token the app's own tool subprocesses hold, whatever the bind
    address - an agent approving its own proposal has nothing to do with the
    network.
  - Three MCP tools (`propose_host_action`, `host_action_status`,
    `host_action_audit`) and deliberately no approval tool. Proposing returns the
    rendered preview rather than JSON, so the model shows the operator the real
    text instead of a paraphrase.
  - Every scufris credential is now stripped from the agent CLI's environment,
    not just the machine API token: the helper's socket secret arrives through
    an `EnvironmentFile`, so without stripping the model would hold it.
  - `examples/host_action.py` drives the whole contract - including the
    one-way and cancelled cases - against a faked host, and
    `nix build .#hostd-vm-test` proves the helper on a real root unit.

- Read-only host inspection well beyond the live stats snapshot, as a new
  `scufris.host` package and twelve MCP tools: systemd units (list, one unit's
  status, failed units, system *and* user scope), bounded journal reads by unit,
  priority and time window, storage (filesystems, the Nix store, system
  generations, largest directories, garbage-collectable paths), network
  (interfaces, listening sockets, the firewall the current generation declares),
  thermals (temperatures plus the kernel's thermal-throttle counters, which show
  throttling a temperature reading cannot), and packages (what provides a binary,
  profile contents, closure diffs between generations, flake-input pin ages).
- The stats page gained failed-units, generations, Nix-store and thermal cards,
  fed by a new `GET /api/host/overview` on its own slower poll
  (`SCUFRIS_HOST_OVERVIEW_SECONDS`, default 30s) with a server-side TTL cache, so
  the subprocess-backed inspection never rides the 2s metrics poll.
- `examples/host_inspect.py` prints every host report against the real machine.
- `SCUFRIS_HOST_CONFIG_REPO` points at this host's NixOS flake, read (never
  written) to report how old its pinned inputs are.

  Every report carries its own availability: a tool that cannot read something
  says why, an empty-but-healthy result says so in words, and a capped result is
  marked truncated - so a blank is never mistaken for "checked, all fine". Two
  cases where that matters concretely: `nix store diff-closures` prints nothing
  at all for an identical closure, which is now reported as "no closure change"
  rather than an empty diff; and the firewall is labelled DECLARED, because the
  live iptables table needs root and is not readable by the service user.
- The dashboard can require an authenticated operator session. A password
  (verified against a stdlib `scrypt` hash supplied as
  `SCUFRIS_AUTH_PASSWORD_HASH`, generated by the new `scufris hash-password`)
  exchanges for an opaque session id in an `HttpOnly`/`SameSite=Lax` cookie,
  backed by a revocable server-side record with an idle timeout and an absolute
  cap. State-changing requests additionally need a per-session CSRF token and a
  same-origin `Origin`/`Referer`, failed logins are throttled per source, and a
  new login rotates the session id. Enforcement is a single deny-by-default
  middleware, so a newly added route is protected without being remembered - a
  test enumerates the app's own routes to prove it.
- `SCUFRIS_AUTH_MODE` selects the posture: `auto` (default) requires
  authentication exactly when the bind address is not loopback, `required`
  always, `disabled` never - and `disabled` is refused on a non-loopback bind.

### Changed

- **Breaking for network deployments**: binding a non-loopback address without
  `SCUFRIS_AUTH_PASSWORD_HASH` configured now refuses to start instead of
  serving the dashboard unauthenticated. Loopback development is unchanged and
  still needs no login.
- The app's own MCP tool subprocesses and the in-process operator tool console
  authenticate to the HTTP API with a per-process bearer token
  (`SCUFRIS_API_TOKEN`, minted at startup, never persisted) rather than relying
  on the API being open.

## [0.1.0] - 2026-07-29

### Added

- Releases are cut by pushing a `vX.Y.Z` tag. A guard job checks that the tag,
  `pyproject.toml` and `CHANGELOG.md` agree, that the version has real release
  notes, that task records pass `tatr check` and that no uncompiled scratch is
  left in `docs/scratch/`; the full gate then re-runs on the tagged commit
  (including the NixOS VM test); and only then is the Python distribution built,
  installed into a clean virtualenv and asked its version before the release is
  published with its changelog section as the notes.
  `scripts/check-release-ready.sh` runs the same guard locally.
- `scufris --version` prints the installed version, and the version itself now
  has a single source: `pyproject.toml`, read at runtime from the installed
  distribution's metadata through `scufris.version`. The API (`app.version` and
  the `/api/agent/health` `scufris_version` field), the dashboard settings view
  and the Telegram health card all report that one value instead of two
  near-copies with different fallbacks. `scripts/cut-changelog.sh` and
  `scripts/release-notes.sh` connect it to this changelog, so a version, its
  notes and its tag cannot disagree.
- Continuous integration: every push to master and every pull request runs the
  full QA gate on a clean checkout - `nix flake check` (ruff, mypy, pytest) plus
  `nix build .#scufris .#web`, and the frontend suite (`npm run ci`). Repository
  task-record conformance (`tatr check`) becomes a flake check too, so
  the same gate runs locally and on the runner. Third-party actions are pinned
  by commit SHA.
- The Telegram bot now supports `/cancel` to stop the current orchestrator
  message, matching the web chat stop control. Telegram turns render in a tracked
  background task so the bot can keep polling and receive `/cancel` while a turn
  is still streaming; a successful cancel stops the local render task and replies
  `Cancelled current message.`
- Chat runs can now be cancelled. While a turn streams, the composer's send
  button becomes a square STOP control (in any chat - the orchestrator landing
  and every sub-agent); hitting it truly aborts the backend turn (the run task is
  cancelled, its backend stream closed - e.g. the Claude subprocess is killed),
  not just detaches the SSE relay. The partial answer streamed so far is kept in
  the transcript, tagged `(cancelled)`, so the conversation can continue with it
  in mind. A user cancel is a new, neutral `cancelled` terminal state (distinct
  from `error`): it does not surface in `pending_agents`. The orchestrator gets a
  `cancel_agent(agent_id)` MCP tool so "cancel that sub-agent" works by
  instruction as well as manually. New endpoint `POST /api/agents/{id}/cancel`
  (works for the orchestrator via its id).
- Sub-agents now have a `report_back(summary)` MCP callback tool alongside
  `request_input`. Where `request_input` signals "I am blocked, decide for me",
  `report_back` signals "I have finished, here is the result": it records a new
  `reported` agent state, surfaces the agent in `pending_agents()` and (when
  `SCUFRIS_AUTO_WAKE` is on) wakes the orchestrator with the summary, so a
  delegated agent's completion is noticed instead of ending silently. The
  orchestrator reads the report and acknowledges it - no resume needed. Sub-agent
  steering now tells the agent to call `report_back` when its task is done.
- The Telegram bot now streams an orchestrator turn live into the chat instead of
  sending one silent reply. It renders message-per-phase: a "thinking" message
  that is edited as the orchestrator's reasoning streams, one widget message per
  tool call as it completes (wrench + tool name + a status check/cross), then the
  final answer as its own message (keeping the `tools:` footer). The thinking and
  tool widgets use emoji + HTML on the Telegram surface only. Set
  `SCUFRIS_TELEGRAM_STREAM=false` for the previous one-final-message-per-turn
  behaviour.
- The Telegram bot's final answer is now rendered from the model's
  GitHub-flavoured markdown into Telegram MarkdownV2 instead of raw text: a
  heading becomes bold, a list becomes bullets, and a table becomes an aligned
  monospace code block. The conversion is done on the bot's side (a
  `markdown_reply` wrapper over `telegramify-markdown`), not by prompting the
  model. It is guarded two ways so a reply is never dropped by formatting: the
  converter falls back to the raw body on any exception, and the send re-sends
  plain text with no parse mode if Telegram rejects the MarkdownV2 message.

### Removed

- The orchestrator settings page no longer renders the separate "System"
  section. Per-tool toggles remain in "MCP tools"; agent enablement, auth mode,
  and sandbox are surfaced elsewhere instead of duplicated as settings rows.
- The settings page no longer has the "MCP servers" operator-config card (adding
  and removing custom MCP servers) or the "Profiles" named-config switcher. Both
  are gone end to end: the `/api/agent/mcp_servers` and `/api/agent/profiles`
  endpoints, the `mcp_servers` runtime config field (`SCUFRIS_MCP_SERVERS` is no
  longer read), and the named-profile machinery in the settings store, which now
  persists a flat `{overrides: {...}}` file. An existing profile-shaped
  `settings.json` is migrated on load by keeping the active profile's overrides.
  The built-in scufris/den/agent servers and the "MCP tools" health/catalog
  section are unaffected.

### Fixed

- A delegated sub-agent no longer errors mid-turn when a backend emits a single
  line larger than 64 KiB. Both the codex `app-server` runner and the claude
  backend launched their subprocess without an explicit `limit=`, so the stdout
  reader used asyncio's 64 KiB default and raised a bare `ValueError` on any
  bigger line - which a real command-output frame (a wide `rg`, a `tatr ls` over
  hundreds of tasks, a large file dump) easily exceeds, killing an agent ~30s
  into orientation on a big repo. The reader limit is now raised to 8 MiB
  (`STREAM_READ_LIMIT`, shared by both backends), and an over-limit line is
  surfaced as a clean, diagnosable `StreamError` instead of an uncaught
  exception.
- The per-agent page (`/agents/<id>`) now reattaches to an in-flight turn on
  load. Its chat used to only rebuild the settled transcript and stream turns the
  browser itself POSTed, so a turn driven from elsewhere (the orchestrator's
  `message_agent`/`run_agent` against a sub-agent, which runs on the shared
  supervisor + event bus) never showed live, and reloading/reselecting mid-turn
  froze on the settled transcript. On mount it now subscribes to
  `GET /api/agents/<id>/events` (gated on an active run so a finished run is not
  replayed as a phantom bubble), streams the in-flight turn to completion, and
  settles the streamed reply into the log (the turn's prompt line comes from the
  mount-time transcript). Restores the SSE reattach the detail-page reshape had
  dropped.
- Agent session ids now live in a persisted, backend-tagged registry
  (`<state_dir>/sessions.json`) keyed by agent id - for ALL agents, the landing
  orchestrator included. The orchestrator's session used to be in-memory only, so
  a server restart lost its conversation and left read paths free to latch onto a
  sub-agent's codex rollout (the observed orchestrator/sub-agent transcript
  mixing). Deleting an agent removes its mapping; switching an agent's backend
  clears the stale wrong-backend id; a legacy `agents.json` `session_id` migrates
  into the registry on first load.

### Removed

- The `tatr_ls`, `tatr_show` and `tatr_new` MCP tools. The orchestrator manages
  tatr tasks with the `tatr` skill via `Bash`, so a dedicated MCP wrapper is
  redundant. The host/observe tools
  (`host_stats`, `disk_usage`, `list_processes`, `list_agents`, `agent_status`) and
  the new control tools remain; the tool-steering preamble no longer mentions tatr.

### Changed

- The single role-scoped `scufris` MCP server is now SPLIT into three
  single-audience servers, registered per turn by audience: `scufris` (the
  orchestrator's agentic tools - host/observe/project + agent control,
  pending/acknowledge), `den` (the operator's the-den journal + macros life tools,
  registered only on an orchestrator turn when a den is configured), and `agent`
  (the sub-agent callbacks `request_input` + `report_back`, the only server a
  regular sub-agent turn gets). The audience boundary is now PHYSICAL - a sub-agent
  turn simply never registers the orchestrator/den servers - so `apply_role` and
  the `SCUFRIS_AGENT_ROLE` env are retired. Both backends (codex `-c`, claude
  `--mcp-config`) register each server with its own auto-approve wildcard.

- The settings page reports MCP health per server, audience-aware, in the
  top-of-page **Health** card: one row per server with its tool count and a
  green/amber/red status (the orchestrator's `mcp: scufris` + `mcp: den`; a
  sub-agent's `mcp: agent`; a backend with no scufris MCP a single "none" row).
  Status is a live in-process probe that lists each server's tools and checks real
  readiness (the `den` server needs a configured den and the `today`/`macros`
  CLIs), so a genuinely broken or unconfigured server shows amber/red instead of a
  false green. A separate "MCP tools" section groups the tools into a collapsible
  block per server (with the operator's enable toggles + "try it" runners) purely
  for organization - no status circles there. Probe endpoints: `GET
  /api/agent/mcp`, `/api/agents/{id}/mcp` (used for the grouping) and the per-server
  rows in `/api/agent/health`, `/api/agents/{id}/health`.

- The landing orchestrator's permission mode now DEFAULTS to `auto` (edit + run
  commands) instead of `manual` (read-only) - it does write work unattended (Bash
  tatr, create projects/agents). Editable at runtime from its settings page or via
  `SCUFRIS_AGENT_PERMISSION_MODE`; project agents are unaffected (their records
  still default to manual).

- The built-in `scufris` MCP server is now ROLE-SCOPED: the landing orchestrator's
  turns get the full surface (host/observe/control tools and the tool-steering
  preamble), while regular project agents get ONLY the `request_input` callback
  (see Added) - not the full toolset they used to receive. They draw the rest of
  their tools from their own project config/skills. This threads an
  `is_orchestrator` role and the agent's own id through the backend `stream` path;
  operator-declared `mcp_servers` still apply to every agent.

### Added

- Claude backend reaches scufris MCP parity with codex: a claude-backed agent now
  gets the built-in role-scoped `scufris` server wired into every turn via
  `--mcp-config` (an inline JSON blob) + `--strict-mcp-config` + `--allowedTools
  mcp__scufris__*`, so a claude sub-agent can call `request_input` (and the
  orchestrator its control tools) unattended - the full comms loop self-heals on
  claude, not just codex. The role env (`SCUFRIS_AGENT_ROLE` / `SCUFRIS_AGENT_ID` /
  `SCUFRIS_DISABLED_TOOLS`) now comes from a backend-agnostic `scufris_mcp_server`
  core that both backends format to their own flavour (codex to `-c` overrides,
  claude to the JSON config), so they cannot drift on what a role exposes. The
  whole-server `mcp__scufris__*` allowlist is role-safe because the server enforces
  the role scope itself.
- Role-scoped per-agent tools view: `GET /api/agents/{id}/tools` returns the tools
  an agent can actually call in its turns - the orchestrator's full surface, a codex
  or claude sub-agent's `request_input` only, and NOTHING for a backend that does not
  wire the scufris MCP (opencode/mock, today) - instead of the global unscoped set the
  UI used to show. Each project agent's settings page now renders a read-only Tools
  card from it, so a sub-agent shows its real tool surface (one tool, not the
  orchestrator's eighteen). The orchestrator keeps its writable operator console
  (`/api/agent/tools`), which stays the full in-process set.
- A runnable end-to-end example ([`examples/comms_loop.py`](examples/comms_loop.py))
  and an acceptance test (`test_stalled_merge_loop_self_heals`, parametrized on
  both wake paths) that replay the stalled-merge scenario against the mock backend:
  a sub-agent blocks (`request_input`), the orchestrator is woken (bridge) or polls
  (`pending_agents`), answers by resuming the sub-agent's session, and the loop
  resolves - proving the bidirectional-comms feature self-heals the case the spike
  exists to fix, not just its pieces (spike 20260723-001256).
- Auto-wake bridge (opt-in via `SCUFRIS_AUTO_WAKE`, off by default): when a
  sub-agent finishes a run awaiting a decision (a `WAITING` outcome from
  `request_input`) or errors, the dashboard grants the orchestrator a turn with the
  question injected, so a stalled loop self-heals without the operator driving it.
  Wakes are deferred while the orchestrator is mid-turn and batched into one turn
  when it goes idle - never dropped, and the waker never holds the orchestrator's
  serialize key. When off, the orchestrator polls `pending_agents` (BC3) instead.
  Completes bidirectional agent<->orchestrator comms (spike 20260723-001256).
- Sub-agents can signal the orchestrator that they are blocked and need a
  decision, via a `request_input` MCP tool - the only scufris tool a regular agent
  gets (see the role scoping under Changed). Calling it records a WAITING outcome
  carrying the question, preserved across the agent's turn-end (so the natural
  completion does not clobber it) - the orchestrator answers later by resuming the
  session. Wired on both the codex and claude backends (see the claude MCP-parity
  entry above). Part of bidirectional agent<->orchestrator comms
  (spike 20260723-001256).
- Orchestrator-only `pending_agents` and `acknowledge` MCP tools (and the
  `GET /api/agents/pending` / `POST /api/agents/{id}/acknowledge` endpoints behind
  them): the orchestrator can poll "which sub-agents need me" - those with an
  unacknowledged `request_input` (WAITING) or ERROR outcome, with their question -
  and clear one once handled, so a blocked sub-agent no longer waits forever.
  Part of bidirectional agent<->orchestrator comms (spike 20260723-001256).
- A durable per-agent run-outcome record (`<state_dir>/outcomes.json`): when a
  run ends, the final message and terminal state are persisted for every agent,
  so the orchestrator can observe a finished agent AFTER its per-run event stream
  has closed - the substrate for bidirectional agent<->orchestrator comms
  (spike 20260723-001256). A new `AgentState.WAITING` ("ended a turn awaiting a
  decision") names the needs-input state, distinct from `BLOCKED` (waiting on an
  approval). Deleting an agent drops its outcome.
- Full CRUD orchestrator control tools on the scufris MCP server: `get_project`,
  `update_project`, `delete_project`, `update_agent` and `delete_agent` join the
  existing create/list/run/message tools, so the orchestrator can edit an agent's
  permission mode (manual|edit|auto), provider (codex|claude) and model, and manage
  projects, all from chat. The PATCH tools send only the fields you pass. The agent
  write tools edit REGULAR agents only - the reserved orchestrator configures itself
  via settings and is refused.
- Orchestrator control tools on the scufris MCP server (orchestrator-only): the
  landing orchestrator can now DO dashboard actions, not just observe. `list_projects`,
  `create_project`, `create_agent`, `run_agent` and `message_agent` call the
  dashboard's own HTTP API at `SCUFRIS_API_BASE` (127.0.0.1:<port>, injected when the
  dashboard spawns the server), reusing each endpoint's validation and lifecycle since
  the MCP subprocess cannot touch the live in-app supervisor. Curated and bounded like
  the existing tools; a non-2xx or network failure returns `error:` text, never an
  exception.
- Settings page: an interactive "try it" runner on each enabled tool card - reveal
  a form generated from the tool's parameter schema, confirm, and run one MCP tool
  in isolation with its result rendered inline, without a chat turn. Backed by a new
  `POST /api/agent/tools/{name}/run` endpoint that runs a single scufris tool
  in-process (bypassing the agent) and refuses a disabled tool (403), an unknown tool
  (404), or bad args (422). The tools listing (`GET /api/agent/tools`) now also
  exposes each tool's typed parameter schema.

[Unreleased]: https://github.com/alexjercan/scufris/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alexjercan/scufris/releases/tag/v0.1.0
