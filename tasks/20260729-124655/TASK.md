# EPIC: Make Scufris a safe NixOS host operator

- PRIORITY: 115
- TAGS: goal, epic, v0.2.0, host, nixos
- KIND: EPIC
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Epic

Give Scufris real agency over the machine it runs on. Today the host surface is
three read-only MCP tools (`host_stats`, `disk_usage`, `list_processes` in
`scufris/mcp_server.py`) and every sub-agent is bound to a project working tree,
so the README's "scuffed Jarvis for one NixOS machine" promise has nothing
behind it: the assistant can describe the box but cannot act on it.

NixOS is what makes acting on it safe enough to attempt. The system is
declarative, a change can be built and diffed before it is activated, and every
activation is a generation that can be rolled back. So every mutating host
action follows one contract:

    propose -> preview -> approve -> apply -> audit -> roll back

Two deployment facts shape this epic. The operator's configuration lives in a
second git repository (`~/personal/nix.dotfiles`: flake-parts, home-manager,
sops-nix, `hosts/nixos`), so a host change is a git flow over that repo plus a
privileged activation step. And Scufris itself is deployed from that repo as a
systemd USER service running as the operator, bound to `0.0.0.0:8000` and opened
to the LAN by an explicit firewall rule, with no HTTP authentication - so
nothing here may gain mutating power before the dashboard is authenticated.

This is the differentiator: a general coding CLI can edit a file, but it does
not know this machine, cannot see its units and generations, and cannot be
trusted to switch its configuration. Scufris can, because it is always on, it
lives on the host, and it can put a human in front of every consequential step.

## Done Means

1. A host agent can inspect units, logs, storage, network, sensors, packages,
   and system generations through typed tools instead of improvised shell
   (test: `test_host_inspection_covers_units_logs_and_storage`).
2. No mutating host action reaches the system without a typed proposal, a
   rendered preview, and an explicit operator approval
   (test: `test_host_action_requires_preview_and_approval`).
3. A NixOS configuration change runs edit -> build -> closure diff -> approve ->
   switch, records the resulting generation, and can be rolled back
   (test: `test_nixos_change_builds_diffs_switches_and_rolls_back`; the real
   activation and rollback are `nix build .#hostd-vm-test`). The EDIT half is a
   project workflow on the config repo, not a scufris surface
   (`tasks/20260729-125035/DECISION.md`); rolling back "from the UI" waits on
   20260729-125040, which renders R3 like any other action.
4. Every requested, denied, approved, and applied host action is durably audited
   with actor, agent, run, command, result, and generation reference
   (test: `test_host_actions_are_audited`).
5. Scheduled host checks reach the operator through Telegram without opening the
   dashboard (test: `test_scheduled_host_digest_is_delivered`).
6. manual: "clean up disk space", "why is this box hot", "add this package to
   my config", and "restart that service" are answerable and actionable from
   chat without dropping into a terminal. (Corrected from "the laptop" during
   20260729-125024: this host is a DESKTOP - chassis_type 3, no battery, no fan
   sensors - so the thermal answer comes from coretemp plus the CPU's
   thermal_throttle counters, not from a battery/fan reading.)

## Child Tasks

- [x] 20260729-125015 (p70, v0.2.0) gate the dashboard behind an authenticated
      session
      landed f7a2b83; 2 review rounds (10 findings, 2 MAJOR); the machine-token
      leak into the agent CLI env was the one worth the review's cost
- [x] 20260729-125020 (p65, v0.2.0) spike: define the host capability privilege
      and safety model
      landed; SPIKE.md + DECISION.md; privilege boundary is a root helper with
      typed verbs, no sudo rules, no shell escape; unblocked 125024 outright
- [x] 20260729-125024 (p60, v0.2.0) expand read-only host inspection beyond
      stats
      landed dc60a51; 3 review rounds (16 findings, 3 MAJOR). The one worth the
      review's cost: shell=False does not stop OPTION injection, so a
      model-supplied unit pattern of `-Hsomeone@host` made systemctl open an
      outbound SSH connection as the service user - in the package whose premise
      is that reading the host cannot do anything. Also stored XSS in the new
      cards (a systemd unit is named by a FILE), and a round-1 fix that
      reintroduced empty-rendered-as-broken.
- [x] 20260729-125029 (p55, v0.2.0) add the host action framework with preview
      approval and audit
      landed 7677b5f; 3 review rounds (25 findings, 5 BLOCKER, 3 MAJOR). The
      ones worth the review's cost: approval originally checked for the wrong
      credential shape, so no credential at all could approve on loopback;
      secret stripping lived in one backend while Claude still inherited the
      root-helper secret; and caller-supplied `agent` text was allowed to key a
      proposal cap. The final framework keeps argv construction, proposal state,
      apply and audit inside `scufris-hostd`, with the app only proposing typed
      actions and approving helper-owned ids.
- [x] 20260729-125035 (p50, v0.2.0) add the NixOS configuration change flow with
      generation rollback
      landed; DECISION.md re-cut the task: the config repo is a PROJECT, so
      scufris owns build/preview/activate/rollback and NOT the edit (typed edit
      verbs rejected). The two worth the review's cost came from the VM test
      rather than from reading: every `nix` new-CLI call needed
      `--extra-experimental-features` (the operator's nix.conf is not ours to
      assume, and `nix path-info`/`nix store gc` already shipped assuming it),
      and a test VM has NO system profile generation, unlike every installed
      host. Also decided mid-build: the preview must NOT run
      `switch-to-configuration dry-activate`, because that executes an
      unapproved configuration's own code as root.
- [x] 20260729-125040 (p45, v0.2.0) add the host operator agent and the approval
      decision core
      landed; DECISION.md settled four forks first (the mutating tools MOVE to a
      host-agent-only server; an allowlisted Telegram chat IS the operator; restart
      recovery reads a new hostd `list_pending` verb; the task re-cut into three)
      and TWO of its own sections were corrected mid-build by measurement. 2 review
      rounds (5 findings, 1 MAJOR). The one worth the review's cost: the refusals
      protecting a pending decision were keyed on the agent's BLOCKED state, which
      nothing clears when a proposal EXPIRES - so one proposal the operator never
      answered left the host agent permanently unreachable and unacknowledgeable,
      with the approval that would have freed it no longer approvable. The other
      worth naming: keying the strong confirmation on `reversal.possible` alone
      demanded a typed acknowledgement for every service restart, because "no undo"
      is the NORMAL answer for R1.
- [x] 20260730-104520 (p44, v0.2.0) add the dashboard host approval queue and
      audit surface
      landed; 2 review rounds (5 findings, 2 MAJOR). Both MAJORs were the same root
      cause and neither was findable by reading: the 4-second poll rebuilt the page
      over whatever the operator was typing (making the type-the-token one-way gate a
      race lost every tick), and the error banner never cleared, so a refused
      decision was still reported after a later success. Serving the page for real
      also found what no test could - `/api/host/actions` answers `200 []` with no
      helper configured, so "not configured" had to be read off the audit endpoint.
- [x] 20260730-104524 (p43, v0.2.0) add host approvals over Telegram
      landed; 2 review rounds (3 findings, 1 MAJOR). The one worth the review's
      cost: the message cap trimmed the TAIL, which is where the undo line and the
      result live - so a long preview (an R3 activation's IS a closure diff) showed
      the operator the diff and not how to undo it. It now shortens the preview and
      says how many lines it dropped. Also settled `decidable()` as the one
      definition of "a decision can still be made here", so no surface offers a
      button the service would refuse.
- [x] 20260729-125046 (p40, v0.2.0) add scheduled host checks and a proactive
      digest
      landed; DECISION.md settled four forks (code not a model turn; silence plus a
      daily heartbeat; an in-process loop with persisted state; escalation built but
      off) and TWO of its sections were amended by measurement. 2 review rounds (5
      findings, 2 MAJOR). Both MAJORs were the same shape and neither was visible in
      a single-pass test: a standing condition was re-sent every 15 minutes (96 a day
      for a disk that had not moved) and, with escalation on, re-proposed a root
      action alongside each message. Also corrected mid-build: nothing may fire on a
      fresh schedule (it made every test boot read the real host) and the run-now
      endpoint must not block (the route sweep spent 38s walking the nix store).

## Decisions

- 20260729-125015 DECISION.md: single operator, password -> scrypt hash in the
  existing sops dotenv, opaque session id in an HttpOnly cookie over a revocable
  server-side record, one deny-by-default middleware, and a per-process bearer
  token for the app's own MCP tool subprocesses (ACCEPTED)
- 20260729-125035 DECISION.md: `~/personal/nix.dotfiles` is a PROJECT - an agent
  edits and commits it through the ordinary project flow, and scufris owns only
  build -> preview -> activate -> rollback; the toplevel is built by the server
  from a resolved rev and `activate` is refused on the generic propose surface;
  no `dry_activate` verb and no dry-activate in the preview (it would run
  unapproved code as root); a Plan carries STEPS so a half-applied activation is
  recordable (ACCEPTED)
- 20260729-125046 DECISION.md: the checks and the digest text are CODE (no model
  turn - the DoD's threshold and named-failure tests are not properties a prose turn
  has); `watch` is silent unless it has news while `daily` always sends one line, so
  silence never means "is it running"; the trigger is an in-process loop with
  persisted state and a missed window is recorded, not stampeded; escalation ships
  OFF and only the R2 cleanup verbs may ever be proposed by a threshold (ACCEPTED)
- 20260729-125040 DECISION.md: the mutating host tools MOVE off the orchestrator
  onto a host-agent-only MCP server (inspection stays on both); an allowlisted
  Telegram chat IS the operator, with one shared approval service and no second
  decision path; restart recovery reads a new read-only hostd `list_pending` verb
  rather than persisting the app's queue; a pending approval is a WAITING agent
  marked operator-bound so the orchestrator cannot answer it; and the strong
  confirmation is required by `reversal.possible`, not by the risk letter. The
  task is re-cut into three children (agent + core, web surface, Telegram)
  (ACCEPTED)
- 20260729-125020 SPIKE.md + DECISION.md: the privileged surface is a
  `scufris-hostd` NixOS system unit running as root with a typed JSON protocol
  over a unix socket and NO sudo rules (it is the only option that can bind an
  approval to the exact store path that was previewed, and its audit log is
  root-written so the app cannot rewrite its own record); five risk classes
  R0-R4 where the verb set IS the taxonomy and the refused class is enforced by
  absence of a verb; `nix store diff-closures` for the config preview and an
  explicit "no honest preview" for service restarts; generations for R3
  rollback, recorded unit state for R1, one-way declared for R2; NO arbitrary
  shell at any privilege under any approval; config changes proposed in a
  sprout worktree over the config repo and committed before they are built
  (ACCEPTED)

## Manual Acceptance

EVERY child task has landed and every automated criterion in Done Means is proven on
master (the five named tests pass; `nix flake check`, `cd web && npm run ci` and
`nix build .#scufris .#web` are green; `nix build .#hostd-vm-test` was run for the
verb it added). This container stays OPEN because Done Means 6 is a MANUAL criterion,
and so are the seven items below - none of them can be closed by building anything.

Two operator actions gate trying any of them, and neither is scufris's to perform:

1. `scufris hash-password`, then add the line plus `SCUFRIS_HOSTD_SECRET` to
   `sops secrets/scufris.env` in `~/personal/nix.dotfiles`.
2. `services.scufris-hostd.enable = true` (the `nixosModules.hostd` module), then bump
   the scufris flake input past the release that carries this work.

Until the secret exists a LAN-bound scufris refuses to start, by design, and every
mutating host endpoint answers "not configured".

- (pending) 20260729-125015: logging in from a phone on the LAN is bearable
  enough that you do not disable it. NOTE: this needs an operator action first -
  run `scufris hash-password`, add the line to `sops secrets/scufris.env` in
  nix.dotfiles, and only then bump the scufris flake input past v0.1.0. Until
  that secret exists, a LAN-bound scufris REFUSES TO START (by design).
- (accepted 2026-07-29) 20260729-125020: the operator accepted the privilege
  model - root helper with typed verbs, no sudo rules, no shell escape, config
  changes proposed in a sprout worktree. This was the gate on writing any
  mutating host code, and it is now open.
- (pending) 20260729-125024: asking the orchestrator "why is this box hot" and
  "what filled the disk" produces a specific, correct answer without a terminal,
  and the four new stats-page cards earn their space.
- (pending) 20260729-125029: the rendered host-action approval prompt states
  plainly what will change and how it can be undone; the framework proof is
  `examples/host_action.py`, while the dashboard and Telegram approval surfaces
  land in 20260729-125040.
- (pending) 20260729-125035: the closure diff makes a change understandable
  before switching, not after; and adding a package through chat - as a project
  task on nix.dotfiles, then one approval - is faster and no scarier than doing
  it by hand. NOTE: `services.scufris-hostd.enable` and the sops secret are
  still the operator actions that gate trying this at all.
- (pending) 20260730-104520: the queue is readable at phone width and the risk
  difference between a service restart and a system switch is obvious at a glance.
  NOTE: this needs a real look. The building session had no browser tooling, so the
  structure, the classes and the media query are tested but the RENDER is unverified.
- (pending) 20260730-104524: approving a real host change from a phone is clear
  enough to do confidently while away from the desk. NOTE: no Telegram account or
  phone existed in the building session, so `examples/telegram_approval.py` is the
  stand-in - it prints every message and button for both the one-tap and the two-tap
  flow. The real check still needs a phone.
- (pending) 20260729-125046: after a week of daily digests, the operator still reads
  them - the scheduled brief is worth reading rather than noise. NOTE: this needs a
  week of living with it AND the deployment. `examples/host_digest.py` prints the
  digest in all five states (boring/watch, boring/daily, something wrong, something
  recovered, a check broken), which is the fastest way to judge the wording first.

## Notes

- Scope discipline: this epic builds the SPECIFIC approval/audit path for host
  actions. The general capability-grant system (20260729-102919) stays in the
  backlog until a second consumer exists to generalize from.
- The dashboard-authentication child is carved out of 20260729-102208, which
  keeps the secret-reference and redaction half for the plugin epic.
- Threat-model honesty, from the spike: `alex` is in the `docker` group, which
  is root-equivalent on this machine, so these controls are NOT a defence
  against a compromised operator account. They defend against the model acting
  unasked, a prompt-injected agent, an approval given without visible
  consequences, and the absence of a record. Tightening the account itself is a
  `nix.dotfiles` change (rootless docker, or dropping the group) and is the
  operator's call, out of scope here.

## Close-out (2026-07-30)

The OPERATOR declared this epic done and closed it. What that does and does not
mean, so a cold session does not misread the record:

- Every child task landed and every automated criterion in Done Means is proven on
  master. That was already true at the previous commit.
- Done Means 6 and the eight items in Manual Acceptance are closed by DECISION,
  not by evidence. They are left exactly as written rather than ticked: the two
  operator actions (the sops secret, `services.scufris-hostd.enable`) and the things
  that need a real phone, a real browser and a week of digests are still unperformed
  at close. Anything they turn up becomes a NEW task, not a reopening of this one.
- `ARCHITECTURE.md` in this folder is the durable map of what the epic produced -
  the processes and trust boundaries, the propose/preview/approve contract, the risk
  taxonomy, the audience split, the R3 flow, the proactive path, and what the design
  explicitly does not claim. It was written at close and verified against the code
  rather than from the task records.

The build phase and the epic are both behind us: 8 of 8 children closed, every
automated proof green on master, and the container closed at the operator's
declaration on 2026-07-30 with the manual items recorded as unperformed.
