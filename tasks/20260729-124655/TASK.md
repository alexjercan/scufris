# EPIC: Make Scufris a safe NixOS host operator

- STATUS: OPEN
- PRIORITY: 115
- TAGS: goal,epic,v0.2.0,host,nixos

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
   switch, records the resulting generation, and can be rolled back from the UI
   (test: `test_nixos_change_builds_diffs_switches_and_rolls_back`).
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
- [ ] 20260729-125035 (p50, v0.2.0) add the NixOS configuration change flow with
      generation rollback
- [ ] 20260729-125040 (p45, v0.2.0) add the host operator agent and its approval
      surfaces
- [ ] 20260729-125046 (p40, v0.2.0) add scheduled host checks and a proactive
      digest

## Decisions

- 20260729-125015 DECISION.md: single operator, password -> scrypt hash in the
  existing sops dotenv, opaque session id in an HttpOnly cookie over a revocable
  server-side record, one deny-by-default middleware, and a per-process bearer
  token for the app's own MCP tool subprocesses (ACCEPTED)
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
- (pending) config flow: the closure diff makes a change understandable before
  switching, not after.
- (pending) digest: the scheduled brief is worth reading rather than noise.

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

## Flow State

- FLOW STEP: PLANNING
