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
6. manual: "clean up disk space", "why is the laptop hot", "add this package to
   my config", and "restart that service" are answerable and actionable from
   chat without dropping into a terminal.

## Child Tasks

- [x] 20260729-125015 (p70, v0.2.0) gate the dashboard behind an authenticated
      session
      landed f7a2b83; 2 review rounds (10 findings, 2 MAJOR); the machine-token
      leak into the agent CLI env was the one worth the review's cost
- [ ] 20260729-125020 (p65, v0.2.0) spike: define the host capability privilege
      and safety model
- [ ] 20260729-125024 (p60, v0.2.0) expand read-only host inspection beyond
      stats
- [ ] 20260729-125029 (p55, v0.2.0) add the host action framework with preview
      approval and audit
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
- Pending the host spike SPIKE.md and DECISION.md: the action taxonomy, the
  privilege boundary (the service runs as the operator; `nixos-rebuild switch`
  does not), the preview and rollback mechanism per action class, and where the
  line against arbitrary shell sits.

## Manual Acceptance

- (pending) 20260729-125015: logging in from a phone on the LAN is bearable
  enough that you do not disable it. NOTE: this needs an operator action first -
  run `scufris hash-password`, add the line to `sops secrets/scufris.env` in
  nix.dotfiles, and only then bump the scufris flake input past v0.1.0. Until
  that secret exists, a LAN-bound scufris REFUSES TO START (by design).
- (pending) config flow: the closure diff makes a change understandable before
  switching, not after.
- (pending) digest: the scheduled brief is worth reading rather than noise.

## Notes

- Scope discipline: this epic builds the SPECIFIC approval/audit path for host
  actions. The general capability-grant system (20260729-102919) stays in the
  backlog until a second consumer exists to generalize from.
- The dashboard-authentication child is carved out of 20260729-102208, which
  keeps the secret-reference and redaction half for the plugin epic.

## Flow State

- FLOW STEP: PLANNING
