# Spike: define the host capability privilege and safety model

- STATUS: OPEN
- PRIORITY: 65
- TAGS: spike,v0.2.0,host,nixos,security

## Story

As the operator, I want a decided model for what Scufris may do to this machine
and how, so that host agency is designed once with its safety rails instead of
accreting one privileged shell call at a time.

The hard question is privilege. Scufris runs as a systemd USER service as the
operator; `nixos-rebuild switch`, `systemctl restart`, and `nix-collect-garbage
-d` on system profiles do not. Something has to bridge that gap, and the choice
determines the blast radius of every later task in this epic.

## Steps

- [ ] Inventory what exists to build on: `scufris/metrics.py`, `processes.py`,
      the MCP tool surface in `mcp_server.py`, the backend permission modes
      (`manual`/`edit`/`auto`), the supervisor's run/cancel machinery, and the
      Telegram bridge.
- [ ] Define the host ACTION TAXONOMY by risk class, each with its preview and
      its reversal: read-only inspection, reversible service control, disposable
      cleanup (garbage collection, caches), declarative system change (NixOS
      config), and irreversible/refused (disk formatting, user deletion, key
      material).
- [ ] Decide the PRIVILEGE BOUNDARY and record the tradeoffs: targeted
      `sudo` NOPASSWD rules declared in `nix.dotfiles`, a separate privileged
      helper unit with a narrow typed IPC, or polkit rules. State how the chosen
      mechanism fails closed and what an attacker with the operator's session
      can reach.
- [ ] Decide the PREVIEW mechanism per class: `nixos-rebuild build` plus
      `nvd diff` or `nix store diff-closures` for config changes, `systemctl
      show`/dry-run for units, dry-run output for garbage collection, and what
      to show when a class has no honest preview.
- [ ] Decide ROLLBACK semantics: system generations for config changes, unit
      state restoration for service control, and which classes are declared
      one-way (and therefore need a stronger confirmation).
- [ ] Draw the line against arbitrary shell: whether a typed-action allowlist is
      the only path, or a free-form command escape exists and under what
      approval. Decide once, in writing.
- [ ] Decide where the config repository is edited: a sprout worktree over
      `~/personal/nix.dotfiles` (reusing the existing worktree machinery) versus
      in-place edits, and how an unpushed/dirty config repo is handled.
- [ ] Decide audit storage and retention, and whether it shares the
      transactional store from 20260729-102147 or is an append-only log of its
      own.
- [ ] Write `SPIKE.md`, record accepted choices in `DECISION.md`, and refine the
      remaining children of this epic against it.

## Definition of Done

- The action taxonomy, its preview, and its reversal are written down per class
  (cmd: `rg -n "read-only|service control|cleanup|declarative|irreversible" tasks/20260729-125020/SPIKE.md`).
- The privilege boundary is decided with its failure mode stated
  (cmd: `test -f tasks/20260729-125020/DECISION.md && tatr check 20260729-125020`).
- The arbitrary-shell question has a recorded answer, not an implicit one
  (cmd: `rg -n "shell" tasks/20260729-125020/DECISION.md`).
- manual: the operator accepts the privilege model before any mutating host
  code is written.

## Notes

- Epic: 20260729-124655.
- Target configuration repository: `~/personal/nix.dotfiles` (flake-parts,
  home-manager, sops-nix, `hosts/nixos`, `home/alex`).
- Prefer typed actions over a shell passthrough. If a shell escape survives the
  spike, it needs a stronger justification than convenience.
- KISS: this epic proves the approval/audit pattern for ONE consumer. Do not
  design the general capability-grant system here (20260729-102919).

## Flow State

- FLOW STEP: PLANNING
