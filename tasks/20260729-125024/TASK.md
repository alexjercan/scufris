# Expand read-only host inspection beyond stats

- STATUS: OPEN
- PRIORITY: 60
- TAGS: feature,v0.2.0,host,mcp,backend

## Story

As the operator, I want to ask the assistant real questions about this machine
and get answers from typed tools, so that "why is the laptop hot", "what filled
the disk", and "did anything fail overnight" are answerable in chat.

Today the host toolset is `host_stats`, `disk_usage`, and `list_processes`. That
covers "what is using memory" and nothing else. This task is pure read-only
expansion: it is useful on its own, it is the input every later mutating action
reasons from, and it needs no privilege decision to ship.

## Steps

- [ ] Add systemd inspection: unit list with state filtering, one unit's status
      and recent invocation result, failed units, and user vs system scope.
- [ ] Add journal query: bounded reads by unit, priority, and time window, with
      an output cap and a clear truncation marker rather than dumping megabytes
      into a model context.
- [ ] Add storage inspection: filesystem usage by mount, largest directories
      under a bounded root, Nix store size, garbage-collectable space, and
      system generations with their dates and sizes.
- [ ] Add network inspection: interfaces, addresses, listening sockets with the
      owning process, and the current firewall rule state.
- [ ] Add sensors and power: temperatures, fan state, battery, and thermal
      throttling indicators (this is a laptop, and "why is it hot" is a real
      question).
- [ ] Add package and generation queries: what provides a binary, what is
      installed in the profile, how the current generation differs from the
      previous one, and whether the flake inputs are behind.
- [ ] Make every tool bounded and honest: explicit limits, timeouts, structured
      output, and a defined result when the underlying command is missing or the
      data is unavailable. No silent empty result that reads as "nothing wrong".
- [ ] Surface the same data in the dashboard where it earns its place, rather
      than only through the agent.

## Definition of Done

- Units, logs, storage, network, sensors, and generations are readable through
  typed tools with bounded output
  (test: `test_host_inspection_covers_units_logs_and_storage`).
- Every tool degrades to an explicit unavailable/unsupported result instead of
  raising or returning a misleading empty value
  (test: `test_host_inspection_tools_degrade_explicitly`).
- Journal and directory reads cannot exceed their configured caps
  (test: `test_host_inspection_output_is_bounded`).
- manual: asking the orchestrator "why is my laptop hot" and "what filled the
  disk" produces a specific, correct answer without a terminal.

## Notes

- Epic: 20260729-124655.
- Read-only: nothing in this task changes system state, so it can land before
  the privilege decision from the host spike.
- SPIKE OUTCOME (`tasks/20260729-125020/DECISION.md`): confirmed and stronger
  than assumed. Every inspection this task needs was MEASURED working as `alex`
  with no privilege at all - `journalctl -u <system unit>` (wheel gets the
  journal ACL), `nixos-rebuild list-generations`, `nix store diff-closures`,
  `systemctl show`. So this task needs no helper, no sudo, and no waiting: it
  can run in parallel with the privileged machinery in 20260729-125029.
- Use `nix store diff-closures` (builtin to nix 2.34.8 here), not `nvd` - `nvd`
  is not installed and the builtin means no new dependency in `nix.dotfiles`.
- Build on `scufris/metrics.py` and `scufris/processes.py` conventions
  (psutil-backed, structured records, tolerant of missing data).
- The MCP tool docstrings are prompt surface: say when to PREFER a tool over
  shell, the way `list_processes` already does.

## Flow State

- FLOW STEP: PLANNING
