# Add the host action framework with preview approval and audit

- STATUS: OPEN
- PRIORITY: 55
- TAGS: feature,v0.2.0,host,security,backend

## Story

As the operator, I want every host change to arrive as a proposal I can preview
and approve, so that an agent with access to my machine is a colleague asking
before it acts rather than a process with root and good intentions.

This is the mechanism the rest of the epic runs on:

    propose -> preview -> approve -> apply -> audit -> roll back

It is deliberately built for ONE consumer (host actions) rather than as a
general capability system. Generalizing waits for a second consumer.

## Steps

### A. The `scufris-hostd` root helper

- [ ] `scufris/hostd/actions.py`: the typed action taxonomy. `ActionKind`
      (`unit_start`, `unit_stop`, `unit_restart`, `unit_reload`, `gc_older_than`,
      `gc_store`), each with a pydantic argument model and a `RiskClass` (R1/R2).
      The HELPER constructs every argv - the caller never supplies one. Validate
      unit names against a charset, refuse a leading `-` explicitly and pass
      positionals after `--` (lesson `shell-false-does-not-stop-option-injection`).
      Enforce the R1 deny-list (`sshd`, `dbus`, `systemd-logind`,
      `NetworkManager`, plus anything matching `scufris` or `scufris-hostd`
      regardless of the list). R4 is enforced by no verb existing.
- [ ] `scufris/hostd/preview.py`: preview per risk class, honest where there is
      none. R1 renders current `ActiveState`/`SubState` plus reverse
      dependencies, LABELLED as state and blast radius rather than a simulation.
      R2 runs the real `--dry-run`, and because that prints only a dead-path
      count, the helper computes the reclaimable size itself (`nix path-info -S`)
      and lists the generations it resolved. A class with no preview says so; an
      empty preview is never rendered as a successful one.
- [ ] `scufris/hostd/actions.py` (R2 floor): `gc_older_than` resolves the
      generation list FIRST and refuses or clamps any request that would remove
      either of the two most recent generations. The `--delete-older-than` flag
      is age-based only and must not be trusted to provide this. Never a bare
      `nix-collect-garbage -d`.
- [ ] `scufris/hostd/proposals.py`: the proposal registry lives IN THE HELPER.
      `propose` returns an id, the rendered preview, a state fingerprint, an
      expiry and a reversal descriptor; `apply(id)` is the only way to act. This
      is what makes "preview X, apply Y" impossible - the app cannot hand the
      helper an argv. TTL 10 minutes, single-use (an applied or denied proposal
      is terminal), and the fingerprint is re-read at apply time so a proposal is
      refused once the system moved underneath it.
- [ ] `scufris/hostd/audit.py`: append-only single-line JSON. Records
      `requested`, `denied`, `approved`, `applied`, `failed`, `cancelled` and
      `reverted` with actor, agent, run, action kind, arguments, argv, result,
      duration, outcome and reversal reference. Redact secret-shaped values.
      Rotate on SIZE (16 MiB active, 5 files kept total), prune oldest-first, no
      verb for deleting entries.
- [ ] `scufris/hostd/protocol.py` + `server.py`: newline-delimited JSON frames
      over a unix socket. Verbs: `hello`, `propose`, `apply`, `cancel`,
      `audit_tail`. Every frame carries the shared secret, compared
      constant-time; an unknown verb, a malformed argument or a bad secret is
      refused with a typed error. `apply` streams output frames and ends with one
      terminal result frame. Reuse `scufris.host.run.Runner` so tests replay
      captured output through `FakeRunner` instead of patching subprocess.
- [ ] `scufris/hostd/main.py` and the `scufris-hostd` console script. Fail
      closed: no secret configured means the helper refuses to start.
- [ ] The reversal path. R1 records prior `ActiveState`/`SubState` and offers
      restoration of it, acknowledging that a restarted daemon lost in-memory
      state. R2 records that it CANNOT be undone, and takes a stronger
      confirmation than the reversible classes rather than a weaker one.

### B. One execution path

- [ ] Make `Supervisor` and `EventBus` generic in their event type
      (`Supervisor[EventT]`, `EventBus[EventT]`), with
      `AgentSupervisor = Supervisor[StreamEvent]` so every existing call site
      keeps its current type. Mechanical: rename + parameterize, no behavior
      change. Sweep every implementor AND every test double in one pass (lesson
      `protocol-signature-change-hits-the-doubles`), and say mypy explicitly when
      claiming green.
- [ ] `scufris/hostclient.py`: the app-side async client for the socket. Apply
      runs through the supervisor with its own event models, so cancellation is
      `supervisor.cancel(run_id)` and the helper kills the child process group
      and records the cancelled outcome.

### C. The app surface, and the approval gate at the execution boundary

- [ ] `auth.py` + the `app.py` middleware: an `OPERATOR_ONLY_PATHS` set consulted
      BEFORE the bearer-token short-circuit at `scufris/app.py:905-908`, so a
      machine token is refused there rather than accepted ahead of the session
      and CSRF checks. This is WORK, not a property inherited from
      20260729-125015 (lesson `enforcement-point-not-the-decision-record`).
- [ ] Endpoints: `POST /api/host/actions` (propose - a session OR the machine
      token, because an agent may propose), `POST /api/host/actions/{id}/approve`
      and `/deny` (operator-only: session + CSRF, bearer REFUSED),
      `GET /api/host/actions`, `GET /api/host/actions/{id}`,
      `POST /api/host/actions/{id}/revert`, `GET /api/host/audit`. Declare static
      segments before parameterized ones (lesson
      `static-route-before-param-route-or-it-is-shadowed`).
- [ ] An MCP tool `propose_host_action` in `mcp_server.py` so an agent can
      propose and read its own proposal, and has no tool that can approve one.
- [ ] Config in `scufris/config.py`: `hostd_socket`, `hostd_secret` (never in
      `os.environ` - it reaches the client through `Settings`, exactly as
      `auth_api_token` does), `hostd_enabled`. With hostd absent, every mutating
      endpoint answers "no privileged helper configured" rather than
      half-working.

### D. Packaging and deployment

- [ ] `[project.scripts] scufris-hostd` in `pyproject.toml`.
- [ ] `nix/scufris-hostd.nix`: a NixOS SYSTEM module exporting
      `services.scufris-hostd` (enable, package, socketPath, group, secretFile,
      auditLog, auditMaxBytes, auditKeep), running as root with
      `RuntimeDirectory` for the socket. Exported as `nixosModules.hostd` -
      separate from `nixosModules.default`, so enabling host agency is its own
      diffable act.
- [ ] Extend `nix/tests/scufris-vm.nix` (or add a sibling) to boot the helper and
      prove the socket appears, an unknown verb is refused, and a proposal
      without approval never executes. Run with `nix build .#vm-test` (not in
      `checks` - lesson `nixos-vm-test-for-on-demand-not-checks`).

### E. Tests, example, docs

- [ ] `tests/test_host_actions.py`: the DoD tests plus the adversarial set -
      forged action id, replay of an applied proposal, approval after fingerprint
      drift, concurrent approvals of the same proposal, cancellation mid-apply,
      secret redaction in the audit, a machine-token approval REFUSED, a unit
      name that would become an option (`-Hsomeone@host`), a deny-listed unit,
      and a `gc_older_than` that would eat the previous generation.
- [ ] `tests/test_hostd_audit.py`: rotation on size, oldest-first pruning, and
      that no code path deletes or rewrites an entry.
- [ ] `examples/host_action.py`: propose -> preview -> approve -> apply -> audit
      -> revert end to end against a `FakeRunner`, printing what an operator
      would see.
- [ ] Docs in the SAME task: the AGENTS.md deployment section (the second sops
      line and the module option), README if it describes the host surface, and
      `CHANGELOG.md` under `[Unreleased]`. Check the surfaces against the diff
      rather than ticking the step.

## Definition of Done

- An action with no effective approval never reaches execution, regardless of
  the path it was requested through
  (test: `test_host_action_requires_preview_and_approval`).
- The machine bearer token cannot approve an action, asserted at the middleware
  that enforces it (test: `test_machine_token_cannot_approve_a_host_action`).
- Approving a stale proposal is refused, and approvals do not replay
  (test: `test_host_action_approval_is_scoped_and_single_use`).
- Every requested, denied, approved, applied and failed action produces a
  durable redacted audit record (test: `test_host_actions_are_audited`).
- Cancelling mid-apply leaves a recorded, consistent outcome rather than an
  unknown state (test: `test_host_action_cancellation_is_recorded`).
- A reversible action records its inverse and the undo path is exercised; a
  one-way action records that it cannot be undone
  (test: `test_reversal_is_recorded_or_declared_impossible`).
- The audit log rotates on size and is pruned oldest-first, and nothing outside
  the helper can delete an entry
  (test: `test_audit_log_rotates_and_never_deletes`).
- The gate is green
  (cmd: `nix flake check && nix build .#scufris .#web`).
- manual: an approval prompt states plainly what will change and how it can be
  undone.

## Notes

- Epic: 20260729-124655.
- Depends on: the host spike (taxonomy, privilege, preview, rollback) and the
  dashboard authentication task. Both are settled - see
  `tasks/20260729-125020/DECISION.md`.
- The three forks the spike deferred are decided in `DECISION.md` next to this
  file, confirmed by the operator: the socket takes a shared secret from the
  sops dotenv (because the agent backends run shell as `alex` and would
  otherwise reach `apply` directly); this task ships R1 + R2 and leaves R3 to
  20260729-125035; and `Supervisor`/`EventBus` become generic in their event
  type rather than the host borrowing the agent's event union.
- SPIKE OUTCOME: the privileged surface is a `scufris-hostd` NixOS SYSTEM unit
  running as root with a typed JSON protocol over a unix socket, NOT sudo rules.
  Building that helper, its protocol, its flake packaging and its module option
  is part of this task. The verb set IS the action taxonomy.
- Audit storage: the helper appends its own root-owned, append-only log, so the
  record survives the app being the thing that misbehaved. This epic does NOT
  inherit a dependency on 20260729-102147. The request side (proposed/denied) is
  app state.
- The approval endpoint must require a real operator session plus CSRF and
  REJECT the bearer token. `scufris/app.py:905-908` short-circuits the
  middleware on a valid bearer token before the session lookup and before the
  CSRF and same-origin checks, on every non-public path and method - and the
  app's own MCP tool subprocesses hold exactly that token.
- Audit retention knob (picked here per the spike): 16 MiB active file, 5 files
  kept total, size-rotated, oldest-first, exposed in the module option.
- Approval TTL: 10 minutes, single-use, plus a fingerprint re-read at apply.
- Reuse `scufris/supervisor.py` for run lifecycle and cancellation instead of
  inventing a second execution path.
- The dashboard approval UI is 20260729-125040; this task's operator-facing
  proof is the rendered preview text plus `examples/host_action.py`.
- The general capability-grant system is 20260729-102919 and stays in the
  backlog.

## Flow State

- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
