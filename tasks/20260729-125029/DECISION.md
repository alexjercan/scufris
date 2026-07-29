# Decision: the three forks the host action framework could not infer

- DATE: 20260729-125029
- STATUS: ACCEPTED
- TASK: 20260729-125029
- TAGS: decision, host, security, privilege, supervisor, v0.2.0

## Context

`tasks/20260729-125020/DECISION.md` settled the privilege model: a
`scufris-hostd` root system unit with typed verbs, no sudo rules, no shell
escape, a root-written append-only audit log, and a verb set that IS the risk
taxonomy. It deliberately left three questions to the implementing task, and
each one changes what gets built rather than how. All three were put to the
operator and confirmed.

## Decision

### 1. The socket requires a shared secret, held only in the app process

The hostd unix socket is root-owned and group-restricted, AND every frame must
carry a secret read from the sops dotenv as `SCUFRIS_HOSTD_SECRET`.

The constraint that forced this: the agent backends run arbitrary shell as
`alex`. A socket protected only by filesystem permissions cannot distinguish
scufris from the codex/claude subprocess scufris itself spawned, so a
prompt-injected model could call `apply` directly and the entire
propose/preview/approve path would be decoration. That is the FIRST threat the
spike names, not a hypothetical one.

The secret reaches the app the same way `SCUFRIS_API_TOKEN` does and under the
same rule (lesson from 20260729-125015): it lives on `Settings`, never in
`os.environ`, so it is not inherited by the agent CLI subprocess or by any shell
command the model runs from it.

Stated plainly, and to be documented rather than oversold: this raises the bar
against the model acting unasked. It is NOT a boundary against a compromised
`alex` account, which is already root-equivalent through the `docker` group.

Consequence: hostd cannot be enabled before the secret exists in
`sops secrets/scufris.env`. hostd refuses to start without it, the same
fail-closed shape as the LAN-bound auth password.

### 2. This task ships R1 and R2. R3 stays in 20260729-125035

The helper's verb set here is `unit_start`, `unit_stop`, `unit_restart`,
`unit_reload` (R1, reversible) and `gc_older_than`, `gc_store` (R2, one-way).

Both classes are needed HERE because the Definition of Done requires that every
applied action record how to undo it *or* record that it cannot be undone. With
R1 alone the second half of that contract has no consumer and ships untested.
R2 is also what makes "clean up disk space" answerable, one of the epic's manual
acceptance sentences.

R3 (`build`, `dry_activate`, `activate`, `rollback`) is left to
20260729-125035, whose whole subject is the configuration change flow and the
closure-diff preview trap the spike measured. Adding those verbs there is
exactly the "a capability is a reviewed code change with a test" path the spike
decided on, so the protocol is designed to be extended by a new verb rather than
generalized in advance.

The dashboard approval UI belongs to 20260729-125040. This task's operator-facing
proof is the server-rendered preview text (the `scufris/host/render.py` pattern)
plus `examples/host_action.py`, which drives propose -> preview -> approve ->
apply -> audit -> revert end to end against a faked runner.

### 3. `Supervisor` and `EventBus` become generic in their event type

`Supervisor[EventT]` and `EventBus[EventT]`, with
`AgentSupervisor = Supervisor[StreamEvent]` so every existing call site keeps its
current type. Host apply defines its own event models and runs through the same
supervisor.

Rejected: widening the `StreamEvent` union (every `match` over it in the agent
paths would have to handle host members, and an agent surface could relay a root
command's output); and mapping apply output onto `StreamTextDelta` (a "model
text delta" carrying root command output is exactly the kind of adjacent-thing
mislabelling this epic rejects elsewhere).

The refactor is mechanical and mypy-checked, and it keeps the task note's
requirement literally: one execution path with one cancellation story, not two.

## Consequences

- `nix.dotfiles` gains a second explicit act: the `SCUFRIS_HOSTD_SECRET` line in
  the sops dotenv, then `services.scufris-hostd.enable = true`. Neither is
  acquired by upgrading scufris.
- 20260729-125035 extends the hostd protocol with the R3 verbs; it does not
  build a second helper.
- 20260729-125040 builds the approval UI over the endpoints landed here.
- The supervisor refactor touches agent call sites, so its diff is reviewed
  alongside a security feature. Keeping it mechanical (rename + parameterize, no
  behavior change) is a requirement on the implementation, not a hope.
