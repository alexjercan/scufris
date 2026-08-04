# `packages/hostctl/` - the host control client

`scufris-hostctl` is the **unprivileged** half of host agency: the client that
drives [`scufris-hostd`](../../../hostd/src/scufris_hostd/README.md). It runs as
the app's own user, in the app's own process, and it cannot change the machine.
Everything it does, it does by asking the root helper over a unix socket.

The three packages split by PRIVILEGE, which is a boundary the operating system
already enforces:

| Package | Privilege | What it does |
|---|---|---|
| `scufris_host` | none | reads the machine |
| `scufris_hostctl` | none | proposes, holds for approval, dispatches, records |
| `scufris_hostd` | root, separate process | owns the verbs, the proposals and the audit log |

- The trust boundaries, the two operator surfaces and the HTTP routes are in
  [`scufris/README.md`](../../../../scufris/README.md).
- The privilege model is
  [`tasks/20260729-125020/DECISION.md`](../../../../tasks/20260729-125020/DECISION.md);
  the configuration-change half is
  [`tasks/20260729-125035/DECISION.md`](../../../../tasks/20260729-125035/DECISION.md).

## 1. The contract, from the client side

The fixed sequence is `propose -> preview -> approve -> apply -> audit -> roll
back`. The helper owns the right-hand half of every arrow; what follows is what
this package does at each step.

**Propose.** A caller (an HTTP route, an MCP tool, the Telegram bot) hands
`HostdClient.propose` a typed action. The helper builds the argv, decides the
risk class and returns a `ProposalView` with a preview and a TTL.
`HostActionStore.record_proposal` writes the app-side record. The client never
constructs a command line; it names a verb.

**Preview.** The proposal carries the helper's own rendering of what would
happen. `render_action` turns a record plus its preview into the one text both
operator surfaces show - the dashboard card and the Telegram message are the
same words, because they are the same function.

**Approve.** `HostApprovalService` is the only decision seam. Approve, deny,
cancel and revert live on it, along with the `decidable()` predicate, and each
surface's only job is to say WHO is deciding - a session id for the dashboard,
`operator:telegram:<chat_id>` for the chat. An action whose reversal is
`NONE` has no ordinary approve control at all: `confirmation_for` demands a
typed acknowledgement, and the server enforces the rule the UI shows.

**Apply.** `HostApprovalService.approve` is the only caller of the apply path.
An action with no approval has no route to execution - not because a check
refuses it, but because nothing else calls it. The apply is a STREAM: output
frames arrive as they are produced and are published on an `EventBus` that a
relay or the Telegram bot subscribes to. Closing that stream sends a cancel
frame and drops the connection, and the helper kills the process group.

**Audit.** The privileged record is root-written and append-only and this
package cannot touch it. What it keeps is the other half - a DECISION JOURNAL:
what was approved, what was denied and why. That outlives the minutes the helper
keeps a proposal, so a restart mid-queue no longer answers "there was never any
such action".

**Roll back.** A reversible action carries its reversal, and `revert` proposes
it as a new action through the same seam - so undoing something is itself
approved, previewed and audited.

## 2. The configuration change flow

`hostconfig/` is the R3 flow, and it is the one place this package runs a
command itself. That is safe because the command is `nix build` as the app's own
user:

1. `resolve` turns a git ref into a commit and a flake URL. Git reads only,
   milliseconds, so the caller answers immediately. The NixOS configuration is
   an EXTERNAL project and Scufris never edits it - the build reads the tree
   from the COMMIT, so it cannot touch the repository's working tree.
2. `ConfigChangeBuilder` runs the build in a supervised background run,
   publishing output on a bus. Minutes to hours; each transition is a further
   write to the `config_change` row.
3. On success it PROPOSES the activation as an ordinary host action. The store
   path it names came from the build it just ran - a caller cannot supply one.
4. Activation, and rollback, are the helper's. A rollback names a generation
   NUMBER, never a store path.

`ConfigChangeService` takes the configured repository and attribute as two
values rather than the app's `Settings`: they are the whole of what it reads.

## 3. State

Two tables, declared in `models.py` against `scufris_core.Base`:

| Table | Holds |
|---|---|
| `host_action` | one proposed action and what the operator decided about it |
| `config_change` | one configuration change: what was built, and what came of it |

Both are private to this package. `scufris/db/migrations/env.py` imports
`scufris_hostctl` before reading `Base.metadata` - the facade pulls `models` in
on the way to `actions` and `hostconfig.changes` - which is what puts them in
front of Alembic; `test_every_package_model_is_registered` is what keeps that
import from being dropped.

## 4. The public surface

`scufris_hostctl` is the whole of it. A caller imports that name and nothing
below it - `test_no_package_imports_a_sibling_private_module` enforces the rule
for every workspace member, and the row classes are not exported at all.

`examples/hostctl_approval_flow.py` drives propose -> preview -> approve ->
apply -> audit against a fake executor over a temporary unix socket, with no
root, no network, no NixOS machine and a temporary database.
