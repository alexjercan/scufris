# Decision: a root helper with typed verbs is the only privileged host surface

- DATE: 20260729-125020
- STATUS: ACCEPTED
- TASK: 20260729-125020
- TAGS: decision, host, nixos, security, privilege, v0.2.0

## Context

Epic 20260729-124655 wants every mutating host action to run
propose -> preview -> approve -> apply -> audit -> roll back. Scufris is a
systemd USER service running as `alex`; `nixos-rebuild switch`, `systemctl
restart` on system units and system-profile garbage collection are root
operations, and there is no bridge today (`sudo` requires a password, and
`hosts/nixos/default.nix` declares no sudo rules). The research, the measured
host facts and the rejected alternatives are in `SPIKE.md` next to this file
and are not repeated here.

Three forks were put to the operator and all three were confirmed: the
privilege mechanism, the arbitrary-shell question, and where the config
repository is edited.

## Decision

### 1. Privilege: a `scufris-hostd` root helper with a typed IPC. No sudo rules.

A NixOS **system** unit running as root, declared in `nix.dotfiles` next to
`programs.scufris`, listening on a unix socket and speaking a small typed JSON
protocol with a closed set of verbs. The helper - never the caller -
constructs every argv, validates every argument, and writes the audit record.
`sudo` on this box stays password-required; no `security.sudo.extraRules` are
added.

Chosen over targeted sudo NOPASSWD rules (rejected: sudo matches an argv
prefix, so any rule useful enough to cover `nixos-rebuild switch --flake` also
covers "run attacker-chosen Nix as root"; the grant lands on the user, not on
scufris; nothing binds the approved preview to the applied action; the audit
record would be written by the same uid that acts), over polkit (rejected: it
cannot express `nixos-rebuild` at all, and this host has no polkit agent
installed), and over running scufris itself as root (rejected: it hands root to
the process containing the model and the LAN-facing HTTP surface, inverting the
epic).

The decisive property is one sudo structurally cannot provide: **the helper can
refuse to activate any toplevel it did not itself build and the operator did
not approve**, closing the window between preview and apply.

**How it fails closed.** No socket (helper not enabled in the config) means no
privileged action is possible at all. An unknown verb is refused. A malformed
or out-of-range argument is refused. There is no shell verb to fall back to,
and no verb exists for the refused class, so those actions have no code path at
all rather than a check that could have a bug.

**What an attacker with the operator's session reaches - stated plainly.** The
socket is reachable by anything running as `alex`, and the helper cannot
distinguish scufris from another process of the same user. It is not a defence
against a compromised operator account, and it must never be described as one:
`alex` is in the `docker` group, which is already root-equivalent on this
machine. What the helper does defend against is the model acting unasked, a
prompt-injected agent, an operator approving something whose consequences were
invisible, and the absence of a record afterwards.

### 2. Action taxonomy: five risk classes, and the verb set IS the taxonomy

| Class | Verbs | Privilege | Preview | Reversal |
|---|---|---|---|---|
| **R0 read-only** | units, logs, storage, network, sensors, packages, generations | none (`alex`) | n/a | n/a |
| **R1 service control** | `unit_start`, `unit_stop`, `unit_restart`, `unit_reload` | helper | `systemctl show` + reverse dependencies, labelled state-not-simulation | restore recorded prior state |
| **R2 disposable cleanup** | `gc_older_than`, `gc_store` | helper | `--dry-run` dead-path count, plus a size the helper computes and the generation list it resolved (see below) | **none - one-way** |
| **R3 declarative config change** | `build` (unprivileged), `dry_activate`, `activate(toplevel)`, `rollback(generation)` | helper for the last three | `nix store diff-closures` + `dry-activate` unit list | switch to a recorded generation |
| **R4 irreversible / refused** | *(no verb exists)* | - | - | - |

R0 needs no privilege work: units, logs, generations, closure diffs and storage
were all measured readable as `alex` (see SPIKE.md). Task 20260729-125024 is
therefore unblocked and can run in parallel with the helper.

R4 - partitioning and filesystem formatting, user and group changes, key
material (age keys, sops secrets, ssh keys), disabling the firewall, and
anything targeting `scufris-hostd`, the `scufris` unit or the auth secret - is
enforced by **absence of a verb**, not by a deny check.

Two constraints on R1: a deny-list of units that would take out the operator's
remote access or the approval path itself (`sshd`, `dbus`, `systemd-logind`,
`NetworkManager`, `scufris`, `scufris-hostd`), and refusal to act on the helper
or on scufris regardless of that list.

One constraint on R2: never a bare `nix-collect-garbage -d`. Only
`--delete-older-than <N>d`, with a floor that keeps the current and the
immediately previous generation, so the one-way class cannot destroy the
rollback target of the R3 class.

The flag does NOT provide that floor and must not be trusted to: measured,
`--delete-older-than` keeps the current generation and is otherwise purely
age-based, so a previous generation older than N days is deleted - exactly the
R3 rollback target this constraint exists to protect. The HELPER enforces the
floor: it resolves the generation list first, and refuses or clamps a request
that would remove either of the two most recent generations.

### 3. Preview: the builtin closure diff, and honesty where there is no preview

R3 previews with `nixos-rebuild build --flake <ref>` (unprivileged) plus
`nix store diff-closures /run/current-system ./result`. `nix store
diff-closures` is builtin to nix 2.34.8 on this host and `nvd` is not
installed, so the builtin is used and no dependency is added to `nix.dotfiles`.
`dry_activate` (root, therefore a helper verb) adds the list of units that
would restart.

A trap for 20260729-125035, measured on this host: when the built toplevel
matches the running system, `nix store diff-closures` exits 0 and prints
NOTHING AT ALL, so "no change" and "the command failed" are byte-identical in
its output. The approval surface must check the exit status first and render an
explicit "no closure change" - never an empty panel that could equally mean the
preview broke.

R2 previews with the real `--dry-run` output, with one honesty correction:
measured, `nix-collect-garbage --dry-run --delete-older-than 3650d` prints a
dead-path COUNT ("7642 store paths would be deleted") and neither a byte total
nor a generation list. The helper therefore computes the reclaimable size
itself (`nix path-info -S` over the dead set) and lists the generations it
resolved. Presenting the raw count as "space freed" would be the same failure
this decision rejects elsewhere - adjacent information dressed up as a
preview.

R1 has **no honest preview** - a restart cannot be simulated. It shows current
unit state plus reverse dependencies, explicitly labelled as a statement of
state and blast radius rather than a simulation. Where a class has no honest
preview, the UI says so rather than presenting adjacent information as one.

### 4. Rollback: generations for R3, recorded state for R1, one-way declared for R2

R3 records the generation number and the toplevel store path in the audit entry
at apply time, which is what makes rollback a targeted activation rather than a
guess. R1 records `ActiveState`/`SubState` before acting and offers restoration
of that state as the inverse (acknowledging a restarted daemon has lost
in-memory state). R2 is declared one-way and therefore takes a **stronger**
confirmation than the reversible classes, not a weaker one.

### 5. Arbitrary shell: no. Typed verbs only.

There is no shell verb, at any privilege level, under any approval. This
extends the stance already written into `scufris/mcp_common.py` ("the allowlist
IS this set of handlers - there is no generic 'run any command' tool").

Rationale, recorded so it is not re-litigated: a root shell verb would make the
taxonomy, the previews, the store-path binding and the audit decorative in a
single call; it has no possible preview by construction, breaking the epic's
central contract at first use; and it is the most attractive prompt-injection
target in the system. The long tail is covered by adding a verb with a test and
a review - that friction is the point. The operator's escape hatch is a
terminal, which they already have.

### 6. Config repository: a sprout worktree, committed before it is built

Host config changes are proposed in a sprout worktree over
`~/personal/nix.dotfiles`, committed on that branch, and built from that
commit. The operator's own checkout is never touched - it is dirty right now,
which is exactly the case in-place editing handles badly. Building from a
commit rather than a dirty tree means what gets activated is an identified
revision rather than an unreproducible snapshot. Merging the branch back into
the config repo's default branch is a separate operator act, mirroring how
scufris's own branches land.

### 7. Audit: append-only, root-written, independent of the app's store

The privileged half of the record (approved / applied / command / result /
generation / toplevel) is appended by the helper as root to its own
append-only log, so it survives the app being the thing that misbehaved. It
deliberately does **not** share the transactional store from 20260729-102147:
that task is still OPEN, the epic should not inherit a dependency it does not
need, and the requirements differ (app stores are mutable, multi-writer and
app-owned; this log is append-only, single-writer and must be trustworthy when
the app is not). The request side - proposed and denied - is app-side state and
may live wherever the app's state lives. The dashboard reads the log for
display.

**Retention.** The log is append-only and root-owned, so the app cannot trim
it and an unbounded file is the default failure. The helper rotates its own
log: one active file plus a bounded number of rotated ones, rotating on size
rather than on a timer so a burst cannot outrun the policy, with the rotated
set pruned oldest-first. Entries are single-line JSON, so a rotation boundary
never splits a record. Nothing outside the helper may delete or rewrite them -
in particular there is no verb for pruning the audit log, because a verb that
erases the record of privileged actions is precisely what an R4 refusal is
for. The concrete size and count are a knob for 20260729-125029 to pick and
put in the module option; the shape (helper-owned, size-rotated, oldest-first,
no external pruning path) is decided here.

### 8. Approval is an operator act, authenticated by the session

Approval requires a real operator session and CSRF token from the mechanism
landed in 20260729-125015.

**This is a REQUIREMENT on the implementing task, not a property the code
already has.** As it stands, `scufris/app.py:840-844` short-circuits the
middleware on a valid bearer token BEFORE the session lookup and before the
CSRF and same-origin checks, for every non-public path and every method. The
app's own MCP tool subprocesses hold exactly that token, so on today's code an
agent could call an approval endpoint. 20260729-125029 must therefore reject
bearer-token authentication explicitly on the approval endpoint and require a
session plus CSRF, with a test that a machine-token approval is refused. An
agent must never be able to approve its own proposal, and nothing in the
current middleware secures that.

## Consequences

- 20260729-125024 (read-only inspection) is unblocked immediately and needs no
  helper.
- 20260729-125029 grows the `scufris-hostd` helper, its typed protocol, its
  packaging in the flake and its NixOS module option, alongside the
  proposal/preview/approval/audit framework.
- Enabling host agency becomes an explicit, diffable act in `nix.dotfiles`
  (turning the helper's module option on), not something a scufris upgrade can
  silently acquire.
- Adding a capability later is a reviewed code change with a test, never a
  configuration line.
- Deferred to the implementing tasks: whether the helper socket also requires a
  shared secret from the sops dotenv (raises the bar against a different
  process of the same user, without pretending to be a boundary); whether
  `dry_activate` is worth a verb in v1; and how a Telegram approval proves
  operator identity (20260729-125046).
- Pre-existing and out of scope here: `alex` is in the `docker` group, which is
  root-equivalent, so the machine's real privilege boundary is weaker than the
  one this epic builds. Fixing it is a change in `nix.dotfiles` (rootless
  docker, or dropping the group) and is the operator's call.
