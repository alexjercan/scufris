# Spike: what may Scufris do to this machine, and through what privilege?

- DATE: 20260729-125020
- STATUS: RECOMMENDED
- TAGS: spike, host, nixos, security, v0.2.0

## Question

The host-operator epic (20260729-124655) wants every mutating host action to
follow one contract:

    propose -> preview -> approve -> apply -> audit -> roll back

Three uncertainties gate every task under it, and this spike exists to reduce
them:

1. **Privilege.** Scufris is a systemd USER service running as `alex`.
   `nixos-rebuild switch`, `systemctl restart` on system units, and
   `nix-collect-garbage` on the system profile are root operations. Something
   has to bridge that gap, and the choice sets the blast radius of everything
   downstream.
2. **The action surface.** Which host actions exist at all, what each one's
   honest preview is, and how each one is reversed.
3. **The shell line.** Whether a typed-action allowlist is the only path, or a
   free-form command escape exists.

A good answer names ONE privilege mechanism concretely (with its failure mode
and what an attacker actually reaches), classifies the actions with a preview
and a reversal each, and says yes or no to arbitrary shell in writing.

## Context

### What exists in Scufris

- **Host surface is read-only and small.** `PsutilCollector`
  (`scufris/metrics.py`) and `PsutilProcessCollector` (`scufris/processes.py`),
  exposed as `host_stats`, `disk_usage`, `list_processes` in
  `scufris/mcp_server.py`. Nothing mutates the host.
- **The shell wrapper is already an allowlist.** `_run` in
  `scufris/mcp_common.py:35` takes a fixed argument list, `shell=False`,
  resolves the executable on PATH, 15s timeout, output capped at 20k. The
  module docstring states the stance outright: "The allowlist IS this set of
  handlers - there is no generic 'run any command' tool." This spike extends
  that stance rather than inventing one.
- **Three single-audience MCP servers** (tasks/20260727-105609/DECISION.md):
  `scufris` (orchestrator agentic), `den` (life tools), `agent` (sub-agent
  callbacks). A host surface has to pick one; only an ORCHESTRATOR turn
  registers `scufris`, so tools there are never advertised to a sub-agent.
- **Permission modes already exist** as a vocabulary: `manual` / `edit` /
  `auto` (`scufris/enums.py:60`), scoped to a project working tree, not to the
  host.
- **Authentication landed** (20260729-125015): opaque session id in an HttpOnly
  cookie over a revocable server-side record, per-session CSRF token, one
  deny-by-default middleware, and a per-process bearer token for the app's own
  MCP subprocesses (`scufris/auth.py`). Without this, no approval surface could
  mean anything - anyone on the LAN would have been the approver.
- **The supervisor** owns background runs with cancel and heartbeat
  (`scufris/supervisor.py`), and the **Telegram bridge** already drives the same
  orchestrator turn path, so an approval can reach the operator off-dashboard.

### How this box is actually deployed

From `~/personal/nix.dotfiles` (flake-parts, home-manager, sops-nix,
`hosts/nixos`):

- `programs.scufris` (home-manager, `nix/scufris-service.nix` with
  `isNixos = false`) - a **user** unit running as `alex`, `After=sops-nix`,
  `Restart=on-failure`, bound to `0.0.0.0:8000`.
- Reachable from the LAN by an explicit rule in
  `hosts/nixos/default.nix:271` (`192.168.0.0/24 -> 8000`).
- The same module can produce a NixOS **system** service (`isNixos = true`,
  `DynamicUser = true`) - that variant exists but is not what runs here.
- Secrets ride a sops dotenv (`secrets/scufris.env`) decrypted into
  `$XDG_RUNTIME_DIR` at activation.

### What the operator account can do today (measured, not assumed)

Probed on this host as `alex`:

| Capability | Result |
|---|---|
| `id` | `wheel`, `docker`, `libvirtd`, `networkmanager`, `audio`, `video`, `dialout` |
| `sudo -n -l` | **password required** - a non-interactive service cannot sudo |
| `security.sudo.extraRules` in `hosts/nixos` | **none** |
| `journalctl -u sshd` | **works** (wheel gets journal ACLs) |
| `nixos-rebuild list-generations` | **works** |
| `nix store diff-closures <gen> <gen>` | **works** |
| `systemctl show sshd -p ActiveState` | **works** |
| `nvd` on PATH | **missing** |
| `pkexec` present, `polkit-agent-helper-1` | present / **missing** |
| nix version | 2.34.8 |

Two consequences fall straight out of that table:

1. **The entire read-only inspection task (20260729-125024) needs no privilege
   work at all.** Units, logs, generations, closure diffs and storage are all
   readable as `alex`. It can be built the moment this spike lands, in parallel
   with the privileged machinery.
2. **`nvd` is not installed, but `nix store diff-closures` is built into nix
   2.34.** The config-change preview should use the builtin and add no
   dependency to `nix.dotfiles`.

### The uncomfortable fact that shapes the threat model

`alex` is in the **`docker` group**, which is root-equivalent on any machine
running the daemon (`docker run -v /:/host --privileged` is a root shell), and
in `libvirtd`, which is close behind. The password prompt on `sudo` is
therefore not a real security boundary for anything already running as `alex` -
it is a boundary against *non-interactive* processes only, which happens to
include the scufris service, but any code running as `alex` can trivially step
around it.

This must be said out loud because it changes what the controls in this epic
are FOR. They are not a defence against an attacker who owns the operator's
account; that attacker has already won, today, with or without this epic. They
are a defence against:

- the model taking an action nobody asked for (hallucinated, or
  over-enthusiastically inferred),
- a prompt-injected agent turning a benign request into a destructive one,
- the operator approving something whose consequences were not visible,
- and the absence of a record afterwards.

Writing that distinction down is itself part of the deliverable: a control that
is sold as stopping a determined attacker, and does not, is worse than one that
is honest about its scope.

## Options considered

### Privilege boundary

**A. Targeted `sudo` NOPASSWD rules declared in `nix.dotfiles`.**

`security.sudo.extraRules` granting `alex` passwordless sudo for a fixed list
of commands (`nixos-rebuild switch --flake ...`, `systemctl restart <unit>`,
`nix-collect-garbage --delete-older-than ...`).

- Pros: smallest possible change; declarative and reviewable in the config
  repo; no new process or protocol; fails closed in the sense that an
  unlisted command still prompts for a password and therefore fails in a
  service context.
- Cons, and they are the deciding ones:
  - sudo's command matching is an **argv-prefix allowlist**, historically the
    weak kind. Any rule permissive enough to be useful is permissive enough to
    be abused: `nixos-rebuild switch --flake <anything>` is literally "run this
    attacker-chosen Nix expression as root", and pinning the flake ref in
    sudoers makes the rule useless the moment the path or the attribute
    changes.
  - The grant is to the **user**, not to scufris. Every process running as
    `alex` inherits it, including an interactive shell and every agent CLI
    subprocess scufris spawns. It converts a password prompt that today stops
    non-interactive code into no prompt at all.
  - Nothing binds the approved preview to the applied action. Between "operator
    approved this closure diff" and "sudo nixos-rebuild switch", the tree can
    change and a different system gets activated. The approval is advisory.
  - The audit record would be written by the same uid that performs the
    action, so it is rewritable by whatever went wrong.

**B. A privileged helper as a NixOS system unit with a narrow typed IPC.**

A `scufris-hostd` system service running as root, listening on a unix socket
(`0660`, group-restricted), speaking a small typed JSON protocol with a closed
set of verbs. The helper - not the caller - constructs every argv, validates
every argument, streams output back, and writes the audit record itself.

- Pros:
  - The boundary is a real process boundary with a **typed** protocol. There is
    no argv to smuggle through; there are only verbs, and a verb that does not
    exist cannot be requested.
  - The helper can enforce invariants sudo structurally cannot. The decisive
    one: **an approval can be bound to the exact store path that was
    previewed.** `activate(toplevel = /nix/store/<hash>-nixos-system-...)` with
    the helper refusing any path it did not itself build (or that does not
    match the approval record) closes the preview-to-apply TOCTOU window that
    option A leaves open.
  - The **audit log is written by root** and is not rewritable by the app, so
    the record survives the thing it is recording.
  - Adding a capability is a reviewed code change with a test, not a sudoers
    line - which is the right amount of friction for "what may this machine be
    told to do".
  - Fails closed at every layer: socket absent (helper not enabled) means no
    privileged action is possible at all; unknown verb is rejected; malformed
    request is rejected; the helper has no shell verb to fall back to.
  - It is declared in `nix.dotfiles` next to `programs.scufris`, so enabling
    host agency stays an explicit, diffable operator act.
- Cons: a second unit, a protocol, packaging and tests. More code than A. (Per
  the repo's own rules, "more code" is not by itself an argument against the
  correct design.)
- Honest limit, which applies to A equally: the socket is reachable by anything
  running as `alex`. The helper does **not** distinguish scufris from any other
  process of the same user, and pretending otherwise would be the exact kind of
  dishonest control described above. Its value is that only typed, validated,
  approval-bound, root-audited verbs exist at all - not that only one caller
  can speak them.

**C. polkit rules.**

A polkit JS rule allowing `alex` to manage specific units without
authentication.

- Pros: the native mechanism for systemd unit management; per-unit granularity;
  no sudoers.
- Cons: covers **only** the systemd slice. `nixos-rebuild` is not
  polkit-mediated at all - it shells out to sudo internally - so polkit cannot
  express the flagship action of this epic and would have to be paired with A
  or B anyway. Same user-wide grant problem as A. And no polkit agent is
  installed on this box (`polkit-agent-helper-1` missing), so anything needing
  interactive authentication cannot work in a headless service; only
  unconditional-YES rules would function, which is the most permissive form.
  Rejected as a *whole* answer; may return later as a narrowing detail.

**D. Run scufris itself as the NixOS system service (root).**

The module already supports it (`isNixos = true`).

- Pros: everything just works; zero IPC.
- Cons: it hands root to the process containing the LLM, the LAN-facing HTTP
  surface, and every agent CLI subprocess. It inverts the epic - the point is
  to constrain what the assistant may do, not to remove the constraint. It also
  breaks the current deployment (`DynamicUser` has no operator home, no sops
  user secrets, no access to `~/personal`). Rejected.

**E. Do nothing - stay read-only.**

- Pros: zero new risk; the read-only inspection task alone already delivers
  most of the "know this machine" differentiator, and needs no privilege.
- Cons: does not deliver the epic. Worth naming because it is the fallback if
  the operator does not accept a privilege model - and because it is genuinely
  *partially* available: inspection ships either way.

### Preview mechanism

- **Config change**: `nixos-rebuild build --flake <ref>` (unprivileged, no root
  needed) followed by `nix store diff-closures /run/current-system ./result`.
  Builtin to nix 2.34; no `nvd` dependency. This is the honest preview - it
  shows package version deltas and closure size change before anything is
  activated. `nixos-rebuild dry-activate` additionally shows which units would
  restart, but needs root, so it belongs to the helper as its own verb.
- **Service control**: there is **no honest dry-run for a restart**. The
  candidates were `systemctl show` (current state) and
  `systemctl list-dependencies --reverse` (blast radius). Both are informative;
  neither predicts the outcome. The right answer is to show both and say
  plainly that the preview is a statement of current state and dependents, not
  a simulation.
- **Cleanup**: `nix-collect-garbage --dry-run` and `nix store gc --dry-run`
  print what would be deleted and how much would be freed. This is a genuine
  preview.
- **Refused class**: no preview, because there is no action.

### Rollback

- **Config change**: NixOS generations. `nixos-rebuild switch --rollback`, or
  activate a specific recorded generation. The generation number and the
  toplevel store path are recorded in the audit entry at apply time, which is
  what makes rollback a targeted operation rather than a guess.
- **Service control**: record `ActiveState` / `SubState` before acting and
  offer "restore prior state" as the reversal. Not perfect (a restarted daemon
  has lost its in-memory state) but it is the true inverse of the unit-level
  change.
- **Cleanup**: **one-way**. Deleted generations do not come back. Store paths
  are re-buildable in principle, but that is not a rollback. This class needs a
  stronger confirmation *because* it cannot be undone, not a weaker one because
  it feels routine.
- **Refused**: not applicable.

### Arbitrary shell

- **Allowlist only.** Every capability is a named verb. The operator's escape
  hatch is a terminal, which they already have.
- **Free-form escape under stronger approval.** Tempting because it covers the
  long tail without a code change per verb.
  - Rejected. A root shell verb makes every other control in this epic
    decorative: the taxonomy, the previews, the store-path binding and the
    audit all exist to constrain what can be requested, and a passthrough
    restores the unconstrained set in one call. It also has no preview by
    construction - "here is a command string" is not a preview - which breaks
    the epic's central contract at the first use. And it is the single most
    attractive target for prompt injection in the whole system.
  - The long-tail argument is real but the answer to it is "add a verb, with a
    test and a review", which is the friction we want, not a hole we keep open
    for convenience.

### Where the config repository is edited

- **In place in `~/personal/nix.dotfiles`.** Simple, but the agent then fights
  the operator for the working tree - and that tree is dirty *right now*
  (uncommitted `sops` in `systemPackages` and a modified `secrets/scufris.env`,
  from the auth task's pending operator step). An agent editing in place would
  either build the operator's half-finished work into a system generation or
  refuse to act until the operator tidied up.
- **A sprout worktree over the config repo.** Reuses machinery the project
  already depends on, isolates the proposal on a branch, and leaves the
  operator's checkout untouched no matter what state it is in.
  - One wrinkle worth stating: building a flake from a dirty worktree is
    possible but not reproducible, and what gets activated should be an
    identified commit. So the proposal is **committed on the sprout branch
    first**, then built from that commit, then previewed, then switched. The
    merge back into the config repo's default branch is a separate operator
    act, exactly as with scufris's own branches.

### Audit storage

- **Share the transactional store from 20260729-102147.** Consistent with the
  rest of the app's state - but that task is still OPEN at p80, so this epic
  would inherit a dependency it does not need, and the requirements differ: the
  app stores are mutable, multi-writer and app-owned; the audit log is
  append-only, effectively single-writer, and must remain trustworthy when the
  app is the thing that misbehaved.
- **Its own append-only log, written by the helper as root.** Independent,
  survives the app, no dependency on 102147, and readable by the dashboard for
  display. The request side (proposed / denied) is app-side state and may live
  wherever the app's state lives; the privileged side (approved / applied /
  result / generation) is written by root.

## Recommendation

**Option B: a privileged helper system unit with a narrow typed IPC, and no
sudo rules at all.** Sudo on this box stays password-required; the only new
privileged surface is a closed verb set.

It beats A because it is the only option that can bind an approval to the exact
artifact that was previewed, construct its own argv instead of trusting a
prefix match, and write an audit record the app cannot rewrite. It beats C
because C cannot express `nixos-rebuild` at all and this box has no polkit
agent. It beats D because handing root to the process containing the model
inverts the epic. E remains the honest fallback, and its most valuable half -
read-only inspection - ships regardless.

The verb set IS the action taxonomy:

| Class | Verbs | Privilege | Preview | Reversal |
|---|---|---|---|---|
| **R0 read-only** | units, logs, storage, network, sensors, packages, generations | none (`alex`) | n/a | n/a |
| **R1 service control** | `unit_start`, `unit_stop`, `unit_restart`, `unit_reload` | helper | `systemctl show` + reverse dependencies, labelled "state, not simulation" | restore recorded prior state |
| **R2 disposable cleanup** | `gc_older_than`, `gc_store` | helper | `--dry-run` output, bytes freed, generations removed | **none - one-way** |
| **R3 declarative config change** | `build` (unprivileged), `dry_activate`, `activate(toplevel)`, `rollback(generation)` | helper for the last three | `nix store diff-closures` + `dry-activate` unit list | switch to recorded generation |
| **R4 irreversible / refused** | *(no verb exists)* | - | - | - |

R4 is enforced by absence, not by a check: partitioning and filesystem
formatting, user/group changes, key material (age keys, sops secrets, ssh
keys), disabling the firewall, and anything targeting the helper's own unit,
the scufris unit, or the auth secret. Absence is the strongest form this can
take - there is no code path to audit for a bug.

Two constraints on R1 that fall out of the helper owning argv: a deny-list of
units that would take the operator's remote access or the approval path itself
down (`sshd`, `dbus`, `systemd-logind`, `NetworkManager`, `scufris`,
`scufris-hostd`), and refusal to act on the helper or on scufris regardless of
the list.

One constraint on R2: never a bare `nix-collect-garbage -d`. Only
`--delete-older-than <N>d` with a floor that keeps the current and the
immediately previous generation, so the rollback target of the *other* class
cannot be destroyed by this one.

The approval itself is an operator act authenticated by the session that
20260729-125015 landed. An agent must never be able to approve its own
proposal; the approval endpoint requires a real session and a CSRF token, which
the machine token deliberately does not satisfy.

## Open questions

- **`alex` is in the `docker` group, which is root-equivalent.** This does not
  block the epic - it is pre-existing, and it is honestly accounted for in the
  threat model above - but it means the machine's real privilege boundary is
  weaker than the one this epic builds. Resolving it is a change in
  `nix.dotfiles` (rootless docker, or dropping the group), not in scufris, and
  it is the operator's call. Worth a task in that repo.
- **Caller authentication on the helper socket.** Peer credentials (`SO_PEERCRED`)
  can confirm the uid, which is all the OS can offer while scufris runs as
  `alex`. Whether to add a shared secret from the same sops dotenv (raising the
  bar for a *different* process of the same user without pretending to be a
  boundary) is a detail for the implementing task, not a spike-level fork.
- **Whether `dry_activate` is worth a verb in v1** or whether the closure diff
  alone is a sufficient preview. Leaning yes, because "which units restart" is
  the question an operator actually asks - but it can be deferred without
  changing any interface.
- **Digest delivery** (20260729-125046) will want an approval that arrives over
  Telegram. The session-authenticated approval above is dashboard-shaped; how a
  Telegram approval proves operator identity is a real question, deferred to
  that task since the allowlisted chat id is already the bot's whole auth model.

## Next steps

This spike refines the existing children of epic 20260729-124655 rather than
seeding new tasks; the epic's plan already carries the work forward:

- 20260729-125024 - expand read-only host inspection beyond stats. **Unblocked
  now**: needs no privilege, no helper, and can proceed in parallel.
- 20260729-125029 - the host action framework (proposal, preview, approval,
  audit) plus the `scufris-hostd` helper and its typed protocol.
- 20260729-125035 - the NixOS config change flow (R3) and generation rollback.
- 20260729-125040 - the host operator agent and its approval surfaces.
- 20260729-125046 - scheduled host checks and the proactive digest.

The accepted choices are recorded in `DECISION.md` next to this file.

## Fix record

This spike seeds no new tasks, but it refines five children of epic
20260729-124655 and their state is what a later cycle needs. Each appends a
line here as it lands.

- 20260729-125024 read-only host inspection - OPEN. Unblocked outright by this
  spike; needs no privilege.
- 20260729-125029 host action framework + `scufris-hostd` - OPEN.
- 20260729-125035 NixOS config change flow + rollback - OPEN.
- 20260729-125040 host operator agent + approval surfaces - OPEN.
- 20260729-125046 scheduled host checks + digest - OPEN.
