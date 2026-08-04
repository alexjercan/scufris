# `packages/hostd/` - the privileged helper

`scufris-hostd` is a separate process, running as **root**, that is the only
thing in Scufris able to change the machine. It speaks a closed set of typed
verbs over a unix socket, builds every command itself, holds every proposal, and
writes its own append-only audit log.

There are **no sudo rules**, and there is **no shell verb at any privilege under
any approval**. Adding a capability is a reviewed code change with a test, never
a configuration line.

- The app-side of this contract (who may propose, who may approve, how the
  decision reaches both surfaces) is in
  [`scufris/README.md`](../../../../scufris/README.md).
- The reasoning behind the privilege model is
  [`tasks/20260729-125020/DECISION.md`](../../../../tasks/20260729-125020/DECISION.md);
  the R3 half is
  [`tasks/20260729-125035/DECISION.md`](../../../../tasks/20260729-125035/DECISION.md).

## 1. Enabling it

Its own NixOS module, deliberately separate from the app's, so granting host
agency is a diffable act in your configuration and never something a scufris
upgrade quietly acquires:

```nix
{
  imports = [inputs.scufris.nixosModules.scufris-hostd];

  services.scufris-hostd = {
    enable = true;
    group = "scufris";                              # a DEDICATED group, not `users`
    secretFile = config.sops.secrets."scufris-hostd-secret".path;
  };
}
```

| Option | Default | What it is |
|---|---|---|
| `enable` | `false` | Turns the root unit on |
| `package` | `scufris.packages.<system>.scufris-hostd` | Provides the `scufris-hostd` binary |
| `socketPath` | `/run/scufris-hostd/hostd.sock` | Where it listens. Must match `SCUFRIS_HOSTD_SOCKET` in the app |
| `group` | (required) | The group given read/write on the socket - the group the scufris service runs as |
| `secretFile` | (required) | A file holding the shared secret, read as root at start. A string path, never a nix path literal: a `./secret` would be copied into the world-readable store |
| `auditLog` | `/var/log/scufris-hostd/audit.jsonl` | The append-only record |
| `auditMaxBytes` | 16 MiB | Rotate once the log would exceed this |
| `auditKeep` | `5` | Files kept in total (active plus rotated). With the defaults the log is bounded at 80 MiB |
| `ttlSeconds` | `600` | How long an operator has to approve a proposal before it goes stale |

Use a **dedicated group**. `users` is the default primary group of every normal
account on NixOS, and reaching the socket is the first step of every attack the
secret exists to slow down.

Two preconditions, both fail-closed:

1. `secretFile` must exist and be non-empty, or the helper refuses to start. A
   root socket with no credential in front of it is reachable by anything running
   as the operator - including the agent CLI subprocesses scufris itself spawns,
   which run arbitrary shell.
2. The same secret must reach the app as `SCUFRIS_HOSTD_SECRET`. Without it the
   app answers every mutating host endpoint with "not configured"; with it, an
   operator password becomes mandatory even on loopback.

Stated plainly, because it must not be oversold: **the secret raises the bar
against the model acting unasked. It is not a boundary against a compromised
operator account**, which on this machine is already root-equivalent through the
`docker` group.

The unit runs with `pkgs.nix`, `pkgs.systemd` and `pkgs.nixos-rebuild` on its
PATH - deliberately not coreutils - so the helper acts with the same tools the
operator would use by hand.

### Running it by hand

The module's `ExecStart` is just the CLI, so a manual run for testing is the same
thing with a socket somewhere writable:

```sh
scufris-hostd \
  --socket /tmp/hostd.sock \
  --secret-file /tmp/hostd.secret \
  --audit-log /tmp/audit.jsonl \
  --group "$(id -gn)" \
  --ttl-seconds 600 \
  --log-level DEBUG
```

`--audit-max-bytes` and `--audit-keep` complete the set. Nothing else is
configurable, on purpose.

## 2. The socket language

Newline-delimited JSON over a unix socket, mode `0660` (root writes, the
configured group reads). **One request per connection.** The response is one or
more frames ending in exactly one terminal frame. Records are single-line, so a
reader never buffers for a frame boundary.

Two properties come from the shape of the models rather than from checks
scattered around the server (`protocol.py`):

- **A caller names a verb, never a command.** No request has an argv field, and
  `ActionKind` is a closed enum - an unknown verb fails to parse at this boundary
  and never reaches the code that builds commands.
- **Every frame carries the secret.** Not a handshake a later frame can skip, so
  a connection cannot be escalated after the fact. Comparison is
  constant-time.

A request larger than 64 kB is not a request this helper has a verb for, and is
refused as such.

### Verbs

These are the PROTOCOL verbs (what a caller may ask the helper to do), which are
not the same list as the ACTION verbs in section 3 (what an action may be).

| Verb | Fields it reads | What it does |
|---|---|---|
| `hello` | - | Returns the protocol version and the action verbs this build implements |
| `list_pending` | - | Every proposal still PENDING, oldest first. Read-only: builds no argv, names no id, changes nothing. This is how the app rebuilds its queue after a restart |
| `propose` | `kind`, `args`, `requester` | Validates, builds the plan and the preview, registers a PENDING proposal, returns it |
| `apply` | `proposal_id`, `approved_by` | Executes the proposal's steps in order, streaming output, then appends the audit record |
| `deny` | `proposal_id`, `approved_by`, `reason` | Terminal: the operator said no |
| `cancel` | `proposal_id` | Stops an apply in flight (or drops a pending proposal) |
| `audit_tail` | `limit` (1-500, default 50) | The last N audit records |

Every request also carries `secret`. `requester` is `{actor, agent, run}` - who
asked; the app fills it from the CREDENTIAL, never from a request body.
`approved_by` is the operator identity the app reports after its own session
check: the helper records it and does not claim to have verified it. What it
verifies is that the action being applied is the one it previewed.

### Frames

| Frame `type` | Sent for | Carries |
|---|---|---|
| `hello` | `hello` | `version`, `verbs` |
| `pending` | `list_pending` | `proposals` |
| `proposal` | `propose` | one `ProposalView` |
| `output` | during `apply` | `stream` (`stdout`/`stderr`) and `text`, live |
| `result` | terminal frame of `apply` | `ok`, `outcome`, `returncode`, `duration_seconds`, `steps_completed`/`steps_total`, `reversal`, `detail` |
| `audit` | `audit_tail` | `records` |
| `error` | any refusal | `code`, `detail` |

Error codes: `unauthorized`, `bad_request`, `refused` (the helper will not do
this at all), `not_found`, `expired`, `drifted`, `already_used`, `internal`.

### A proposal

`propose` answers with a `ProposalView`, and that object IS what the operator
approves:

| Field | Meaning |
|---|---|
| `id`, `kind`, `risk`, `args` | which action, in which risk class |
| `steps` | **every command this action would run, in order**, each with a `label` and a `timeout`. A caller approves THESE, not a description of them |
| `summary` | one line in the operator's language |
| `preview` | `kind` (`simulation` / `state` / `none`), a `label` saying what the lines ARE, the `lines`, and an `Availability`. A `simulation` is the system's own answer to "what would happen"; a `state` preview describes the world as it is now and lets the operator infer the consequence. Rendering the second as the first is the failure this taxonomy exists to prevent |
| `reversal` | `possible`, a `summary`, and the inverse action's `kind`/`args` when there is one. `possible=false` is a first-class answer, not a missing value |
| `fingerprint` | the state the preview was taken against, re-read at apply time |
| `requester`, `created_at`, `expires_at`, `state` | provenance and lifecycle |

`state` moves through `pending` -> `applying` -> `applied` / `failed`, or to
`cancelled` / `expired` / `drifted`. Every state but `pending` is terminal;
`applying` is the atomic claim, which is what makes two approvals racing on one
id produce one execution and one refusal.

### An example exchange

```jsonc
// -> propose
{"verb":"propose","secret":"...","kind":"unit_restart",
 "args":{"unit":"nginx"},"requester":{"actor":"agent:host","agent":"host"}}

// <- proposal (one line, expanded here for reading)
{"type":"proposal","proposal":{
  "id":"01J...","kind":"unit_restart","risk":"r1",
  "args":{"unit":"nginx.service"},
  "steps":[{"argv":["systemctl","restart","--","nginx.service"],
            "label":"restart nginx.service","timeout":120.0}],
  "summary":"restart the nginx.service unit",
  "preview":{"kind":"state","label":"current state and blast radius","lines":["..."]},
  "reversal":{"possible":false,"summary":"a restart cannot be un-restarted"},
  "state":"pending","expires_at":1753900000.0}}

// -> apply (a new connection, the secret again)
{"verb":"apply","secret":"...","proposal_id":"01J...",
 "approved_by":"operator:telegram:123456789"}

// <- output frames, then exactly one result
{"type":"output","stream":"stderr","text":"..."}
{"type":"result","proposal_id":"01J...","ok":true,"outcome":"ok",
 "returncode":0,"steps_completed":1,"steps_total":1}
```

`scufris_hostctl`'s `client` is the app's side of this: connect, send one authenticated
request, read frames until the terminal one. An apply is a stream that can be
cut, which is how a stop button works.

## 3. The action verbs

The verb set IS the risk taxonomy (`actions/`). R0 needs no privilege and lives
in [`packages/host/`](../../../host/src/scufris_host/README.md).
**R4 is enforced by no verb existing** -
partitioning, users, key material, the firewall and anything targeting scufris
itself have no code path here, rather than a deny check that could have a bug.

| Verb | Risk | Arguments | Runs | Reversal |
|---|---|---|---|---|
| `unit_start` | R1 | `unit` | `systemctl start -- <unit>` | the recorded prior unit state |
| `unit_stop` | R1 | `unit` | `systemctl stop -- <unit>` | as above |
| `unit_restart` | R1 | `unit` | `systemctl restart -- <unit>` | often none, and that is NORMAL |
| `unit_reload` | R1 | `unit` | `systemctl reload -- <unit>` | as above |
| `gc_older_than` | R2 | `days` (1-3650) | `nix-env --profile <system> --delete-generations <numbers>` | none. ONE-WAY |
| `gc_store` | R2 | - | `nix store gc` | none. ONE-WAY |
| `activate` | R3 | `toplevel`, `repo`, `rev` | two steps: point the system profile at the toplevel, then run its `switch-to-configuration switch` inside a transient systemd unit | roll back to the generation it replaced |
| `rollback` | R3 | `generation` (a NUMBER) | the same two steps, aimed at that generation | activate the generation it came from |

R2 and R3 are multi-step or destructive, so a few details are load-bearing:

- **`gc_older_than` names the generations in the argv.** It deliberately does not
  use `nix-collect-garbage --delete-older-than`, which is purely age-based and
  keeps only the CURRENT generation - on a box whose previous generation is older
  than the cutoff, that flag deletes the exact rollback target the preview said
  would be kept. Naming the numbers makes the command and the preview the same
  statement. The two most recent generations are excluded by POSITION before age
  is considered, and a generation whose date cannot be parsed is kept.
- **A collection that would delete nothing is refused.** An operator should never
  be asked to approve an empty act.
- **`activate` cannot be proposed by a caller.** Its argument is a store path,
  and a caller who chose that path would be choosing what the machine boots while
  the closure diff faithfully described their choice. The only code path that
  reaches it builds the path itself from a resolved git revision
  (`../hostconfig/`), both propose surfaces refuse `kind=activate` outright, and
  `actions/validate.py` still validates the path structurally: a store-path ROOT (not a
  subpath), known and valid in this store, carrying a `nixos-version` (which is
  `nixos-rebuild`'s own precondition, and skipping it is how a machine ends up
  unable to boot) and a `bin/switch-to-configuration`.
- **`rollback` names a number, never a path**, and the helper resolves which
  store path that generation is - so "roll back" cannot be spelled as "activate
  this other thing". `nixos-rebuild --rollback` ("whatever is previous") is
  deliberately not used.
- **The switch runs in a transient systemd unit**, the same
  `nixos-rebuild-switch-to-configuration` name `nixos-rebuild` itself uses. It has
  to survive restarting the units scufris runs in - and systemd refusing a second
  `systemd-run --unit=<name>` while one is live means a hand-run `nixos-rebuild
  switch` and this helper cannot activate two configurations at once. The
  collision is a feature.
- **`activate` and `rollback` have a real partial state.** If step 2 fails after
  step 1 succeeded, the profile points at the new configuration while THIS boot
  still runs the old one: roll back or fix the activation and switch again,
  before rebooting. `steps_completed < steps_total` is how the record says so.

### What a configuration preview does NOT include

The unit-restart list. Producing it means running
`<toplevel>/bin/switch-to-configuration dry-activate` - as root, from a
configuration nobody has approved yet. So the preview shows the `nix store
diff-closures` output, the generation and store path the system is on now, and
the revision it was built from, and it says why the unit list is absent. Read the
commit's diff in git for what changed; that is the real review surface.

(`diff-closures` prints nothing and exits 0 for identical closures, so "no
closure change" is stated explicitly rather than shown as empty output.)

## 4. What it refuses, and why there

Validation raises `ActionRefused`; by the time an action has a `Plan` it is a
command the helper is willing to run once approved.

- **The unit TYPE, not a list of names.** R1 acts on `.service`, `.socket`,
  `.timer`, `.path` and `.mount` only. `.target`, `.slice` and `.scope` have no
  code path, because `emergency.target` kills sshd without naming sshd and
  `user@1000.service` ends the session the scufris service lives in. Refusing the
  type is a boundary; enumerating the dangerous names inside it is a game of
  catch-up. A name with no suffix normalises to `.service`.
- **A deny-list, as the second line.** Units whose loss takes out remote access,
  the desktop session or the approval path itself: sshd, dbus, logind, the network
  managers, polkit, the user session and its manager, getty, the display manager,
  journald, udevd, nix-daemon. Compared on the STEM, case-insensitively, and on
  the template part of an instance (`user@1000.service` is caught by `user`).
- **Anything named `scufris`.** A verb that can restart the approval path can end
  an approval mid-flight, and one that can restart the helper can drop the record
  of what it was doing.
- **An argument that would become a flag.** `shell=False` with an explicit argv
  answers a different question: measured on this repo, a unit named
  `-Hsomeone@host` made `systemctl` open an outbound SSH connection. Every value
  is charset-validated, a leading `-` is refused by name, and positionals are
  passed after `--`.

At apply time there are exactly four refusals, and each is a different failure:

| Code | Means |
|---|---|
| `not_found` | an id the helper never issued |
| `already_used` | a proposal that has already been used. Approvals do not replay |
| `expired` | its window closed (`ttlSeconds`) |
| `drifted` | the fingerprint moved: the preview described a system that has since changed. Terminal, so the operator re-proposes and reads a fresh preview instead of approving an old description of a new world |

## 5. Cancellation

A `cancel` frame on the live connection, or simply dropping the connection, mean
the same thing: a root command must not keep running for a caller who is no
longer there.

Cancellation kills the whole **process group**, not the child.
`nixos-rebuild` and `nix-collect-garbage` spawn their own children; killing only
the parent leaves root-owned work running with nothing watching it. The group
gets 5 seconds to die politely before SIGKILL.

The R3 exception is recorded honestly rather than hidden: the switch runs in a
transient unit that is NOT in the helper's process group, so cancelling stops
WATCHING it, not the activation. Nothing can safely stop a
`switch-to-configuration` halfway. `Plan.cancel_detail` carries that sentence to
the operator.

## 6. The audit log

Written by root, to its own log, so it survives the app being the thing that
misbehaved. It deliberately does not share the app's store.

- **Append-only.** The only write path is `append`. There is no update, no
  delete, and no protocol verb reaches the module - a verb that erases the record
  of privileged actions is precisely what an R4 refusal is for.
- **Bounded.** Rotation is by SIZE, not on a timer, so a burst cannot outrun the
  policy. Records are single-line JSON, so a boundary never splits one. The
  oldest rotated file is pruned first, and that pruning is the only deletion in
  the module.
- **Redacted.** Anything secret-shaped is replaced before it is written, by key
  name and by value, so the log never becomes the leak.

Events: `requested`, `refused`, `denied`, `approved`, `applied`, `failed`,
`cancelled`, `expired`. `refused` and `denied` are kept apart on purpose - the
first is the helper declining to build the action at all, the second is the
operator saying no to something it was willing to do.

Each record carries the timestamp, the action id, kind and risk class, the args,
every step's argv, the requester, the outcome and returncode, how many steps
completed, what the reversal is (or the recorded reason there is none), the
restore point, and a detail line. `/api/host/audit` reads the tail of this
through the `audit_tail` verb, which is how the dashboard shows actions the page
itself never saw.

## 7. Module map

| Module | Role |
|---|---|
| `protocol.py` | the wire contract: verbs, frames, states, error codes |
| `actions/` | `taxonomy` (the verb set and risk classes), `models` (argument models, `Step`, `Plan`), `validate` (every value that reaches an argv, and the deny-lists), `plans` (`build_plan`) |
| `preview.py` | R1 and R2 previews, the reversal, the fingerprint, the honesty labels |
| `nixos.py` | the R3 preview: closure diff, current generation, and what is deliberately not shown |
| `engine.py` | the proposal registry and the four apply refusals |
| `executor.py` | the ONE place this package spawns a process, plus cancellation |
| `audit.py` | the append-only, rotated, redacted record |
| `files.py` | the filesystem seam (`is_file`, `is_executable`, `resolve`), injectable like the runner |
| `server.py` | the unix socket: framing, per-frame authentication, size limits |
| `main.py` | the unit entry point and its CLI |

## 8. How it is tested

The Python suite injects an `Executor`, a `Runner` and a `Files`, so the whole
path - including cancellation and a half-applied R3 - runs as an ordinary user
with no root and no NixOS underneath.

The half that cannot be faked has its own VM test:

```sh
nix build .#scufris-hostd-vm-test   # a real root unit, a real socket, a REAL
                                    # activation and rollback of a second
                                    # toplevel
```

It needs KVM, so it is not in CI - it guards the release pipeline.
`examples/host_action.py` prints the whole contract against a scripted executor,
including an action with no undo and one stopped mid-apply.
`examples/hostd_socket_roundtrip.py` is the socket-boundary proof: it drives
propose -> preview -> approve -> apply -> audit over a raw unix socket, so the
frame contract this README documents is exercised end to end.
