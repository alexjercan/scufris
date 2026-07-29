# Notes: the host action framework

Design and fix record for the change that landed on `feat/host-action-framework`.
The decisions are in `DECISION.md` next to this file; this is what was actually
built, what fought back, and what a later session should know.

## Shape

Two processes, and the split is the whole design.

- `scufris/hostd/` is the privileged half. It runs as ROOT under
  `services.scufris-hostd` (`nix/scufris-hostd.nix`, exported as
  `nixosModules.hostd`) and is the only thing on this machine that scufris can
  reach root through. It owns the verbs, the argv construction, the proposals,
  the previews and the audit log.
- `scufris/hostclient.py`, `scufris/host_actions.py` and the `/api/host/actions`
  endpoints are the app half. They can name a verb and, later, an id the helper
  itself issued. They cannot name a command.

Placing the proposal registry in the HELPER rather than the app is what makes
"preview one thing, apply another" impossible rather than merely discouraged.
The app has no way to express the second thing.

## The four apply refusals

Each is a different failure, and each got its own test:

| refusal | what it stops |
|---|---|
| `not_found` | a forged or expired-and-reaped id |
| `already_used` | an approval replayed |
| `expired` | approving a preview taken ten minutes ago |
| `drifted` | approving a description of a system that has since moved |

## Things that fought back

**The concurrency check spanned an await, so it was not a check.** The first
version of `HostdEngine.apply` did `_require_pending()`, then `await` the drift
re-read, then burned the proposal. Two approvals racing on one id therefore both
passed the check and both ran the command.
`test_concurrent_approvals_of_one_proposal_run_it_once` caught it on the first
run. The fix is a transient `APPLYING` state claimed SYNCHRONOUSLY, before the
first await - the claim and the check are now one operation. Generalisable:
a state check followed by an await followed by a state change is three
operations, not one, and the middle one is where a second caller gets in.

**A supervisor cancel arrives as `GeneratorExit`, not `CancelledError`.** The
apply stream is an async generator; `supervisor.cancel` ends up calling
`aclose()` on it, which throws `GeneratorExit` at the yield. An
`except asyncio.CancelledError` around the loop - which is what the shape
suggests - never fires. The cancel frame is therefore written from the
generator's `finally`, and written SYNCHRONOUSLY (`writer.write`, no
`await drain()`), because an async generator being closed must not await a
yield-capable operation. `test_cancelling_a_live_apply_is_recorded` drives the
real socket to prove it, which is the only way this would have been caught.

**`TestClient` without `with` cancels background work.** The approval endpoint
starts a supervisor run and returns; the run outlives the request by design
(ADR-001). A `TestClient(app)` not entered as a context manager tears its portal
down after each request, so the apply was cancelled before it ran - and the test
saw the action "settle" with nothing having executed. It passed or failed
depending on timing. The `make_client` fixture holds the client open for the
whole test and says why in its docstring.

**The gc floor was in the preview and not in the command.** `build_plan`
resolved the generations the two-generation floor allows, displayed them, and
then emitted `nix-collect-garbage --delete-older-than Nd` - a flag that keeps
only the CURRENT generation and is otherwise purely age-based. On the repo's own
fixture the preview listed generation 190 under "generations kept" while the
command would have deleted it: the rollback target the floor exists to protect.
Found in review (R1.4). The fix names the generations in the argv, so the
preview is DERIVED from the command instead of computed beside it.

The test is the part worth remembering: it asserted `plan.generations_removed`,
the display list, and passed the whole time. An assertion on the thing that is
shown cannot catch a disagreement between what is shown and what runs. It now
asserts `plan.argv`.

**`nix path-info -S` would have been a fiction.** The spike specified `-S` for
the reclaimable size. `-S` is CLOSURE size, and closures overlap, so summing it
over a dead set counts shared dependencies once per referrer - the number would
have been several times the space actually freed, printed in a field labelled
"space this would free". Measured on this host before writing the code
(9127 dead paths; a 500-path batch answers in ~0.2s). The implementation sums
each path's own `narSize` instead and says so in the docstring. This is exactly
the failure the epic rejects elsewhere, and it was one keystroke away from being
shipped inside the module whose job is refusing it.

**`nix-collect-garbage --dry-run` and `nix store gc --dry-run` both print only a
count.** Re-measured here; the spike's finding holds. So the `gc_store` preview
computes the size itself, per path. `gc_older_than` shows NO size at all after
the R1.4 fix: it deletes generation links, which frees nothing by itself, so any
figure there would describe a different action. It says that in words instead.
The count is a count and the size is a size; neither wears the other's name, and
neither appears next to a command it does not describe.

**A `log` variable shadows the NixOS test driver's logger.** In a
`testScript`, `log` is an `AbstractLogger` global. Assigning `log = machine.succeed(...)`
fails the driver's own type check with five confusing diagnostics.

## The generic supervisor

`Supervisor` and `EventBus` are now generic in their event type. The refactor
was mechanical (rename + parameterize) and produced ZERO mypy errors in existing
code; the only call-site change is `Supervisor()` -> `agent_supervisor()`.

The supervisor needed exactly two things from the event type, and naming them is
what made the refactor small: how to publish a terminal failure
(`error_event`), and how to recognise one a stream produced itself
(`error_detail`). Everything else was already lifecycle.

The alternative - widening `StreamEvent` - would have meant a root command's
output was a member of the union every chat surface renders. Rejected in
`DECISION.md` before writing any of it.

## What review round 1 found, and why

Two out-of-context reviewers found four independent ways a privileged action
could be taken, or its record destroyed, without the operator. All four were
proven with a reproduction before being fixed. The full round is in REVIEW.md;
these are the ones with a lesson in them.

**The approval gate asked the wrong question (R1.1).** The check was "is a
bearer token present, on an operator-only path?" - so a caller that sent NO
Authorization header at all fell past it, hit the `if not auth_on` short-circuit,
and executed a root command anonymously. On a loopback deployment that is any
process on the machine, including the shell the model runs its own commands in:
`curl -XPOST http://127.0.0.1:8000/api/host/actions/<id>/approve`.

My own test walked straight over it, because it sent `Bearer anything` - it was
written to prove the machine token is refused, and a test that always presents a
credential cannot discover that presenting none is enough. The lesson is about
the shape of the check: a positive test ("this credential is refused") does not
cover the absence of the credential, and an authorization check must be written
as "require an identity", never as "reject these identities".

Fixed in two layers: `create_app` refuses to build an app with host agency and
no operator credential, and the middleware requires a session on those paths
whatever the bind address, with CSRF and origin in the same self-contained block
rather than in the generic one that the auth-off path skips.

**The secret arrived by the route I had ruled out (R1.3).** Three comments in
this diff said the hostd secret is "NEVER in os.environ", copied from the
machine API token's design where it is true because the token is MINTED
in-process. The hostd secret is DELIVERED through the environment - an
`EnvironmentFile` is how a sops secret reaches a unit - so it was in
`os.environ` by construction, and `_codex_env` copied `os.environ` wholesale
while popping only the API token. The model held the credential for the root
socket.

The lesson: a property inherited from a similar-looking value is a hypothesis.
"Kept out of the environment" and "delivered through the environment" are
opposite facts about two things I described with one sentence. The fix is a
declared `SECRET_ENV_VARS` set with a test that every secret-shaped `Settings`
field is in it, so the next credential is covered by a failing test rather than
by someone remembering - and it closed the same exposure for the Telegram token,
the password hash and the provider credentials, which were pre-existing.

That fix was itself incomplete, in both halves, and round 2 found both. The
strip went into `_codex_env` - ONE call site - while `backends.ClaudeBackend`
spawned with no `env=` at all, so with `SCUFRIS_AGENT_BACKEND=claude` the model
still held the socket credential (R2.1). And one of the three false comments
survived the sweep, in `hostclient.py`, the module most about the secret (R2.3).
A set that names the secrets does not help a spawn that never consults it: the
strip is now the seam `agent.agent_subprocess_env`, and
`test_no_agent_subprocess_is_spawned_without_the_stripped_environment` walks the
package's AST and fails any `create_subprocess_*` that does not pass it or is
not explicitly exempted with a reason. Fixing a leak at the site it was found
leaves the other sites; fixing it at the seam, with a structural test, does not.

**The audit could be erased by the caller most likely to want that (R1.2).**
Reaching the socket needs no secret, and every unauthenticated frame wrote a
record. Measured by the reviewer at ~4000 connections/second: enough to rotate
the entire 80 MiB history off disk in about ninety seconds. "Nothing outside the
helper can delete an entry" was true of the API and false in effect. Refusals
now coalesce per window with a count.

The lesson is that a retention policy is an attack surface when anyone can write
to the log. Bounded + append-only + attacker-writable = deletable.

**The deny-list enumerated names inside a type it should have refused (R1.5).**
`emergency.target` drops the box to single-user and kills sshd without ever
naming sshd; `user@1000.service` ends the operator session that scufris itself
runs in, which `_SELF_MARKER` claimed to make impossible. R1 now acts on
services, sockets, timers, paths and mounts only. Refusing the TYPE is a
boundary; listing the dangerous names inside an allowed type is a game of
catch-up that the attacker picks the next move in.

**And one about the record itself (R1.6):** "who asked" was read from a
caller-supplied body field, so every agent proposal was audited as the operator.
The one question the audit exists to answer must not be answerable by the
caller; it now comes from the credential.

## Where the enforcement actually is

`auth.OPERATOR_ONLY_PATTERN`, consulted in `app.py`'s middleware BEFORE the
bearer short-circuit and BEFORE the `auth_on` early return.

The second half of that placement is worth keeping: deciding it only when auth
is on would mean a loopback deployment lets an agent approve its own proposal,
and an agent approving its own proposal has nothing to do with the bind address.
`test_a_machine_token_cannot_approve_even_with_auth_off` pins it.

The absence of an approve MCP tool is pinned separately
(`test_the_agent_has_no_tool_that_approves_a_host_action`), because a
convenience tool added later would silently undo the expensive half.

## What is deliberately not here

- R3 (the NixOS configuration change: `build`, `dry_activate`, `activate`,
  `rollback`) is 20260729-125035. The protocol is designed to be extended by a
  new verb, which is the "a capability is a reviewed code change with a test"
  path the spike decided on.
- The dashboard approval UI is 20260729-125040. This task's operator-facing
  proof is `render_action` plus `examples/host_action.py`, which is where the
  approval wording was actually read and revised.
- Durable app-side proposal state. The helper expires proposals in minutes and
  the audit log is the record that has to survive, so `HostActionStore` is
  in-memory and bounded.

## Verification

`nix flake check`, `nix build .#scufris .#web`, and `nix build .#hostd-vm-test`
(a real root unit restarting a real service in a VM, with the audit log checked
for the secret) all green, 766 tests passing after review round 1. The VM test is now also a step in the release
pipeline - it is the only place the privileged half is proven to actually run a
command.
