# Review: Add the host action framework with preview approval and audit

- TASK: 20260729-125029
- BRANCH: feat/host-action-framework

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context (two, with different lenses: correctness/security and
  spec/tests/docs. Findings merged and renumbered here; each was re-derived
  in-session before being adopted, and the two BLOCKERs I could reproduce were
  reproduced before any fix was written.)

Four independent ways a privileged action could be taken or a record destroyed
without the operator. The framework's central claim did not hold on the code as
written, and in two cases the tests I wrote stepped over the hole.

- [x] R1.1 (BLOCKER) scufris/app.py:993 - the operator-only check only fires
  when an `Authorization` header is PRESENT, and it sits above
  `if not auth_on: return await call_next(request)`. On a loopback deployment
  (the default) a caller that simply omits the header reaches
  `/api/host/actions/{id}/approve` and a root command executes with no operator
  involved. Reachable from the agent's own shell: the model runs
  `curl -XPOST http://127.0.0.1:8000/api/host/actions/<id>/approve`. Require a
  real session on operator-only paths UNCONDITIONALLY, and refuse to construct
  an app that has host agency but no configured operator credential.
  - Response: Fixed, and reproduced first: with auth off and no headers at all, approve
    returned 200 and the executor ran `systemctl restart -- nginx.service`. Two
    layers now. `validate_auth_config` refuses to build an app with
    `hostd_secret` set and no `auth_password_hash` (host agency with nobody to
    approve is not a deployment), and the middleware requires a real session on
    operator-only paths before the `auth_on` short-circuit, with CSRF and origin
    checked in the same self-contained block rather than the generic one that
    auth-off skips. Pinned by
    `test_an_anonymous_caller_cannot_approve_with_auth_off` (which sends NO
    header - the old test sent `Bearer anything` and stepped over the hole) and
    `test_host_agency_without_an_operator_credential_refuses_to_start`.
- [x] R1.2 (BLOCKER) scufris/hostd/server.py:155 - every unauthenticated
  connection writes an `AuditEvent.REFUSED` record, with no rate limit. Anyone
  who can reach the socket (no secret needed) can rotate the entire audit
  history off disk: measured at ~4000 connections/second against
  `max_bytes=4096, keep=2`, which erased a genuine record; at production
  defaults that is ~90 seconds of flooding to erase everything the box has ever
  been asked to do. "Nothing outside the helper can delete an entry" (README,
  AGENTS.md) stops being true in effect. Coalesce refusal records (one per
  window with a count) and account them against a budget that cannot evict
  authenticated history.
  - Response: Fixed. Refusals coalesce: the first in a window is written immediately (the
    operator needs to know someone is trying), and the rest become one summary
    record carrying the suppressed count when the window closes.
    `test_a_refusal_flood_cannot_rotate_the_history_away` drives 5000 refusals
    against a 4 KiB/keep=2 log and asserts a genuine record survives, with
    `test_a_different_refusal_is_still_recorded_during_a_flood` as the paired
    guard so the rate limit cannot hide everything instead.
- [x] R1.3 (BLOCKER) scufris/agent.py:134 - `SCUFRIS_HOSTD_SECRET` arrives
  THROUGH the environment (an `EnvironmentFile` is how a sops secret reaches the
  unit) and `_codex_env` copied `os.environ` while popping only the API token,
  so the agent CLI - and every shell command the model runs from it - held the
  credential for the root helper's socket. That is the exact threat
  `DECISION.md` chose the secret for, and the comments claiming "NEVER in
  os.environ" were false for a value sourced from env. Strip it, as a declared
  set rather than a `pop` someone has to remember.
  - Response: Fixed before the round was written, since it was reproducible in one command.
    `config.SECRET_ENV_VARS` is stripped in `_codex_env`, and
    `test_every_secret_setting_is_stripped_from_the_agent_environment`
    enumerates secret-shaped `Settings` fields so the next credential fails the
    test rather than needing to be remembered. That also closed the same
    exposure for the Telegram token, the password hash and the provider
    credentials - pre-existing, and deliberately included: a set named "secrets
    the model must not see" that knowingly omits secrets is worse than the small
    scope widening.
- [x] R1.4 (BLOCKER) scufris/hostd/actions.py:349 - the two-generation floor is
  computed and displayed but not enforced: the argv was
  `nix-collect-garbage --delete-older-than Nd`, a flag that keeps only the
  CURRENT generation and is otherwise purely age-based. On this repo's own
  fixture the preview listed generation 190 under "generations kept" while the
  command would have deleted it - the rollback target the floor exists to
  protect. Name the generations in the argv so the preview and the command are
  one statement, and assert on `plan.argv` rather than on the display list.
  - Response: Fixed. `gc_older_than` now emits
    `nix-env --profile /nix/var/nix/profiles/system --delete-generations N...`,
    naming exactly the generations the floor allows, so the preview is derived
    from the command instead of computed beside it. It no longer claims to free
    space, because deleting generation links does not - `gc_store` does, with its
    own preview and approval - and a request with nothing old enough is refused
    rather than emitting a no-op. The test asserted `generations_removed` (the
    display list), which is exactly why this survived; it asserts `plan.argv`
    now.
- [x] R1.5 (MAJOR) scufris/hostd/actions.py:78 - the R1 deny-list matches unit
  STEMS while `_UNIT_SUFFIXES` admits `.target`, `.slice` and `.scope`, so the
  units that actually take out remote access and the approval path are not
  covered. Verified to pass validation: `emergency.target`, `rescue.target`,
  `multi-user.target`, `network.target`, `user.slice`, `user@1000.service`,
  `init.scope`, `display-manager`, `systemd-journald`, `nix-daemon`.
  `unit_start emergency.target` drops the box to single-user and kills sshd - the
  outcome the `sshd` entry exists to prevent, through a name the list does not
  cover - and `unit_stop user@1000.service` kills the scufris USER service that
  `_SELF_MARKER` claims to protect. Refuse `.target`/`.slice`/`.scope` for the R1
  verbs outright and extend the deny set.
  - Response: Fixed by refusing the TYPE. R1 acts on services, sockets, timers, paths and
    mounts only; `.target`, `.slice` and `.scope` have no code path. The deny
    set also gained the session and pid-1 plumbing entries, and a templated
    instance is checked against its template so `user@1000.service` is refused
    by the `user` entry. `test_targets_slices_and_scopes_have_no_code_path_at_all`
    and `test_a_templated_instance_is_refused_like_its_template`, with
    `test_the_units_an_operator_actually_means_still_work` as the paired guard so
    tightening until nothing is allowed cannot pass.
- [x] R1.6 (MAJOR) scufris/app.py:1257, scufris/mcp_server.py:463 -
  `actor = "agent" if body.agent else _operator_identity(request)` takes the
  actor from a caller-supplied BODY field, and the MCP tool never sends it. A
  bearer caller has no session, so `_operator_identity` returns the literal
  `"operator"`: every agent-originated proposal is written into the root-owned
  audit as having been asked for by the operator. "Who asked" is the one field
  the audit exists to answer. Derive the actor from the CREDENTIAL, and have
  `propose_host_action` pass its agent id.
  - Response: Fixed. `_requester_identity` derives the actor from the credential: a session
    is `operator:<id>`, a bearer token is `agent`, neither is
    `unauthenticated`. A body field now only says WHICH agent. The MCP tool
    passes `SCUFRIS_AGENT_ID` (defaulting to `orchestrator`).
    `test_a_machine_proposal_is_never_audited_as_the_operator` sends no `agent`
    field - the real MCP shape - and also asserts a body claiming
    `operator:deadbeef` cannot promote a machine caller, with an operator-side
    paired guard.
- [x] R1.7 (MAJOR) scufris/auth.py:101 - the comment claims
  "`tests/test_auth.py` enumerates app.routes to prove every mutating host route
  is either here or read-only". No such test exists; the existing sweeps only
  check session/CSRF with auth ON. The pattern is also an explicit
  `(approve|deny|revert|cancel)` alternation, so a future `/force` or `/retry`
  silently accepts a machine token - "covered by existing" is false. Add the
  sweep the comment promises, including an auth-OFF variant that sends no
  `Authorization` header at all (the header-present shape is what let R1.1
  through).
  - Response: Fixed both halves. `test_every_mutating_host_route_is_operator_only`
    enumerates `app.routes` and asserts every non-GET route under `/api/host/`
    is operator-only (propose explicitly excepted), and the comment now says
    what is true: the alternation is explicit, so the TEST is what keeps the
    pattern in step with the routes. Your note that the sweep alone would not
    have caught R1.1 is right, which is why the anonymous-caller test exists
    separately.
- [x] R1.8 (MINOR) scufris/hostd/audit.py:201 - with `keep=1`,
  `_rotate_if_needed` unlinks the log entirely instead of rotating, and
  `nix/scufris-hostd.nix` types `auditKeep` as a bare `types.int`, so
  `auditKeep = 1` is an accepted configuration meaning "delete the whole audit on
  overflow". Clamp to a minimum of 2 and constrain the module option.
  - Response: Fixed. `AuditLog` clamps `keep` to a minimum of 2, so the unlink branch is
    unreachable, and `auditKeep` is `types.ints.between 2 100` in the module
    (`auditMaxBytes` is bounded too).
- [x] R1.9 (MINOR) scufris/app.py:1383 - `deny_host_action` marks the app record
  DENIED before awaiting `hostd.deny`. If the helper is unreachable the endpoint
  503s, the app shows it denied and `AlreadyDecided` blocks a retry, while the
  helper's proposal stays PENDING and appliable for the rest of its TTL. Call the
  helper first, or roll the local decision back on failure.
  - Response: Fixed. The helper burns the proposal first and the local record is marked
    only after it confirms, since the helper's state is what decides whether the
    action can still run. A non-pending record is refused with 409 before either
    side is touched.
- [x] R1.10 (MINOR) scufris/hostd/engine.py:135 - `propose` is neither
  rate-limited nor concurrency-capped, and the `gc_store` preview runs
  `nix-store --gc --print-dead` (measured at 35s) which takes the global nix GC
  lock. An agent holding only the machine token can loop `propose_host_action`
  and keep the GC lock contended as root with nothing ever approved. Serialize
  the preview work and cap pending proposals per requester.
  - Response: Fixed. Previews are serialized behind a lock (they are the expensive half and
    the R2 ones hold the nix GC lock), and a requester may hold at most 5 pending
    proposals - the refusal says why, naming the cost.
- [x] R1.11 (MINOR) scufris/host_actions.py:169 - `render_action`'s docstring
  says it is "Used by the MCP tool (so an agent shows the operator the real
  preview rather than its own paraphrase)". It is not - `propose_host_action`
  returns raw JSON - and `HostActionStore.render` has no caller at all. Wire the
  tool through it (which is what the tool's own instruction needs) or correct the
  claim and drop the method.
  - Response: Fixed by making the docstring true rather than by deleting the claim:
    `propose_host_action` now returns the RENDERED preview plus an explicit "you
    cannot approve this" line, which is what the tool's own instruction needs. A
    non-JSON `error: ...` answer passes through unchanged. The uncalled
    `HostActionStore.render` is gone.
- [x] R1.12 (MINOR) nix/scufris-hostd.nix:63 - the `group` example is `"users"`,
  the default primary group of every normal account on NixOS, which with socket
  mode 0660 exposes the root helper's socket to every human user on the box - and
  reaching it is all R1.2 needs. Example a dedicated group, and say what a shared
  one widens.
  - Response: Fixed: the example is a dedicated `scufris` group, and the description says
    what a shared group like `users` widens and why reaching the socket matters.
- [x] R1.13 (MINOR) tests/test_host_action_api.py:262 -
  `test_cancelling_a_live_apply_is_recorded` is flaky: one failure (with a
  fixture-teardown error pointing at the hung apply still being in flight) in ten
  full-suite runs, passing in the other nine plus twelve targeted runs. A DoD test
  for the cancellation path that fails ~10% of the time will be read as noise the
  first time it fires in CI. Replace the sleep-polling with explicit awaits and
  make the fixture teardown drain in-flight handlers.
  - Response: Fixed. The polling waits on conditions with a generous deadline
    (`_until`) instead of a fixed number of short sleeps, and the fixture
    teardown cancels in-flight handlers before stopping the loop - which is what
    the teardown ERROR was pointing at. Ten consecutive runs of the file, zero
    failures. Also noted: your finding that the cancellation test does not pin
    the cancel-frame fix specifically (EOF covers it too) is correct, and
    NOTES.md now says so instead of overstating it.
- [x] R1.14 (MINOR) scufris/hostd/server.py:212 - `_run_verb` falls through to
  `_apply` for any verb it does not explicitly handle. The enum makes that safe
  today, but "an unhandled verb defaults to executing something" is the wrong
  default in this file specifically. Dispatch `Verb.APPLY` explicitly and raise
  `BAD_REQUEST` on the fall-through.
  - Response: Fixed. `Verb.APPLY` is dispatched explicitly and an unhandled verb raises
    `BAD_REQUEST`.
- [x] R1.15 (MINOR) scufris/config.py:243 - TASK.md's step C ticks
  `hostd_enabled` as delivered config; only `hostd_socket` and `hostd_secret`
  exist. The behaviour is delivered (an empty secret means 503 "not configured"),
  so amend the step rather than adding a redundant knob.
  - Response: Amended the step rather than adding the knob: an empty secret already means
    "not configured", and a second flag that could disagree with it is a state to
    get wrong. The step now also records the corrected reason the secret cannot
    be kept out of `os.environ`.
- [x] R1.16 (NIT) nix/scufris-hostd.nix:72 - `secretFile` is `types.path`, so
  `secretFile = ./secret;` type-checks and copies the shared secret into the
  world-readable nix store, contradicting the option's own description. Use
  `types.str` so a store path is a configuration error.
  - Response: Fixed: `secretFile` is `types.str`, so a nix path literal is a configuration
    error rather than a silent copy into the store.
- [x] R1.17 (NIT) tasks/20260729-125029/TASK.md - step A names
  `scufris/hostd/proposals.py`; the registry landed in `scufris/hostd/engine.py`.
  The step is delivered, only the filename differs - correct the record so a cold
  session can find it.
  - Response: Fixed in TASK.md - the step names `scufris/hostd/engine.py`.

### What the reviewers verified

Both reviewers ran the full gate independently: `ruff`, `mypy` and 753 tests
green, `nix build .#scufris .#web` green. `nix flake check` is red only on
`checks.records` with `closed-missing-review` / `closed-missing-retro`, which
are artifacts of the task being CLOSED before this file and RETRO.md exist.

All seven `test:` proofs named in the Definition of Done were run and pass. The
spec reviewer sabotage-tested four load-bearing tests: neutering the
operator-only check, moving the `APPLYING` claim back after the drift await, and
disabling audit rotation each turned the corresponding tests red. The
cancellation test proved weaker than claimed: removing the client's cancel frame
alone left it green, and disabling the server's EOF detection alone also left it
green - the two paths are redundant, so the test does not pin the cancel-frame
fix specifically. That is a defensible belt-and-braces design, but NOTES.md
overstated what the test proves.

No existing test was weakened or deleted; the supervisor/eventbus changes are
mechanical and every implementor and test double was updated.

### Pending user checks

- manual: an approval prompt states plainly what will change and how it can be
  undone. Both reviewers read the rendered text from `examples/host_action.py`
  and confirmed it carries the preview label and the UNDO / NO UNDO line, but
  whether it is GOOD ENOUGH is the operator's judgement, not the reviewers'.


## Round 2

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context (a fresh reviewer against the new diff, with no sight
  of the implementing session. It re-verified each round 1 finding against the
  CURRENT code rather than against the Response text, and sabotage-tested six of
  the claimed pins. The in-session pass re-derived R2.1 and R2.2 from the source
  before adopting them.)

Every round 1 finding was re-checked. Fifteen of the seventeen are genuinely
fixed, with the four BLOCKER pins sabotage-verified (reverting each fix turns its
named test red). Two are only PARTIALLY fixed, and both partials have the shape
of the finding they came from: R1.3's strip was applied at ONE call site and a
second backend spawn still leaks the secret, and R1.10's cap is keyed on a field
the caller supplies - the exact mistake R1.6 was about.

- [x] R2.1 (BLOCKER) scufris/backends.py:664 - the claude backend calls
  `asyncio.create_subprocess_exec` with no `env=`, so the claude CLI - and every
  shell command the model runs from it - inherits `os.environ` verbatim,
  `SCUFRIS_HOSTD_SECRET` included (it arrives through the unit's
  `EnvironmentFile`). This is R1.3's hole, unclosed for a supported backend: with
  `SCUFRIS_AGENT_BACKEND=claude` a prompt-injected model calls `apply` on the
  root socket directly and the whole propose/preview/approve path is decoration.
  AGENTS.md's "STRIPPED from every subprocess environment" is false here.
  Confirmed in-session: `_codex_env` is the only stripper, and it is referenced
  at `scufris/agent.py:583` and `:750` only. Pass the stripped environment at
  this call site through a shared seam (`agent_subprocess_env(settings)`), and
  widen `test_every_secret_setting_is_stripped_from_the_agent_environment` to
  cover EVERY backend spawn - a per-call-site strip is the thing that was
  already forgotten once.
  - Response: Fixed. `agent_subprocess_env(settings)` is now the single
    shared scrubber for model-driven subprocesses, and `ClaudeBackend` passes
    it to `asyncio.create_subprocess_exec`. The regression coverage has two
    layers: `test_claude_backend_strips_secrets_from_the_cli_environment`
    proves the claude spawn drops `SCUFRIS_HOSTD_SECRET`, and
    `test_no_agent_subprocess_is_spawned_without_the_stripped_environment`
    walks production subprocess call sites so the next backend cannot forget
    the same strip.
- [x] R2.2 (MAJOR) scufris/hostd/engine.py:405 - `_refuse_a_flood_of_proposals`
  keys the cap on `requester.agent or requester.actor`, and `agent` comes
  straight from the caller-supplied `HostActionRequest.agent` body field
  (`scufris/app.py:1279`). A machine caller varying that string per request never
  hits the cap: reproduced against the engine, 20 pending proposals accepted with
  `agent=orchestrator-<i>` against `MAX_PENDING_PER_REQUESTER = 5`. The costs the
  cap exists for remain - unbounded growth of `self._proposals` in the ROOT
  process (`_reap` only drops terminal entries) and one uncoalesced `REQUESTED`
  audit record per accepted proposal, which is R1.2's rotation shape on an
  authenticated path. This repeats R1.6's own lesson: the caller must not control
  the identity the server makes decisions on. Key the cap on `requester.actor`
  (the credential-derived value) and add a test that varying `agent` does not
  raise the cap.
  - Response: Fixed. `_refuse_a_flood_of_proposals` keys only on
    `requester.actor`, which is derived from the credential, while
    `requester.agent` remains attribution text. Pinned by
    `test_a_machine_caller_cannot_raise_its_pending_cap_by_varying_agent_name`;
    the paired guards still prove the same actor is capped and different
    credential-derived requesters do not share one bucket.
- [x] R2.3 (MINOR) scufris/hostclient.py:9 - "**The secret never enters
  `os.environ`.** It arrives on `Settings` and stays there, exactly like
  `auth_api_token`" is unchanged since the original commit. That is the precise
  false claim R1.3 was about, in the module most about the secret, while
  NOTES.md reads as if it was corrected everywhere. Rewrite it to say the secret
  arrives THROUGH the environment and is stripped from subprocess environments
  via `config.SECRET_ENV_VARS`.
  - Response: Fixed. `hostclient.py` now says the secret can arrive through the
    process environment, because that is how the deployed unit receives its
    `EnvironmentFile`, and that model-driven child processes must receive
    `agent.agent_subprocess_env(settings)` rather than raw `os.environ`.
- [x] R2.4 (MINOR) scufris/mcp_server.py:464 - neither half of the
  `propose_host_action` change is pinned: nothing asserts the tool sends `agent`,
  and nothing asserts it returns rendered prose rather than JSON.
  `test_a_machine_proposal_is_never_audited_as_the_operator` passes with the tool
  unchanged, because `_requester_identity` already defaults a bearer caller's
  agent to `"orchestrator"`. Add a unit test over `_render_host_action` (a record
  payload renders with the "you cannot approve this" line; a non-JSON
  `error: ...` passes through) and one asserting the request body carries
  `SCUFRIS_AGENT_ID`.
  - Response: Fixed. `test_propose_host_action_returns_the_rendered_preview`,
    `test_propose_host_action_preserves_non_json_api_errors`, and
    `test_propose_host_action_sends_the_agent_id` pin the tool behavior: the
    operator-facing preview prose is returned to the model, non-JSON API errors
    are not hidden, and the request body carries `SCUFRIS_AGENT_ID`.
- [x] R2.5 (MINOR) tasks/20260729-125029/TASK.md:73 - step A's last bullet is
  ticked claiming R2 "takes a stronger confirmation than the reversible classes".
  Nothing in the diff varies the approval by risk class: `approve_host_action`
  (scufris/app.py:1359) is one POST for R1 and R2 alike and `risk` appears
  nowhere on the approval path. The other half (R2 records it cannot be undone)
  IS delivered and tested. Amend the step the way R1.15 was amended, or
  implement it here.
  - Response: Fixed as a record correction and a handoff, not by adding the UI
    behavior to this framework task. `TASK.md` now says the framework records
    `risk` and `reversal.possible`, and that differentiated confirmation
    belongs to the approval interfaces. The receiving UI/Telegram task,
    `tasks/20260729-125040/TASK.md`, now carries the explicit R2 requirement
    and `test_one_way_action_requires_stronger_confirmation`.
- [x] R2.6 (NIT) scufris/app.py:1035 - `if machine_forbidden:` inside the bearer
  branch is unreachable dead code now that the block at `app.py:987` returns on
  every operator-only path. Delete it, so the one enforcement point stays the
  only place a reader has to trust.
  - Response: Fixed. The unreachable bearer-branch check is gone, leaving the
    operator-only path block as the single machine-token refusal point.
- [x] R2.7 (NIT) scufris/hostd/actions.py:406 - the comment cites "review round
  1, R1.2" for the gc floor, which is the audit-flood finding; the floor is R1.4.
  Same drift at scufris/config.py:253, scufris/agent.py:147 and
  tests/test_auth.py:1015 (all cite R1.1 for what REVIEW.md numbers R1.3), and
  tests/test_host_actions.py:530 (R1.2 for R1.4). Renumber to match REVIEW.md.
  - Response: Fixed. The comments now cite R1.3 for secret stripping and R1.4
    for the generation-floor command fix.
- [x] R2.8 (NIT) .env.example:39 - the `SCUFRIS_HOSTD_SECRET` comment does not
  mention that setting it makes `SCUFRIS_AUTH_PASSWORD_HASH` mandatory even on
  loopback (`scufris/auth.py:247`). An operator following this file alone gets a
  service that refuses to start. Add the sentence README.md:123 already has.
  - Response: Fixed. `.env.example` now states that enabling the host helper
    makes `SCUFRIS_AUTH_PASSWORD_HASH` mandatory even on loopback, matching the
    startup check.

### What the reviewer verified

`ruff check .`, `mypy .` (97 files) and `python -m pytest` (766 tests, run four
times) green, `nix build .#scufris .#web` green, `python examples/host_action.py`
renders the preview label and the UNDO / NO UNDO lines. All seven `test:` proofs
from the Definition of Done were run by name and pass. `nix flake check` is red
only on `checks.records` with `closed-not-approved` / `closed-missing-retro`,
which this round and the retro resolve.

Six sabotage edits (each reverted) confirmed the round 1 pins: reverting the
operator-only ordering, the refusal coalescing, the gc argv, the unit-type
refusal, the template check and the credential-derived actor each turned the
named test red. R1.8, R1.9 and R1.14 are correct in code but unpinned by a test;
that is recorded rather than raised, since in each case the dangerous branch no
longer exists to regress into.

Two observations that are not findings: AGENTS.md's "The NixOS VM test" is
singular now that `.#hostd-vm-test` exists alongside `.#vm-test` (the deployment
section does cover it), and `GET /api/host/actions/{id}/events` and
`GET /api/host/audit` are reachable with the machine token, relaying approved
argv and output back to the agent - read-only and apparently deliberate, since
`host_action_audit` is an MCP tool.

### Pending user checks

- manual: an approval prompt states plainly what will change and how it can be
  undone. Carried forward from round 1 - still the operator's judgement.


## Round 3

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh reviewer against the current branch diff with
  special attention to the Round 2 findings. The reviewer returned findings
  only and found no blocker, major, minor, or nit issues worth recording.)

Round 2 is resolved. The reviewer verified from source that `ClaudeBackend`
passes `env=agent_subprocess_env(settings)`, the pending proposal cap keys on
credential-derived `requester.actor`, and the documentation, task records, and
MCP tool tests reflect R2.3 through R2.8.

### What the reviewer verified

Focused Round 2 regression tests:

- `tests/test_auth.py::test_the_claude_backend_strips_every_secret_from_its_child_environment`
- `tests/test_auth.py::test_no_agent_subprocess_is_spawned_without_the_stripped_environment`
- `tests/test_host_actions.py::test_varying_the_agent_name_does_not_raise_the_proposal_cap`
- `tests/test_mcp_server.py::test_the_host_action_tool_returns_the_rendered_preview_not_json`
- `tests/test_mcp_server.py::test_a_host_action_tool_error_passes_through_unrendered`
- `tests/test_mcp_server.py::test_the_host_action_tool_names_the_agent_it_is_running_as`

Broader host-action/auth files:

- `tests/test_host_actions.py`
- `tests/test_host_action_api.py`
- `tests/test_hostd_audit.py`
- `tests/test_mcp_server.py`
- `tests/test_auth.py`

Result: 168 passed.

Residual test gaps carried forward: the reviewer did not run the full flake or
VM gates in this pass, and R1.8, R1.9, and R1.14 remain correct in code but not
directly pinned by narrow regression tests.
