# Review: Extract the backend-aware orchestrator diagnostics service

- TASK: 20260729-102148
- BRANCH: fix/agent-diagnostics-service

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) scufris/app.py:3592 - the legacy `/api/agent/account` maps "the
  agent is disabled" onto `Capability[UsageQuota].unsupported()`, but the contract
  this task introduces (`scufris/backends/base.py:31`, `scufris/README.md` "The
  per-agent diagnostics contract") defines `supported: false` as "this backend has
  no such reader". A disabled codex agent's backend HAS one, so the envelope's
  central distinction is wrong on its first legacy caller. Use
  `Capability[UsageQuota].read(None)` there; the same payload already carries
  `enabled: false`.
  - Response: fixed in this round's commit - `/api/agent/account` now builds
    `Capability[UsageQuota].read(None)` when the agent is disabled, so
    `supported: false` keeps meaning "this backend has no such reader".
- [x] R1.2 (MINOR) scufris/app.py:3585 - Step 4's ticked text says "Legacy
  `/api/agent/*` keeps its current behaviour by importing the moved helpers", but
  `/api/agent/account` changed its wire shape (`quota: UsageQuota | null` -> the
  envelope). The CHANGELOG discloses it honestly; the tick does not. Amend the Step
  text to name the legacy account shape change (or give that route its own bare
  response model).
  - Response: fixed in this round's commit - Step 4 now names the
    `/api/agent/account` `quota` shape change explicitly;
    `scufris/README.md`'s "legacy routes keep their older shapes" sentence
    carried the same overclaim and was corrected too.
- [x] R1.3 (MINOR) tests/test_app.py:1892 -
  `test_account_quota_null_when_disabled` no longer asserts a null quota, so its
  name misdescribes what it pins. Rename to
  `test_account_quota_unsupported_when_disabled`, or to whatever R1.1 settles on.
  - Response: fixed in this round's commit - renamed to
    `test_account_quota_empty_reading_when_disabled`, asserting `{"supported":
    true, "value": null}` per R1.1.
- [x] R1.4 (NIT) scufris/agent_diagnostics.py:92 - the function-local
  `from .mcp_health import servers_for_audience` (and `probe_server` at line 121)
  is a carry-over from `app.py`. `mcp_health` imports only `os`, `shutil`, `typing`
  and `.enums` at module level - it defers the server modules itself - so hoist
  both to the module top and delete the two inline imports.
  - Response: fixed in this round's commit - `probe_server` and
    `servers_for_audience` hoisted to the module top; both inline imports
    deleted.
- [x] R1.5 (NIT) scufris/agent_diagnostics.py:119 - a leaf module's docstring
  points at the private `app._ensure_den_path`. State the precondition without
  naming an app-private symbol: "the den path must already be bridged by the
  caller".
  - Response: fixed in this round's commit - the docstring now says the den
    path must already be bridged by the caller, with no app-private symbol
    named.

Process signal: Step 4's tick is intent-shaped. The "legacy keeps current
behaviour" clause is contradicted by the diff and by the branch's own CHANGELOG
entry. The disclosure is honest; the tick is not.

Process signal: the Notes' base-branch probes visibly changed the plan (DoD 4 was
rewritten from "static settings drift" to "capability follows the record"), and
the shipped test pins the rewritten claim rather than the original intent.

Verified in-session, independently of the out-of-context reviewer:

- `ruff check .` clean; `mypy .` clean (194 files); `python -m pytest` exit 0;
  `cd web && npm run ci` green through webpack.
- The DoD grep proof passes; `scripts/check_file_size.py` silent with no new
  allowlist entry (`agent_diagnostics.py` 196 lines, `test_agent_diagnostics.py`
  278).
- Mutation check: all four named DoD tests were copied onto a clean `master`
  worktree and all four FAIL there, then pass on the branch. They pin the change,
  not merely execute it.
- Legacy `/api/agent/usage` and `/api/agent/memory` do keep their pre-branch
  shapes (`app.py:3571`, `app.py:3578`); only `/api/agent/account` moved.
- Doc-surface sweep: no stale mentions of `_agent_is_codex`,
  `_agent_has_scufris_mcp`, `_tool_parameters`, `_tools_for_servers`,
  `_mcp_servers_for_audience` or `_probe_servers` outside `tasks/`.
- `AccountInfo.quota` has no renderer in `web/src`, so its type change is inert
  in the dashboard; the `usage`/`memory` envelopes are unwrapped at the fetch
  boundary and the panels render unchanged.

Not a finding, recorded so the next run does not re-chase it: one full fixed-order
suite run (`pytest -p no:randomly`) on this branch failed
`tests/test_host_action_api.py::test_cancelling_a_live_apply_is_recorded`. Two
further fixed-order runs on the branch passed, the test passes alone, and the diff
touches no host-action code. It is a second pre-existing flake alongside the one in
Notes (`test_agent_fork_reverts_single_session`), not a branch regression.

Not verified: `nix flake check` and `nix build .#scufris-web` were not re-run; the
close-out's reproduce-on-master claim for both is taken on the two filed follow-ups
(20260803-022018, 20260803-022030).

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

No findings. All five round-1 findings are ticked: the fixes are present in
`7255ed0` and each `Response:` line matches the diff.

- R1.1: `scufris/app.py:3596` builds `Capability[UsageQuota].read(None)` on the
  disabled branch, so `supported: false` keeps meaning "this backend has no such
  reader". The route docstring states the distinction instead of going stale.
- R1.2: Step 4 names the `/api/agent/account` quota shape change, and
  `scufris/README.md:331-334` no longer claims the legacy routes are untouched.
  `/api/agent/usage` and `/api/agent/memory` do still return their pre-branch
  bare shapes (`app.py:3571`, `app.py:3578`), so the corrected sentence is true.
- R1.3: `tests/test_app.py:1892` is `test_account_quota_empty_reading_when_disabled`
  and asserts `{"supported": true, "value": null}`; its comment was rewritten to
  match rather than left contradicting the assertion.
- R1.4: both inline imports are gone; `scufris/agent_diagnostics.py:27` holds the
  single module-level import. No cycle, and no test monkeypatches
  `mcp_health.probe_server`/`servers_for_audience`, so the hoist broke no patch
  target.
- R1.5: the `probe_servers` docstring states the precondition without naming an
  app-private symbol.

The fix commit's only behavioural change is the legacy disabled-account branch,
and its blast radius is contained: `AccountInfo.quota` has no renderer in
`web/src`, and `/api/agent/account` has no other Python or TS consumer.

Verified in-session, independently of the out-of-context reviewer:

- `ruff check .` clean; `mypy .` clean (194 files); full `python -m pytest`
  exit 0; `cd web && npm run ci` green through webpack.
- The no-name-comparison DoD grep passes; `scripts/check_file_size.py` silent
  with no new allowlist entry (`agent_diagnostics.py` 193 lines,
  `test_agent_diagnostics.py` 278).
- Re-derived R1.1 from the source rather than the Response: the disabled branch
  at `app.py:3593-3596` is the only quota path that skips the rollout read, and
  the sole web consumer (`agent-settings-view.ts:223`) renders a null quota
  regardless of the envelope's `supported` flag.
- Working tree clean at `d4c2c99`.

Process signal: the fix commit chased the same overclaim into `scufris/README.md`
unprompted, which R1.2 had asked for only in TASK.md.

Not verified: `nix flake check` and `nix build .#scufris-web` were not re-run;
the close-out's reproduce-on-master claim for both still rests on the two filed
follow-ups (20260803-022018, 20260803-022030).
