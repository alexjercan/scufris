# Split the oversized test suites under the size cap

- STATUS: OPEN
- PRIORITY: 70
- TAGS: refactor, v0.2.0, testing, maintainability
- KIND: TASK
- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want oversized test suites split by domain, so that working
on one area loads only that area's tests.

## Steps

- [ ] Record the pre-move baseline on the sprout BEFORE the first edit, so any
      later flag is provably introduced rather than inherited:
      `python scripts/check_file_size.py` exit 0;
      `python -m pytest --collect-only` reports 896 tests collected;
      `nix flake check` all 5 green; `cd web && npm ci && npm run ci` green at
      22 files / 258 tests. `npm` needs `nix develop`, and a fresh sprout needs
      `npm ci` first. Read any base file with `git show master:<path>` into a
      scratch file, NEVER `git checkout master` - it cannot succeed from a
      sprout and fails silently inside a `bash -c` chain.
- [ ] Write two throwaway checks under the scratchpad and run BOTH on every
      commit below:
      (a) a test-NAME set difference - `python -m pytest --collect-only`
      prints nodeids, so strip the `<file>::` prefix from both sides and
      difference the sorted multisets; a dropped or renamed test is reported
      rather than noticed. The count alone is not the proof. `npx vitest list`
      prints `file > describe > it` for the frontend; strip the file prefix the
      same way.
      (b) a move proof - normalize each base file and the union of its new
      files to a multiset of stripped, non-blank, non-comment, non-import lines
      and difference them, so every remaining entry is a rewording or a rename
      that can be named. Report it PER COMMIT in a table, not as prose.
- [ ] Before each commit, grep the private names that commit moves, not only
      the module paths: `_login`, `_settings`, `_propose`, `ORIGIN`, `SECRET`,
      `PASSWORD`, `_make_bot`, `_events_bot`, `_capture_sends`,
      `_record_calls`, `_Recorder`, `_fake_settings_ops`, `_update`, `_ok`. A
      `monkeypatch.setattr("mod.NAME", ...)` target is a STRING and fails
      silently when the new home happens to bind the name.
- [ ] Commit 1 - `tests/test_host_action_api.py` (1285). Lift `PASSWORD`,
      `ORIGIN`, `SECRET`, `_settings`, `_login` and `_propose` into
      `tests/conftest.py` and repoint `tests/test_host_digest.py`,
      `tests/test_telegram_approvals.py` and `tests/test_nixos_config_change.py`
      to import them from `conftest` in the SAME commit. Split the remainder in
      two: `test_host_action_api.py` keeps who may propose, approve, revert and
      cancel, the audit trail, and the helper being absent or presenting a wrong
      secret; new `tests/test_host_action_decisions.py` takes the decision core
      (`:701-1285`) - confirmation strength, the approval race, expiry, restart
      durability, operator binding, delivery back to the requesting agent, and
      the queue row. Delete the `tests/test_host_action_api.py` ALLOWLIST entry
      in this commit.
- [ ] Commit 2 - `tests/test_nixos_config_change.py` (1044), AFTER commit 1
      because it consumes the helpers commit 1 moves. New
      `tests/test_nixos_activation.py` takes the plan, the preview, rollback and
      apply against the helper (`:152-522`); `test_nixos_config_change.py` keeps
      building a commit, proposing it, and the HTTP surface (`:523-1044`). Its
      imports from `tests/test_host_actions.py` (`host_runner`, `host_files`,
      `NIX`, `BUILT_SYSTEM`, `OLD_SYSTEM`, `RUNNING_SYSTEM`) carry over
      unchanged to whichever half needs them - that file is not in scope. Delete
      the entry.
- [ ] Commit 3 - `tests/test_auth.py` (1219). Three files by behavior:
      `test_auth.py` keeps the password hash, the loopback fail-closed policy,
      login/logout, session lifetime and the session store;
      `tests/test_auth_boundary.py` takes what a request must carry (session,
      CSRF, origin), the login throttle, the browser redirects, and the
      streaming and Telegram bridges; `tests/test_auth_machine.py` takes machine
      callers - MCP tools under auth, the machine token, and every secret
      stripped from an agent's environment. The `# --- review round 1
      regressions` block (`:841-1219`) is a HISTORY grouping, not a behavior:
      redistribute its 15 tests by what each asserts, do not move it as a unit.
      Delete the entry.
- [ ] Commit 4 - `tests/test_host_inspection.py` (1076).
      `tests/test_host_thermal.py` takes throttling and the thermal report
      (`:313-539`, the one block that is contiguous AND single-module);
      `tests/test_host_nix_store.py` takes the nix store, packages, flake status
      and closure-diff parsing; `test_host_inspection.py` keeps the four DoD
      properties across the six domains plus units, journal, network and render.
      The `# --- review round 1 regressions` block (`:901-1076`) is
      redistributed by assertion, as in commit 3. Decide at work time: if
      `test_host_nix_store.py` measures under ~150 lines, fold it back and ship
      two files - record the measurement either way. Delete the entry.
- [ ] Commit 5 - `tests/test_telegram.py` (1760). Four files:
      `test_telegram.py` keeps the long-poll transport, command dispatch, the
      typing action AND the shared bot harness (`:86-251`);
      `tests/test_telegram_stream.py` takes the live thinking bubble, tool
      widgets and final answer (`:633-949`); `tests/test_telegram_render.py`
      takes the pure formatters and the `/stats` and `/settings` renderers
      (`:950-1106`, `:1583-1686`); `tests/test_telegram_app.py` takes in-process
      launch, the orchestrator callbacks, end-to-end, and the read-only commands
      (`:1107-1512`, `:1687-1760`). The harness stays in `test_telegram.py` and
      the three new modules import it - see DECISION.md section 3 for why it
      does NOT go to `conftest.py`. Delete the entry.
- [ ] Commit 6 - `tests/test_agent_store.py` (937). Three files:
      `test_agent_store.py` keeps CRUD, validation, the read-only gate, the
      reserved orchestrator, backend and model defaults and on-load migrations;
      `tests/test_agent_sessions.py` takes session mapping through the store and
      `SessionRegistry` history and ownership (`:333-572`);
      `tests/test_agent_outcomes.py` takes the durable run outcome,
      `request_input`, `report_back`, pending and acknowledge (`:573-937`).
      Delete the entry.
- [ ] Commit 7 - `web/src/agent-chat-view.test.ts` (1181).
      `agent-chat-log.test.ts` takes `renderChatLog`, `messageMeta` and
      `transcriptReply` (`:102-284`); `agent-chat-composer.test.ts` takes the
      slash palette and image attachments (`:764-854`); the view file keeps
      `createAgentChat`, cancel/stop, reattach, edit-to-fork and
      `startAgentChat`. Duplicate a fixture factory a new file needs when it is
      under ~10 lines; extract a shared `web/src/*-fixtures.ts` only if two
      files need more than ~20 lines of shared setup, and record the
      measurement. Delete the entry.
- [ ] Commit 8 - `web/src/host-view.test.ts` (997). `host-proposal.test.ts`
      takes the pending queue and the one-way gate (`:209-423`); the view file
      keeps escaping, the edges, decided actions, the record, `startHost`, the
      review-round fixes and digests. Delete the entry.
- [ ] Apply the epic comment policy (`AGENTS.md`, and the table in
      20260731-171411) to every module docstring and section comment that
      moves: each new file gets a docstring stating what behavior it covers,
      task-ID lore is dropped while its invariant is kept, and no new task ID is
      introduced. Confirm the set with a grep at work time, not from this list.
- [ ] Before EVERY commit: `ruff format <the files you edited>` /
      `cd web && npx prettier --write <the files you edited>` - scoped, never a
      whole dir. Generator-written import blocks are the reliable trigger and
      this reached x3 in 20260731-171431. Then re-measure the four near-cap
      siblings that are NOT in scope - `tests/test_host_digest.py` (884),
      `tests/test_host_actions.py` (882), `tests/test_telegram_approvals.py`
      (860), `tests/test_agent.py` (852) - because a re-wrapped import line can
      push one over 900, which the guard fails as a NEW violation.
- [ ] `git add` every new test file AND every task record before any
      `nix flake check` or `nix build`: they evaluate only git-TRACKED files,
      and this bit 20260731-171431 from the record side.
- [ ] Prove the guard at every commit, not only the tip:
      `git rebase master --exec 'python scripts/check_file_size.py'`. If a gate
      fires on MOVED code, measure before recording a cause:
      `git show master:<path>` into a scratch file and run the check against
      that.

## Definition of Done

- The size guard passes and its ALLOWLIST holds `scufris/app.py` and
  `tests/test_app.py` and nothing else
  (cmd: `python -c "import sys; sys.path.insert(0, 'scripts'); import check_file_size as c; sys.exit(0 if sorted(c.ALLOWLIST) == ['scufris/app.py', 'tests/test_app.py'] and c.check(c.REPO_ROOT, c.ALLOWLIST) == [] else 1)"`).
- No file under `tests/` or matching `web/src/**.test.ts` exceeds 900 lines
  except `tests/test_app.py`, which 20260729-103712 owns
  (cmd: `python scripts/check_file_size.py`).
- The guard is green at every commit on the branch, not only at the tip
  (cmd: `git rebase master --exec 'python scripts/check_file_size.py'`).
- Every test NAME collected on `master` still collects and the suite is 896 or
  more, reported as a set difference per commit rather than as a count
  (cmd: the name-set difference from Steps, base vs branch, for both
  `python -m pytest --collect-only` and `npx vitest list`).
- Each split is a move: the normalized code-line multiset difference between
  each base file and its new file set is empty except for named rewordings
  (cmd: the move proof from Steps, reported per commit in a table).
- Both canonical gates and both package builds pass
  (cmd: `nix flake check && nix build .#scufris .#scufris-web`).
- The frontend gate passes at its new file count
  (cmd: `cd web && npm run ci`).
- No comment in a file this task splits cites a task, spike or decision ID as
  its only justification - 8 such sites exist on `master`, in
  `test_host_action_api.py`, `test_agent_store.py`, `test_host_inspection.py`
  and `test_auth.py`; the command must print nothing
  (cmd: `rg -n "2026[0-9]{4}-[0-9]{6}" tests web/src --glob 'test_auth*.py' --glob 'test_agent_store.py' --glob 'test_agent_sessions.py' --glob 'test_agent_outcomes.py' --glob 'test_host_action_api.py' --glob 'test_host_action_decisions.py' --glob 'test_host_inspection.py' --glob 'test_host_thermal.py' --glob 'test_host_nix_store.py' --glob 'test_telegram*.py' --glob 'test_nixos_*.py' --glob 'conftest.py' --glob '*.test.ts'`).

## Notes

- Epic: 20260731-171411. Decision: `tasks/20260731-171432/DECISION.md`.
- Depends on: 20260731-171420. Run after the source splits so tests move once,
  not twice - all five source children have landed (`1253dfd`).
- The load-bearing choice is SPLIT BY BEHAVIOR, not by the package mirror,
  refuted by attribution measurements on three files; DECISION.md section 1.
  The mirror is taken only where a submodule owns a contiguous behavior block
  (`host/thermal.py`, `telegram/render.py`).
- Shared setup goes to `tests/conftest.py` only when its consumers are already
  cross-domain; the Telegram bot harness stays domain-local. DECISION.md
  section 3.
- Baselines to beat: 896 pytest tests, 22 web test files / 258 web tests, guard
  exit 0, `nix flake check` 5 green.
- `tests/test_app.py` (3813) and `scufris/app.py` stay allowlisted and belong to
  20260729-103712. Do not touch either.
- `tests/test_host_actions.py` (882) publishes fixtures to two siblings and is
  NOT in scope; its cross-test-module imports are left alone.
- Line estimates in DECISION.md section 4 are pre-split arithmetic on measured
  regions and are re-measured at work time; only the caps are load-bearing.
- A behavior fix or a genuinely broken test found on the way becomes its own
  task, not a fold-in.
- 20260731-233221 owns promoting the recurring lessons into repository guards;
  none of that work folds in here.
