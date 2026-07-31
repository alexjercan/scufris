# Decision: split the test suites by BEHAVIOR, not by the package mirror

- DATE: 2026-08-01
- STATUS: ACCEPTED
- TASK: 20260731-171432
- TAGS: refactor, maintainability, kiss, testing
- EPIC: 20260731-171411

## Context

Eight test files are over the 900-line test cap and allowlisted in
`scripts/check_file_size.py`. Confirmed on `1253dfd`, not trusted from the epic
record:

| File | Lines | Over by |
|-|-|-|
| `tests/test_telegram.py` | 1760 | 860 |
| `tests/test_host_action_api.py` | 1285 | 385 |
| `tests/test_auth.py` | 1219 | 319 |
| `web/src/agent-chat-view.test.ts` | 1181 | 281 |
| `tests/test_host_inspection.py` | 1076 | 176 |
| `tests/test_nixos_config_change.py` | 1044 | 144 |
| `web/src/host-view.test.ts` | 997 | 97 |
| `tests/test_agent_store.py` | 937 | 37 |

Baseline before any edit, to be re-established on the sprout: `python
scripts/check_file_size.py` exits 0; `python -m pytest --collect-only` reports
**896 tests collected**; `cd web && npm run ci` is green at **22 files / 258
tests**; `nix flake check` passes all 5 checks.

The five landed epic children moved the code under every one of these tests.
`scufris/telegram`, `scufris/host`, `scufris/auth`, `scufris/agent_store`,
`scufris/hostd` and `scufris/mcp_host_tools` are now packages, and the four
frontend views are now flat sibling modules. So every file in scope tests a
package or a sibling set rather than the single module it was written against,
and the load-bearing question is whether the tests should MIRROR that new
structure or split by behavior independently of it.

## Decision

### 1. Split by BEHAVIOR. The package mirror is refuted by measurement, not by taste

The mirror is the attractive answer - one test file per source module, a
mechanical rule, no judgement. It was tested by attributing every test function
to the submodule whose public names it references (`ast` walk over each test
body against each package's module-level names), weighting by the test's own
line span. Three independent files, three refutations:

| Test file | Attribution | Why the mirror fails |
|-|-|-|
| `tests/test_auth.py` (1219) | `?` 449, `policy` 405, `store` 113, `credentials` 21 | 449 lines - the largest share - reference NOTHING in `scufris/auth/`. They test the deny-by-default middleware in `scufris/app.py` and the secret stripping in `scufris/agent/env.py` and the backends. The mirror has no file to put them in. |
| `tests/test_telegram.py` (1760) | `?` 655, `render` 182, `bot` 177, `text` 93 | Same shape, worse: the transport tests drive the bot through `respx` HTTP mocks and the orchestrator callbacks through `scufris.app.build_telegram_callbacks`. 37% of the file names no `scufris/telegram/` symbol at all. |
| `tests/test_agent_store.py` (937) | `store` 649, `registry` 72, `records` 47 | The opposite failure: the mirror does not SPLIT. `store.py` is the facade every call goes through, so a mirror yields one 649-line file and two stubs, and the outcome tests - a real, separately-named behavior living in `outcomes.py` - land in the `store` bucket because they are spelled `store.request_input(...)`. |

Two failure modes, both fatal, and they are the same fact seen twice: a test
addresses a package through its FACADE and its HTTP surface, so the call
surface a test uses is evidence about the API, not about what the test is
about. `tests/test_host_inspection.py` shows it a third way - its four biggest
tests (`..._covers_units_logs_and_storage` 94 lines, `..._output_is_bounded`
61, `..._degrade_explicitly` 52, `..._distinguishes_empty_from_broken` 50)
assert one property ACROSS all six domains at once and are unfilable under any
single module by construction.

So the split axis is the behavior under test, which is what the task's own
Notes already say ("split by what is being tested, not by fixture"). This is
the mirror image of 20260731-171431's frontend answer and for a symmetric
reason: there, the compiler made the structural rule safe, so structure won;
here, the structure is not what the tests are organized around, so it loses.

### 2. Where a submodule DOES own a contiguous behavior block, the two agree - take it

The rule is not "ignore the package". Two blocks measured contiguous and
single-module, and both become their own file under their module's name:

- `scufris/host/thermal.py`: `tests/test_host_inspection.py:313-539`, 10
  consecutive tests, 179 attributed lines, every one attributed to `thermal`
  alone. Becomes `tests/test_host_thermal.py`.
- `scufris/telegram/render.py`: `tests/test_telegram.py:950-1106` and
  `1583-1686`, the pure formatters (`format_reasoning`, `format_tool`,
  `render_reply`, `markdown_reply`, the `/stats` and `/settings` renderers),
  182 attributed lines, no fixture beyond a `ToolCall` factory. Becomes
  `tests/test_telegram_render.py`.

Both are cases where the module IS the behavior. Neither is a mirror applied on
principle.

### 3. Shared setup goes to `tests/conftest.py` only when its consumers are already cross-domain; otherwise it stays in a domain module and siblings import it

`tests/test_host_action_api.py` currently publishes `ORIGIN`, `_login` and
`_settings` to three sibling modules (`test_host_digest.py`,
`test_telegram_approvals.py`, `test_nixos_config_change.py`), and
`tests/test_host_actions.py` publishes `host_runner`, `host_files`, `NIX`,
`BUILT_SYSTEM`, `OLD_SYSTEM`, `RUNNING_SYSTEM` to two. Splitting
`test_host_action_api.py` moves the definition site of the first set, so it has
to land somewhere stable.

The rule, and its two halves:

- **Cross-domain -> `tests/conftest.py`.** `ORIGIN`/`_login`/`_settings`/
  `_propose` are consumed by the host-action, host-digest, telegram-approvals
  and nixos-change domains - four domains, and after the split, five modules.
  That is the definition of shared setup and `conftest.py` (296 lines, already
  home to `_Helper`, `helper`, `make_fixture_stats` and the autouse
  `_isolate_state_dir`) is the existing place for it. This is the task's own
  Step, and it deletes three cross-test-module imports as a side effect.
- **Domain-local -> stays put, siblings import it.** The Telegram bot harness
  (`_update`, `_ok`, `_Recorder`, `_make_bot`, `_events_bot`, `_capture_sends`,
  `_record_calls`, `_fake_settings_ops`, ~166 lines at
  `tests/test_telegram.py:86-251`) is read only by Telegram modules. Lifting it
  into `conftest.py` would make every unrelated test file in the repo carry
  `respx`, `httpx` and the bot harness in its collection context - the exact
  context cost this epic exists to remove. It stays in `tests/test_telegram.py`
  and the three new Telegram modules import from it, which is the repo's
  measured idiom (`from test_host_actions import host_runner`, three existing
  instances).

`tests/test_host_actions.py` (882) is NOT in scope and its fixtures are not
moved; `tests/test_nixos_config_change.py`'s imports from it carry over
unchanged to whichever half needs them.

### 4. The target shape: eight files become twenty, one commit per over-cap file

Line estimates are pre-split arithmetic on the measured regions and will be
re-measured at work time; only the caps are load-bearing.

| Commit | File | Becomes | ~Lines |
|-|-|-|-|
| C1 | `tests/test_host_action_api.py` (1285) | `test_host_action_api.py` - who may propose, approve, revert, cancel; audit; the helper being absent or wrong-secret | 640 |
| | | `test_host_action_decisions.py` - the decision core: confirmation strength, the approval race, expiry, restart durability, operator binding, delivery back to the agent, the queue row | 600 |
| | | + `ORIGIN`/`_login`/`_settings`/`_propose` to `conftest.py`; 3 siblings repointed | |
| C2 | `tests/test_nixos_config_change.py` (1044) | `test_nixos_activation.py` - the plan, the preview, rollback, apply, against the helper | 500 |
| | | `test_nixos_config_change.py` - build a commit, propose it, and the HTTP surface | 560 |
| C3 | `tests/test_auth.py` (1219) | `test_auth.py` - the password hash, the loopback fail-closed policy, login/logout, session lifetime and the session store | 450 |
| | | `test_auth_boundary.py` - what a request must carry: session, CSRF, origin, throttle, browser redirects, the streaming and Telegram bridges | 480 |
| | | `test_auth_machine.py` - machine callers: MCP tools under auth, the machine token, and every secret stripped from an agent's environment | 400 |
| C4 | `tests/test_host_inspection.py` (1076) | `test_host_inspection.py` - the four DoD properties across the six domains, plus units, journal, network, render | 640 |
| | | `test_host_thermal.py` - throttling and the thermal report | 290 |
| | | `test_host_nix_store.py` - the nix store, packages, flake status and closure diff parsing (decide at work time: fold back if it measures under ~150) | 230 |
| C5 | `tests/test_telegram.py` (1760) | `test_telegram.py` - the long-poll transport, command dispatch, the typing action, and the shared bot harness | 640 |
| | | `test_telegram_stream.py` - the live thinking bubble, tool widgets and the final answer | 380 |
| | | `test_telegram_render.py` - the pure formatters and the `/stats` and `/settings` renderers | 330 |
| | | `test_telegram_app.py` - in-process launch, the orchestrator callbacks, end-to-end, and the read-only commands | 560 |
| C6 | `tests/test_agent_store.py` (937) | `test_agent_store.py` - CRUD, validation, the read-only gate, the reserved orchestrator, backend and model defaults, on-load migrations | 450 |
| | | `test_agent_sessions.py` - session mapping through the store and `SessionRegistry` history and ownership | 250 |
| | | `test_agent_outcomes.py` - the durable run outcome, `request_input`, `report_back`, pending and acknowledge | 390 |
| C7 | `web/src/agent-chat-view.test.ts` (1181) | `agent-chat-view.test.ts` - `createAgentChat`, cancel/stop, reattach, edit-to-fork, `startAgentChat` | 750 |
| | | `agent-chat-log.test.ts` - `renderChatLog`, `messageMeta`, `transcriptReply` | 250 |
| | | `agent-chat-composer.test.ts` - the slash palette and image attachments | 180 |
| C8 | `web/src/host-view.test.ts` (997) | `host-view.test.ts` - escaping, the edges, decided actions, the record, `startHost`, the review-round fixes, digests | 740 |
| | | `host-proposal.test.ts` - the pending queue and the one-way gate | 280 |

Each commit deletes exactly its own `ALLOWLIST` entry, so the guard is green at
every commit and not only at the tip - proved with
`git rebase master --exec 'python scripts/check_file_size.py'`, not asserted.
C2 lands after C1 because it imports the helpers C1 moves. The other six are
independent. After this task the allowlist holds `scufris/app.py` and
`tests/test_app.py` and nothing else.

### 5. Frontend test fixtures are duplicated, not extracted, unless a measurement says otherwise

`agent-chat-view.test.ts` carries five small factories (`tool`, `reply`,
`config`, `flush`, `blobText`, ~30 lines total) and `host-view.test.ts` carries
its own. Vitest has no `conftest`, so sharing them means a new
`web/src/*-fixtures.ts` module - which the size guard would cover at the
600-line SOURCE cap and which webpack would not bundle (it pins nine entry
files). A new module for one three-line factory is the abstraction a single
caller does not demand. Rule at work time: duplicate a factory a new file needs
if it is under ~10 lines; extract a shared module only if two files need more
than ~20 lines of shared setup, and record the measurement if so.

## Alternatives considered

- **Mirror the package structure** - one test module per source submodule.
  Rejected on the measurement in section 1: it cannot place 449 lines of
  `test_auth.py` or 655 of `test_telegram.py` anywhere, and on
  `test_agent_store.py` it does not split the file at all (649 of 937 lines
  attribute to the `store` facade).
- **Split each file at its `# ---` section markers, mechanically.** Tempting,
  and it is where most of the cuts land anyway - but `test_auth.py:841-1219`
  and `test_host_inspection.py:901-1076` are both a "review round 1
  regressions" bucket, which is a HISTORY grouping, not a behavior. Those two
  regions are redistributed by what each test asserts. Taking the markers
  literally would preserve exactly the fixture-shaped grouping the task's Notes
  reject.
- **Split only far enough to clear 900.** Cheapest: `test_agent_store.py` needs
  to shed 37 lines. Rejected - the epic's cap is a ratchet and clearing it by
  one line is not clearing it (20260731-171430 settled this for `auth.py`), and
  a 899-line test file is not a component that fits one context.
- **Lift the Telegram bot harness into `conftest.py` too**, for one rule
  instead of two. Rejected in section 3: 166 lines of `respx`/`httpx` bot
  machinery in the root conftest is paid by every test module in the repo.
- **Move `test_host_actions.py`'s fixtures to `conftest.py` in the same pass**,
  finishing the cross-test-module imports. Rejected as scope: that file is 882
  lines, not allowlisted, and nothing in this task's DoD requires it. It is a
  candidate for a later task if the pattern recurs.
- **One squashed commit for all eight files.** Rejected: the files are
  independent (only C2 depends on C1), so the 20260731-171430 shape applies -
  per-file commits let `git rebase --exec` prove the guard at each one. Nothing
  here becomes a package, so no intermediate state is unrepresentable.
- **Do nothing.** The epic's Done Means 3 requires the allowlist to hold only
  `scufris/app.py` and `tests/test_app.py`; deferring leaves the epic open and
  leaves eight files that cannot be read and changed in one context.

## Consequences

- Twelve new test files. `tests/` goes from 39 to 48 Python modules and
  `web/src` from 22 to 25 `.test.ts` files, so `npm run ci`'s file count moves
  22 -> 25 and the pytest per-file collection summary gets nine more rows. Both
  numbers are recorded baselines and will need updating in the close-out.
- Test NAMES are preserved but their FILE changes, so every `tasks/` record and
  `LESSONS.md` entry citing `tests/test_auth.py::test_...` by path stops
  resolving. Those are history and are not rewritten - the same call
  20260731-171431 made for `common.ts:<line>` citations. `AGENTS.md`'s live
  citation of `tests/test_host_mcp_server.py::test_the_agent_has_no_tool_...`
  is NOT in scope and is unaffected; any other live doc citation into a split
  file must be repointed in the same commit
  (`verify-a-doc-citation-by-running-the-grep`).
- `conftest.py` grows by roughly 45 lines and becomes the definition site of
  the host-action login helpers, so a change to the login flow is one edit
  rather than four. It also becomes a file every test collection loads - the
  reason section 3 draws a line at cross-domain consumers.
- Four sibling test files sit near the cap and are NOT in scope:
  `test_host_digest.py` (884), `test_host_actions.py` (882),
  `test_telegram_approvals.py` (860), `test_agent.py` (852). C1 edits import
  lines in three of them. An import repoint that `ruff format` re-wraps can
  push one over 900, which the guard fails as a NEW violation, not a stale
  entry. Every commit re-measures all four before it is made.
- A split that drops or weakens a test is the regression this task is most
  exposed to, and a line-count check cannot see it. The proof is a test-NAME
  set difference, not a count: `python -m pytest --collect-only` prints
  nodeids, and stripping the file prefix makes the sets comparable across a
  move. `npx vitest list` does the same for the frontend. Both are in the DoD.
