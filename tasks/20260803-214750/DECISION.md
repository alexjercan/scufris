# Decision: split the singular agent surface rather than deleting it

- DATE: 20260804-041340
- STATUS: ACCEPTED
- TASK: 20260803-214750
- TAGS: v0.2.0, architecture, api, storage

## Context

The task as written says the `/api/agent/*` surface "exists only for backwards
compatibility" and instructs deleting the whole router, plus the pre-database
JSON import, plus four of the five Alembic revisions.

The JSON import half holds up. The router half does not. Three independent
sources in the tree contradict it:

- The module's own docstring (`scufris/api/legacy_agent.py:1-19`): "Three things
  live only here, because the console is their only caller: the effective-config
  view and its whitelisted PATCH (the settings page), the in-process 'try it'
  tool runner, and the orchestrator's session switcher."
- `scufris/README.md:316` describes it as "the orchestrator-scoped singular
  surface **kept for the console**".
- Route diffing against `scufris/api/agent_runs.py:461-510`. Of sixteen routes,
  four have a plural equivalent.

| Route | Plural equivalent | Verdict |
|---|---|---|
| GET `/api/agent/usage` | `/api/agents/orchestrator/usage` | alias - docstring says so |
| GET `/api/agent/memory` | `/api/agents/orchestrator/memory` | alias - docstring says so |
| GET `/api/agent/account` | `/api/agents/orchestrator/account` | alias - docstring says so |
| GET `/api/agent/health` | `/api/agents/orchestrator/health` | alias - "delegates to the same service" |
| GET `/api/agent/info` | none | console view: model + auth mode + enabled |
| GET/PATCH `/api/agent/config` | none | the `WRITABLE_KEYS` whitelist |
| GET `/api/agent/tools` | `/api/agents/{id}/tools` is DIFFERENT | console audience; the docstring contrasts the two explicitly |
| GET `/api/agent/mcp` | `/api/agents/{id}/mcp` is DIFFERENT | the orchestrator's scufris+den servers |
| POST `/api/agent/tools/{name}/run` | none | the in-process "try it" runner |
| GET `/api/agent/sessions` | none | the session switcher |
| POST `/api/agent/session` | none | new / switch |
| POST `/api/agent/session/fork` | `/api/agents/{id}/fork` forks a RUN | different subject |
| GET `/api/agent/context` | none | window + token snapshot |
| GET/DELETE `/api/agent/session/{id}` | none | transcript, delete |

And the console is live. `web/src/agent-view.ts` (the landing orchestrator page,
entry `agent.ts`) and `web/src/agent-settings-view.ts` (the unified settings
page, mounted at BOTH `/settings` and `/agents/<id>/settings`, for EVERY agent)
call the console-only routes today. `/api/agent/config` is fetched for every
agent, not just the orchestrator (`web/src/agent-settings-view.ts:352`).

Nothing schedules their repair. The task's NOTES.md claims "the pages that
render it belong to `20260729-102157`", but that epic's Child Tasks are the
project-task API, the flow board and the artifact viewer. None of them deletes
or repoints the agent console.

So deleting the whole router breaks the settings page, the tool runner and the
session switcher with no scheduled repair - and `nix flake check` stays GREEN
while it does, because the vitest suites mock `fetch` and never touch a real
route.

## Decision

**D1. Split the surface: delete four routes, keep twelve.** Apply the task's own
stated principle - "the SAFE half: code with no replacement to wait for" - to
what the code actually is.

- Delete the four alias routes and their tests. No replacement to wait for; the
  plural routes already answer.
- Keep the twelve console-only routes. They DO have a replacement to wait for
  (the console rewrite), so by this task's own rule they are out of scope.
- Move them out of a module named `legacy_`: `scufris/api/legacy_agent.py` ->
  `scufris/api/console.py`, `LegacyAgentDeps` -> `ConsoleDeps`,
  `build_legacy_agent_router` -> `build_console_router`,
  `tests/test_legacy_agent_router.py` -> `tests/test_console_router.py`. The
  DoD's "no legacy surface survives" grep is then satisfied honestly rather than
  by deleting a working console.

**D2. The URL prefix stays `/api/agent/*`.** Renaming the singular prefix would
touch ~20 call sites across four web modules and the route-contract table and
buys nothing this task needs. The module name carried the false "legacy" claim;
the URL is just a URL. The console rewrite decides the URL shape when it decides
the pages.

**D3. `test_writable_keys_match_the_api_update_model` survives, re-pointed.**
`api/agents.py`'s `AgentUpdate` (`:63-72`) is not an equivalent: it carries
per-agent ROW fields (name, backend, model, description, goal, task_id,
permission_mode). `AgentConfigUpdate` mirrors `settings_store.WRITABLE_KEYS`,
the global settings whitelist. Different subjects. Under D1 the question
dissolves - `AgentConfigUpdate` moves to `scufris/api/console.py` with its
routes and the test re-points at the new module path. The cross-check is kept,
not lost.

**D4. `tests/test_projects.py:326` goes, and the behavior change is recorded.**
The test asserts a damaged `projects.json` beside a database fails startup by
name rather than showing an empty Projects page. That promise is a property of
the IMPORT: once `import_legacy_state` is gone nothing reads `projects.json`, so
a damaged one is neither refused nor silently believed - it is ignored, and the
Projects page reads the database. The failure cannot occur. Delete the test and
state in the CHANGELOG that a leftover `projects.json` is ignored entirely.

**D5. Refuse a pre-v0.2.0 database by naming the five squashed revisions.**
`scufris/db/migrate.py:178` already has `_known_revision`, and `:211-215`
already refuses an unknown revision with "written by a newer version" - which a
v0.1.x database now hits, and which is the opposite of the truth. Rather than
replacing that message, keep a frozenset of the five ids being squashed
(`8f8087f3cc9c`, `9b6587dab793`, `380a27d7fddb`, `3a5161b39846`,
`e054a39a5fae`) and branch on it. Both refusals stay true, and
`test_a_database_from_a_newer_scufris_is_refused_without_a_backup` keeps its
subject.

**D6. `examples/host_agent.py` patches `scufris.orchestrator.runs`.** Its flow
posts to `/api/host/actions` and `/api/agents/{id}/chat` (`:203,246`). Planning
put the patch target at `scufris.api.agent_runs` on the strength of the router
that serves those paths; implementation found that wrong. `api/agent_runs.py`
imports `get_backend` for a diagnostics read (`:456`), but the call that LAUNCHES
the turn - and therefore the one the example's `RecordingBackend` has to
intercept for the resumed prompt to be its recording - is
`scufris/orchestrator/runs.py:201`. All three bind sites `tests/conftest.py:193`
lists are real; the example needs the launching one.

Verified by RUNNING the example, which is what caught the mistake. The example
is broken on master for an unrelated reason (see Consequences), so the run was
done with `PYTHONPATH=packages/hostd/tests` standing in for the `sys.path` fix
`20260804-041340` owns; green end to end, with the recorded denial prompt in the
output. The DoD proof stays blocked on that task landing.

**D7. The backup-before-migrate proof gets its second revision from a STAGED
`versions/`, not from the shipped tree.** Squashing to one baseline removes the
"behind head at a revision this build knows" state, and
`test_the_backup_is_taken_on_the_real_migration_path` is the only test that
reaches `backup_database` through `upgrade_to_head`
(`scufris/db/migrate.py:241`) - the other two call it directly. Round 1 caught
the first attempt retiring it behind a `pytest.skip`, which left the wiring
unproven and `migrate.py:241` deletable with the suite green (REVIEW.md R1.1).

Instead the `behind_head` fixture copies the shipped Alembic environment to a
tmp directory, writes one throwaway follow-on revision into the copy, and
monkeypatches `migrate._migrations_dir` at it. Every entry point in `migrate.py`
reads that one function, so `head_revision`, `_alembic_config` and
`upgrade_to_head` all move together and the production path is exercised
unmodified. Nothing about the property depends on what the throwaway revision
does, only that head is one step past the database - it creates one table, which
is also what discriminates a pre-migration copy from a post-migration one.

Rejected: shipping a second real revision purely to give the test a step (it
would have to do something, and the DoD says exactly one revision exists), and
recording the lost proof as a consequence for the next revision to restore
(the invariant is a security property - a copy before an irreversible schema
move - and leaving it unenforced for a release is how it quietly stops
happening).

## Alternatives considered

**Delete the whole router and accept a broken console until the rewrite.**
Rejected: no task owns the repair, and the gates would not catch it. A green
`nix flake check` over a settings page that 404s is the worst available outcome.

**Delete the whole router AND the console pages here.** Rejected: that is the
demolition half this task explicitly excludes, and the operator loses the
settings page and session switcher with nothing to replace them.

**Rename the URL prefix to `/api/console/*` while renaming the module.**
Rejected under D2: churn across four web modules and the route contract, for a
naming question the console rewrite will reopen anyway. YAGNI.

**Leave the module named `legacy_agent.py` and narrow the DoD grep.** Rejected:
the name is what made the plan wrong in the first place. Fixing the name is the
cheap part and it stops the next planner from making the same read.

## Consequences

Bought: the JSON import and four dead aliases go, five revisions become one, and
the tree carries no module named `legacy` - while the operator console keeps
working. The console's twelve routes are now named for what they are, so the
next reader is not told they are compatibility shims.

Cost: this task no longer deletes ~1050 lines of router and router tests; those
lines MOVE. The real deletion is the JSON import stack (~1600 lines:
`db/legacy/` 740, `test_db_legacy.py` 619, `state_migration.py` 232) plus the
four aliases and their cases. NOTES.md's "~2600 lines deleted" estimate was
built on the premise this record overturns.

Foreclosed: any upgrade path from v0.1.x data. That is the accepted v0.2.0
position (`20260801-154211`: "the old database is dropped, not migrated"), and
this task is where it becomes irreversible.

Deferred: deleting the console routes, which happens when their replacement is
live. `20260729-102157` does not currently own that; if it stays out of scope,
the console keeps these routes and someone must schedule the rewrite.

Discovered and seeded: `examples/host_agent.py` and
`examples/telegram_approval.py` are broken on master. `6d998c8` (the hostd
carve) moved `tests/test_host_actions.py` to `packages/hostd/tests/` and did not
update the examples' own `sys.path` inserts, and `flake.nix:250-268` runs no
example, so nothing caught it. Filed as `20260804-041340`; D6's proof depends on
it.
