# Retro: BC2 request_input sub-agent callback + role-scoped MCP tools

- TASK: 20260723-094303
- BRANCH: feat/request-input-tool
- REVIEW ROUNDS: 1 (APPROVE)

## What went well

- Store-first: building the correct primitive (`AgentStore.request_input` + the
  run-id-keyed WAITING preservation in `mark_finished`) BEFORE the mcp/app/agent
  wiring meant those upper layers were thin, obviously-correct glue over an
  already-tested contract. The hardest bit (mid-turn signal vs turn-end clobber)
  was solved and pinned once, at the bottom.
- The role model from `DECISION.md` (Option B) paid off exactly as predicted:
  generalizing the `is_orchestrator` gate into an audience - not a second server -
  means BC3's orchestrator-only `pending_agents`/`acknowledge` will drop in as
  another `orchestrator`-audience entry with zero new plumbing.
- The out-of-context reviewer APPROVEd with no MAJOR after specifically trying to
  break the preservation logic (neutralized the predicate, hunted edge cases).
  The design held because the edge cases (empty run_id, stale run, ERROR,
  acknowledged, delete-mid-run) were each guarded and tested up front.

## What went wrong

- Three same-file test failures: `main()` now role-scopes the process-global
  `mcp` tool registry (`apply_role`), and `test_main_configures_logging_and_runs`
  invoked `main()` without restoring it, leaking a trimmed tool set into every
  later test in the file. Root cause: I added a global-singleton mutation to a
  function a test already exercised, without checking that test isolated the
  mutation (the file already had a `restore_tool_registry` fixture for exactly
  this - I just didn't apply it to the newly-mutating caller).
- Five test doubles broke with `TypeError: ... unexpected keyword argument
  'agent_id'`: widening the shared `stream` protocol signature keeps the
  production impls compiling (defaulted param) but every hand-written fake with an
  explicit signature (`fake_app_server`, `FakeBackend.stream`) still breaks at
  call time. Root cause: I changed a shared contract without sweeping its stubs.

## What to improve next time

- When adding a mutation of a process-global singleton to a function under test,
  check that its test snapshots/restores the singleton (or reuse the existing
  restore fixture) in the SAME edit.
- When widening a shared Protocol/ABC method signature, grep for its test doubles
  (`def fake_...`, class `.stream(` stubs) and update them in the same change -
  a defaulted param compiles the impls but not the explicit-signature fakes.

## Action items

- [x] Ledger: `global-singleton-mutation-needs-its-tests-restore-fixture` (x1) and
  `widening-a-shared-signature-needs-a-test-double-sweep` (x1).
- No follow-up code tasks. Tracked non-blocker (in TASK.md Notes): claude
  `--mcp-config` parity so `request_input` reaches claude sub-agents.
</content>
