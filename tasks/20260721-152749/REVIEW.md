# Review: use enums/Pydantic for stringly-typed options

- TASK: 20260721-152749
- BRANCH: feature/typed-option-enums

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings from a fresh subagent with no sight
  of the implementing session; the in-session pass re-ran both suites and adopted
  R1.2, re-verifying mypy stays clean without the ignore)

A behavior-preserving refactor. Both suites green with NO pydantic serialization
warning (the reviewer grepped explicitly): backend `ruff` + `mypy` + `pytest`
(279 passed) and web `npm run ci` (frontend untouched - `git diff --stat web/`
empty, as the wire-unchanged claim requires). The five StrEnums serialize as their
raw string, so JSON/wire + `==` are unchanged; `test_enums.py` pins this (and
would fail under a plain `Enum`). Every `run.state`/`mark_*`/`model_copy` raw-string
site was swept and converted; the two validation-bypassing paths (`mark_finished`
coercion, `normalize_permission_mode` returning the enum) are handled. Legacy
`app_server|exec -> codex` still loads; the API input stays strict. The
converted-vs-left inventory matches the diff.

- [x] R1.2 (NIT) scufris/agent_store.py:245 - the `# type: ignore[arg-type]` on
  `normalize_permission_mode(...)` is stale now that the fn returns `PermissionMode`
  and the field is `PermissionMode`. Drop it.
  - Response: Removed. Confirmed `mypy scufris/` still clean (no issues in 20
    files) without the ignore.

- [ ] R1.1 (NIT) scufris/enums.py:46 - `AgentState.BLOCKED` is defined but never
  assigned today (no `mark_blocked`). Not a regression (the prior
  `AgentLifecycle = Literal[..., "blocked", ...]` had it too); a forward-looking
  member.
  - Response: Kept intentionally - preserves the pre-existing lifecycle member set
    and reserves the approval-waiting state for the run machinery; documented as
    such on the enum.

No pending manual DoD items (a pure code-quality refactor; the DoD is machine-proved
by test_enums.py + the unchanged full suite).
