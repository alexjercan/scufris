# Review: BC2 request_input sub-agent callback + role-scoped MCP tools

- TASK: 20260723-094303
- BRANCH: feat/request-input-tool

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings), in-session pass verified and recorded

The out-of-context reviewer ran the full suite (360 passed, ruff + mypy clean),
ran each DoD proof by name, neutralized the `preserve_waiting` predicate to
confirm `test_waiting_survives_same_run_completion` genuinely fails without it,
and tried to construct a WAITING-preservation misbehavior (empty run_id, stale
run, acknowledged, ERROR, delete-mid-run, mid-turn ordering) - none found. In
session I re-derived the load-bearing claim: the empty-`run_id` guard
(`bool(existing.run_id)`) is what stops a `run_id==""` WAITING from being wrongly
preserved, and `agent_runs.get(agent_id, "")` supplies the live run id so the
mid-turn WAITING and the turn-end DONE share it. No BLOCKER/MAJOR.

- [x] R1.1 (MINOR) CHANGELOG.md - within the same `[Unreleased]` section, the
  pre-existing Changed/Removed entries claimed the scufris server "is now
  ORCHESTRATOR-ONLY" while the new Added entry says "ROLE-SCOPED"; the two
  contradict in unshipped notes.
  - Response: fixed this round. Reconciled the section to the role-scoped
    end-state: the Changed entry now reads "ROLE-SCOPED" (orchestrator full
    surface, regular agents only `request_input`); the tatr-removal rationale
    dropped its stale whole-server "orchestrator-only" clause; my Added entry
    trimmed to the `request_input` capability + WAITING preservation (no duplicate
    role-model restatement). The line-69 "control tools ... (orchestrator-only)"
    stays - control tools ARE orchestrator-audience under role scoping, so it is
    accurate.
- [ ] R1.2 (NIT) scufris/mcp_server.py - local `import os` inside `_self_agent_id`
  and `_role` is redundant with a module-level import.
  - Response: declined with reasoning. `os` is NOT imported at module top - the
    file imports it LOCALLY in five places (`_api_base`, `_disabled_tools`,
    `main`, and these two). The local imports MATCH the file's established
    convention; a lone top-level import for two of five call sites would be the
    inconsistency. Left as-is.
- [ ] R1.3 (NIT) scufris/agent_store.py - `assert existing is not None` in
  `mark_finished` is a mypy-narrowing aid, dead at runtime.
  - Response: left as-is (the reviewer agreed it is not worth changing). It
    documents the `preserve_waiting` -> non-None narrowing at the read site.

No open `manual:` DoD items for this task (all proofs are `test:`/`cmd:`).
</content>
