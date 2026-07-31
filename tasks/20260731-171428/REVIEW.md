# Review: Split the agent runtime modules under the size cap

- TASK: 20260731-171428
- BRANCH: refactor/split-agent-runtime-modules

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tasks/20260731-171428/TASK.md:160 - the close-out's
  Difficulties paragraph claims "the pre-split file was one of 17 in the repo
  that the formatter would rewrite". Independently re-derived at the merge-base:
  `ruff format --check` on `scufris/{agent,agent_store,backends,sessions}.py` at
  022e472 reports "4 files already formatted". The 17 are unrelated files and
  none of the four split modules is among them, so the recorded diagnosis is
  false - the reformat was caused by lines written during the split, not
  inherited. Replace the sentence with what was measured: the four pre-split
  modules were format-clean, and `ruff format` rewrote the newly written code in
  `agent_store/outcomes.py`.
  - Response: Confirmed and corrected. The Difficulties paragraph now states the measured fact (all four pre-split modules format-clean at the merge-base) and attributes the reformat to the delegating calls written during the split.
- [x] R1.2 (MINOR) scufris/backends/__init__.py:27 - `_context_from_status` is
  re-exported by the facade with no consumer of that path: `claude.py` and
  `opencode.py` both import it directly from `.base`, and nothing outside the
  package references it (re-derived by grep across the tree). The comment at
  :114 justifies it as "the shared adapter helper" but names no facade caller.
  Delete it from the `from .base import ...` line (:27), from `__all__` (:116),
  and drop that clause from the comment (:114).
  - Response: Confirmed and removed from the facade import, `__all__` and the comment. `claude.py` and `opencode.py` keep importing it from `.base`.
- [x] R1.3 (NIT) scufris/agent_store/reserved.py:46 - `orchestrator_record` and
  `host_record` each call `orch_backend(settings)` twice (lines 46/47 and
  75/76); the pre-split methods computed it once. Hoist
  `backend = orch_backend(settings)` to the top of each function and use it for
  both `backend=` and `default_model_for(settings, backend)`.
  - Response: Hoisted `backend = orch_backend(settings)` in both functions.
- [x] R1.4 (NIT) scufris/agent_store/reserved.py:20 - dropping "(from B5c)" from
  `_ORCHESTRATOR_DESCRIPTION` changes a user-visible string, not a comment.
  Nothing asserts on it and the close-out declares it, so keep the change, but
  record it as a deliberate string edit rather than filing it under comment-lore
  cleanup.
  - Response: Recorded. The close-out now calls the `_ORCHESTRATOR_DESCRIPTION` edit out separately as a deliberate user-visible string change, not comment-lore cleanup.

### Verified in this round

- Re-derived R1.1 and R1.2 in-session rather than accepting them: ran
  `ruff format --check` against the four modules extracted from 022e472 (4 files
  already formatted), and grepped every `_context_from_status` reference in the
  tree (8 hits, all inside `scufris/backends/`).
- Reran the full gate in-session: `check_file_size.py` green, `ruff check`
  green, `mypy` clean, `pytest` 896 passed / exit 0 - the same count the
  close-out records as the baseline. `nix flake check` built ruff, mypy, pytest
  and filesize; only `records` failed, with the documented tatr-0.1.0
  `unplanned-in-progress` false positive.
- Out-of-context reviewer additionally confirmed by AST sweep: every name any
  file imports from the four packages still resolves, no submodule imports
  through its own package `__init__`, no cycle, layering intact,
  `sessions/models.py` imports nothing from `scufris`. Method bodies are
  byte-identical old vs new except the 8 `AgentStore` methods that now delegate,
  each of which preserves its guards and field values.
- Monkeypatch targets bind to the globals the read sites actually use, so none
  of the 5 repointed strings is a silent no-op. `git diff --stat -- tests/` is
  2 files / 5 lines, no import line changed.
- Both decide-once points landed as recorded: `agent/appserver.py` 466 (no
  `sandbox.py`), `agent_store/store.py` 595 after shedding the signal writers.
- No pending `manual:` proofs on this task.

## Round 2

- REVIEWER: in-session (the four fixes are mechanical edits to lines Round 1
  named; each was re-derived and the full gate rerun in-session, so a second
  cold read would add nothing)
- VERDICT: APPROVE

All four Round 1 findings verified fixed; no fix introduced a regression.
Re-derived R1.1's correction (`ruff format --check` on the four modules at
022e472: "4 files already formatted") and R1.2's removal (`_context_from_status`
now has 5 references, all inside `scufris/backends/`, none through the facade).
Full gate rerun after the fixes: `check_file_size.py` green, `ruff check` green,
`ruff format --check` clean on all four packages, `mypy` clean, `pytest` 896
passed / exit 0. No pending `manual:` proofs.
