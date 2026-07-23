# Review: mypy baseline cleanup (enum-vs-str call sites)

- TASK: 20260723-182253
- BRANCH: chore/mypy-baseline-cleanup
- DATE: 20260723
- REVIEWER: out-of-context agent (round 1)
- VERDICT: APPROVE

## Round 1 - VERDICT: APPROVE

Reviewed against the actual code; reviewer re-ran the full gate under
`nix develop`: mypy clean (0 errors, 46 files), ruff clean, full pytest green.

### Findings

1. [verified-ok] Enum-value correctness: every substituted member equals the
   string it replaced (`Backend.CODEX="codex"`, `.MOCK="mock"`, `.CLAUDE="claude"`,
   `AuthMode.API_KEY="api_key"`, `AgentState.DONE="done"`), so each test exercises
   the same path.
2. [verified-ok] Raw-string `==` assertions still hold (StrEnum compares equal to
   its string): `.backend == "mock"`, `.state == "done"`, `s.agent_backend ==
   "claude"` untouched and passing. No test meaning changed.
3. [verified-ok] test_enums.py coercion intent preserved: the three coercion
   inputs stay raw strings with scoped `# type: ignore[arg-type]` + rationale,
   including the legacy `"app_server" -> Backend.CODEX` fold. Not wrongly
   converted.
4. [verified-ok] (a) `outcome()` (`RunOutcome | None`) narrowed with
   `assert outcome is not None` before `.acknowledged` - correct, masks no real
   bug (prior lines already establish the outcome exists). (b)
   `dict[str, object] -> dict[str, Any]` scoped to the ONE handler that subscripts
   the body; the other 5 helpers only do whole-dict `==` and stay `object`. The
   difference is load-bearing.
5. [verified-ok] No collateral sed damage: surviving raw strings are the
   `mark_finished(backend="codex")` param (typed `str | None`, correctly a string)
   and the deliberate test_enums.py inputs. No assertion/session_id/project name
   rewritten.
6. [verified-ok] Imports: each file references only enums it imports; ruff F401
   finds no unused imports.

No substantive issues. Ship it.
