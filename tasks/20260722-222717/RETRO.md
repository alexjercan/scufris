# Retro: T1 - orchestrator-only scufris MCP scoping

- TASK: 20260722-222717
- BRANCH: feature/orchestrator-only-mcp
- REVIEW ROUNDS: 1 (APPROVE, out-of-context; one declined NIT)

See TASK.md for what changed and why; this is process only.

## What went well

- Test-first with the right red: the new tests failed with `TypeError:
  unexpected keyword argument 'is_orchestrator'` before the impl existed - a
  red for the right reason, not a passing-before-code test.
- The argv-level harness test (a fake `codex` that dumps its own `sys.argv`)
  proved the flag reaches the process command line, which the `_mcp_overrides`
  unit test alone cannot. The out-of-context reviewer specifically called this
  out as a genuine end-to-end proof. Worth the extra fake.
- Caught the pre-existing mypy red early by comparing error COUNTS on master vs
  the branch (44 == 44) instead of assuming my change caused it - so it did not
  derail the cycle, and the DoD note is honest about it.
- Grepping `def stream` up front caught a 4th backend (OpenCode) the plan's
  "three implementations" missed, before it could bite.

## What went wrong

- The test doubles (`_stream_app_server` / backend `stream` fakes in
  test_backends.py and test_app.py) were discovered by `TypeError` at test time,
  not swept up front - and the app-chat failures (503 / RunPhase.ERROR) were the
  same root cause wearing a different mask. Root cause: I grepped implementors
  but not test stand-ins when changing a Protocol signature. This is the exact
  recurring lesson `protocol-signature-change-hits-the-doubles`, now at its 3rd
  occurrence.

## What to improve next time

- When changing a Protocol/interface method signature, in ONE sweep grep for
  every implementor AND every test double (`def <method>`), update them together,
  and run mypy explicitly - do not let the doubles surface as TypeErrors. (Now
  promoted to Pending promotions -> work skill.)

## Action items

- [x] Bumped `protocol-signature-change-hits-the-doubles` to x3 and moved it to
      LESSONS.md Pending promotions with a `-> work skill` target.
- No follow-up code tasks: T2/T3 already queued; the pre-existing mypy red is
  already tracked by task 20260720-174021.
