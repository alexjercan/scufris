# worktree pytest guard: enforce python -m pytest in sprouts

- STATUS: CLOSED
- PRIORITY: 0
- TAGS: backlog, bug

## Story

As a scufris developer, I want a guard (or clear enforced convention) that
prevents bare `pytest` in a sprout worktree from silently importing the MAIN
checkout, so that the worktree import-shadowing trap stops costing cycles. In
the nix dev shell `import scufris` resolves to CWD, and the console-script
`pytest` does not put CWD first on sys.path, so bare `pytest` in a worktree runs
against the wrong tree. The fix is `python -m pytest`.

## Steps

- [x] Add a conftest.py assertion (or a check-script) that fails fast if tests are importing scufris from a path outside the current worktree.
- [x] Chose the conftest guard over a wrapper (fails fast at collection; no wrapper to bypass).
- [x] Verify: the guard fires when scufris resolves outside cwd; `python -m pytest` passes (203); `python -m pytest` passes.
- [x] Document in AGENTS.md.

## Definition of Done

- Running tests from a sprout worktree cannot silently exercise the main checkout (manual: reproduce in a sprout and confirm the guard fires).

## Notes

- Promotes the `nix-devshell-import-resolves-to-cwd-source` lesson to a guard.
