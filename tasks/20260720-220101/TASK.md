# worktree pytest guard: enforce python -m pytest in sprouts

- STATUS: OPEN
- PRIORITY: 0
- TAGS: backlog,bug

## Story

As a scufris developer, I want a guard (or clear enforced convention) that
prevents bare `pytest` in a sprout worktree from silently importing the MAIN
checkout, so that the worktree import-shadowing trap stops costing cycles. In
the nix dev shell `import scufris` resolves to CWD, and the console-script
`pytest` does not put CWD first on sys.path, so bare `pytest` in a worktree runs
against the wrong tree. The fix is `python -m pytest`.

## Steps

- [ ] Add a conftest.py assertion (or a check-script) that fails fast if tests are importing scufris from a path outside the current worktree.
- [ ] Alternatively/additionally, provide a thin test wrapper that always invokes `python -m pytest`.
- [ ] Verify: bare `pytest` in a sprout worktree is caught; `python -m pytest` passes.
- [ ] Document in AGENTS.md.

## Definition of Done

- Running tests from a sprout worktree cannot silently exercise the main checkout (manual: reproduce in a sprout and confirm the guard fires).

## Notes

- Promotes the `nix-devshell-import-resolves-to-cwd-source` lesson to a guard.
