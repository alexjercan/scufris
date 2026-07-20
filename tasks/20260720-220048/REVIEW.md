# Review

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

What I tried to break: I attacked the `set -euo pipefail` + grep interaction (the classic footgun where grep exiting 1 on no-match aborts the hook), the grep anchoring for false-positives, and the actual end-to-end guard behavior against a real staged symlink. I staged `web/node_modules` as a symlink and confirmed the commit was refused; I then ran a normal commit to make sure the guard does not block everything; and I fuzzed the regex against near-miss paths like `web/node_modules_notes.md`, `notweb/node_modules`, and `aweb/node_modules`.

Verification notes. Reproduced the guard in the worktree: creating a symlink `web/node_modules`, `git add`ing it, and attempting `git commit` fails with the clear multi-line message and leaves HEAD unchanged (no commit created). A normal commit with no `web/node_modules` staged succeeds and leaves HEAD advanced as expected; I reset it back so the branch is left as found. The `set -euo pipefail` + grep concern is a non-issue: the grep is the condition of an `if`, and `set -e` does not fire on a command whose exit status is being tested, so a no-match (exit 1) simply falls through to a clean `exit 0`. The anchor `(^|/)web/node_modules(/|$)` behaves correctly: it matches `web/node_modules` and paths beneath it (including a nested `.../web/node_modules/...`, which is desirable), and does NOT match `web/node_modules_notes.md`, `web/node_modulesX`, `notweb/node_modules`, or `aweb/node_modules`. The hook is tracked at mode 100755 (executable in git). `core.hooksPath` is set to the relative `hooks`, which git resolves per-worktree, so each sprout worktree uses its own versioned hook, and the shellHook wires it via `git config core.hooksPath hooks` guarded with `|| true`. The AGENTS.md doc is accurate: `.gitignore`'s dir-only `node_modules/` pattern does not match a `web/node_modules` symlink, which is exactly why the footgun exists and why this guard is warranted. ruff, mypy, and pytest all pass via the devShell (all checks passed, 27 source files clean, full test suite green).

- No findings.
