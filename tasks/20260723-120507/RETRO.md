# Retro: harden pending-agents poll onto a collision-proof path

- TASK: 20260723-120507
- BRANCH: fix/pending-agents-path
- REVIEW ROUNDS: 1 (APPROVE, clean)

## What went well

- Diagnosed before "fixing": rather than assuming the operator's `404 "no such
  agent"` was a route-ordering bug, I booted a REAL server and proved the landed
  ordering was correct (`/api/agents/pending -> [] 200`). That reframed the report
  from "code bug" to "the tool is hitting a build without the route" - which is the
  actually-actionable finding.
- The hardening is the right kind of fix: it removes the CLASS of problem
  (ordering-dependence) rather than patching a symptom, so a stale-build mismatch
  now degrades to an honest 404 instead of a misleading "no such agent".
- The repo's "every /api route is tagged" invariant test (`test_openapi_docs_are_
  organized`) caught the missing OpenAPI tag on the new route immediately - a good
  example of an invariant test earning its keep on an unrelated change.

## What went wrong

- My first "definitive" boot was WRONG: `nix develop --command scufris` (invoked
  at the main checkout) serves the nix-BUILT package from the main-checkout source,
  NOT the worktree's uncommitted edits. So it showed master's behavior and I
  briefly mis-read the result as the new path being broken. Root cause: I treated
  the console-script entrypoint like `python -m pytest`, but only the `python -m`
  form puts CWD first on sys.path; the console script runs the built package.
  Corrected by booting `cd <worktree> && python -m scufris`.

## What to improve next time

- To verify a LIVE route/behavior against worktree edits, boot with
  `cd <tree> && python -m scufris` (CWD-first), never `nix develop --command
  scufris` from elsewhere - the same reason the repo mandates `python -m pytest`
  in a sprout. This also IS the operator's root cause: a running `scufris` won't
  pick up landed code unless its build target has it.

## Action items

- [x] Ledger: `nix-develop-command-runs-the-built-package-not-worktree-source`
  (x1) - boot worktree code with `python -m <app>` (CWD-first), like
  `python -m pytest`.
- No follow-up code tasks. Returns to BC4 (the wake bridge) next.
</content>
