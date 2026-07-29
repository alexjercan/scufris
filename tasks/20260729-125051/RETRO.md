# Retro: continuous integration for every push and pull request

- DATE: 20260729
- TASK: 20260729-125051
- REVIEW ROUNDS: 2 (REQUEST_CHANGES, then APPROVE)

## What went well

- **Probing the runner early paid off immediately.** The lesson
  `probe-runtime-on-target-host-early` says a reasoned verdict about a
  dependency is a hypothesis until it runs live. The whole risk in this task was
  "can a hosted runner afford `nix flake check`", and the answer arrived from a
  real run 15 minutes in: under two minutes, green, nothing to cache. Every
  later decision rested on a measurement instead of an estimate.
- **Proving red, not just green.** Pushing a commit that deliberately broke ruff
  AND prettier at once, watching both jobs fail, then reverting and watching
  both pass, cost three runs and converted "CI exists" into "CI discriminates".
  The `records` check was proven the same way locally, because corrupting a task
  record on a pushed branch would have left the break in the history.
- **Putting conformance in the flake rather than in the workflow.** `tatr check`
  as a `checks.records` derivation means the local gate and the CI gate are the
  same object. The alternative - a `tatr` step in the workflow only - would have
  been faster to write and would have created exactly the drift this epic
  exists to end.

## What went wrong

- **The first draft mistook `nix flake check` for the whole gate.** It builds
  `checks` but only EVALUATES `packages`, so a stale `npmDepsHash` would have
  passed CI green while `nix build .#web` was broken for every flake consumer.
  Caught by the round-1 reviewer, not by me, and not by any run - the runs were
  all green. A green check that checks less than you think is the most expensive
  kind of wrong.
- **Security posture was left at the defaults.** No `permissions:` block, and
  the actions pinned to a mutable ref (`@main`, `@v4`) in the same diff that
  argued for pinning tatr via `flake.lock`. The inconsistency was sitting in one
  file and I did not see it.
- **The evidence lived on GitHub, not in the repository.** Timings and the
  red-run proof existed only as run IDs in my context until the reviewer pointed
  out the task record contained none of it. A DoD item that says "recorded in
  the task" is not satisfied by knowing the answer.

## Lessons

- `nix-flake-check-does-not-build-packages`: `nix flake check` builds `checks`
  and only evaluates `packages`, so package derivations (stale `npmDepsHash`, a
  broken build) stay green. Any gate that claims to protect consumers of the
  flake needs an explicit `nix build .#<pkg>` alongside it.
- `prove-a-new-gate-red-before-trusting-it-green`: a gate that has only ever
  been observed passing has not been observed at all. Break each class of check
  it claims to cover once, watch it fail, revert. Cheap, and it is the only
  thing that distinguishes a gate from a decoration.
- `pin-ci-actions-by-sha-like-any-other-dependency`: a workflow's `uses:` refs
  are dependencies with no lockfile. Pin by commit SHA with the version in a
  trailing comment, and declare `permissions:` explicitly - the same argument
  already accepted for `flake.lock` applies, and applying it to one and not the
  other in the same diff is the tell.

## What to do differently next time

Write the evidence into the task record AS it is gathered, not after. Each of
the three CI runs was a fact worth recording the moment it completed; batching
that into a NOTES.md at the end meant the round-1 review correctly found the
DoD unmet while the underlying work was actually done.

And when a diff contains an argument for pinning one dependency, grep the same
diff for every other dependency it introduces.
