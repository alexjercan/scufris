# Lessons

The compressed memory of mistakes this repo has already paid for. One or two
lines per lesson; `/compound` appends here after a task's retro. Grep this for
your area before starting work. At 3+ occurrences a lesson moves to the
Pending promotions section at the bottom; the user decides whether it gets
promoted into AGENTS.md, a skill, or the tooling itself.

## Build / environment

- `write-a-procedure-in-failure-order-not-thought-order` (x1): order a
  documented procedure so the last IRREVERSIBLE action comes after every check
  and every reversible one - push the branch, then tag it. Then read it as a
  stranger: which checkout, which branch, which shell? The v0.1.0 recipe tagged
  before pushing master and never said "main checkout, on master, inside nix
  develop", so followed literally from a sprout worktree it would have tagged a
  feature branch and published it. 20260729-125107.
- `alternatives-are-prose-not-a-code-fence` (x1): a fenced shell block reads as
  "paste me". Never put mutually exclusive options in one fence, especially
  destructive ones - the yank section listed demote / delete-release /
  delete-tag as three consecutive lines, and pasting it would have deleted the
  tag and broken every flake consumer pinned to it. 20260729-125107.
- `filter-a-gh-run-lookup-by-what-you-actually-want` (x1): `gh run list
  --limit 1` means "most recent run of anything" - in this repo it resolved to
  a failed dispatch from another task, so the documented watch command would
  have reported the wrong release's fate. Filter `--branch <tag>` or `--event`,
  and pass `--exit-status` to `gh run watch` so a red run fails the command
  instead of exiting 0. Note `--branch <tag>` matches tag-triggered runs only;
  a `workflow_dispatch` run's head branch is master. 20260729-125107.
- `delete-a-branch-you-pushed-for-evidence` (x1): `sprout land` squash-merges
  and removes only the LOCAL branch, so a branch pushed to origin (for a PR, a
  CI probe) survives, never shows as merged, and can carry temporary workflow
  files - one here still had a `kvm-probe.yaml` with an `on: push` trigger.
  Deleting the remote copy is a separate explicit act at the end of the task.
  20260729-125107.

- `nix-flake-check-does-not-build-packages` (x1): `nix flake check` builds the
  `checks` derivations but only EVALUATES `packages`, so a stale `npmDepsHash`
  or a broken package derivation passes green while `nix build .#web` is broken
  for every flake consumer. Any gate claiming to protect consumers needs an
  explicit `nix build .#scufris .#web` next to the check (CI does this; so
  should a local pre-release pass). 20260729-125051.
- `prove-a-new-gate-red-before-trusting-it-green` (x1): a gate only ever
  observed passing has not been observed at all. Break one thing per class it
  claims to cover (lint, format, test, records), watch it fail, revert. Three
  CI runs turned "CI exists" into "CI discriminates" - and the local-only
  variant (corrupt a record, build just that check) avoids pushing a break into
  branch history. 20260729-125051.
- `pin-ci-actions-by-sha-like-any-other-dependency` (x1): a workflow's `uses:`
  refs are dependencies with no lockfile - `@main` and `@v4` both move. Pin by
  commit SHA with the human version in a trailing comment, and declare
  `permissions:` explicitly instead of inheriting the repo default. Tell that
  this was missed: the same diff argued for pinning tatr via `flake.lock`.
  20260729-125051.
- `ci-jobs-must-pin-the-commit-not-the-ref` (x1): a workflow that resolves a tag
  NAME per job builds whatever that name points at when each job starts.
  Resolve once, emit the SHA, check out the SHA downstream, then assert the tag
  still names it. Caught in review: all three release jobs checked out
  `github.ref`, so a `workflow_dispatch` run would have gated and shipped
  master while claiming to release the tag. 20260729-125101.
- `never-interpolate-workflow-input-into-a-run-body` (x1): `${{ }}` expands
  BEFORE bash sees the script, so it cannot be quoted - a crafted
  `workflow_dispatch` input executes. Pass untrusted values through `env:` and
  use `"$VAR"`; inside a nested `bash -c '...'` let the inner shell read the
  environment rather than splicing. Validate with `[[ =~ ]]`, not
  `grep -E '^...$'` (grep anchors a LINE, and dispatch inputs may be
  multi-line). 20260729-125101.
- `publish-last-create-as-a-draft` (x1): a multi-step publish must create the
  artifact invisible, fill it, and flip it visible in the FINAL step. Publishing
  first means a later failure leaves a live, empty release that watchers were
  notified about, under a permanent version number. Also: `gh release view
  <tag>` does not see DRAFTS, so use `gh release list` when probing for an
  existing release, or a retry creates a second one. 20260729-125101.
- `probe-produces-answers-a-decision-did-not-list` (x1): a probe's job is not to
  pick between the branches you wrote down, it is to find the one you did not.
  The KVM decision listed "runner has it" and "runner does not"; the truth was
  `/dev/kvm` PRESENT but root:kvm 0660 and unusable - which makes the obvious
  `if [ -e /dev/kvm ]` guard pass and then fail. Sharpens
  `probe-runtime-on-target-host-early`. 20260729-125101.
- `resume-existing-sprout-state` (x1): when `sprout new <feature>` finds an
  existing worktree, inspect its branch, status and task diff before deciding it
  is stale. If it belongs to the same task, continue from that state and preserve
  its changes. 20260724-012212.
- `edit-from-the-worktree-path-not-the-planning-read` (x2): edits meant for the
  branch landing in the MAIN checkout. (1) a file Read at its main-checkout path
  during planning, then Edited in the work phase, lands the edit in the main tree
  (the Edit reuses the stale path). (2) TASK.md planning edits (Flow State, plan,
  DECISION.md) made in the main checkout BEFORE `sprout new` are not on the branch
  at all - the worktree cuts from committed HEAD. Fix: sprout FIRST, then edit the
  task record inside the worktree (or commit before sprouting); after `sprout new`
  re-Read from the worktree path before the first Edit. 20260723-001251,
  20260726-215910.
- `sprout-new-and-cd-is-denied-run-it-alone` (x1): the flow/`sprout` skill's
  `cd "$(sprout new <feature>)"` one-liner is blocked by the harness EnterWorktree
  guard (the combined create-and-`cd` shape). Run `sprout new <feature>` on its
  own to create the worktree, then operate on absolute worktree paths (the Bash
  cwd resets between calls anyway). 20260726-215910.
- `recheck-head-before-committing-in-a-user-touched-repo` (x1): when a cross-repo task
  edits a repo the USER may be working in concurrently (here their personal
  nix.dotfiles), the checkout's branch/HEAD can move under you between reading it and
  committing - the operator merged a feature branch to master and switched, so an edit
  planned for the branch committed to master (`git commit` printed `[master ...]`). Net
  result was fine, but re-run `git branch --show-current` IMMEDIATELY before the commit,
  not once at task start; do not trust a branch read from earlier in the cycle.
  20260727-011526.
- `no-backticks-in-git-commit-m` (x1): a `git commit -m "...`var(--bg)`..."` with
  backticks (or `$()`) in the message runs command substitution in the shell - the
  backticked text is EXECUTED and vanishes from the message, silently mangling it
  (here `background was , undefined`). When a commit body contains code punctuation
  (backticks, `$(...)`, `!`), write it with `git commit -F <file>` or a quoted
  heredoc, never an inline `-m` double-quoted string. 20260722-104048.
- `grep-touched-files-for-non-ascii-before-commit` (x1): the repo is ASCII-only
  (no arrows, em-dashes, smart quotes) yet a stray typographic char slips into
  user-facing affordance text by reflex (here a U+2192 "->" in a "go here" link);
  the check gate does not catch it, only a reviewer did. Before committing any
  file where you wrote user-facing text, `grep -nP "[^\x00-\x7f]"` the touched
  files. 20260721-234644.
- `absence-grep-must-not-be-extension-scoped` (x2) -> work skill removal/doc-sweep:
  an absence-proving sweep narrowed by extension globs OR by a hand-listed set of
  "the doc surfaces" skips tracked dotfiles that carry commands (`.env.example`,
  `.gitignore`), so a stale reference survives the "one-pass grep". Run it as
  `git grep` over every TRACKED file with PATH exclusions only; a review caught a
  stale `["tatr_new"]` in `.env.example` and a `nix build .#web` in `.gitignore`
  this way. 20260722-222729, 20260730-164048.
- `rerun-the-gate-after-the-last-record-edit` (x1): `checks.records` reads
  `tasks/`, so editing the task record (STATUS, Flow State) AFTER a green `nix
  flake check` invalidates it - flipping STATUS to CLOSED before REVIEW.md/RETRO.md
  existed made the gate red while the session believed it green. Run the gate as
  the LAST action before the commit. 20260730-164048.
- `a-comments-only-step-can-hide-load-bearing-code` (x1): a rename step scoped to
  "the comments that name this output" missed the one place it was CODE
  (`defaults.web` at nix/scufris-service.nix:63, reached via
  `self.packages.${pkgs.system}`) because the grep looked for the dotted literal
  `packages.web`. Grep the bare attribute name in `nix/`, not the dotted path.
  20260730-164048.
- `flake-parts-coerces-nixosmodules-not-homemanagermodules` (x1): `builtins.isAttrs`
  is a valid module probe only for `nixosModules` (flake-parts runs those through
  its `deferredModule` option type); `homeManagerModules` is undeclared and passes
  through as the raw module FUNCTION, so `isAttrs` is false there. Probe with
  `m: builtins.isFunction m || builtins.isAttrs m`, and note `==` on Nix functions
  is always false, so you cannot prove two module attrs are the same value.
  20260730-164048.
- `dont-split-on-a-char-the-payload-contains` (x1): a script re-aligning a comment
  column split each line on `"#"` - also the char in `nix build .#scufris` - and
  rewrote eleven AGENTS.md lines into nonsense. Split on the separator with its
  spacing (` +# `) or rewrite the block explicitly, and re-read the produced text:
  the artifact, not the tool's success report, is the proof. 20260730-164048.
- `scope-absence-greps-to-the-diff-not-the-file` (x1) -> plan skill DoD greps
  (sibling of `absence-grep-must-not-be-extension-scoped`): an absence-proving DoD
  grep ("no new non-ASCII", "no stale symbol") run over a WHOLE file self-matches
  pre-existing content the diff never touched, so the cmd reads red while the
  intent holds. Scope it to the diff: `git diff <base>... -- <path> | grep -nP
  ...`. A "no new non-ASCII" DoD hit two pre-existing glyphs (arrow, middot) this
  way. 20260723-225621.
- `dod-kfilter-proof-must-select-tests` (x1) -> plan skill DoD proofs (sibling of
  `scope-absence-greps-to-the-diff-not-the-file`): a `-k`/grep DoD proof written
  at plan time guesses future test names and can select ZERO tests (here `-k
  "fallback"` matched nothing; tests were named `..._falls_back_to_plain`). A
  proof over an empty selection "passes" while verifying nothing. Confirm each
  `-k`/grep DoD selects its intended tests before closing, or name tests to match
  the planned filter. 20260726-205809.
- `review-md-needs-a-bare-VERDICT-line` (x1): `tatr check` flags a CLOSED task
  `closed-not-approved` unless its REVIEW.md carries a machine-readable
  `- VERDICT: APPROVE` line (a list item), not prose like `Verdict: **APPROVE**`.
  Write the bulleted `- VERDICT: <APPROVE|REQUEST_CHANGES>` line so the Finish
  conformance pass is green. 20260727-095441.
- `format-before-the-check-gate` (x2): a combined `fmt --check && lint && test`
  suite aborts at the formatter step, so a stray unformatted line wastes the whole
  run before mypy/pytest execute. Run the WRITING formatter (`ruff format` /
  `prettier --write`) before invoking the check gate, not after it complains. Seen
  on a frontend (prettier, 20260719-210723) and a backend (ruff, 20260719-212203)
  task; at x3 promote to a pre-commit hook or AGENTS.md. (Reviewed 2026-07-20,
  task 20260720-220116: still x2, remains a watch - promote when it recurs.)
- `nix-flake-check-sees-only-tracked-files` (x2) -> work skill verify-step: `nix
  flake check` on a dirty tree evaluates only git-TRACKED files, so a branch that
  ADDS modules checks a STALE tree (fails on the pre-change file, ignores the new
  ones) until you `git add`/commit them - local `python -m pytest`/`ruff` see the
  working dir and pass, so the two disagree confusingly. `git add` new files before
  the flake gate. The error names the SANDBOX path, not the cause: a new
  `scripts/*.py` check failed with `can't open file '/build/work/...'`.
  20260727-105609, 20260731-171420.
- `a-pinned-tool-must-be-bumped-when-its-data-format-moves` (x1): pinning a
  conformance tool by `flake.lock` protects the gate from upstream churn, but a
  commit that migrates the DATA the tool reads (task records to the tatr v2
  schema, 9d78ebe) silently strands the pin - 0.1.0 cannot parse `PLAN STATUS`,
  so it reported every IN_PROGRESS record as unplanned and the gate was red for
  the whole duration of every task. A schema migration owes the pin a bump in
  the same change. 20260731-171420.
- `match-a-path-exclusion-in-the-domain-it-names` (x1): an exclusion rule whose
  values are all DIRECTORY names still matches basenames if the code walks
  every `split("/")` component - `result-view.ts` was silently exempt from the
  new file-size cap. Say the domain in the code (`split("/")[:-1]`), not only
  in the comment, and pin it with a test whose fixture is a legitimate file the
  rule would wrongly match. A guard that silently exempts a file is worse than
  no guard. 20260731-171420.
- `ruff-format-is-not-lint-fix` (x1): `ruff format` does NOT sort imports (I001) or
  fix other lint - only `ruff check --fix` does, and the flake gate runs `ruff
  check`, so a format-only pass can leave an I001 the gate then rejects. Run BOTH
  (scoped to touched files): `ruff format <files> && ruff check --fix <files>`.
  20260727-105609.
- `argparse-global-flag-read-from-argv` (x1): a global flag that must work BOTH
  before and after a subcommand (`prog --debug sub` and `prog sub --debug`) is
  unreliable via `parents=[common]` on the top parser AND the subparsers - the
  subparser default clobbers a value set at the parent, and `default=SUPPRESS` +
  `set_defaults` does not fully fix it. Put the flag on a shared parent only so
  argparse ACCEPTS it anywhere, then read the effective value straight from argv
  (`"--debug" in argv`), not from `args.<dest>`. 20260719-235504.
- `set-e-plus-grep-c-aborts-scripts` (x1): under `set -e`, a `grep`/`grep -c` that
  matches nothing exits non-zero and aborts the script (even inside `$(...)`). Use
  `grep -co ... || true`, drop `set -e` around greps, or test the count
  separately. (The AGENTS.md "no pipe eats the exit code" rule, for grep.)
  20260719-190549.

- `symlink-node_modules-into-fresh-worktrees` (x3, GUARDED 2026-07-20 ->
  hooks/pre-commit rejects a staged `web/node_modules`, task 20260720-220048;
  the setup how-to below stays guidance): a sprouted worktree has no
  `web/node_modules`, so `npm run ci` fails until deps exist; `ln -s
  <main>/web/node_modules <worktree>/web/node_modules` is instant and webpack/
  vitest resolve through it fine - no reinstall. The `.gitignore` `node_modules/`
  (dir-only, trailing slash) does NOT match the symlink, so it shows as
  untracked; stage the real source files explicitly, never `git add -A`.
  20260719-182915. Cleanup cost (20260719-223105): the same untracked symlink
  makes `sprout rm` fail on "modified or untracked files" - and it deletes the
  branch BEFORE bailing on the worktree, leaving a half-torn-down state. Remove
  the symlink first, or finish with
  `rm -f web/node_modules && git worktree remove --force && git worktree prune`.
  Recurred (20260720-184148): a reflex `git add -A` STAGED the symlink into the
  commit (the `.gitignore` dir-only `node_modules/` never matches it). Never
  `git add -A` in a worktree - stage explicit paths; if it slips in,
  `git rm --cached web/node_modules` + amend, then delete the symlink before
  landing. Recurred again (20260724-152157): reached for `npm ci` (works, but a
  full reinstall) instead of the instant symlink - prefer the `ln -s` as the first
  frontend act in a fresh worktree.
- `dep-change-needs-nix-develop-rebuild` (x1): the active dev shell runs a fixed
  nix-store uv2nix venv, so a new dependency added with `uv add` is invisible to
  a bare `pytest`/`mypy`. Run checks via `nix develop --command ...` (or re-enter
  the shell) so the venv rebuilds from the updated `uv.lock`. 20260719-154420.
- `nix-devshell-import-resolves-to-cwd-source` (x3 -> PROMOTE): in the nix dev
  shell, `import scufris` resolves to the CWD's `scufris/` source (shadowing the
  venv install), so any in-process smoke / `python -c` check must run from the
  BRANCH's own directory - never `os.chdir` into another checkout before
  importing, or you silently test that checkout's code. Symptom: a route/behavior
  pytest passes but a smoke reports missing (was testing master, not the branch).
  20260719-212205. Corollary (20260720-184136): the CONSOLE-SCRIPT `pytest` does
  NOT put CWD first on sys.path, so in a sprout worktree bare `pytest` imports
  scufris from the MAIN checkout (editable install's abs path) - a new branch
  symbol then ImportErrors at collection though mypy is green. Run
  `python -m pytest` from the worktree (it prepends CWD); verify with
  `inspect.getfile(scufris.<mod>)`. Third occurrence (20260723-120507): the SERVER
  console script has the same trap - `nix develop --command scufris` boots the
  BUILT/main-checkout package, not a worktree's edits, so a live route check
  silently exercises master (misread a hardened route as broken). Boot worktree
  code with `cd <tree> && python -m scufris`. Same operator-facing footgun: a
  running `scufris` won't serve landed code unless its build target has it. At x3,
  promote to AGENTS.md verify step (see Pending promotions).
- `in-place-mutation-beats-a-provider-rewire` (x1): to make config captured in
  many closures live-mutable, mutate the ONE shared `Settings` object in place
  (pydantic `validate_assignment=True` validates each write) instead of
  rewiring N readers through a `get_settings()` provider - every reader already
  holds that object, so the in-place path is both smaller and not weaker. Only
  BUILD-TIME selectors (which agent impl) need more: wrap them in a
  protocol-implementing handle that rebuilds. Count the readers before adopting
  a plan's "route through a provider" step. 20260720-184136.
- `new-scufris-module-needs-package-init` (x1): mypy errors with "Source file
  found twice under different module names" when a `scufris/` module has no
  package `__init__.py`. `scufris/__init__.py` now exists; keep it.
  20260719-154420.
- `run-repo-checks-inside-nix-develop` (x2): the flake dev shell is the ONLY place
  the toolchain lives - not just ruff/mypy/pytest but `node`/`npm`/`npx` too. On the
  bare PATH even `./node_modules/.bin/vitest` dies "node: No such file or directory"
  (and `npx`/`npm` are "command not found"), so symlinking `web/node_modules` is not
  enough - the whole web suite (`npm run ci`, vitest) must ALSO run as
  `nix develop --command bash -c 'cd web && ...'`. Invoke every check that way or the
  first call dies and wastes a turn. 20260723-153609, 20260726-215847.
- `nix-develop-pytest-pipe-eats-the-summary` (x1): piping
  `nix develop -c ... python -m pytest` through `tail`/`grep` drops the final
  `N passed in Xs` line (only the progress dots survive), so you cannot confirm
  green by grepping the tail. Confirm via the EXIT CODE instead
  (`... >/dev/null 2>&1; echo $?`). All-dots-and-`[100%]` with no `F`/`E` is also
  conclusive. 20260723-153609.
- `scufris-mypy-baseline-is-red` (x1, RESOLVED 20260723-182253): `mypy .` (and the
  `nix flake check` mypy check) WAS red on master with ~58 pre-existing errors -
  test files passing plain `str` where `Backend`/`AuthMode`/`AgentState` StrEnums
  are expected. The baseline is now GREEN (task 20260723-182253). Keep the durable
  wisdom: a "mypy green" DoD is only literal when the tree is already green - if a
  baseline is red, "green" means "adds no NEW errors", so verify your CHANGED files
  are clean rather than chasing the whole tree. See
  `strenum-fields-take-the-member-not-the-raw-str-in-typed-callers` for the fix
  pattern. 20260723-153609.
- `strenum-fields-take-the-member-not-the-raw-str-in-typed-callers` (x1): a
  production field/param typed with an `enum.StrEnum` (`Backend`/`AuthMode`/
  `AgentState`) is REJECTED by mypy when a caller passes a plain `str`, even though
  pydantic/StrEnum coerce it fine at runtime (`Backend.CODEX == "codex"`). In
  callers - tests included - pass the ENUM MEMBER; downstream `== "codex"`
  assertions still hold because the member equals its string. Reserve a raw string
  ONLY where the coercion itself is under test (e.g. test_enums.py, the legacy
  `"app_server" -> CODEX` fold) and mark those `# type: ignore[arg-type]` with a
  why. Do NOT convert the coercion tests to enums - that leaves them green but
  proving nothing. 20260723-182253.
- `class-method-shadows-builtin-in-annotations` (x1): a class that defines a method
  named after a builtin generic (`list`, `dict`, `type`, `id`) shadows it inside
  class-scope annotations, so a later method's `-> list[str]` resolves to the METHOD,
  not the type (mypy: "Function ... is not valid as a type", then "not iterable" at
  the call site). Methods textually BEFORE the shadowing def are unaffected, which
  hides it. Fix: annotate that builtin via a module-level alias bound outside the
  class (`SessionIdList = list[str]`), not `typing.List` (ruff UP006). 20260724-111947.
- `sprout-worktree-needs-npm-ci-for-the-web-suite` (x2): a fresh sprout worktree has
  NO `web/node_modules` (the python venv is flake-provided, the node deps are not),
  so `npm run test` / `npm run ci` die "vitest: command not found" until you run
  `npm ci` in `web/` first. Do it once per worktree before touching the frontend.
  Run it inside the shell too: `nix develop .#default --command bash -c 'cd
  <worktree>/web && npm ci && npm run ci'`. 20260723-193216, 20260727-095441.
- `tick-a-compound-step-conjunct-by-conjunct` (x1) -> work skill: a step reading
  "decide X **and** Y" gets ticked off the section you wrote (X), not off the
  step's own text, so Y silently ships undecided - here "audit storage and
  retention" landed with no retention policy. Re-read the step when ticking it.
  20260729-125020.
- `run-the-command-before-documenting-its-output` (x2): a sentence describing
  what a tool prints is a claim - run it once. Two review findings came from
  writing what `nix-collect-garbage --dry-run` (prints a path COUNT, no bytes)
  and `nix store diff-closures` (prints NOTHING when unchanged, so "no change"
  and "failed" are identical) ought to print. A figure observed EARLIER in the
  same session is the same claim with an expiry date: re-run every command a
  close-out quotes, at close-out time, because the tree moved underneath it
  (a test count, and a line count the task's own sweep changed). And never read
  a chained `a && b ; c` verification through `tail` - the middle command's
  non-zero exit is invisible, which is how `ruff format --check .` got recorded
  as clean while exiting 1. 20260729-125020, 20260731-171420.
- `deleting-a-reference-orphans-what-cited-it` (x1): a sweep that removes
  citations must also fix the prose that pointed AT them. Deleting a record
  link left "the fix for R1.3 stripped codex's environment ... (review round 2,
  R2.1)" eleven lines below with no antecedent - the same dangling lore the
  sweep existed to remove. After a deletion pass, grep for what referenced the
  deleted thing. 20260731-171420.

## Testing

- `verify-a-doc-citation-by-running-the-grep` (x2): a doc sentence naming the test
  that proves a property ("`tests/test_mcp_server.py` asserts the absence") is a
  CLAIM, and copying it forward propagates it - `AGENTS.md` named the wrong file for
  the test proving no agent can approve a host action (it is
  `tests/test_host_mcp_server.py::test_the_agent_has_no_tool_that_approves_a_host_action`),
  and later tasks read it without opening the file. A wrong pointer is worse than
  none: the next reader checks the empty file and concludes nothing proves it. Grep
  for the named symbol in the named file before writing OR repeating a citation, and
  when a doc claims a test exists, that grep is the review step.
  20260729-124655, 20260731-131543.
- `notification-features-need-a-repetition-test` (x1): for anything that fires on a
  schedule, the first test is "what does an UNCHANGED world produce over N ticks" -
  every single-pass test passed while a 96%-full disk produced 96 messages a day and a
  fresh root-action proposal every 15 minutes, which is how a notification feature gets
  muted. Measured, not reasoned: drive four ticks. 20260729-125046.
- `drive-the-instance-the-app-started-on-its-own-loop` (x1): to test an in-process
  background surface (a telegram bot, a poller, a worker), use the object the app's
  lifespan created and dispatch through the client's portal - a second instance built
  beside it splits its state from production's (announcements landed in one bot's map
  while taps hit another's), and awaiting on the TEST's loop leaves
  `supervisor.start`'s `create_task` on a loop nobody runs. 20260730-104524.
- `assert-a-credential-rule-with-only-that-credential` (x1): a test for a rule
  derived from WHICH credential a caller presents must send only that credential -
  `TestClient` keeps the operator's session cookie from an earlier `_login`, so a
  request carrying both a cookie and a bearer token tests an operator with an
  Authorization header, not an agent. Use a second client with its own cookie jar.
  20260729-125040.
- `assert-the-wrong-rendering-is-absent-not-just-the-right-one` (x1): a pin that
  only asserts the good output passes against the bug too. The store-path regex
  test asserted the package directory parses - true with the broken permissive
  pattern as well; it became a pin only once it asserted the pattern must REFUSE
  the `.../bin/foo` form. Same for every honesty test: assert the misleading
  rendering is ABSENT ("the window is empty" not in text), not just that the right
  one is present. Three tests in this task were unfalsifiable as first written -
  a conditional behind `if "unavailable" not in out`, a tautology over a title
  every path emits, and `assert x or not y` inside a DoD proof. Companion to
  `revert-the-fix-to-prove-the-test` and `dod-named-tests-deserve-the-most-scrutiny`.
  20260729-125024.
- `a-fixture-that-cannot-express-the-bug-blesses-it` (x1): a test whose fixture
  lacks the STRUCTURE the bug lives in does not merely miss it - it certifies the
  wrong behaviour as correct, and the next reader trusts it. The throttle test
  built cpu directories with no `topology/`, so there were no hyperthread
  siblings to deduplicate, the doubled sum was trivially "right", and two review
  rounds read past it. Build the fixture from the real thing's SHAPE (siblings,
  sockets, missing files), not only its values - then revert the fix and watch it
  fail, which is the only proof the shape is sufficient. Companion to
  `revert-the-fix-to-prove-the-test` and `capture-real-cli-output-for-parser-tests`.
  20260729-205145.
- `assert-the-property-not-the-environments-answer` (x1): when several outcomes
  are all CORRECT, assert the invariant they share, not the one your dev box
  happens to give. `assert thermal.battery.ok` passed on this desktop (empty
  `/sys/class/power_supply` -> "no battery") and failed in the nix build sandbox
  (no such directory -> "unreadable") - both correct degradations, and only the
  sandbox's different `/sys` exposed it. The real property was "the report always
  carries a message and is never silently blank", which holds everywhere and
  still fails against a bare empty report. Corollary: the sandbox is a DIFFERENT
  environment, so anything reading `/sys`, `/proc` or PATH needs its test written
  for both. 20260729-125024.
- `revert-the-fix-to-prove-the-test` (x1): a test written to pin a bug is
  unproven until you revert the fix and watch it FAIL. Two defects survived a
  green suite in one cycle: a fence test whose fixture was indented so the `^##`
  anchor never matched (it passed with fence detection fully disabled), and a
  re-date fix that regressed idempotence with no test covering it. One edit and
  one run catches both. 20260729-125056.
- `dod-named-tests-deserve-the-most-scrutiny` (x1): a test named in a Definition
  of Done is the one nobody re-reads - its NAME does the arguing. All three
  named here compared the app's version against the same call the app makes
  (green even when everything reported `0.0.0+unknown`), and the single
  cross-source assertion sat behind an `if` that skipped in exactly that failure
  mode. Assert against an INDEPENDENT source, and never let the real assertion
  be conditional. Related: `dod-proof-must-exercise-the-named-claim`.
  20260729-125056.
- `a-fix-can-break-the-property-it-was-protecting` (x1): after fixing an edge
  case, re-test the INVARIANT the surrounding code claims, not just the edge
  case. Adding automatic re-dating satisfied a review comment and broke
  "idempotent" - a property stated in the DoD, the script header and the notes.
  Corollary: when a review asks for an escape hatch, add the hatch; do not
  change the default. 20260729-125056.
- `transport-cancel-needs-live-receive-loop` (x1): for chat/long-poll transports,
  a cancel command test must prove the receive loop can accept `/cancel` while a
  previous turn is still active, not only that command dispatch recognizes it.
  20260728-175659.
- `isolate-state_dir-in-tests-that-assert-config` (x3, PROMOTED 2026-07-27 -> conftest autouse `_isolate_state_dir` fixture): a test that constructs `Settings()` and asserts a field is
  defaulted/absent silently reads a REAL external override - the
  `~/.local/state/scufris` store (state_dir) OR the repo `.env` file - which wins
  over the constructor, so it is green on CI (`nix flake check` has neither) and
  red on a dev box whose override disagrees. Isolate the baseline:
  `state_dir=tmp_path` for the store, `_env_file=None` (mypy: `# type:
  ignore[call-arg]`) for the `.env`. (2) `_enabled()` asserting no SCUFRIS_DEN_PATH
  in the MCP env reddened once the operator put SCUFRIS_DEN_PATH in `.env`. (3) the
  APPEND-only reasoning sidecar GREW a real-home file across test runs (overwrite
  stores hid it); fixed by an autouse `_isolate_state_dir` conftest fixture that
  points SCUFRIS_STATE_DIR at a per-test tmp dir. 20260723-233337, 20260727-003852,
  20260726-215910.
- `append-only-store-amplifies-real-state_dir-leak` (x1): an append-only
  per-session store makes the latent "tests build Settings() without state_dir and
  write to real ~/.local/state/scufris" leak HARMFUL (the file grows every run),
  where overwrite-based stores hid it. When adding a persisted store, design test
  isolation with it (or grep the real state dir after a run). 20260726-215910.
- `sidecar-alignment-needs-a-fingerprint-guard` (x1): merging a scufris-owned
  sidecar back onto provider transcript messages by position alone mislabels when
  the sidecar is partial (feature deployed mid-session) or drifted; tail-align and
  guard each pair with a whitespace-normalized fingerprint of the answer, breaking
  at the first mismatch so it degrades to no-data instead of wrong-data.
  20260726-215910.
- `check-the-base-suite-before-you-start` (x1): run the FULL check suite on the
  pristine base commit BEFORE implementing, and note pre-existing reds in TASK.md
  up front - otherwise an inherited failure surfaces at verify time as a
  surprise and costs a diagnosis detour to prove it is not yours. Here
  `test_agent_config_omits_builtin_server_when_tools_disabled` was red on master
  (reads the real `~/.local/state/scufris` because it omits an isolated
  `state_dir`); knowing that from minute one would have made it a non-event.
  20260723-225616.
- `grep-new-files-for-a-stray-write-tag` (x1): the Write tool occasionally appends
  a stray closing tag (`</content>`) as the last line of a NEW file; in a `.py`
  this SyntaxErrors at pytest collection (`invalid syntax` on the tag line). After
  Write-ing a new file, glance at its tail (or `grep -n '</content>'`) before
  running it - same reflex as the non-ASCII sweep. Bit wake.py + test_wake.py in
  one cycle. 20260723-094313.
- `external-cli-tests-skipif-not-flake-coupling` (x2): when an integration test drives
  an external binary that is NOT in the `nix flake check` sandbox PATH (only leaks in
  via the user nix profile under `nix develop`), do NOT couple the flake to an
  unpublished/local repo to get it there. Split the coverage: deterministic argv/gating
  tests that stub the shell-out (always green, incl. sandbox) PLUS real end-to-end
  tests guarded by `skipif(shutil.which('<bin>') is None)` that run where the tool
  exists and skip loudly otherwise. Keeps the source-of-truth gate green while still
  pinning the real contract. `today` (journal), then `macros` (food lookup).
  20260720-122514, 20260727-010447.
- `wrap-env-derived-cli-with-a-temp-home-fixture` (x1): when a wrapped CLI resolves its
  data file from an ENV var (`macros` reads `$HOME/.local/share/nvim/macros.csv`), make
  the real-CLI tests HERMETIC by seeding a temp store and redirecting via that env, not
  by reading/writing the operator's live data. `_run`/`subprocess.run` with no `env=`
  inherits `os.environ`, so `monkeypatch.setenv("HOME", tmp)` + a seeded file points the
  CLI at the temp copy - which also lets a WRITE subcommand (`macros -i`) be tested
  without touching real data. 20260727-010447.
- `commit-before-sabotage-or-the-restore-eats-the-fix` (x2) -> work skill A/B rule
  (already prose there; recurred anyway): sabotage-testing a fix by mutating a file
  then `git checkout -- <file>` to restore RESTORES TO HEAD, so if the fix itself is
  not yet committed the checkout silently reverts it - and a later `git add -A`
  re-stages the reverted file, landing a broken tree (here app.py called a
  `mark_finished(backend=...)` whose param had been reverted out of agent_store.py;
  the persist callback raised, sessions never persisted). Caught only by the
  full-suite-on-master gate at flow Finish. COMMIT the fix before any sabotage; or
  stash/restore the sabotage hunk alone, never `checkout --` a file holding
  uncommitted work. Recurred 20260729-125015 in its INDEX form: `git checkout
  <file>` restores from the index, so a second sabotage round after an earlier
  `git add`/commit silently reverted a whole round of review fixes in the two
  files touched. The rule is per-sabotage-ROUND, not once per task.
  20260723-001251, 20260729-125015.
- `enumerate-a-credentials-readers-before-picking-its-carrier` (x2): a secret put
  in `os.environ` is readable by EVERY subprocess, and one in a module global by
  every app in the process. Pick the carrier from who must NOT see it; if
  deployment forces env delivery, strip it at every recipient boundary, not just
  the first backend that leaked. 20260729-125015, 20260729-125029.
- `assert-a-leak-against-the-recipient-not-the-structure-you-wrote` (x2): a test
  that a secret is absent must check the thing that RECEIVES it, with the ambient
  source deliberately seeded. Structural config/env assertions are not enough;
  inspect the actual subprocess env for every model-driven spawn. Sibling of
  `dod-named-tests-deserve-the-most-scrutiny`. 20260729-125015,
  20260729-125029.
- `constant-time-compare-raises-on-non-ascii-str` (x1): `hmac.compare_digest`
  raises `TypeError` on a non-ASCII `str`, and Starlette decodes headers as
  latin-1 - so an UNAUTHENTICATED caller sending `Authorization: Bearer \xff`
  turned the auth check into a 500 plus a traceback. Encode both sides
  (`surrogatepass`) so the comparison is total. Reaching for a primitive for one
  property (constant time) still means asking about its domain. 20260729-125015.
- `api-preserving-refactor-still-drops-an-old-contract` (x1): a refactor that keeps
  the whole observable API green (here moving session ids from `agents.json` to a
  registry - zero existing tests changed) still silently RETIRES an old contract
  (the "session_id round-trips via agents.json" behavior), which now nothing
  asserts. An all-green existing suite is not proof the retirement was intended.
  Before trusting it, name the old contract you dropped and add a test that pins
  the NEW mechanism carries it (the four registry tests here). Flagged in review.
  20260723-001251.
- `assert-a-renamed-field-is-populated-not-just-absent` (x1): when a change
  renames/replaces a data field (here `codex_version` -> neutral `backend_version`),
  the tests proved the OLD name was gone and the null case worked, but every case
  used a missing CLI so the new field was always None - the positive path (the new
  field carries the RIGHT value) went untested. When you introduce/rename a field,
  add at least one test that it is POPULATED with a real value on the happy path,
  not only that the old name is absent and null behaves. Caught by out-of-context
  review. 20260722-104034.
- `dod-proof-must-exercise-the-named-claim` (x2): a DoD "(test: X)" is a proof only
  if X ASSERTS that specific claim. (1) Order/quantity claims need the fixture made
  distinguishable (distinct mtimes) and the order asserted, not set membership
  (20260724-111947). (2) A USER-FACING RENDERING claim ("renders the parent chat in
  its table") needs the test to assert the rendered STRING - testing only the
  underlying API field left the tool's table unrendered and the claim false
  (20260724-132830). Data-present != displayed. A/B the assertion (red with the
  mechanism removed?). Both caught by out-of-context review.
- `moving-a-read-behind-a-seam-needs-the-fakes-updated` (x1): routing a
  previously-hardcoded read through an existing abstraction (fork's
  `read_transcript(codex_home)` -> `backend.read_transcript`) makes tests that stub
  that seam return their EMPTY default, silently dropping coverage - the fork test
  seeded a codex rollout but the FakeBackend transcript was empty, so the seed lost
  its prior context. When you move a read behind a seam, grep the tests that
  fake/mock that seam and populate them in the same edit. 20260724-124236.
- `decide-sync-async-from-the-io-boundary` (x1): pick a new protocol/interface
  method's sync-vs-async shape from where its I/O actually LIVES (a blocking file
  read vs an async HTTP client), not from "match the sibling method". A backend
  `delete_session` first drafted sync (blocking httpx) then flipped async so
  opencode's delete rides its async `OpencodeClient`; the flip rippled through the
  impl, the app `await`, and the test (sync def -> async) and cost several re-runs.
  Decide it up front and record it in the DECISION so the shape does not churn
  after tests exist. Accept a sync/async asymmetry across an interface when each
  method's I/O boundary differs. 20260724-124236.

- `directory-invariant-guard-enumerate-cwd-cases` (x1): a guard that checks
  "is X under the current directory" (e.g. the conftest scufris-import guard)
  must enumerate every cwd case before shipping: repo root, a SUBDIRECTORY of
  the repo, an unrelated tree, symlinked paths. The subdirectory case is the
  one to get backwards - accept cwd when the target is `== cwd` OR an ancestor
  of cwd (`_pkg_root in _cwd.parents`), not the reverse. Shipped reversed once;
  out-of-context review caught the subdirectory false-fire. 20260720-220101
- `test-the-net-new-route-not-the-reused-path` (x1): when a task adds NEW
  endpoints alongside an existing one that shares logic (here incremental
  `POST`/`DELETE /api/agent/mcp_servers` beside the whole-list config `PATCH`),
  the reused path's tests do NOT cover the new routes' own branches
  (409/404/403/422). Write direct tests for each new route; a green suite over
  the old path is not coverage of the new one. Caught by out-of-context review.
  20260720-184148. -> review skill (verify each new route has its own test).

- `type-test-fixtures-by-protocol` (x1): annotate injected test doubles by the
  protocol they satisfy (e.g. `Collector`), not the concrete fake class, so tests
  need no cross-test class import - mypy can't resolve `from .conftest import X`
  because `tests/` is not a package. 20260719-154544.
- `test-streaming-over-a-real-socket-not-asgitransport` (x1): httpx
  `ASGITransport` and Starlette `TestClient` buffer the whole response body, so
  they assert an SSE stream's CONTENT but never its TIMING - they always look
  "buffered". To prove a response streams in real time, run a real uvicorn on a
  port and read it with a socket client, timestamping chunks. Cost two false
  "it buffers" diagnoses before switching. 20260720-020356.
- `self-loopback-blocking-call-needs-a-real-socket-test` (x1): an in-process
  handler that makes a BLOCKING call which can loop back to its OWN server (here
  the operator tool console running an HTTP-backed MCP tool - FastMCP runs sync
  tools with `return fn(...)` ON the loop, and the tool's blocking httpx hits this
  same server) HANGS the event loop: the loopback request can never be served. Run
  such a tool OFF the loop (`asyncio.to_thread(lambda: asyncio.run(...))`) and
  prove it with a REAL uvicorn socket - respx/ASGITransport reply instantly and
  PASS while production hangs. Sibling of the real-socket lessons above.
  20260723-141026.
- `mock-transport-drive-the-step-not-the-loop` (x1): to e2e-test a long-poll /
  retry loop whose transport is respx-stubbed, drive the SINGLE-STEP seam
  (`poll_once`) with the loop wrapper (`run`) stubbed to a no-op, NOT the
  free-running loop - the stub returns instantly so the loop never blocks and
  busy-spins, hanging the process (a 200s timeout kill here). Sibling of the
  respx-replies-instantly lessons above. 20260722-222739.
- `settings-test-must-disable-env-file` (x2): a test that builds a real
  `Settings(...)` to assert defaults/explicit inputs must pass
  `_env_file=None  # type: ignore[call-arg]`, or a dev box's real `.env` leaks in
  and the test passes only from a checkout WITHOUT one (e.g. a sprout worktree),
  reddening the suite from the main checkout. Config tests were fixed in T4
  (444f627); the `test_telegram.py` lifespan tests were missed, caught at the T5
  Finish. 20260722-222734, 20260726-195211.
- `os-environ-setdefault-in-test-leaks-past-monkeypatch` (x2): a production
  function that MUTATES `os.environ` directly via `setdefault` (`_ensure_api_base`,
  `_ensure_den_path`) leaks that key for the rest of the run, and `_env_file=None`
  does NOT shield a later `Settings()` - it disables the `.env` FILE, not the
  leaked os.environ var. Two shapes: (1) monkeypatch teardown of a LATER `setenv`
  reverts to the LEAKED value not absent (19 respx tests reddened, 20260723-141026);
  (2) an app-creating test seeds `SCUFRIS_DEN_PATH` from the dev `.env` into
  os.environ, then `test_backends::_hermetic()` reads it and wires `den` when it
  expected scufris-only - green from a sprout worktree (no `.env`), red from the
  main checkout. The worktree-vs-main-checkout green/red split is the tell; diff
  the failing test against the default branch before blaming your change. Fix in
  a conftest autouse fixture that snapshots/restores `SCUFRIS_*` keys, not per
  test. RESOLVED 20260727-130139: `_isolate_state_dir` now snapshots/restores all
  `SCUFRIS_*` os.environ keys around each test. 20260723-141026, 20260727-123342,
  20260727-130139.
- `env-dependent-bug-repro-needs-a-temp-dotenv-in-the-worktree` (x1): a bug that
  only fires with an ambient `.env` (e.g. a leaked `SCUFRIS_*` var) will look
  GREEN in a sprout worktree, which has no `.env`. Reproduce and verify the fix
  by seeding a temporary `.env` in the worktree root, running the failing case,
  then deleting it - do not trust worktree-green as proof the leak is closed.
  20260727-130139.
- `tool-reachable-by-two-runners-needs-a-test-per-runner` (x1): an MCP tool reachable
  by BOTH the agent (MCP subprocess, env injected by `agent.scufris_mcp_server`) AND
  the in-process operator console (`POST /api/agent/tools/{name}/run`, reads the
  dashboard's own `os.environ`) can pass one runner and fail the other. The journal
  tools worked from an agent turn but returned "not configured" from the console
  because `SCUFRIS_DEN_PATH` was only injected into the subprocess; tests drove the
  tool function directly (monkeypatched env) and never the console endpoint, so the
  gap was invisible. Test each runner: add a `.../run` endpoint test, not only a
  direct-call test. Fix mirrored `_ensure_api_base` with `_ensure_den_path`.
  20260727-005013.
- `concurrent-request-test-needs-async-httpx-not-testclient-stream` (x1): to test
  "a second request is refused (409) while the first is still in flight" against
  an ASGI app, you CANNOT hold the first request open with `TestClient.stream` +
  a second sync call - both Starlette's TestClient and httpx's ASGITransport
  BUFFER the whole response body before returning, so a held-open streaming turn
  never returns and the portal deadlocks (hung pytest >3 min). Drive concurrent
  requests on one loop with `httpx.AsyncClient(ASGITransport)` (async test):
  `create_task` the first turn (its backend blocked on an `asyncio.Event`),
  bounded-poll `/status` until running, fire the second expecting 409, then
  release in a `finally`. Sibling of `test-streaming-over-a-real-socket-not-asgitransport`
  (buffering bites request concurrency too, not just streaming timing). 20260721-112436.
- `tests-that-lean-on-a-default-break-when-it-flips` (x1): a test that asserts
  "disabled" behavior while relying on the config DEFAULT being disabled is
  really testing the default, not the behavior - flipping the default reds it.
  Set the precondition explicitly (`agent_enabled=False`) so the test states its
  own intent and survives a default change. 20260720-020402.
- `guard-a-contract-by-capability-not-source-text` (x1): a test that asserts "this
  code never does X" (e.g. sesh.py spawns no tmux/subprocess) by substring-scanning
  the module SOURCE is fooled by the module's OWN docstring/comments naming X.
  Assert the CAPABILITY instead - the module imported no spawning machinery
  (`not hasattr(mod, "subprocess")`) - or strip comments before scanning for
  `Popen`/`os.system`. 20260721-112440.
- `assert-a-distinct-value-not-the-default` (x1): to prove a field returns X (the
  per-agent/effective value) and NOT its fallback Y (a global default), set X to a
  value DISTINCT from Y - if you leave X at the default, the assertion passes for
  BOTH the correct and the buggy impl, so it verifies nothing. Caught in review as
  a vacuous `account.model` check. 20260721-234609.
- `verified-notes-arent-review-findings` (x1): `tatr check` parses any
  `- [ ] Rn.n (SEVERITY) ...` line in REVIEW.md as a finding and rejects any
  severity outside BLOCKER|MAJOR|MINOR|NIT. Write round verification notes ("what
  I checked, no finding") as plain prose bullets; reserve the checkbox-finding
  syntax for the four real severities. -> review skill. 20260720-174021.
  Extension (20260720-184137): out-of-context review SUBAGENTS also invent
  non-canonical severities (LOW/INFO seen), which fail `tatr check` after
  landing - constrain the reviewer prompt to BLOCKER|MAJOR|MINOR|NIT, or remap
  before committing REVIEW.md.
- `fullmatch-not-match-dollar-for-id-validation` (x1): `re.match(r"^\w+$", s)`
  ACCEPTS a trailing newline (`"fs\n"`) because Python `$` matches before a
  final `\n`; for whole-string id/key validation use `re.fullmatch` (or
  `\A...\Z`). Bit an MCP-server-id guard that then persisted a malformed TOML
  key. Keep one shared pattern imported by every boundary so they can't drift.
  20260720-184137.
- `strenum-field-needs-coercion-on-unvalidated-writes` (x1): a pydantic field typed
  as a `StrEnum` can silently hold a BARE STRING when set through a path that skips
  validation - `model_copy(update={"state": "done"})`, `model_construct`, a direct
  attr-assign, or an enum-typed param called with a raw str. mypy + a casual test
  pass; it only shows as a `PydanticSerializationUnexpectedValue` warning at
  serialize time. Coerce (`Enum(value)`) at those boundaries, or have the helper
  RETURN the enum. Grep pytest output for serializer warnings after enum-typing a
  field. 20260721-152749.
- `tightening-a-type-strands-its-type-ignore` (x1): making a previously-loose call
  well-typed (a helper now returns the concrete enum, a field narrows) leaves any
  `# type: ignore[...]` on it dead - mypy still passes WITH it, so it hides. Grep
  for `type: ignore` near a changed signature and drop the stale ones.
  20260721-152749.
- `error-frames-use-json-dumps-not-model-dump-json` (x1): the SSE error frame is
  built with `json.dumps` (spaces after colons: `"kind": "error"`) while event
  frames use pydantic `model_dump_json` (compact: `"kind":"start"`). A test
  asserting the compact form on an error frame fails on the space. Assert on the
  actual serializer's output for the frame you are testing. 20260720-144530.
- `global-singleton-mutation-needs-its-tests-restore-fixture` (x1): adding a
  process-global-singleton mutation (here `apply_role` trimming the module-level
  `mcp` tool registry) to a function a test already invokes (`main()`) leaks the
  mutated state into every later test in the file - three same-file failures. If
  the file has a snapshot/restore fixture (`restore_tool_registry`), apply it to
  the newly-mutating caller in the SAME edit; check who else calls the function.
  20260723-094303.
- `widening-a-shared-signature-needs-a-test-double-sweep` (x1): adding a defaulted
  param to a shared Protocol/ABC method (`backend.stream`, `_stream_app_server`)
  compiles every production impl but breaks every hand-written test double with an
  explicit signature (`TypeError: unexpected keyword argument`). Grep for the
  stubs (`def fake_...`, `.stream(` fakes) and update them in the same change - a
  green mypy is not proof the fakes still accept the call. 20260723-094303.
- `acceptance-assert-the-end-state-not-the-cleanup-return` (x1): when a loop can
  reach its resolved state by more than one mechanism, assert the OBSERVABLE END
  STATE, not the return of one mechanism. A BC5 example asserted
  `acknowledge()["acknowledged"] is True`, which passed by luck: answering a
  blocked sub-agent by resume (a new run) overwrites its WAITING outcome with DONE,
  so by ack-time acknowledge often returns False. Asserting `pending == []` holds
  under every callback interleaving; the bool did not. A green test that encodes a
  race is still wrong. 20260723-094318.
- `mark_finished-preserves-waiting-only-within-the-same-run` (x1): a `WAITING`
  outcome (from `request_input`) is kept through turn-end ONLY when the finishing
  run's id equals the run that set it (`agent_store.py` `preserve_waiting`); any
  later/other run's terminal state overwrites it. So a `message_agent` resume (a
  NEW run) finishing DONE naturally clears the sub-agent from `pending_agents` -
  answering IS the clear, and `acknowledge` is idempotent belt-and-suspenders. Test
  the loop around this, not against it. 20260723-094318.
- `out-of-context-review-misses-cross-layer-timing` (x1) -> review skill: an
  out-of-context reviewer who reads only the changed (frontend) files can APPROVE a
  design that races the OTHER layer. Here a reattach that reconciled by re-fetching
  `/transcript` on the `done` frame looked clean, but the backend persists the
  (new) session id in a post-turn `on_complete` callback that runs in the
  supervisor's `finally` AFTER the terminal SSE frame - so the reload could read an
  empty transcript and drop a first turn. Found only by tracing
  `_launch_agent_turn.persist` + `supervisor._execute` ordering, not by the green
  suite or the reviewer. When a UI reconcile depends on WHEN the backend persists,
  trace the callback order across the seam; settle from data the event already
  carries (the `done` reply) rather than a read that races the write.
  20260723-001301.
- `test-the-throttle-suppresses-not-just-that-edits-happen` (x1): a live-render
  test run with the throttle disabled (`edit_interval=0`) proves ORDERING but
  stays green if the throttle or the unchanged-body guard were deleted. Add a
  large-interval test asserting intermediate updates are SUPPRESSED (and the tail
  is force-flushed on the phase boundary) plus a no-op-update test for the
  unchanged guard - the reviewer caught this gap in the Telegram live-stream
  (DoD said "throttled and skipped when unchanged"). 20260726-201901.
- `merge-default-before-out-of-context-review` (x1) -> review/flow skill (sibling
  of `recheck-head-before-committing-in-a-user-touched-repo`): when the review
  will `git diff <default>` and a concurrent session may be moving the default
  branch, MERGE the default into the feature branch BEFORE requesting the
  out-of-context review, not just at land. Here master advanced two commits after
  the branch was sprouted, so `git diff master` showed spurious "reverts" of
  unrelated frontend + LESSONS files and the reviewer spent a MAJOR finding on
  base drift; `git diff HEAD` (feature-only) was clean. Update-from-default first
  so the reviewer sees only the feature diff. 20260727-102452.
- `import-used-only-in-monkeypatch-string-is-unused` (x1): a module symbol
  referenced SOLELY through a `monkeypatch.setattr("mod.NAME", ...)` STRING is
  not an import use - ruff F401 rejects the `from mod import NAME` and the flake
  gate (lint) fails. Either drop the import (the string is enough) or actually
  reference the symbol. Caught by `nix flake check` after a green local pytest.
  20260727-133302.
- `assert-terminal-outcome-on-the-durable-record-not-status` (x1):
  `/api/agents/{id}/status` reports the LIVE supervisor RunPhase when a run record
  exists (else the persisted state), so a StreamError-terminated turn reads
  `state: done` there while the PERSISTED `AgentState` is `error`. A test that
  polls `/status` for `"error"` never converges and times out to `done`. Assert a
  turn's terminal OUTCOME on the durable record (`GET /api/agents/{id}` or the
  OutcomeStore / `pending`), not `/status`: RunPhase (did the stream finish) and
  AgentState (did the turn succeed) are independent axes. 20260727-140443.

## Backend

- `no-work-on-a-startup-path-a-test-boot-also-walks` (x1): a convenience that does
  real work when a component first sees empty state ("run once so the feature proves
  itself") runs in every test that boots the app and on every restart of a crash loop -
  here it made the suite read the real host and doubled its runtime. Arm on first
  sight, act on the next window, and give the operator a run-now button instead.
  20260729-125046.
- `rule-from-a-field-name-needs-real-data` (x1): a policy keyed on a boolean whose
  NAME reads right ("needs a strong confirmation iff `reversal.possible` is false")
  is a hypothesis - measured, "no undo" is the NORMAL answer for a service restart,
  so the rule demanded a typed acknowledgement for the most routine act. Run the
  rule against the real producer's answers before wiring it into a caller.
  20260729-125040.
- `state-keyed-guard-needs-a-clearer-on-every-path` (x2): any state that GATES
  something - a refusal keyed on it, a banner rendered from it - needs its clearing
  path written in the same edit, for every way the state can end. The one that gets
  missed is "nobody ever acts" (an expiry, a timeout, a success after a failure): no
  code path represents it, so the transitions you are writing will not remind you.
  Measured twice: a BLOCKED-keyed guard locked an agent out for good after a proposal
  nobody answered, and a `lastError` with no reset reported a stale "409 already
  decided" forever. 20260729-125040, 20260730-104520.
- `shell-false-does-not-stop-option-injection` (x1): `shell=False` with an explicit
  argv answers ONE question. A positional that starts with `-` is still parsed as
  a FLAG by the program you hand it to - measured, `systemctl <verb> -Hme@host`
  opens an outbound SSH connection with the caller's credentials. When the value
  can come from a model (which may have just read attacker-influenced text), pass
  positionals after `--` AND refuse a leading `-` explicitly. Ask "can this
  argument become a flag" of every argv, separately from shell safety.
  20260729-125024.
- `two-endpoints-when-one-answer-would-lie` (x1): when one endpoint is asked to
  serve two genuinely different questions, split it rather than scope the shared one.
  `GET /api/agent/tools` is the orchestrator's IN-PROCESS operator console (it really
  can run all ~18 tools locally); a sub-agent's settings page needs a DIFFERENT
  answer ("what does THIS agent's turn advertise", role+backend scoped). The bug was
  a MISSING scoped endpoint (`GET /api/agents/{id}/tools`), not a wrong shared one -
  role-scoping the console would have made it lie. Extract the shared core
  (`role_tool_names`) so the two never drift. 20260723-193216.
- `static-route-before-param-route-or-it-is-shadowed` (x1): a STATIC path segment
  (`GET /api/agents/pending`) declared AFTER a same-prefix parameterized route
  (`GET /api/agents/{agent_id}`) is shadowed - FastAPI/Starlette match in
  declaration order, so the static path resolves as `agent_id="pending"` -> 404.
  Declare the static route FIRST (the repo already does this for
  `/api/agents/backends`), and pin it with a test that a shadowed route would fail
  (assert the real list body, not just a 2xx). 20260723-094308.
- `trust-runtime-shape-over-annotation` (x1): a dependency's type annotation can lie
  about its runtime shape - FastMCP's `mcp.call_tool` is annotated
  `-> Sequence[ContentBlock] | dict` but actually returns the 2-tuple
  `(content_blocks, structured_dict)`. Probe the real return value live before
  unpacking it, and unpack defensively (`cast(Any, ...)` + a shape check) so a future
  version bump degrades gracefully instead of 500-ing. 20260720-134545.
- `derived-default-must-follow-its-source-on-update` (x1): a field DERIVED from
  another at CREATE time (here the per-backend default model via
  `default_model_for`) must be recomputed on every UPDATE path that can change
  its source - not only in create(). The model was defaulted per-backend at
  create but `update()` only wrote it when explicitly sent, so a backend switch
  kept the stale model (claude showing "gpt-5.5"). Fix: follow the EFFECTIVE
  source on update (explicit value wins; blank/omitted-on-change re-derives),
  and pin it with a "change the source, assert the derived value followed" test.
  20260721-133047.
- `web_dist-via-__file__-is-dev-only` (x1): the FastAPI `web_dist` default
  (`<repo>/web/dist` from `__file__`) works for the editable dev install but not
  a packaged wheel; bundling built assets into the nix closure is still open.
  20260719-154544. RESOLVED (20260721-140156): build `web/dist` as its own
  `pkgs.buildNpmPackage` derivation (`packages.web`) and point
  `SCUFRIS_WEB_DIST` at it from the module - the closure now carries the built
  frontend independent of the Python wheel.
- `buildnpmpackage-static-site-needs-dontNpmInstall` (x1): for a webpack/vite
  app that emits STATIC files (not a publishable npm package), `buildNpmPackage`
  needs `dontNpmInstall = true` + a custom `installPhase` that copies the build
  output to `$out`; the default install/pack phase has no package to install and
  fails. Pair with `npmBuildScript = "build"`. Bootstrap `npmDepsHash` with the
  all-`A` fake sha256 and read the real one from the "got:" mismatch.
  20260721-140156.
- `new-config-field-updates-all-its-surfaces` (x1): a new `SCUFRIS_` setting has
  more than one home - the `config.py` field AND `.env.example` (its discoverable
  doc), plus the settings-store whitelist if it is runtime-mutable. The env-doc
  file is the easy miss (caught by review R1.1 for `SCUFRIS_PROJECT_BASE_DIRS`);
  update them in the same commit. 20260721-112440.
- `expanduser-path-config-at-use-time` (x1, sibling of
  `new-config-field-updates-all-its-surfaces`): a new filesystem-path config knob
  must `.expanduser()` at USE time (pydantic stores a `~` env value verbatim -
  `Path('~/x')` is not expanded), mirroring `app.py`/`projects.py`/`sesh.py`; and the
  value you put in `.env.example` is a TEST INPUT, not just doc prose - feed that exact
  form through the real path in a test. A `SCUFRIS_DEN_PATH=~/personal/the-den`
  (the documented example) silently disabled every journal tool because `~` was never
  expanded; out-of-context review caught it. 20260720-122514.
- `list-env-field-needs-nodecode-for-a-before-validator` (x1): a pydantic-settings
  field typed as a list/dict/model has its ENV value JSON-decoded at the SOURCE
  (`EnvSettingsSource`) BEFORE any `field_validator(mode="before")` runs, so a
  non-JSON env string ("123,456" for a `list[int]`) raises `SettingsError` and the
  validator never sees it. Annotate the field `Annotated[T, NoDecode]` so the raw
  string reaches the validator, which must then parse BOTH the delimited and the
  JSON-array forms itself. Mirroring `project_base_dirs`'s bare before-validator
  (which has the same latent bug - its colon form works only via a constructor
  list, not via env) reproduced it; caught at test time, fixed with NoDecode on
  `telegram_allowed_chat_ids`. Prove a "also accepts X" config path through the
  intended channel (env vs constructor) before trusting it. 20260722-222734.
- `systemd-user-service-ignores-hm-session-vars` (x1): a systemd USER service does
  NOT inherit `home.sessionVariables` (those land in the login-shell env via HM, not
  the systemd user manager), so a value the interactive shell has (here
  `DEN_PATH=~/personal/the-den`) is absent from the service. Set it explicitly on the
  unit (`programs.scufris.settings.den_path` -> `SCUFRIS_DEN_PATH`); confirm by
  grepping the RENDERED unit, not by assuming the shell env carries. 20260726-225845.
- `scufris-web-server-module-is-env-driven` (x1): the new scufris is ONE
  `scufris serve` web server configured entirely via `SCUFRIS_` env vars, not
  the old bot's server+bot split. The service module maps a flat `settings`
  attrset to `SCUFRIS_<UPPER>`, injects `SCUFRIS_WEB_DIST` from `packages.web`,
  and puts codex/claude/git on the service PATH (operator tools, not deps).
  20260721-140157.
- `dynamicuser-needs-explicit-state-and-home` (x1): a systemd service with
  `DynamicUser=true` has no writable `$HOME`, so an app that defaults its state
  dir to `Path.home()/...` fails at runtime. Set `SCUFRIS_STATE_DIR`/`HOME` to
  the `StateDirectory` (`/var/lib/<name>`). The home-manager USER service is
  immune (real home); the trap is nixos-system-service only. 20260721-140157.
- `render-hm-unit-file-not-eval` (x1): to verify a home-manager systemd unit,
  BUILD the `activationPackage` and read the generated `.service` file; eval of
  a single-valued `Service.ExecStart` returns a one-element list that `--raw`
  refuses to coerce (use `--json`/`builtins.head`). 20260721-140157.
- `hm-user-unit-renders-under-home-files-systemd-user` (x1): after building the HM
  `activationPackage`, the rendered user unit is at
  `result/home-files/.config/systemd/user/<name>.service` (readlink through the
  symlink), NOT at `result/<name>.service`. A bare `find ./result -name X.service`
  can miss it if you assume the top level; grep the `home-files/.config/systemd/user`
  path. 20260727-093957.
- `name-the-conditions-a-nix-equivalence-depends-on` (x1): when a fix relies on two
  nix attribute paths resolving to the SAME derivation (here
  `linuxPackages.nvidia_x11.bin` == the host's `hardware.nvidia.package`, so
  nvidia-smi matches the loaded kernel module), "same nixpkgs input" is NOT the
  whole guarantee - it also assumed the host runs the DEFAULT kernel with
  `nvidiaPackages.stable`. Prove the equivalence by comparing store paths, and in
  the comment name the exact conditions it depends on (+ the escape hatch if the
  host pins a non-default kernel or beta/legacy driver), not just the shared input.
  20260727-093957.
- `flake-cant-see-untracked-new-files` (x1): a dirty-tree flake evaluation
  includes modifications to TRACKED files but not brand-new untracked files;
  `nix build` fails with "Path ... is not tracked by Git". `git add` the new
  file (explicit path, never `-A` in this repo) before building. And do not end
  a build with `; echo EXIT=$?` - the echo's 0 masks the build's real exit.
  20260721-141458.
- `nixos-vm-test-for-on-demand-not-checks` (x1): expose a
  `pkgs.testers.nixosTest` as `packages.vm-test` (Linux-only via
  `lib.optionalAttrs pkgs.stdenv.isLinux`), NOT a `checks` entry, so the fast
  lint/type/test gate is not dragged down by a full VM boot; run it deliberately
  with `nix build .#vm-test`. It gives a boot-and-serve proof of the nixos
  module (unit active, `/` serves the dashboard, DynamicUser state dir writable).
  20260721-141458.
- `reserve-serialize-slot-synchronously` (x1): a background task that acquires
  its serialize lock only WHEN IT RUNS leaves a window where another caller
  (a reset arriving right after the turn was started) grabs the free lock and
  jumps ahead of the very turn it should follow. Claim the slot SYNCHRONOUSLY
  when the run is started (a FIFO reservation: append a Future to a per-key
  chain, return the predecessor to await), not inside the scheduled task. Caught
  by out-of-context review of the supervisor. 20260720-221922.
- `supervisor-endpoints-must-be-async` (x1): a FastAPI endpoint that schedules
  background work (`asyncio.create_task`, e.g. via `supervisor.start`) or needs
  the running loop MUST be `async def` - a SYNC endpoint runs in an AnyIO worker
  thread with no event loop, so `create_task`/`get_event_loop` raises "no current
  event loop in thread 'AnyIO worker thread'". Treat "calls supervisor.start" as
  a hard signal for `async def` (like `/api/chat/stream`). 20260720-221942.
- `serialize-then-launch-self-deadlocks-on-shared-key` (x1): an endpoint that
  holds `supervisor.serialized(K)` and then LAUNCHES a turn via a helper that
  reserves the SAME key inside `supervisor.start(serialize_key=K)` deadlocks -
  the per-key FIFO lock is non-reentrant, so the launch waits on the caller's own
  unreleased slot forever (fork held `serialized(ORCHESTRATOR_ID)` around
  `_launch_agent_turn`, hung pytest with no timeout plugin). Endpoints that only
  MUTATE state hold the lock; endpoints that LAUNCH a turn must not (the launcher
  already serializes + 409-guards). When you swap what a held lock's body calls,
  re-derive the lock safety - a lock safe around the old body is not safe around
  a new body that acquires the same key. 20260721-180208.
- `bound-any-per-request-registry` (x1): an in-memory dict keyed by a fresh id
  per request (uuid run_id) that is never pruned is a guaranteed leak on a
  long-lived server. Write the reaping policy (cap + drop-oldest-terminal) in the
  SAME commit as the insertion; each `_Run` there also owned an EventBus buffer,
  so the leak compounded. Caught by out-of-context review. 20260720-221922.
- `moving-logic-off-a-scope-drops-its-incidental-guarantees` (x2): when you move
  work OUT of, or RETIRE, a scope/surface that silently provided a property (a
  request-held lock, a `with` block, a render's read-only gate), enumerate what it
  was providing and re-establish each explicitly BEFORE deleting it. Moving chat
  turns off the held `chat_lock` dropped turn-vs-mutation ordering (20260720-221922);
  retiring settings-view's `renderSettings` dropped its `config.writable` read-only
  gate, so global write controls rendered live+403 on a read-only server
  (20260721-234632, R1 MAJOR). The guarantee you forget is the one never written down.
- `retire-a-path-map-callgraph-and-reroute-shared-tests` (x1): before deleting a
  code path (the codex-exec runners), map its call graph and count each helper's
  usages to split exec-ONLY (delete) from SHARED-with-the-survivor (keep) - so you
  neither orphan dead code nor nick the app-server path. Then re-POINT the deleted
  path's tests that actually covered SHARED behavior (missing-binary, cwd, image
  attach) onto the surviving runner rather than dropping them; coverage must
  survive the retirement, not leave with it. 20260721-180224.
- `cap-what-must-survive-not-just-the-length` (x1): a cap over a RENDERED document
  is a "what must survive" problem, not a length problem - trimming the tail of a
  host-action message cut the undo line and the result (its last two lines) exactly
  on the action class whose preview is longest. Name the load-bearing parts, trim the
  unbounded middle, and pin it with data the real system produces. Sibling of
  `cap-message-length-after-escaping-not-before`. 20260730-104524.
- `cap-message-length-after-escaping-not-before` (x1): to enforce a hard length
  cap (e.g. Telegram's 4096-char message) on text that will be HTML/entity-escaped,
  trim AFTER escaping - `html.escape` expands one char up to ~6 (`&` -> `&amp;`),
  so a raw-length trim does not bound the final message. Cutting the TAIL of
  escaped text is safe from a bare `&` (the cut only ever drops a leading `&...`).
  Test the cap with escapable chars, not plain letters. 20260726-201901.
- `share-one-renderer-so-two-surfaces-cannot-drift` (x1): to make two UI surfaces
  over the same data look identical, EXPORT and reuse the existing (usually the
  read-only) render fn rather than restyling the second surface's markup. Bonus:
  when the interactive bits live in a WRAPPER (toggle/runner in `toolControlCard`/
  `renderToolControls`, not in `toolCard`), the bare renderer is read-only by
  construction, so the read-only guarantee carries over for free. Frontend twin of
  `two-endpoints-when-one-answer-would-lie` (extract the shared core so the two
  never drift). Check for the shared component BEFORE writing CSS. 20260727-101518.
- `enforcement-point-not-the-decision-record` (x2): a DECISION.md states INTENT
  and reads exactly like a description of working code - so a security property
  must be verified at the guard that enforces it, with a file:line, never from
  a decision record or a module docstring. Require the exact principal at the
  execution boundary; a test that rejects one bad credential can still miss no
  credential at all. 20260729-125020, 20260729-125029.
- `security-identity-must-come-from-credential` (x1): audit actor, rate-limit
  bucket, and authorization identity must come from the session or bearer
  credential, never from request body attribution. A caller-provided `agent`
  string may label the proposal, but it must not key caps or answer "who asked."
  20260729-125029.
- `scope-auth-rules-to-their-transport` (x1): security guidance must name the
  transport and credential; an HTTP session rule does not describe an allowlisted
  Telegram chat credential. 20260731-131543.
- `pin-the-input-a-caller-should-not-choose` (x1): when a security property reads
  "the SERVER builds what it activates", audit every input to that build for who
  supplies it. Blocking a caller-supplied store path but accepting a
  caller-supplied REPOSITORY path is the same hole one step removed - an agent
  can commit its own flake anywhere it can write. Noting in the decision record
  that the control is imperfect ("an allowlist would not stop bad Nix") is not a
  reason to skip it. 20260729-125035.
- `a-preview-must-not-execute-what-it-previews` (x1): a better preview obtained by
  RUNNING the unapproved artifact's own code - here
  `<toplevel>/bin/switch-to-configuration dry-activate`, as root, at propose time -
  trades the whole approval for a nicer panel. Where the only honest preview is
  narrower, ship the narrow one and print what is missing and why. Sibling of
  `where-a-class-has-no-honest-preview-say-so`. 20260729-125035.
- `ask-who-owns-it-before-asking-what-shape-it-is` (x1): when a feature touches a
  repo/project/surface something else already owns, the first planning question is
  "whose job is this", not "which of my three designs". Three artifact options for
  something you should not build at all is a well-formed question with no right
  answer - here the config repo was a PROJECT, and the answer deleted five planned
  steps. The signal was in the task's own Story (three narrow verbs = an editor)
  and in the orchestrator spike's Projects model. 20260729-125035.
- `read-the-callers-timeout-before-putting-work-in-a-request` (x1): a synchronous
  probe is only free if every caller can wait. `mcp_common._API_TIMEOUT` is 15s
  and a flake evaluation is seconds-to-minutes, so probing inside the request made
  the one tool an agent always calls report a timeout for a build that was in fact
  running. Long work belongs in the run the request starts, where a failure lands
  on a record. 20260729-125035.
- `the-operators-nix-conf-is-not-yours-to-assume` (x1): nix's new CLI (`nix
  path-info`, `nix store gc`, `nix store diff-closures`, `nix build`) is behind
  experimental features, and whether they are on is the operator's config. Pass
  `--extra-experimental-features "nix-command flakes"` in every argv, as
  `nixos-rebuild` does; found by the hostd VM test, where already-shipped R2 verbs
  turned out to be broken on any host that had not opted in. 20260729-125035.
- `a-nixos-test-vm-has-no-system-profile` (x1): `nix-env -p
  /nix/var/nix/profiles/system --list-generations` is EMPTY in a nixos test VM (it
  boots its toplevel directly), while every installed host has at least one
  generation - so a VM test about generations must create the state a real machine
  already has. Its switch also cannot install a bootloader (grub on the test
  image's ext2 root: "will not proceed with blocklists"), so set
  `boot.loader.grub.enable = false` when the test switches configurations.
  20260729-125035.

## Frontend (web/)

- `poll-render-wipes-the-input-under-the-operator` (x1): a page that rebuilds itself
  on a poll (`replaceChildren`) is fine read-only and wrong the moment it has an
  input - a partially typed value AND the focus vanish on the next tick, which made a
  type-this-token-to-approve gate a race lost every 4s. Gate the POLL's render on
  "is an input inside the page focused"; keep refreshing the data. 20260730-104520.
- `backend-invariants-do-not-cross-into-the-frontend` (x1): a property the backend
  enforces structurally (bounded, honest-about-failure, escaped) does NOT travel
  with the data - the view has to enforce its own half. Having built a package
  whose whole point is "a blank never reads as fine", the cards over it piped
  machine-controlled strings into `innerHTML` (a systemd unit is named by a FILE,
  so `~/.config/systemd/user/<img src=x onerror=...>.service` is stored XSS) and
  rendered a capped count and an indefinitely stale snapshot as if complete.
  Crossing the language boundary reset the attention. When a backend guarantees
  something, write down what the CONSUMER must also guarantee and check that
  list. Structural fix beats discipline: give the view textContent-only helpers
  so there is no HTML sink left to remember. Repeat of
  `escape-only-host-strings-in-element-content`. 20260729-125024.
- `el-helper-returns-htmlelement-not-the-subtype` (x1): the `el(tag, cls, html)`
  helper is typed `HTMLElement`, so `.disabled`/`.value`/`.files` don't exist on
  its result - tsc reds it. Create any element whose subtype-specific property you
  will touch with `document.createElement("button"|"input"|...)` (precise type);
  reserve `el()` for plain container/text nodes. 20260721-180222.
- `interface-method-shorthand-trips-unbound-method` (x1): declaring a callback
  member of a config/deps interface as METHOD shorthand (`forkTurn?(...): void`)
  makes eslint `@typescript-eslint/unbound-method` fire the moment you extract it
  into a `const` (`const fork = config.forkTurn`). Declare such members as
  function-typed PROPERTIES (`forkTurn?: (...) => void`) instead. 20260721-180222.
- `ui-reshape-silently-drops-a-wired-capability` (x1): when a component is
  replaced by a reshaped one, a capability wired into the OLD surface can vanish
  while its backend half survives and stays green. The per-agent SSE reattach
  (an `EventSource` on `/api/agents/<id>/events`) shipped in the old inline run
  panel (F0, 20260721-112428) but the F1/F2/F3 detail-page reshape dropped it -
  the backend relay + its tests stayed, so nothing went red; the page just
  stopped continuing in-flight turns on reload. After a UI reshape, check that
  each capability the old surface had is re-wired, not just that tests pass.
  20260723-001301.
- `forward-typed-null-tracker-resolves-to-never` (x1): a `let x: T | null = null`
  declared BEFORE class `T` resolves its annotation to `null` under the webpack
  ts-loader build (a forward type reference esbuild/vitest tolerate but ts-loader
  does not), so a later `if (!x) throw` guard narrows `x` to `never` and every
  member access reds. Sibling: calling a block-scoped class method as
  `es.emit(...)` trips typed-eslint `no-unsafe-call`. For a construction-tracking
  test double, keep an explicitly-typed module-level `const created: T[] = []`
  (push in the ctor) and read `created[created.length-1]`, with a free helper for
  any call - not a `let` before the class or a class method. 20260723-001301.
- `webpack-dev-server-compression-buffers-sse` (x1): webpack-dev-server defaults
  `compress: true`, which injects the gzip `compression` middleware in front of
  the proxy. It buffers small (sub-1KB) streaming chunks to the end of the
  response (it holds them waiting to reach its size threshold before deciding to
  gzip), so an SSE token stream arrives in one lump on the dev port (:8090) even
  though the backend port (:8000) streams. Set `compress: false` on devServer for
  any SSE endpoint. 20260720-020356.
- `dont-gate-streaming-render-on-a-single-raf` (x1): throttling a live render
  with ONE queued `requestAnimationFrame` is fragile - a later synchronous
  re-render (here `onDone` -> `renderLog`, which detaches the pending node) can
  fire before the rAF paints, so a buffered burst shows nothing until the end.
  Paint eagerly (first update immediate) and time-throttle, don't depend on a rAF
  that something else can clobber. 20260720-020356.
- `curl-streams-browser-doesnt-suspect-the-path-between` (x1): when `curl` (local,
  direct, no `Accept-Encoding`) streams an SSE endpoint but the browser shows it
  all at once, the buffering is in the transport BETWEEN them - a reverse proxy,
  a dev-server, or compression - not the server or the app code. Bisect by layer
  with timestamped probes rather than editing the render. 20260720-020356.
- `tailwind-preflight-strips-defaults` (x1): Tailwind's Preflight base reset (from
  `@import "tailwindcss"`) removes user-agent defaults - notably `list-style: none`
  on ul/ol and native form-control styling (`font: inherit`, `border-radius: 0`,
  transparent bg) - so anything rendered as real markdown/HTML must restore its
  defaults explicitly (`.md ul { list-style: disc }`). When a styled element looks
  "unstyled", grep the BUILT bundle for the Preflight rule before guessing.
  20260719-232155.
- `web-fetch-json-cast-generic` (x1): eslint `recommendedTypeChecked` rejects the
  `any` from `resp.json()`; wrap fetches in a `fetchJson<T>` helper doing a single
  `as T` cast instead of scattering unsafe assignments. 20260719-154539.
- `frontend-verify-needs-e2e-serve` (x2): a green webpack build proves
  compilation, not wiring - serve the bundle through the backend and curl `/` +
  `/api/*` to prove the slice runs, and check WHICH status each endpoint really
  answers (the host page read "not configured" off the queue endpoint, which answers
  `200 []` when the helper is absent; only the audit endpoint 503s - every test was
  green with that wrong). No headless browser here, so visual render is
  user-eyeballed. 20260719-154539, 20260730-104520.
- `side-effect-free-module-for-jsdom-tests` (x1): to unit-test frontend render
  logic, keep it in a module with NO import-time side effects (no auto-start, no
  CSS import) + a thin entry that wires it up; otherwise importing under vitest
  kicks off fetch/timers. `vitest` + `jsdom` drop into the TS/webpack project and
  wire into `npm run ci`. 20260719-160924.
- `build-dom-not-parse-html-for-untrusted-markdown` (x1): to render untrusted
  markdown (e.g. LLM replies) safely, do NOT parse it to HTML and sanitize
  (marked -> DOMPurify) - tokenize the markdown and BUILD the DOM with
  `createTextNode` for every text run + a fixed element whitelist, scheme-validate
  link hrefs. No `innerHTML` of model output = no XSS surface to filter, and zero
  deps. Pin with hostile-input jsdom tests (raw HTML, script-in-fence, javascript:
  link). 20260719-223102.
- `escape-only-host-strings-in-element-content` (x1): when interpolating into
  innerHTML, escape only untrusted STRINGS for their context (element content
  needs `< > &`; attributes also quotes); numbers via `toFixed` are safe. Prove
  it with a jsdom test that a hostile value creates no element. 20260719-160924.
- `webpack-multipage-htmlplugin-per-page` (x1): for a multi-page frontend, use
  one `entry` + one `HtmlWebpackPlugin` (explicit `chunks`) per page + a
  `historyApiFallback` rewrite per sub-route; FastAPI `StaticFiles(html=True)`
  then serves `/` and `/<page>/` with NO backend change. 20260719-180543.
- `route-sensors-to-their-card-not-a-dump` (x1): a flat "all sensors" card reads
  as a text wall; route each reading to the card it describes (core temps onto the
  CPU load squares, drive temps into Disks) and consolidate related cards
  (Memory+swap, Disks=usage+io+temp). Use a `card__subhead` to section a card.
  20260719-190533.
- `stable-rows-with-dash-beats-conditional-sections` (x1): a card that shows/hides
  subsections by "has data this poll" resizes and jars; render a STABLE row set
  (filtered once to the real entities, e.g. base disks via a strict-prefix rule
  dropping partitions + loop/ram noise) and show `-` for absent values; a `.card`
  min-height damps the rest. 20260719-192214.
- `separate-usage-reset-from-log-reset` (x1): a single "reset the chat state"
  helper that clears BOTH the running usage indicator AND the message log is a
  trap for any flow that rebuilds the log and then resets usage (e.g. fork, which
  builds `_messages` then resets the token counter). Keep a narrow `resetUsage()`
  distinct from the full `_resetAgentState()`; call the narrow one when the
  messages must survive. 20260719-224101. RETIRED 20260719-223106: the head
  `ctx · out` indicator was deleted (redundant with the API-driven context box),
  so `resetUsage` no longer exists - this lesson has no referent in the current
  code; kept only as history.
- `dont-shadow-browser-globals-with-domain-words` (x1): a local named `window`,
  `document`, `name`, `status`, `length`, etc. shadows a global other code in the
  same module relies on (here `const window` for a rate-limit window descriptor,
  next to `window.confirm`/`window.setTimeout`). eslint's default config does NOT
  flag it, so it slips through `npm run ci`. Suffix the domain word
  (`windowLabel`). 20260719-223106.
- `prefer-one-authoritative-render-over-a-parallel-client-counter` (x1): a
  client-side accumulator that shadows a number the API already returns
  authoritatively WILL drift (the head `ctx · out` counter only summed turns done
  in the current tab; the context box reads cumulative totals from disk). When an
  endpoint carries the truth and every mutation path already refreshes from it,
  delete the parallel counter instead of syncing it - it removed state
  (`applyUsage`/`resetUsage` + two module vars), not just a widget. 20260719-223106.
- `full-rebuild-render-resets-scrolltop` (x1): a render that does
  `container.replaceChildren()` throws away the scroll position (scrollTop -> 0).
  A "don't yank the user" scroll policy must CAPTURE scrollTop before the rebuild
  and RESTORE it when not auto-scrolling - merely skipping the scroll leaves the
  reader flung to the TOP, because the rebuild already moved them. jsdom cannot
  catch this (scrollTop is a static 0 with no layout), so reason about it or test
  in a browser. 20260719-223111.
- `aria-live-on-a-rebuilt-region-over-announces` (x1): `aria-live` on a container
  that is re-rendered via `replaceChildren` makes assistive tech treat the whole
  thing as new each turn (with `aria-relevant="additions"`, every child is a fresh
  "addition"). To announce just the new reply, wrap the live region around the
  incrementally-appended content, not a wholesale-replaced log. 20260719-223111.
- `flex-display-defeats-the-hidden-attribute` (x1): a rule like
  `.block { display: flex }` overrides the UA `[hidden] { display: none }`, so
  `element.hidden = true` will NOT hide it. Add `.block[hidden] { display: none }`
  and pin it with a "hides when empty/null" jsdom test. 20260719-212207.
- `dispatch-only-known-kinds-not-else-error` (x1): when switching on a
  discriminated union's `kind` (e.g. SSE stream events), do NOT put the
  error/fallback in the final `else` - a newly added variant then silently routes
  to the error path (adding `text_delta` made every token call `onError`). Match
  each known kind explicitly (including `error`) and IGNORE unknown ones, so a new
  variant is additive, not a regression. 20260720-002621.
- `clickable-container-guards-both-activation-paths` (x1): a clickable container
  (`role=button tabindex=0` card) wrapping an interactive child (a delete button)
  has TWO bubbling channels - pointer `click` and keyboard `keydown`. Guarding
  only the click (`ev.stopPropagation()` on the button) still lets Enter/Space on
  the focused child bubble to the container handler and fire it too. Guard the
  container's keydown with `ev.target !== card` (or handle only
  `target===currentTarget`), and test the keyboard path, not just the mouse.
  Caught by out-of-context review. 20260721-112434.
- `assert-form-control-value-not-textcontent` (x1): when a field migrates from a
  read-only text row to a form CONTROL (`<input>`/`<textarea>`/`<select>`), its
  live value is a PROPERTY (`.value`), not child text - `textContent`/`innerHTML`
  do NOT reflect a set `.value`. A `text.toContain(value)` assertion then goes
  vacuous (passes on an EMPTY control). Assert `.value` (or `.selectedOptions`)
  and migrate the assertion in the same edit as the field. 20260721-112435.
- `re-rendered-element-use-onhandler-not-addeventlistener` (x1): registering a
  handler with `addEventListener` on an element that is RE-RENDERED in place (a
  pure render called on every open/poll) STACKS a new listener each time - a
  leak (here the modal backdrop-close handler). Use the `on<event>` property
  (`root.onclick = ...`), which overwrites, OR remove the prior listener first.
  Caught by out-of-context review. 20260721-152728.
- `persistent-widget-needs-its-own-root-not-a-polled-region` (x1): a widget that
  must survive across polls (a chat log, a live editor) cannot live inside a DOM
  region that a status/poll loop rebuilds with `replaceChildren` - the rebuild
  wipes it mid-interaction. Give it its OWN root element (a sibling container the
  poll never touches) and mount it once. Here the per-agent chat got its own
  `#agent-chat` beside the polled `#agent-detail`. 20260721-112438.
- `reuse-the-shared-primitive-not-the-globalized-shell` (x1): when a task says
  "reuse component X", check whether X is genuinely reusable or welded to module
  globals. The landing chat's render/composer was tied to agent-view module state
  (sessions, fork, image, slash); only the STREAMING (parseSseFrames + the SSE
  consume loop) was truly shared. Extracting that primitive (a URL-parameterized
  `chat-stream.ts`) + re-implementing a lean stateful shell beat de-globalizing
  the tangled module. Name the split at plan time. 20260721-112438.
- `leaked-global-stub-fails-the-NEXT-test-not-its-own` (x1): a vitest test that
  `vi.stubGlobal`s a browser global without restoring it leaks into the following
  describe, whose own `afterEach(vi.unstubAllGlobals)` runs only AFTER its first
  test has already run - so the victim is someone else's first test, and it only
  fails in a whole-FILE run. Tell: passes with `-t <name>`, fails in the file. Fix
  at the source (restore in the describe that stubs), never by loosening the
  victim's assertions; instrument the ambient globals rather than theorizing.
  20260729-125015.
- `persistent-ui-state-needs-a-test-reset-hook` (x1): module-level UI state
  (expanded set, sort key) that must survive poll re-renders leaks across jsdom
  test cases; export a small reset and call it in `beforeEach`. 20260719-182901.
- `client-side-rolling-window-beats-backend-history-for-live-graphs` (x1): for a
  btop-style live sparkline, accumulate samples in a bounded client-side ring
  buffer over the poll the page already runs (`/api/stats`), NOT a backend
  sampler + `/api/history`. The backend design only earns its complexity
  (lifespan task, memory bounds, endpoint) when cross-reload/cross-client
  persistence is an actual requirement - btop history is since-start anyway.
  Inline SVG (area polygon + polyline, viewBox + `preserveAspectRatio=none` +
  `vector-effect: non-scaling-stroke`) needs no canvas/dep and scales to any
  card width. 20260719-182915.

- `escape-client-strings-before-glob` (x1): any client-controlled string
  interpolated into a `glob`/`Path.rglob` pattern must be `glob.escape`d first, or
  a metacharacter value (e.g. a session id of `*`) silently matches unintended
  files. "Local single-user app" is not a reason to skip it. Pin with a `"*"`-id
  test. 20260719-212203.

## Monitoring / collector

- `distinct-loop-vars-for-different-types` (x1): don't reuse a loop variable name
  across two loops whose elements are different nominal types (e.g. psutil
  `snetio` vs `sdiskio`) - mypy binds one type to the name and the second loop's
  attribute access fails. Name them apart. 20260719-182846.
- `tatr-r-walks-up-and-needs-tasks-dir` (x1): `tatr -r <dir> <cmd>` changes to
  `<dir>` then searches UPWARD for the nearest `tasks/` - it does not create one
  (`tatr -r <dir> new` errors "No 'tasks' directory found in hierarchy"). To
  dir-scope tatr to a project, gate on `<dir>/tasks` existing (return empty
  otherwise) so it cannot surface a PARENT's tasks, and mkdir `<dir>/tasks`
  before a test `tatr new`. 20260720-210645.
- `tatr-ids-are-second-resolution` (x1): tatr task IDs are `YYYYMMDD-HHMMSS`, so
  two `tatr new` in the same second COLLIDE (the second fails "already exists",
  since 0.2.0). Any test or tool that creates multiple tasks in a row must space
  them (`sleep(1.1)`) or expect-and-retry the collision - do not chain rapid
  creates. 20260719-224058.
- `tatr-new-body-file-omits-the-header` (x1): `tatr new -b <body-file>` injects
  its OWN `STATUS/PRIORITY/TAGS` header from the title/`-p`/`-t`, so a body file
  that also starts with those three lines yields a DUPLICATED header needing a
  hand edit. Start the body file at the first `##` section (or the Goal); let
  tatr own the header. 20260727-020723.
- `sysfs-per-cpu-counters-are-not-per-cpu-quantities` (x1): everything under
  `/sys/devices/system/cpu/cpu*/` LOOKS per-logical-cpu, but the value can belong
  to the core, the package or the socket and simply be republished on every cpu
  that shares it. `core_throttle_count` is per PHYSICAL core, so summing it over
  logical cpus reports exactly 2x with SMT on (measured: 162 where the truth was
  81). Ask what hardware a value belongs to before aggregating, and dedup by
  `topology/core_id` + `physical_package_id` (core_id is unique only within a
  package). Reduce duplicates with MAX, not last-write-wins: each cpu's own
  interrupt handler writes these, so siblings can be out of step - visible here
  as package counters reading 78/80/82 across cpus of one package. The insight
  was already applied to the package counters four lines away and not to the
  core ones. 20260729-205145.
- `re-measure-output-when-you-swap-the-command` (x1): when you replace a CLI for
  a correctness or safety reason, the parser downstream was written against the
  OTHER program - re-run the new one and look at what it actually prints. Swapping
  `nix-collect-garbage --dry-run` for `nix-store --gc --print-dead` (right call:
  the former also trims profile generations) kept the old summary-line
  assumption, so a healthy EMPTY store would have reported "no count reported" -
  empty rendered as broken, inside the package built to prevent that. Sibling of
  `capture-real-cli-output-for-parser-tests`: capture the fixture again, do not
  port the old one. 20260729-125024.
- `capture-real-cli-output-for-parser-tests` (x1): when parsing a CLI's output,
  run it once and pin a REAL captured line as the test fixture (nvidia-smi CSV,
  incl. `[N/A]`), so the parser is written against reality. 20260719-182846.
- `psutil-process-iter-caches-cpu-percent` (x1): `psutil.process_iter` reuses
  Process objects internally, so `cpu_percent` is a real delta across calls with
  no per-pid cache of your own - prime it once (iterate at startup) and read per
  sample. 20260719-182901.

## Agent / Codex

- `synchronous-request-read-timeout-is-a-total-cap` (x1): an httpx READ timeout
  on a SYNCHRONOUS single request (a blocking POST answered only when done, or a
  buffered `httpx.request` over a whole SSE body) is NOT a per-chunk idle bound -
  for any silent stretch it caps the whole turn. So "make the timeout idle-based"
  cannot be done by tuning the read bound; disable it (`read=None`, keep
  connect/write/pool) and bound the turn out-of-band instead. Trace the call
  shape before choosing a timeout model. 20260724-081804.
- `read-none-needs-a-backstop-at-every-entry-point` (x1): disabling a read
  timeout (`read=None`) is only safe if something else bounds the turn (here the
  supervisor heartbeat). A guarantee that holds on the main path can be ABSENT on
  a CLI/one-shot/test path that drives the same function directly - grep every
  caller and confirm each runs under the backstop. Out-of-context review caught a
  `scufris chat` + opencode hang-forever this way. 20260724-081804.
- `stream-turn-timeout-is-idle-not-wallclock` (x1): a per-turn wall-clock
  deadline over a streaming subprocess kills a turn that is actively producing
  output the moment it runs long (here `_stream_app_server` cut any turn past
  120s mid-stream, so a slow sub-agent "finished" as an error). Bound each
  `readline` with a per-read IDLE timeout instead - silence is the failure
  signal, not total duration; a genuinely hung stream is still cut. This was a
  leftover contradicting the supervisor's own no-output stall guard (ADR-001).
  20260724-011406.
- `reword-shared-config-doc-grep-its-readers` (x1): rewording the docstring/
  semantics of a config field that has MORE THAN ONE reader (here
  `agent_timeout_seconds`, read by the codex runner AND the opencode backend's
  httpx client) leaves the doc true for the field you edited and false for the
  other consumer. `grep -rn <field> scufris/` its readers before rewording so
  the doc stays true for all of them. Caught by out-of-context review.
  20260724-011406.
- `run-completion-callback-keys-by-launch-snapshot-not-current-config` (x1): a
  callback that persists run state at turn-END (here `mark_finished` writing the
  session id) must key by the config the run LAUNCHED with, not whatever the config
  is now - a config edit (backend switch via `update_agent`, which is NOT serialized
  against in-flight turns) can land mid-run, so re-reading the current backend
  mislabels the finishing session. Thread the launch-time snapshot's value into the
  callback. Caught by out-of-context review. 20260723-001251.
- `completion-callback-write-after-existence-check` (x1): a NEW persisted write
  added to a run-completion callback, keyed by an entity id (here `mark_finished`
  writing the run outcome), must sit AFTER the existence guard (`_raw`) and be
  pinned by a delete-mid-run test - the callback can fire after the entity was
  deleted (the code even documented this path), so a write placed BEFORE the guard
  resurrects a stale record that survives restart. Mirror where the sibling
  store's write already sits, not just its class shape. Caught by out-of-context
  review. 20260723-094258.
- `claude-mcp-config-is-variadic-bound-it-with-a-flag` (x1): `claude --mcp-config
  <configs...>` is GREEDY - it swallows every following token as another config path
  until the next `--flag` (a probe's `--mcp-config "$JSON" mcp list` failed with
  "config file not found: .../mcp"). In the backend argv, always follow
  `--mcp-config <json>` with a flag (`--strict-mcp-config` / `--allowedTools` /
  `-p`), never a positional. `--mcp-config` accepts an INLINE `{"mcpServers":{...}}`
  JSON string (not just files). Probed live, claude 2.1.193. 20260723-193218.
- `claude-mcp-tool-approval-is-allowedTools-not-permission-mode` (x1): to run an MCP
  tool UNATTENDED on claude, `--permission-mode` is not enough - allowlist the tool
  by its `mcp__<server>__<tool>` name via `--allowedTools` (+ `--strict-mcp-config`
  to ignore project/global MCP config). Then `--permission-mode default` does not
  hang. Proven live: with `--allowedTools mcp__scufris__request_input`, claude
  exposed AND called the scufris `request_input` tool with no approval prompt.
  20260723-193218.
- `codex-tool-choice-only-steers-via-the-turn-prompt` (x1): to make codex prefer an
  MCP tool over its built-in shell, the instruction MUST ride the turn prompt.
  Probed live (0.142.2, "tell me about this host" with the scufris MCP server):
  strengthened tool descriptions -> 0 MCP; `-c experimental_instructions_file` ->
  0 MCP; `AGENTS.md` via `-C <dir>` -> 0 MCP; a preamble prepended to the prompt ->
  0 shell / 3 MCP. codex ignores the "soft" instruction channels for tool choice.
  If the preamble must stay out of the visible transcript, sentinel-wrap it
  (`[scufris-tools]...[/scufris-tools]`) and strip it on read in the title +
  transcript path (strip at the READ boundary so fork seeds stay clean too).
  20260720-102559. Reapplied to steer sub-agents to `request_input`
  (20260723-153609), the orchestrator's comms poll (20260723-153615), the
  den-journal/macros chain (20260727-020723) and agent delegation
  (20260727-022121) - all needed the instruction on the turn prompt.
- `steer-permission-mode-for-implementing-agents` (x1): when steering the
  orchestrator to spawn+run an agent that must CHANGE files, also steer the
  `permission_mode` - `create_agent` defaults to `manual` (read-only:
  enums.py, codex `read-only` / claude `default`), so a delegated agent left at
  the default makes 0 tool calls no matter how good its goal. Demand `edit`/`auto`
  for implementation work. 20260727-022121.
- `plan-locates-transform-from-the-call-site-not-the-model` (x1) -> plan skill: a
  plan step asserted WHERE a transform runs (steering "added inside the backend per
  turn", so the captured prompt carries it) from an architecture model; in fact
  `_steer` runs downstream at agent.py:583, so the prompt captured at
  _launch_agent_turn is already raw. Caught in work by reading the call site. When a
  step claims before/after/which-layer, grep + cite the call site or phrase it
  verify-first - the plan skill's cite-the-mechanism rule covers "where", not just
  "what". 20260724-141430.
- `close-stdin-when-probing-codex-exec-with-an-arg-prompt` (x1): `codex exec
  "<prompt>"` still blocks ("Reading additional input from stdin...") unless stdin
  is closed - pass `</dev/null` (the app uses a set stdin; a shell probe does not).
  Live codex turns take 1-3 min, so run probes in the BACKGROUND (Bash
  run_in_background) or they trip the 3-min foreground command timeout.
  20260720-102559.
- `codex-app-server-for-token-streaming` (x1): `codex exec --json` is turn-level
  (no token deltas - proven by probing real turns + grepping all rollouts).
  Token-by-token text + reasoning come only from the experimental `codex
  app-server` JSON-RPC-over-stdio protocol. Drive it: `initialize` -> `thread/start`
  (or `thread/resume {threadId}` for multi-turn) -> `turn/start {threadId, input:
  [{type:text,text,text_elements:[]}]}`; the request RESPONSE returns immediately
  and the stream arrives as NOTIFICATIONS (`item/agentMessage/delta {delta}`,
  `item/reasoning/textDelta`, `item/completed`, `thread/tokenUsage/updated`,
  `turn/completed`). Method/event shapes come from `codex app-server generate-ts`.
  PROBE the handshake before building; gate behind a flag (experimental).
  20260720-002619.
- `sse-streaming-from-a-subprocess-in-fastapi` (x1): to stream a slow subprocess
  to the browser: (1) read stdout line-by-line (`await proc.stdout.readline()`)
  with a per-read IDLE timeout (reset each line), not a per-turn wall-clock
  deadline (a shared deadline kills a still-streaming turn - fixed
  20260724-011406) and not `communicate()`; (2) yield events from an async
  generator and kill the proc in `finally` for early close (client disconnect);
  (3) serve via `StreamingResponse(gen(), media_type="text/event-stream")` emitting
  `data: <json>\n\n`, holding any turn lock for the whole stream; (4) client-side
  read `resp.body.getReader()` and parse frames incrementally, carrying the
  partial-frame remainder across chunks. Keep the non-streaming path intact +
  additive. 20260719-223103.

- `codex-binary-breaks-uv2nix-venv` (x1): `openai-codex` bundles a prebuilt
  `codex` CLI that fails auto-patchelf in the uv2nix build (`libtinfo.so.6`).
  Keep it operator-installed and lazy-imported, never a pinned dep. A NixOS
  runtime (nix-ld/FHS/nixpkgs codex) is a separate follow-up. 20260719-162356.
- `optional-dep-vs-deps-all` (x1): the uv2nix dev venv is built from
  `workspace.deps.all`, so a dep that must NOT be in the venv cannot be a
  pyproject optional-extra either - it has to stay out of the workspace
  entirely (document an out-of-band install instead). 20260719-162356.
- `introspect-sdk-not-spike-paraphrase` (x1): for a post-cutoff SDK, install the
  wheel no-deps into a throwaway dir and `inspect.signature` the real classes
  before coding - a spike's method names are a paraphrase, close but wrong in
  specifics. 20260719-162356.
- `codex-exec-is-the-nixos-path` (x1): drive Codex via the nixpkgs `codex` CLI
  (`codex exec --sandbox read-only --skip-git-repo-check --ephemeral
  --output-last-message <file>`, shared `~/.codex` auth), NOT the openai-codex
  SDK whose bundled binary breaks the uv2nix venv. `pkgs.codex` in the dev shell.
  20260719-164418.
- `codex-resume-rejects-sandbox` (x1): `codex exec resume` inherits the original
  session's sandbox and errors on a repeated `--sandbox`; pass session-scoped
  flags (`--sandbox`) only on the FIRST turn, not on resume. A fake that ignores
  unknown args won't catch it - only a live run does. 20260719-162406. INVERSE
  of `resume-must-re-send-per-turn-runtime-settings` (app-server path) - do not
  carry this exec lesson across transports.
- `resume-must-re-send-per-turn-runtime-settings` (x1): scufris spawns a FRESH
  `codex app-server` process per turn, so `thread/resume` restores conversation
  state but NOT the process-level sandbox - it reverts to read-only. The runner
  MUST re-send `sandbox` (and any session-scoped runtime setting: model,
  approval, cwd) on `thread/resume {threadId, sandbox}`, exactly as on
  `thread/start`; `ThreadResumeParams` accepts it (`generate-ts`). Symptom: an
  auto/edit agent writes on turn 1 then goes read-only on every resume turn.
  This is the INVERSE of exec's `codex-resume-rejects-sandbox` - the transport
  decides, so read the contract, don't reason by verb name. 20260721-183828.
- `probe-cli-json-shape-before-scoping-streaming` (x1): check a CLI's `--json`
  event granularity before promising "streaming". `codex exec` emits turn-level
  events (`thread.started`/`turn.completed`), not token deltas, so chat is
  honestly turn-based, not token-streamed. 20260719-162406.
- `codex-mcp-register-via-c` (x1): register an MCP server per-invocation with
  `codex exec -c 'mcp_servers.<id>.command=...' -c '...args=[...]'` - NO
  `~/.codex/config.toml` edit needed; confirm with `codex mcp list -c ...`.
  20260719-162419.
- `codex-exec-mcp-approval` (x1): unattended `codex exec` auto-cancels MCP tool
  calls ("user cancelled MCP tool call"); enable them WITHOUT dropping the
  sandbox via `-c mcp_servers.<id>.default_tools_approval_mode="approve"` +
  `-c approval_policy="never"`, keeping `--sandbox read-only`. Never
  `--dangerously-bypass-approvals-and-sandbox`. 20260719-162419.
- `codex-total-vs-last-token-usage` (x1): codex's `token_count.info` carries BOTH
  `total_token_usage` (cumulative across all turns, grows unbounded) and
  `last_token_usage` (the last request). For "how full is the context window" use
  `last_token_usage.input_tokens / model_context_window`; `total_*` overcounts and
  can exceed the window (a 2-turn session read ~23% vs a true ~6%). Verify any
  percent-of-capacity figure on MULTI-turn data where the two diverge, not a
  one-shot session where they happen to be equal. 20260719-212207.
- `harvest-the-stream-you-already-run` (x1): before adding endpoints/extra
  subprocess calls to expose a tool's internals, check what its existing output
  already carries. `codex exec --json` already held per-turn `mcp_tool_call`
  items + `turn.completed.usage`; the agent parsed one field and dropped the
  rest, so surfacing tool-calls + token usage was just extending the parse.
  20260719-201720.
- `codex-per-server-env-filters-mcp-tools` (x1): codex registers whole MCP
  SERVERS, not individual tools, so to hide one tool of a server pass a signal
  to the server via codex's per-server env
  (`-c mcp_servers.<id>.env.KEY=<json>`) and have the server drop that tool
  from its registry at startup (FastMCP `mcp._tool_manager.remove_tool`) - the
  UI "enabled" flag is only a mirror, the real guard is the server not
  advertising it. Probe `codex mcp list -c mcp_servers.x.env.KEY=...` first
  (the Env column populates). 20260720-184137.
- `backends-tag-provenance-differently` (x1): `codex exec` and `codex app-server`
  write different session `originator` values - exec uses codex's default
  "codex_exec", app-server uses the `clientInfo.name` sent on `initialize`
  ("scufris"). Any code that scopes by originator (the session switch list) must
  accept the whole set scufris produces, or switching backends silently changes
  what is visible. 20260720-020345.
- `check-disk-before-assuming-data-loss` (x1): when records vanish from a UI list
  ("are my sessions deleted?"), confirm the underlying files still exist BEFORE
  touching anything - a missing list entry is far more often a filter/scope
  mismatch (here an originator filter) than a real deletion. 20260720-020345.

- `narrowing-a-persisted-enum-needs-a-coercion-validator` (x2): changing the
  members of a persisted/config enum (`agent_backend`) BRICKS startup for any
  state/env still holding an old value, because the Literal rejects it on load.
  Add a pydantic `field_validator(mode="before")` that folds the old value to its
  replacement so existing state loads, while keeping the API INPUT model strict
  (reject the old value on new writes -> 422, pinned by a test). Same shape whether
  narrowing (exec dropped, 20260721-152746) or widening (app_server|exec -> codex
  when `agent_backend` became codex|claude|mock, 20260721-180224).

- `recon-then-recut-an-architectural-umbrella` (x1): when a seeded task turns out
  to conflate several architectural changes (B5 = retire an abstraction + unify
  session storage + converge UI + retire a runner), buy an out-of-context recon
  map FIRST, then re-cut into ordered sub-tasks with explicit SCOPE GUARDS
  ("does NOT touch X", "two paths coexist temporarily") and land the safe slice
  first - rather than grinding a 2000-line mega-change. Surface the re-cut to the
  user. Corollary: defer a sub-seam to the slice that OWNS it (B5a's editable
  config -> B5b) instead of shimming it early. 20260721-112439.
- `always-present-synthetic-item-invalidates-empty-assertions` (x2): a synthetic
  member in a collection (the reserved orchestrator in the agent list) drives a
  whole class of assertions at once, in BOTH directions. Adding it breaks every
  "empty"/"== []"/"no X" assertion + empty-state UI; later REMOVING it from the
  list (making it a hidden default) breaks the mirror "is present"/"is first"/
  "len == N" assertions and re-enables the empty state. Grep the whole class up
  front and flip them in one pass. 20260721-112439, 20260721-234558.
- `query-service-status-not-os-proxy` (x1): to know an external service's state
  (a model "loading", a job's progress), query the service's own status API, not
  an OS-level proxy. A llama-server model load showed FLAT process RSS
  (~80-190MB) the whole time because `cudaSupport=true` loads weights into VRAM,
  not RSS - so RSS was structurally incapable of answering "is it loading", yet I
  inferred "not loading / downloading" from it and burned 15min + two detours.
  The authoritative sources (`GET /v1/models` `status.value`, the HF blobs dir)
  were there all along. Generalizes the AGENTS.md "verify the mechanism, don't
  infer from a proxy" rule to external services. 20260722-135520.
- `establish-the-real-gate-and-its-baseline` (x1): find the repo's ACTUAL check
  gate and its current pass/fail state at task START, not at verify time. Here the
  gate is `nix flake check` -> `mypy .` (not the light `mypy scufris/`), and it is
  RED on master with 44 pre-existing tests/ arg-type errors (no `pydantic.mypy`
  plugin). Not knowing that cost a "did I regress?" detour; the fix is to baseline
  master (44==44 -> zero net-new) rather than chase absolute mypy failures. Filed
  20260722-153555 to green it. 20260722-135525.
- `hf-refetches-on-upstream-revision-change` (x1): the host `llama-cpp` service
  (`hf-repo`/`hf-file`) re-downloads a GGUF when the upstream HF repo revision
  changes, even with an older blob cached - so a model that "worked yesterday"
  can cold-load for tens of minutes (~26GB) on next use. Budget agent turn
  timeouts for it; pin a revision or `HF_HUB_OFFLINE=1` to avoid surprise
  refetch; `huggingface-cli delete-cache` reclaims the orphaned blob.
  20260722-135520.
- `codex-workspace-write-protects-dot-git` (x1): codex's `workspace-write`
  sandbox (our `edit` mode) makes the workspace writable but carves `.git` back
  out as READ-ONLY, so a commit fails with `.git/index.lock: Read-only file
  system` while `tasks/` etc. stay writable - that split is the fingerprint of
  `edit`, NOT `auto` (`danger-full-access` has no such block). The app-server
  `sandbox` param is a plain 3-value enum (rejects any structured payload), so
  re-granting git must go on the app-server ARGV as `-c
  sandbox_workspace_write.writable_roots=[...]`, resolved via `git rev-parse
  --path-format=absolute --git-dir --git-common-dir` (a sprout worktree needs
  BOTH: its own gitdir under `.git/worktrees/<name>` and the parent's shared
  common `.git`). No-op for `manual`/`auto`. See `_sandbox_overrides`.
- `subprocess-line-reader-needs-explicit-limit` (x1): any
  `asyncio.create_subprocess_exec` whose stdout is consumed with `readline()`
  MUST pass an explicit `limit=` - asyncio's 64 KiB default raises a bare
  `ValueError` ("Separator is not found, and chunk exceed the limit") on any
  longer line, which for an LLM app-server / stream-json frame (a wide `rg`, a
  `tatr ls` over hundreds of tasks, a big file dump) is routine, not
  exceptional. Both scufris backends had this latent; a codex sub-agent died
  ~30s into orientation on a large repo. Raise it (`STREAM_READ_LIMIT` = 8 MiB,
  shared) AND wrap the `readline` so an overflow is a diagnosable `StreamError`,
  not an opaque supervisor crash. 20260727-133302.
- `streamerror-ends-a-turn-done-not-error` (x1): a backend that ends a turn by
  YIELDING a terminal `StreamError` (idle timeout, over-limit line, thread-setup
  failure) completes the stream normally, so `supervisor._drain` publishes the
  event and the run settles `RunPhase.DONE` - the `_execute` except-clauses never
  fire and `run.error` stays None. After the parent >64 KiB fix turned an uncaught
  ValueError into a yielded StreamError, a FAILED turn started persisting as DONE
  with an empty message (looked successful), invisible to `pending_agents`. Record
  the detail on `run.error` in `_drain` (last-wins; leave RunPhase alone) and let
  the persist chokepoint map `run_state.error` -> `AgentState.ERROR` with the
  detail as the durable outcome message. 20260727-140443.
- `flipping-a-terminal-state-needs-a-consumer-and-reserved-member-sweep` (x1):
  when a change lets an agent reach a terminal STATE it could not before (here
  ERROR via a StreamError), sweep every reader of that state - especially ones
  that treat the reserved orchestrator specially. `AgentStore.list()` hid the
  orchestrator but `pending_outcomes()` did not, so the orchestrator could
  self-appear in its OWN `pending_agents` "who needs me" poll until the exclusion
  was added. Caught by out-of-context review. 20260727-140443.
- `error-outcome-message-beats-a-captured-reply` (x1): on a FAILED turn the
  durable outcome message must be the failure detail, not a stale captured success
  reply - a rogue backend can emit a `StreamDone` then a trailing `StreamError`,
  and "captured reply wins" would then show a success message on an errored row.
  Prefer `run_state.error` over the captured reply when the run failed. Caught by
  out-of-context review. 20260727-140443.
- `sync-read-inline-on-a-latency-loop-stalls-it` (x1): a provider awaited INLINE
  in a dispatch/poll loop (the telegram bot's `/settings`,`/stats`) must off-load
  SYNCHRONOUS I/O - `read_usage`'s rollout rglob, `collector.sample`'s psutil -
  via `asyncio.to_thread`, else it blocks the event loop and stalls the next
  long-poll and any concurrent streaming. Sibling of
  `self-loopback-blocking-call-needs-a-real-socket-test`. 20260728-222321.
- `grep-every-call-site-before-changing-a-built-signature` (x1) -> work skill: a
  new required (kw-only) param breaks every constructor; grep ALL call sites AND
  scan per-site arg variations FIRST - a `replace_all` on the "identical" block
  silently skips the site whose Nth arg differs (here `idle_cancel` vs
  `on_cancel`), surfacing only as a later test error. 20260728-222321.
- `assert-a-new-formatter-against-its-real-output` (x1): write substring
  assertions for a brand-new render/format function against its ACTUAL printed
  output, not your mental image - two false failures here came from test-data
  (a redundant "codex" prefix) and asserting no backtick in a body whose own code
  fence uses them. 20260728-222321.

## Pending promotions (3+ occurrences, user decides)

- `format-only-the-files-you-edited-not-whole-dirs` (x3) -> work skill verify-step:
  scope every `ruff format` / `ruff check --fix` / `prettier --write` to the files
  you edited, never `.` or a whole dir - the repo-wide form reflows unrelated
  formatter-version drift into the diff, forcing a revert dance (the flake gate is
  `ruff check` lint-only, so the drift is never a gate failure). Candidate guard: a
  work-skill verify note, or a wrapper that formats only `git diff --name-only`.
  20260724-141430, 20260724-152157, 20260727-105609, 20260727-123342.
- `orchestrator-steering-is-one-block-two-clauses` (x3) -> ALREADY GUARDED by a
  tool (tests): `STEERING_PREAMBLE` and `AGENT_STEERING_PREAMBLE` must each stay
  a SINGLE `[scufris-tools]...[/scufris-tools]` block (`strip_steering` removes
  only the first leading block, `count=1`); add new guidance as a CLAUSE inside
  the one block, never a second block. `test_orchestrator_steering_stays_a_single_block`
  and `test_agent_steering_stays_a_single_block` assert one open/close per
  preamble, so a second-block regression fails CI - promotion effectively done
  via the test guard; user only confirms no further template/skill change is
  wanted. 20260723-153615, 20260727-020723, 20260727-022121.
- `ground-steering-text-in-the-real-tool-signatures` (x3) -> work/plan skill or a
  test: before writing turn-prompt steering that names a tool, read its actual
  name+signature in `mcp_server.py` and match verbatim
  (`macros_lookup(query)`, `create_agent(name, project_id, backend, permission_mode)`);
  a typo'd name steers to a call that cannot succeed - worse than no steering.
  Candidate guard: a test asserting every backticked `tool_name(` in the
  preambles resolves to an `@mcp.tool()` def. 20260723-153615, 20260727-020723,
  20260727-022121.
- `isolate-state_dir-in-tests-that-assert-config` state_dir half PROMOTED
  2026-07-27 (autouse `_isolate_state_dir` conftest fixture); OPTIONAL remaining
  decision for the user: fold the `.env` half (`_env_file=None`) into a shared
  hermetic-Settings helper too. Entry annotated in the Testing section.
- `nix-devshell-import-resolves-to-cwd-source` (x3) -> AGENTS.md verify-step: a
  console-script entrypoint in the nix dev shell (`pytest`, `scufris`) runs the
  BUILT/main-checkout package, NOT a worktree's edits; only the `python -m` form
  puts CWD first on sys.path. Verify branch code with `python -m pytest` (tests)
  and `cd <tree> && python -m scufris` (live server), never the bare console
  script from elsewhere. Operator corollary: a running `scufris` won't serve
  landed code unless its build target has it. 20260719-212205, 20260720-184136,
  20260723-120507.
- `protocol-signature-change-hits-the-doubles` (x3) -> work skill verify-step: changing
  a `Protocol`/interface method signature reds every test DOUBLE that reimplements it
  (fixed arity or `**kwargs` that omit the new param), not just the real impls mypy
  flags - and mypy drift is invisible to a passing pytest run. Before running, grep for
  every implementor AND every test stand-in (`def <method>`) and update them in one pass
  instead of discovering each by a `TypeError`; a "green" claim must name mypy explicitly.
  In 20260722-222717 the impls were grepped up front (caught a 4th backend the plan
  missed) but the doubles were still found by TypeError - so make the double-sweep part
  of the same step. 20260720-144530, 20260720-174021, 20260722-222717.
- `optional-trailing-param-silently-dropped-by-structural-impls` (x1) -> work skill
  verify-step (variant of `protocol-signature-change-hits-the-doubles`): adding an
  OPTIONAL trailing param to a shared TS callback/config interface does NOT error the
  implementers that omit it - structural typing accepts a narrower function, so
  mypy/tsc/webpack all stay green while a bespoke impl silently ignores the new arg
  (here the orchestrator `forkTurn` dropped the cancel `AbortSignal`, caught only by
  review). Grep every implementer of the interface and thread the param through each;
  the compiler will not find the gaps. 20260728-134840.
- `type-change-fails-strict-tsc-not-vitest` (x3) -> AGENTS.md verify-step line (or a
  pre-commit/check hook): after changing a shared TS interface (add/remove/retype a
  field), run the webpack BUILD (`npm run build` / `npm run ci`), not just `vitest` -
  esbuild transpiles without type-checking, so a fixture that constructs the type
  breaks only at the ts-loader gate. 20260720-122517, 20260721-180222,
  20260720-134545.
- `render-rewrite-orphans-its-css` (x3) -> lint/build check or frontend AGENTS.md:
  a render rewrite that drops DOM structure (or retires a whole surface, e.g. a
  modal), OR just STOPS emitting a state class (e.g. dropping an `--active`
  selection highlight), leaves the classes it no longer emits as dead CSS - the
  removal sweep must reach `.css`, not stop at TS/HTML. After changing what a
  render emits, grep the stylesheet for the old classes and delete the orphans in
  the same diff (keep any still used by a sibling view). Related: when you change
  an element's TAG (button -> anchor), re-check the shared class's CSS for
  tag-default assumptions (anchor underline/color). 20260721-112434,
  20260721-234621, 20260722-104043. Promotion candidate: a check that greps for
  classes emitted-but-unstyled or styled-but-unemitted.
- `probe-the-stateful-path-not-the-one-shot` (x1): when an external tool "works
  standalone but fails inside the app", reproduce the app's STATEFUL invocation
  (session resume, continuation, cached state), not just the one-shot call. A
  claude agent failed with `error_during_execution` while a plain `claude -p`
  worked; the difference was `--resume <id>` on a session claude could not find
  (a stale cross-backend id after a backend switch). Three probes (plain turn,
  same-backend resume, unknown-uuid resume) isolated it fast; the "invalid model"
  theory was a red herring (the backend never passed --model). Corollary: don't
  DEVNULL a subprocess's stderr when its turn can fail - that message is the
  diagnosis (tee to a debug log instead). 20260721-152034.
- `probe-runtime-on-target-host-early` (x3) -> spike/plan skill: run the external
  tool on the real host before committing a design around it - a reasoned verdict
  about a dependency's behavior/capability is a hypothesis until run live.
  (1) 20260719-164418: one live `codex exec` reframed a whole task; the spike's
  SDK pick was right on capability, wrong on NixOS installability. (2)
  20260720-144530: make the tool emit its own wire contract (`codex app-server
  generate-ts`, `codex exec --help`) before a cross-cutting signature change; a
  model capability (see an image) is only proven by a live round-trip, never unit
  tests. (3) 20260720-221935: the spike generalized "the agent runs `/flow`", but
  a live probe showed codex is ALREADY agentic and `/flow` is a Claude-Code-only
  skill - the cross-tool generalization was wrong until probed. Proposed
  promotion: a spike-skill line "probe a dependency's real behavior/capability
  before generalizing a design across tools; a cross-tool assumption is a
  hypothesis until a live run confirms it."
