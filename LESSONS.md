# Lessons

The compressed memory of mistakes this repo has already paid for. One or two
lines per lesson; `/compound` appends here after a task's retro. Grep this for
your area before starting work. At 3+ occurrences a lesson moves to the
Pending promotions section at the bottom; the user decides whether it gets
promoted into AGENTS.md, a skill, or the tooling itself.

## Build / environment

- `edit-from-the-worktree-path-not-the-planning-read` (x1): a file Read at its
  MAIN-checkout path during planning, then Edited in the work phase, lands the edit
  in the main checkout instead of the sprout worktree (the Edit reuses the stale
  path by reflex). Caught by `git status` on the main tree and reverted, but a redo.
  After `sprout new`, re-Read the file from the worktree path before the first Edit.
  20260723-001251.
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
- `absence-grep-must-not-be-extension-scoped` (x1) -> work skill removal/doc-sweep:
  an absence-proving sweep narrowed by `--include=*.py --include=*.md ...` globs
  silently skips extensionless/dot config files (`.env.example`, `Dockerfile`,
  `Makefile`), so a stale reference survives the "one-pass grep". Scope the sweep by
  PATH (`--exclude-dir=tasks --exclude-dir=node_modules`), not by extension; a review
  caught a stale `["tatr_new"]` in `.env.example` this way. 20260722-222729.
- `scope-absence-greps-to-the-diff-not-the-file` (x1) -> plan skill DoD greps
  (sibling of `absence-grep-must-not-be-extension-scoped`): an absence-proving DoD
  grep ("no new non-ASCII", "no stale symbol") run over a WHOLE file self-matches
  pre-existing content the diff never touched, so the cmd reads red while the
  intent holds. Scope it to the diff: `git diff <base>... -- <path> | grep -nP
  ...`. A "no new non-ASCII" DoD hit two pre-existing glyphs (arrow, middot) this
  way. 20260723-225621.
- `format-before-the-check-gate` (x2): a combined `fmt --check && lint && test`
  suite aborts at the formatter step, so a stray unformatted line wastes the whole
  run before mypy/pytest execute. Run the WRITING formatter (`ruff format` /
  `prettier --write`) before invoking the check gate, not after it complains. Seen
  on a frontend (prettier, 20260719-210723) and a backend (ruff, 20260719-212203)
  task; at x3 promote to a pre-commit hook or AGENTS.md. (Reviewed 2026-07-20,
  task 20260720-220116: still x2, remains a watch - promote when it recurs.)
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

- `symlink-node_modules-into-fresh-worktrees` (x2, GUARDED 2026-07-20 ->
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
  landing.
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
- `run-repo-checks-inside-nix-develop` (x1): ruff/mypy/pytest are NOT on the bare
  PATH - they live only in the flake dev shell. Invoke every check as
  `nix develop -c bash -c '...'`, or the first `ruff`/`mypy` call dies "command
  not found" and wastes a turn. 20260723-153609.
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
- `sprout-worktree-needs-npm-ci-for-the-web-suite` (x1): a fresh sprout worktree has
  NO `web/node_modules` (the python venv is flake-provided, the node deps are not),
  so `npm run test` / `npm run ci` die "vitest: command not found" until you run
  `npm ci` in `web/` first. Do it once per worktree before touching the frontend.
  20260723-193216.

## Testing

- `isolate-state_dir-in-tests-that-assert-config` (x1, conftest-autouse-fixture
  candidate if it recurs): a test that constructs `Settings()` without
  `state_dir` and asserts on `/api/agent/config` (backend/tools_enabled/
  mcp_servers/disabled_tools) or `/api/agent/profiles` reads the REAL
  `~/.local/state/scufris` override store, which silently wins over the
  constructor arg - green on CI, red on a dev box whose override disagrees. Pass
  `state_dir=tmp_path` (or a helper that does). This was the root of the
  `check-the-base-suite-before-you-start` red. 20260723-233337.
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
- `commit-before-sabotage-or-the-restore-eats-the-fix` (x1) -> work skill A/B rule
  (already prose there; recurred anyway): sabotage-testing a fix by mutating a file
  then `git checkout -- <file>` to restore RESTORES TO HEAD, so if the fix itself is
  not yet committed the checkout silently reverts it - and a later `git add -A`
  re-stages the reverted file, landing a broken tree (here app.py called a
  `mark_finished(backend=...)` whose param had been reverted out of agent_store.py;
  the persist callback raised, sessions never persisted). Caught only by the
  full-suite-on-master gate at flow Finish. COMMIT the fix before any sabotage; or
  stash/restore the sabotage hunk alone, never `checkout --` a file holding
  uncommitted work. 20260723-001251.
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
- `os-environ-setdefault-in-test-leaks-past-monkeypatch` (x1): a test of a
  function that MUTATES `os.environ` directly (here `_ensure_api_base`'s
  `setdefault`) cannot lean on monkeypatch to clean up - monkeypatch does not track
  the raw write, so its teardown restore of a LATER `setenv` reverts to the LEAKED
  value, not to absent. Symptom: 19 unrelated respx tests (which assumed the
  default base) reddened. Snapshot the key and restore it in a `finally`.
  20260723-141026.
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

## Backend

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

## Frontend (web/)

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
- `frontend-verify-needs-e2e-serve` (x1): a green webpack build proves
  compilation, not wiring - serve the bundle through the backend and curl `/` +
  `/api/*` to prove the slice runs. No headless browser here, so visual render is
  user-eyeballed. 20260719-154539.
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
- `capture-real-cli-output-for-parser-tests` (x1): when parsing a CLI's output,
  run it once and pin a REAL captured line as the test fixture (nvidia-smi CSV,
  incl. `[N/A]`), so the parser is written against reality. 20260719-182846.
- `psutil-process-iter-caches-cpu-percent` (x1): `psutil.process_iter` reuses
  Process objects internally, so `cpu_percent` is a real delta across calls with
  no per-pid cache of your own - prime it once (iterate at startup) and read per
  sample. 20260719-182901.

## Agent / Codex

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
  (20260723-153609) and the orchestrator's comms poll (20260723-153615) - both
  needed the instruction on the turn prompt, not the tool description.
- `orchestrator-steering-is-one-block-two-clauses` (x1): the orchestrator's
  `STEERING_PREAMBLE` must stay a SINGLE `[scufris-tools]...[/scufris-tools]`
  block, because `strip_steering` removes only the first leading block
  (regex `count=1`) - a second sentinel-wrapped block would survive uncleaned in
  titles/transcripts. Add new orchestrator guidance as another CLAUSE inside the
  one block (host-tools clause + comms clause composed with `\n`), never as a
  second block. 20260723-153615.
- `ground-steering-text-in-the-real-tool-signatures` (x1): before writing
  turn-prompt steering that tells the model to call a tool, read that tool's
  actual name and signature in `mcp_server.py` and match them verbatim
  (`message_agent(agent_id, message)`, `acknowledge(agent_id)`,
  `pending_agents()`). A typo'd name or wrong arg steers the model to a call that
  cannot succeed - worse than no steering. 20260723-153615.
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

## Pending promotions (3+ occurrences, user decides)

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
