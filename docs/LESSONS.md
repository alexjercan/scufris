# Lessons

The compressed memory of mistakes this repo has already paid for. One or two
lines per lesson; `/compound` appends here after a task's retro. Grep this for
your area before starting work. At 3+ recurrences a lesson is a candidate to
promote into AGENTS.md, a skill, or the tooling itself.

## Build / environment

- `format-before-the-check-gate` (x2): a combined `fmt --check && lint && test`
  suite aborts at the formatter step, so a stray unformatted line wastes the whole
  run before mypy/pytest execute. Run the WRITING formatter (`ruff format` /
  `prettier --write`) before invoking the check gate, not after it complains. Seen
  on a frontend (prettier, 20260719-210723) and a backend (ruff, 20260719-212203)
  task; at x3 promote to a pre-commit hook or AGENTS.md.
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

- `symlink-node_modules-into-fresh-worktrees` (x1): a sprouted worktree has no
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
- `dep-change-needs-nix-develop-rebuild` (x1): the active dev shell runs a fixed
  nix-store uv2nix venv, so a new dependency added with `uv add` is invisible to
  a bare `pytest`/`mypy`. Run checks via `nix develop --command ...` (or re-enter
  the shell) so the venv rebuilds from the updated `uv.lock`. 20260719-154420.
- `nix-devshell-import-resolves-to-cwd-source` (x1): in the nix dev shell,
  `import scufris` resolves to the CWD's `scufris/` source (shadowing the venv
  install), so any in-process smoke / `python -c` check must run from the
  BRANCH's own directory - never `os.chdir` into another checkout before
  importing, or you silently test that checkout's code. Symptom: a route/behavior
  pytest passes but a smoke reports missing (was testing master, not the branch).
  20260719-212205.
- `new-scufris-module-needs-package-init` (x1): mypy errors with "Source file
  found twice under different module names" when a `scufris/` module has no
  package `__init__.py`. `scufris/__init__.py` now exists; keep it.
  20260719-154420.

## Testing

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
- `tests-that-lean-on-a-default-break-when-it-flips` (x1): a test that asserts
  "disabled" behavior while relying on the config DEFAULT being disabled is
  really testing the default, not the behavior - flipping the default reds it.
  Set the precondition explicitly (`agent_enabled=False`) so the test states its
  own intent and survives a default change. 20260720-020402.

## Backend

- `web_dist-via-__file__-is-dev-only` (x1): the FastAPI `web_dist` default
  (`<repo>/web/dist` from `__file__`) works for the editable dev install but not
  a packaged wheel; bundling built assets into the nix closure is still open.
  20260719-154544.

## Frontend (web/)

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
  messages must survive. 20260719-224101.
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
  with a wall-clock DEADLINE, not `communicate()`; (2) yield events from an async
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
- `probe-runtime-on-target-host-early` (x1): for an external-tool integration,
  run the tool on the actual target host before committing to a client (SDK vs
  CLI). One live `codex exec` reframed a whole task; the spike's SDK pick was
  right on capability, wrong on NixOS installability. 20260719-164418.
- `codex-resume-rejects-sandbox` (x1): `codex exec resume` inherits the original
  session's sandbox and errors on a repeated `--sandbox`; pass session-scoped
  flags (`--sandbox`) only on the FIRST turn, not on resume. A fake that ignores
  unknown args won't catch it - only a live run does. 20260719-162406.
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
