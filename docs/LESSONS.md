# Lessons

The compressed memory of mistakes this repo has already paid for. One or two
lines per lesson; `/compound` appends here after a task's retro. Grep this for
your area before starting work. At 3+ recurrences a lesson is a candidate to
promote into AGENTS.md, a skill, or the tooling itself.

## Build / environment

- `dep-change-needs-nix-develop-rebuild` (x1): the active dev shell runs a fixed
  nix-store uv2nix venv, so a new dependency added with `uv add` is invisible to
  a bare `pytest`/`mypy`. Run checks via `nix develop --command ...` (or re-enter
  the shell) so the venv rebuilds from the updated `uv.lock`. 20260719-154420.
- `new-scufris-module-needs-package-init` (x1): mypy errors with "Source file
  found twice under different module names" when a `scufris/` module has no
  package `__init__.py`. `scufris/__init__.py` now exists; keep it.
  20260719-154420.

## Testing

- `type-test-fixtures-by-protocol` (x1): annotate injected test doubles by the
  protocol they satisfy (e.g. `Collector`), not the concrete fake class, so tests
  need no cross-test class import - mypy can't resolve `from .conftest import X`
  because `tests/` is not a package. 20260719-154544.

## Backend

- `web_dist-via-__file__-is-dev-only` (x1): the FastAPI `web_dist` default
  (`<repo>/web/dist` from `__file__`) works for the editable dev install but not
  a packaged wheel; bundling built assets into the nix closure is still open.
  20260719-154544.

## Frontend (web/)

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
- `escape-only-host-strings-in-element-content` (x1): when interpolating into
  innerHTML, escape only untrusted STRINGS for their context (element content
  needs `< > &`; attributes also quotes); numbers via `toFixed` are safe. Prove
  it with a jsdom test that a hostile value creates no element. 20260719-160924.
- `webpack-multipage-htmlplugin-per-page` (x1): for a multi-page frontend, use
  one `entry` + one `HtmlWebpackPlugin` (explicit `chunks`) per page + a
  `historyApiFallback` rewrite per sub-route; FastAPI `StaticFiles(html=True)`
  then serves `/` and `/<page>/` with NO backend change. 20260719-180543.

## Monitoring / collector

- `distinct-loop-vars-for-different-types` (x1): don't reuse a loop variable name
  across two loops whose elements are different nominal types (e.g. psutil
  `snetio` vs `sdiskio`) - mypy binds one type to the name and the second loop's
  attribute access fails. Name them apart. 20260719-182846.
- `capture-real-cli-output-for-parser-tests` (x1): when parsing a CLI's output,
  run it once and pin a REAL captured line as the test fixture (nvidia-smi CSV,
  incl. `[N/A]`), so the parser is written against reality. 20260719-182846.

## Agent / Codex

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
