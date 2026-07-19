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
