# Notes: gating the dashboard behind an authenticated session

Design and fix record for the shipped change. The decision itself (and the
alternatives that were rejected) is in `DECISION.md`; this is what was built,
what went wrong along the way, and what a later task needs to know.

## What changed

| File | What |
|---|---|
| `scufris/auth.py` (new) | scrypt hash/verify, revocable `SessionStore`, `LoginThrottle`, CSRF/origin helpers, the policy functions (`auth_required`, `validate_auth_config`), and the public-path allowlists. |
| `scufris/app.py` | One deny-by-default `enforce_auth` HTTP middleware, the three `/api/auth/*` routes, the machine-token mint, and `validate_auth_config` at startup. |
| `scufris/config.py` | `auth_mode`, `auth_password_hash`, session lifetime and throttle knobs. Deliberately NOT in `settings_store.WRITABLE_KEYS`. |
| `scufris/enums.py` | `AuthPolicy` (`auto`/`required`/`disabled`), distinct from the existing backend-login `AuthMode`. |
| `scufris/cli.py` | `scufris hash-password`; `serve` reports `AuthConfigError` as one line, not a traceback. |
| `scufris/agent.py`, `scufris/mcp_common.py` | The per-process `SCUFRIS_API_TOKEN` reaches the MCP servers that call the API, and `_api_call` presents it as a bearer token. |
| `web/src/common.ts` | `apiFetch`: the single frontend seam (CSRF header + 401 redirect + same-origin credentials). Every previously-bare `fetch` call site routes through it. |
| `web/src/login*.{ts,html}`, `nav.ts`, `_header.html`, `style.css`, `webpack.config.js` | The login page (its own entry, standalone), the sign-out control, and the styles. |
| `examples/auth_session.py` | Drives the whole boundary over a real uvicorn socket and prints each refusal with its reason. |

## The non-obvious part: the app authenticates to itself

The thing that would have broken silently: `scufris/mcp_common.py::_api_call`
means the MCP tool subprocesses reach `/api/*` over loopback with no credential,
and the operator tool console does the same IN THIS PROCESS. A cookie-only
scheme would have left `create_agent`, `run_agent`, `report_back` and every
project tool answering 401 with nobody to log in.

Hence the second identity: a token minted per process in `create_app`, exported
both to `os.environ` (for the in-process console) and into the MCP subprocess env
(`agent.scufris_mcp_servers`), presented as `Authorization: Bearer`.

Two properties worth preserving:

- **Loopback is not trusted.** Exempting `127.0.0.1` would have been shorter and
  would have made every process on the machine a trusted caller. The token is
  the identity; the source address is not.
- **Bearer callers skip CSRF, session callers do not.** CSRF exists because a
  browser attaches cookies ambiently. A bearer caller has no ambient credential
  to ride, and requiring a CSRF token from it would break every tool.

## Bugs and surprises found along the way

1. **The existing OpenAPI test caught the new routes as untagged.** `/api/auth/*`
   had no tag, and `test_openapi_docs_are_organized` asserts every `/api/`
   operation is tagged. Added an `auth` tag section and its `_route_tags` branch.
   A good gate: it noticed a doc surface I had not thought about.

2. **The route sweep walked into `/api/auth/logout` and logged itself out.**
   Logout is (correctly) not public, so the enumerate-all-routes sweep called it
   with a valid CSRF token halfway through and every later assertion saw 401. The
   sweep now re-logs-in after that one route. Worth remembering for any future
   enumerate-the-surface test: the surface contains session-destroying verbs.

3. **A leaked global stub in `agent-chat-view.test.ts`, surfaced by this change.**
   One test does `vi.stubGlobal("URL", {createObjectURL, revokeObjectURL})` and
   never restores it; the next describe's `afterEach(vi.unstubAllGlobals)` only
   runs AFTER its own first test. So the first test of the following block ran
   with a non-constructible global `URL`. It happened to pass before and failed
   once the fetch path went through `apiFetch`. Fixed at the source (restore
   globals in the describe that stubs them), not by weakening the victim's
   assertions - the first attempt (a `vi.waitFor`) was a workaround and was
   reverted once the real cause was found.

4. **`run_server` announced the start before validating.** It logged "starting
   scufris on 0.0.0.0:8123" and only then refused. `validate_auth_config` now
   runs before that log, so `journalctl` reads honestly.

## Verification

- `nix flake check` (ruff + mypy + pytest + records) green; `nix build .#scufris
  .#web` green (a flake check alone only evaluates packages).
- `cd web && npm run ci` green: 21 files, 204 tests.
- `examples/auth_session.py` exits 0.
- End-to-end against the REAL built bundle (`nix build .#web`), bound to
  `0.0.0.0`: refuses to start with no credential; `GET /` redirects `303` to
  `/login/?next=/`; `/login/` and `/login.js` serve; `/agent.js` is `401`; a curl
  login then reaches `/` and `/api/stats`.
- `scufris hash-password` by hand, including the mismatch and empty-password
  refusals.
- **The gate was proven to discriminate**, not merely observed green - five
  sabotages, each producing its own targeted red: removing the CSRF check
  (2 tests), inverting `auth_required` (15), removing the fail-closed raise (1),
  dropping the machine token from the MCP env (1), and leaving a whole route
  family ungated (the sweep, naming `GET /api/agents`).

## What the operator must do

The deployed service will NOT start after the flake input is bumped past v0.1.0
until `SCUFRIS_AUTH_PASSWORD_HASH` exists in `secrets/scufris.env`:

```sh
scufris hash-password         # prompts, prints the env line
sops secrets/scufris.env      # paste it in
```

`~/personal/nix.dotfiles` is committed (not pushed, not activated) with
`auth_mode = "required"` on `programs.scufris.settings` plus the instructions in
`home/alex/default.nix` and `secrets/README.md`.

## For the tasks that follow in this epic

- The enforcement point is ONE middleware. Host action routes get gated by
  existing; do not add per-route auth dependencies, and do not extend
  `PUBLIC_PATHS` without a very good reason.
- `test_authenticated_session_and_csrf_boundary` will fail the moment an
  ungated route appears. That is the intended alarm, not an obstacle.
- Approvals over Telegram (20260729-125040) cannot reuse the session model:
  Telegram authenticates by chat id, entirely outside the HTTP gate. The task's
  requirement that it enforce "the same gate" means the same server-side
  approval check, reached by a different door - not the same cookie.
- There is still no TLS. Anything that makes the dashboard worth attacking from
  outside the LAN needs that conversation first.
