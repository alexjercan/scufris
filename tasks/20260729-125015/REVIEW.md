# Review: gate the dashboard behind an authenticated session

Task: 20260729-125015. Branch: `feature/dashboard-auth`.

## Round 1

Out-of-context adversarial review of the full `git diff master`, with the
reviewer running the real check suite and writing throwaway probe scripts
against a live uvicorn to attempt bypasses.

- VERDICT: REQUEST_CHANGES

### Findings

1. **MAJOR - unauthenticated 500 from a non-ASCII `Authorization` header.**
   `scufris/auth.py::token_matches`. `hmac.compare_digest` raises `TypeError`
   when a `str` argument contains a byte > 0x7F, and Starlette decodes headers
   as latin-1, so the raw byte reaches it. Demonstrated over a real socket:
   `Authorization: Bearer \xff\xff` -> HTTP 500 plus a traceback in the log. Any
   device on the LAN, with no credential, can make the enforcement point throw
   and flood `journalctl`. Reachable on the CSRF header too, with a session.
   Not a bypass (the exception denies), but an unauthenticated crash inside the
   one function that must never be surprising.

2. **MAJOR - the machine token is ambient in every child process, defeating its
   own stated scoping.** `scufris/app.py` exported the token to `os.environ`,
   and `agent._codex_env` returns `dict(os.environ)` (the claude launch inherits
   it implicitly). So the agent CLI - and therefore every shell command the
   model runs, every sub-agent, and the den server regardless of its permission
   mode - held the operator's full-privilege API credential one `env` away. The
   code's own comment and `test_agent_env_carries_the_machine_token` claim the
   den server does not get the token; that assertion checked the declared MCP
   env dict, not the process environment, so it was vacuous. This epic is about
   to hang `nixos-rebuild switch` off that same API.

3. **MINOR - `os.environ[API_TOKEN_ENV]` clobbers across apps in one process.**
   Creating a second app overwrote the variable, so the first app then answered
   401 to the token in the environment. Test-only today, but a latent footgun.

4. **MINOR - `LoginThrottle` does not match its own docstring or DECISION.md**
   ("a global backstop", "a growing delay"): it is per-source only, fixed
   window, no delay, no global counter. Also `self._failures[source]` is never
   removed, so distinct sources accumulate forever - unbounded memory from
   unauthenticated requests, trivially driven from an IPv6 /64 which also evades
   the per-source limit.

5. **MINOR - `/api/auth/login` is exempt from the origin check**, so any page the
   operator visits can fire cross-origin logins at the LAN address and burn the
   lockout window, locking the legitimate operator out.

6. **MINOR - the session store never prunes.** A record is only expired when its
   id is presented, so an abandoned session persists past its cap forever and
   the whole file is rewritten on every request.

7. **MINOR - `result` and `result-1` (nix build-output symlinks into
   `/nix/store`) were committed.**

8. **NIT - the "one frontend seam" guard is narrower than its claim**
   (`web/src/auth-fetch.test.ts`): the regex only matches `fetch(` and the
   directory walk is non-recursive, while `chat-stream.ts` opens an
   `EventSource` outside the seam. Harmless today (same-origin `EventSource`
   sends cookies; a 401 closes it rather than looping) but the comment
   overstates the invariant.

9. **NIT - the DoD sweep skips non-`APIRoute` routes**, so `/openapi.json`,
   `/docs`, `/redoc` and the static `Mount` are outside the alarm. The reviewer
   verified all four are gated today; a future `add_route`/`Mount` would slip
   past.

10. **NIT - `scufris/enums.py` is still `ruff format` red** on a pre-existing
    line, in a file this branch touched.

### Suspicions tested and found NOT to be issues

Recorded because their absence is the substance of the review:

- **Path-based bypass**: `/login/../agent.js`, `/login/%2e%2e/agent.js`,
  `//api/stats`, `/api/stats/`, `/API/stats`, `/api%2fstats`, `/./api/stats`,
  `/api/auth/session;/../stats`, an absolute-URI request target - all 401. The
  allowlist and the router read the same `scope["path"]`, so there is no
  decode/route mismatch.
- **Static mount, docs, HEAD/OPTIONS**: `/`, `/index.html`, `/agent.js`,
  `/openapi.json`, `/docs`, `/redoc`, and `HEAD`/`OPTIONS` on an API path are
  all gated.
- **Middleware ordering**: confirmed from a live traceback as
  `log_requests -> enforce_auth -> route`, so denials are logged and routes
  registered after the decorator are still gated.
- **`same_origin` spoofing**: `http://evil.com@testserver`,
  `http://testserver@evil.com`, `http://testserver:80` vs host `testserver`,
  `http://testserver.evil.com`, `Referer: http://testserverX/`, `Origin: null`,
  and a missing Host all return False. It is netloc equality, not a prefix
  check.
- **CSRF binding**: the token comes from the server-side session record, not the
  cookie. All four unsafe methods are covered, and all 37 GET-only routes were
  enumerated as read-only.
- **Session fixation**: login revokes the presented cookie and mints a new id.
- **Forged cookies**: empty, 10 KB, and path-traversal values all 401 without
  crashing.
- **`is_loopback_host` gaps**: `0.0.0.0`, `::`, `""`, `127.0.0.2` and
  loopback-resolving hostnames are all treated as NON-loopback, i.e.
  auth-required. Every gap errs closed.
- **Blocking `_flush` on the event loop**: 0.065 ms per session read with 50
  sessions; irrelevant at single-operator scale.
- **Login timing oracle**: the fast "no credential configured" path is
  unreachable whenever anything is actually protected, because
  `validate_auth_config` guarantees a hash exists.
- **Open redirect**: `//evil`, `/\evil`, `https://evil`, `javascript:` all
  collapse to `/` on both the server and the client; nothing off-origin found.
- **`hash_password`/`verify_password`**: the `maxmem` computation correctly
  covers scrypt's `128*n*r`; every malformed hash returns False without raising.
- **Fail-closed startup**: all three refusal shapes are covered by real
  `pytest.raises`, and the CLI reports one line rather than a traceback.
- **Existing behavior**: full suite green, SSE streams under auth, the Telegram
  bridge starts and drives a real turn, and the MCP tools reach the API over a
  real socket with the bearer token and are refused without it.
- **DoD test quality**: the route sweep genuinely enumerates `app.routes` (61
  routes), has non-vacuity guards, and asserts both directions. The only
  genuinely vacuous assertion found was the den-token one in finding 2.

## Round 1 responses

All ten addressed.

1. **Fixed.** `token_matches` and the CSRF comparison now encode with
   `surrogateescape` before `compare_digest`, so any byte sequence compares
   false instead of raising. Pinned by `test_non_ascii_credentials_are_refused_not_crashed`,
   which drives the raw bytes over a REAL socket (a TestClient refuses to send
   them, so a TestClient-only test would have proven nothing).
2. **Fixed.** The token no longer touches `os.environ`. It lives on the app's
   own `Settings` instance (`auth_api_token`, runtime-minted, never configured,
   never persisted - it is not in `WRITABLE_KEYS`), `agent.scufris_mcp_servers`
   reads it from there, and the in-process tool console passes it through a
   `ContextVar`. `_codex_env` additionally strips the variable as
   belt-and-braces. The den assertion is now real:
   `test_agent_cli_env_does_not_carry_the_machine_token` asserts against the
   env actually handed to the CLI, with the variable deliberately seeded in
   `os.environ` first so the test would fail if inheritance leaked it.
3. **Fixed** by 2 - each app carries its own token on its own Settings, so two
   apps in one process no longer clobber each other. Pinned by
   `test_two_apps_do_not_clobber_each_others_machine_token`.
4. **Fixed.** The throttle now prunes empty and aged source entries and enforces
   a global failure ceiling as well as the per-source one, so an IPv6 /64 sweep
   is bounded in both memory and attempts. The docstring and DECISION.md now
   describe what actually ships (a fixed window and a lockout, NOT a growing
   delay).
5. **Fixed.** `POST /api/auth/login` is now origin-checked like any other
   state-changing request; a cross-origin attempt is refused before it can count
   toward the lockout.
6. **Fixed.** `SessionStore` sweeps expired records on load and on create.
7. **Fixed.** `result`/`result-1` removed from the index and `result*` added to
   `.gitignore`.
8. **Fixed.** The guard now recurses into subdirectories and covers
   `EventSource` and `XMLHttpRequest`, with `chat-stream.ts` explicitly
   allowlisted for its `EventSource` and the reason recorded.
9. **Fixed.** The sweep now covers every `starlette.routing.Route` as well, and
   asserts the public static paths explicitly.
10. **Fixed.** `scufris/enums.py` formatted.

## Round 2

Re-verified after the fixes: each of the six behavioral fixes was sabotaged in
turn and the test named for it went red, so none of them is a test that merely
watched itself pass. Full gate re-run green (`nix flake check`,
`nix build .#scufris .#web`, `cd web && npm run ci`, `examples/auth_session.py`).

- VERDICT: APPROVE
