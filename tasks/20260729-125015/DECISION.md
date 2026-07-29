# Decision: how the dashboard authenticates

- STATUS: ACCEPTED
- DATE: 2026-07-29
- TASK: 20260729-125015
- EPIC: 20260729-124655

## Context

Scufris is deployed from `~/personal/nix.dotfiles` as a home-manager systemd
USER service running as the operator. It binds `0.0.0.0:8000` and the host
firewall accepts `192.168.0.0/24 -> 8000`. There is no HTTP authentication: any
device on the LAN can create agents, run turns, drive the Telegram-connected
orchestrator, and mutate settings. The host operator epic puts service control
and `nixos-rebuild switch` behind that same surface, so authentication is its
hard prerequisite.

Three facts constrain the choice:

1. There is exactly ONE user. Multi-user identity, roles, and registration are
   not requirements and would be cost with no payer.
2. A secret delivery path already exists and works: the sops dotenv
   `secrets/scufris.env` is decrypted at activation into `$XDG_RUNTIME_DIR` and
   passed to the unit as `EnvironmentFile`. Every `SCUFRIS_*` secret rides that
   one file.
3. The app calls ITSELF over HTTP. The MCP tool servers
   (`scufris/mcp_common.py::_api_call`, base injected at `scufris/agent.py`) and
   the in-process operator tool console reach `/api/*` on loopback with no
   credential. A browser-only auth scheme silently breaks `create_agent`,
   `run_agent`, `report_back`, and every project tool.

## Deployment boundary

Supported:

- **Loopback development.** Bound to `127.0.0.1`, authentication optional. This
  is what `pytest`, the examples, and the mock backend run in.
- **Authenticated LAN.** Bound to a non-loopback address on a trusted home
  network, authentication mandatory and enforced fail-closed.

NOT supported, and explicitly out of this task's threat model:

- **Public internet exposure.** There is no TLS termination here, no account
  lockout policy worth the name against a determined attacker, and no audit of
  authentication events. Put it behind a VPN or a reverse proxy with TLS.
- **A shared host.** The session store and the machine token live in files
  readable by the operator's uid. Any process running as the operator can read
  them, and that is by design - it is the same uid the service runs as.
- **Untrusted LAN** (cafe wifi, a network with devices you do not control).
  Traffic is plaintext HTTP; the password and the session cookie are visible to
  anyone on the path.

## Decision

### Identity: one operator, a password, a server-side session

A single operator authenticates with a password. The password is never stored;
a `scrypt` hash of it is.

- **Hashing: stdlib `hashlib.scrypt`.** Rejected `argon2-cffi` and `bcrypt`:
  both are new dependencies that would churn `uv.lock`, the uv2nix venv, and the
  flake for a single call site. `scrypt` is memory-hard, in the standard
  library, and needs no build. Parameters `n=2**15, r=8, p=1` with a 16-byte
  random salt, encoded as `scrypt$<n>$<r>$<p>$<salt-b64>$<hash-b64>` so the
  parameters travel with the hash and can be raised later without invalidating
  the format. Verification uses `hmac.compare_digest`.
- **Credential source: `SCUFRIS_AUTH_PASSWORD_HASH`**, added to the existing
  sops dotenv `secrets/scufris.env`. Consistent with how
  `SCUFRIS_TELEGRAM_BOT_TOKEN` is already delivered - one file, one mechanism,
  no second secret store. The hash, not the password, is what lives there.
- **Hash generation: `scufris hash-password`.** Prompts without echo, prints the
  encoded hash. The password never reaches the repository, a log, an agent
  transcript, or this decision record.

### Session: opaque id, server-side record

The cookie carries an opaque 256-bit random session id and nothing else. The
record (id, created, last-seen, expiry) lives in a JSON file under `state_dir`
at mode 0600.

Rejected: a signed/encrypted stateless cookie (JWT-shaped). It needs a signing
key to exist and be managed, and it cannot be revoked without a server-side
denylist - which is the server-side store, arrived at by a longer road. A
server-side record makes revocation and expiry trivially real, and single-user
scale means the store is a handful of rows.

Cookie properties:

- `HttpOnly` - JavaScript cannot read it, so an XSS bug cannot exfiltrate it.
- `SameSite=Lax` - a cross-site POST never carries it.
- `Path=/`.
- `Secure` when the request arrived over HTTPS, absent otherwise. Hardcoding
  `Secure` would break the supported plaintext-LAN deployment entirely; deriving
  it means a TLS-terminated deployment gets it automatically.
- Sliding idle expiry with an absolute cap. A session is renewed on use until
  the absolute cap, then it dies regardless of activity.

Revocation: `POST /api/auth/logout` deletes the record. Deleting the store file
revokes everything. A new login **rotates the session id** (the old record is
dropped), which is what closes session fixation.

### Machine identity: a per-process bearer token

At startup the app generates a random `SCUFRIS_API_TOKEN` and injects it into
the MCP subprocess environment next to `SCUFRIS_API_BASE`, and into `os.environ`
for the in-process console path. `_api_call` sends it as
`Authorization: Bearer <token>`.

- Generated per process, never persisted, never configured. It dies with the
  server, so there is no long-lived credential at rest and no rotation story to
  get wrong.
- It is a machine credential, not an operator one: it authenticates the app's
  own tool subprocesses, which already run with the operator's full privileges.
  It grants nothing they did not already have.

Rejected: exempting loopback from authentication. That would make every process
on the machine - and anything that can be tricked into issuing a loopback
request - a trusted caller, and it fails open exactly when the deployment is
most exposed.

### Enforcement: one middleware, deny by default

A single `@app.middleware("http")`, registered so it runs OUTSIDE the request
logger. It denies every request that is neither an authenticated session nor a
valid bearer token, except a small explicit public allowlist:

- `POST /api/auth/login`, `GET /api/auth/session` (so the frontend can ask
  "am I logged in" without a 401 loop),
- the login page and the static assets it needs.

Deny by default is the point: a route added tomorrow is protected by existing,
not by remembering to decorate it. Rejected per-route `Depends(...)` across 58
routes for exactly that reason - the task's own note says so, and the DoD proves
it with a sweep derived from `app.routes` rather than a hand-written list, so a
new unprotected route fails the suite.

### CSRF: double submit plus origin validation

Both, not either:

- A readable (non-`HttpOnly`) `scufris_csrf` cookie, echoed by the frontend in
  an `X-Scufris-CSRF` header on every state-changing method
  (`POST`/`PUT`/`PATCH`/`DELETE`). A cross-site attacker can cause the cookie to
  be sent but cannot read it to build the header.
- `Origin`/`Referer` validation on the same methods, rejecting a cross-origin
  request outright.

`SameSite=Lax` alone was rejected as sufficient: it is a browser behavior, not a
server check, and it is exactly the kind of assumption that ages badly.

### Throttling

Failed logins are throttled per source address and globally, with a growing
delay and a lockout window. Constant-time comparison, and a uniform failure
message that does not distinguish "no credential configured" from "wrong
password".

### Development mode: `SCUFRIS_AUTH_MODE`

- `auto` (default): authentication is REQUIRED when the bind host is not
  loopback, and open when it is. This makes the deployed configuration secure
  without the operator opting in, and keeps `pytest`, the examples, and the mock
  backend free of a login dance.
- `required`: always on, including on loopback.
- `disabled`: off - and **refused at startup on a non-loopback bind**.

Fail-closed: a non-loopback bind with no `auth_password_hash` configured refuses
to start. It does not warn and serve.

## Consequences

- After this lands, the deployed service will NOT start until
  `SCUFRIS_AUTH_PASSWORD_HASH` is present in `secrets/scufris.env`. That is the
  intended behavior of a fail-closed gate, and it is a deliberate one-time
  operator action: run `scufris hash-password`, `sops secrets/scufris.env`, add
  the line, rebuild.
- Traffic stays plaintext HTTP on the LAN. The password crosses the network on
  login and the session cookie crosses it on every request. Accepted for a
  trusted home network; a public or untrusted deployment needs TLS in front and
  is declared unsupported above.
- The Telegram bridge is unaffected: it is an in-process long-poll client and
  its authentication is the chat-id allowlist. That independence is now pinned
  by a test so a later change cannot break it silently.
- This decides authentication only. The secret-reference store and output
  redaction remain in 20260729-102208.
