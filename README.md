# Scufris

[![ci](https://github.com/alexjercan/scufris/actions/workflows/ci.yaml/badge.svg)](https://github.com/alexjercan/scufris/actions/workflows/ci.yaml)

Scufris ("Scuffed Jarvis") is a self-hosted dashboard and assistant for ONE
NixOS machine. It shows live stats about the box, answers questions about it
through LLM-backed agents, and - once you enable the privileged helper - lets an
agent PROPOSE changes to the machine that only a human can approve.

Three things it does:

- **Watches the host.** CPU, memory, disk, network, processes, temperatures,
  failed units, journals, nix store and generations. Plus a scheduled pass that
  messages you on Telegram when something needs attention.
- **Runs agents.** Project-bound assistants (codex / claude / a self-hosted
  opencode + llama.cpp), an orchestrator that delegates to them, and one
  reserved host agent bound to the machine instead of a project.
- **Changes the host, under approval.** Every mutating action goes through
  `propose -> preview -> approve -> apply -> audit -> roll back`, executed by a
  separate root helper that only understands a closed set of typed verbs.

Not supported: public internet exposure, an untrusted network, or a shared host.
Traffic is plain HTTP - put a VPN or a TLS-terminating proxy in front of it if
the dashboard has to leave a trusted LAN.

## Where to read more

| Document | What is in it |
|---|---|
| [`scufris/README.md`](scufris/README.md) | The architecture: processes, trust boundaries, the approval contract, which agent holds which tools, the HTTP surface |
| [`scufris/host/README.md`](scufris/host/README.md) | The read-only inspection package: what it can read, and the rules it reads by |
| [`scufris/hostd/README.md`](scufris/hostd/README.md) | The root helper: how to enable it, the socket language, every verb and its arguments, the audit log |
| [`web/README.md`](web/README.md) | The dashboard frontend: pages, build, tests |
| [`.env.example`](.env.example) | Every setting, annotated, with its default |
| [`AGENTS.md`](AGENTS.md) | Working ON Scufris: commands, conventions, task workflow, security invariants |
| [`docs/RELEASING.md`](docs/RELEASING.md) | Release, retry, and yank procedure |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed per version |
| [`examples/`](examples/) | Runnable scripts that drive one component end to end |
| [`tasks/`](tasks/) | The design record: one folder per task, with the `DECISION.md` behind each fork. [`tasks/20260729-124655/ARCHITECTURE.md`](tasks/20260729-124655/ARCHITECTURE.md) is the map of the host-operator work as it was built |

# How to set up Scufris

Four steps, and only the first two are mandatory: run it, deploy it, then enable
the optional features one at a time. Every feature is off until you configure
it, and none of them half-work.

## 1. Run it from a checkout

Python project managed with [`uv`](https://docs.astral.sh/uv/), built
reproducibly through Nix via
[`uv2nix`](https://github.com/pyproject-nix/uv2nix). Nix with flakes enabled is
the only prerequisite.

```sh
nix develop            # dev shell: interpreter, locked venv, uv, node, codex, tatr
cd web && npm ci       # frontend deps, once per checkout
npm run build          # build web/dist (the app serves it at "/")
cd ..

scufris serve          # the dashboard on http://127.0.0.1:8000
```

Or without a dev shell at all:

```sh
nix run github:alexjercan/scufris   # build and run the packaged app
```

On loopback there is no login, no root helper and no bot: a bare `scufris serve`
is a read-only dashboard plus, if a backend CLI is authenticated, the chat.

The CLI has five subcommands:

| Command | What it does |
|---|---|
| `scufris serve` | run the dashboard web server |
| `scufris chat "<prompt>"` | one orchestrator turn, printed to the terminal |
| `scufris login` | authenticate the codex backend (Sign in with ChatGPT) |
| `scufris hash-password` | prompt for a password, print `SCUFRIS_AUTH_PASSWORD_HASH=...` |
| `scufris mcp-server` | run the MCP tool server over stdio (the app spawns this itself) |
| `scufris --version` | the installed version |

## 2. Deploy it as a service

The flake exports three modules. Pin it to a released tag rather than tracking
`master`:

```nix
{
  inputs.scufris.url = "github:alexjercan/scufris/v0.1.0";
}
```

| Output | What it is |
|---|---|
| `homeManagerModules.scufris` | `programs.scufris` - a `systemd.user` unit running as you |
| `nixosModules.scufris` | `services.scufris` - a NixOS system unit with `DynamicUser` |
| `nixosModules.scufris-hostd` | `services.scufris-hostd` - the ROOT helper. Separate on purpose (see step 4) |
| `packages.scufris`, `packages.scufris-web` | the app and the built dashboard assets |

Every output is named with a `scufris` prefix so it cannot be confused with
another flake's when several are imported side by side. `default` is kept as the
conventional alias for the app package, the app itself and the two service
modules (`packages.default`, `apps.default`, `nixosModules.default`,
`homeManagerModules.default`), so `nix run .` and a plain `.default` import keep
working.

A home-manager deployment, which is what this host runs:

```nix
programs.scufris = {
  enable = true;
  # Anything from scufris/config.py, lowercased. Each key `foo` becomes
  # SCUFRIS_FOO.
  settings = {
    host = "0.0.0.0";
    port = 8000;
    agent_backend = "codex";
    host_config_repo = "/home/alex/personal/nix.dotfiles";
  };
  # Secrets go HERE, never in `settings` (which lands in the nix store).
  environmentFile = config.sops.secrets."scufris-env".path;
  # The agent shells out to these; they are operator-installed binaries,
  # not Python dependencies.
  path = [pkgs.codex pkgs.claude-code pkgs.git];
};
```

Both module shapes take the same options: `settings` (the flat env surface),
`environmentFile` (the secrets), `path` (binaries the agent needs), `stateDir`,
and `hostTools`. `webPackage` is wired to `SCUFRIS_WEB_DIST` for you, because
the Python wheel deliberately does not ship `web/dist`.

`hostTools` defaults to true and puts the host-inspection toolchain (`systemd`,
`nix`, `nixos-rebuild`, `iproute2` - the commands `scufris/host` shells out to)
on the service PATH. Leave it alone unless you intend to supply those commands
yourself through `path`: with them missing, the unit still starts and serves,
but every host page reports "not installed on this host". They are pinned into
the unit's closure rather than picked up from an ambient profile, which is what
makes the user-service and system-service deployments behave identically -
a `systemd --user` unit's profile is `~/.nix-profile/bin`, and on NixOS no
system tool is ever installed there.

## 3. Configure it

Settings come from the environment with a `SCUFRIS_` prefix, or a `.env` file in
the working directory. [`.env.example`](.env.example) is the annotated full
list; below is what actually decides how the deployment behaves.

Most fields are also editable at runtime from the `/settings/` page and persist
across restarts under `SCUFRIS_STATE_DIR`. Env values are the first-boot seed;
persisted overrides layer on top. Two deliberate exceptions:
`SCUFRIS_AUTH_MODE` and the secrets are env-only - a security posture must not
be changeable through the surface it protects.

### Server

| Variable | Default | What it decides |
|---|---|---|
| `SCUFRIS_HOST` | `127.0.0.1` | The bind address, and therefore whether authentication is mandatory |
| `SCUFRIS_PORT` | `8000` | The port |
| `SCUFRIS_WEB_DIST` | `<repo>/web/dist` | The built frontend served at `/` |
| `SCUFRIS_STATE_DIR` | `~/.local/state/scufris` | Persisted settings, sessions, agent records, and the state database |
| `SCUFRIS_SETTINGS_WRITABLE` | `1` | `0` makes a read-only server: the writable-config endpoints answer 403 |
| `SCUFRIS_LOG_LEVEL` | `INFO` | Verbosity (`scufris --debug` forces DEBUG) |
| `SCUFRIS_POLL_SECONDS` | `2.0` | How often the dashboard polls `/api/stats` |
| `SCUFRIS_HOST_OVERVIEW_SECONDS` | `30.0` | How long a host overview snapshot is cached (it shells out, so it polls far slower) |
| `SCUFRIS_HOST_CONFIG_REPO` | `~/personal/nix.dotfiles` | This host's NixOS flake. Read only - nothing here writes to it |
| `SCUFRIS_PROJECT_BASE_DIRS` | `~/personal:~/personal/_tests:~/work:~/third-party` | Directories the Projects page scans one level deep |

### Authentication (required off loopback)

| Variable | Default | What it decides |
|---|---|---|
| `SCUFRIS_AUTH_MODE` | `auto` | `auto` = required on a non-loopback bind, open on loopback. `required` = always on. `disabled` = off, and REFUSED on a non-loopback bind |
| `SCUFRIS_AUTH_PASSWORD_HASH` | unset | The operator's password hash. Required whenever authentication is on |
| `SCUFRIS_AUTH_SESSION_IDLE_SECONDS` | `43200` | A session unused this long stops working |
| `SCUFRIS_AUTH_SESSION_MAX_SECONDS` | `604800` | Absolute lifetime from login, however actively used |
| `SCUFRIS_AUTH_LOGIN_MAX_FAILURES` | `10` | Failed logins from one source before it locks out |
| `SCUFRIS_AUTH_LOGIN_WINDOW_SECONDS` | `900` | The window that count is measured over |

### Privileged host actions

| Variable | Default | What it decides |
|---|---|---|
| `SCUFRIS_HOSTD_SECRET` | unset | The shared secret on every frame to the root helper. Unset = NO privileged surface at all |
| `SCUFRIS_HOSTD_SOCKET` | `/run/scufris-hostd/hostd.sock` | Where the helper listens. Must match `services.scufris-hostd.socketPath` |
| `SCUFRIS_HOST_QUEUE_REFRESH_SECONDS` | `3.0` | How often the approval queue reconciles with the helper |
| `SCUFRIS_HOST_CONFIG_ATTR` | the hostname | Which `nixosConfigurations.<attr>` of the config repo this machine is |
| `SCUFRIS_HOST_CONFIG_BUILD_TIMEOUT` | `7200.0` | Wall clock for one system build (a kernel rebuild is genuinely that slow) |
| `SCUFRIS_API_BASE` | `http://127.0.0.1:$PORT` | The URL the app's own MCP tools call back on. Set only when the dashboard is reachable elsewhere than it binds |

### Agents

| Variable | Default | What it decides |
|---|---|---|
| `SCUFRIS_AGENT_ENABLED` | `1` | `0` serves the dashboard with no chat at all |
| `SCUFRIS_AGENT_BACKEND` | `codex` | The orchestrator's backend: `codex`, `claude`, `opencode`, or `mock` |
| `SCUFRIS_AGENT_MODEL` | `gpt-5.5` | Default model for codex agents |
| `SCUFRIS_CLAUDE_MODEL` | `claude-opus-4-8` | Default model for claude agents |
| `SCUFRIS_AGENT_PERMISSION_MODE` | `auto` | The orchestrator's write posture: `manual` (read-only), `edit`, `auto` (edit + run commands). Project agents carry their own, defaulting to `manual` |
| `SCUFRIS_AUTO_WAKE` | `0` | Grant the orchestrator a turn when a sub-agent needs a decision. Off by default: a wake runs it unattended |
| `SCUFRIS_DISABLED_TOOLS` | `[]` | Built-in tools to drop, as JSON. A dropped tool cannot be called, not merely hidden |
| `SCUFRIS_AGENT_TIMEOUT_SECONDS` | `120.0` | Idle guard between backend output lines (not a cap on the turn) |
| `SCUFRIS_AGENT_MAX_CONCURRENT` | `4` | Concurrent supervised runs; the rest queue |
| `SCUFRIS_DEN_PATH` | unset | The-den journal directory for the orchestrator's `journal_*` tools. Unset leaves them inert |
| `SCUFRIS_ENABLE_MOCK_BACKEND` | `0` | Offer the offline `mock` backend in the create picker (dev/demo) |
| `SCUFRIS_CODEX_BIN`, `SCUFRIS_CLAUDE_BIN` | PATH | Where the backend CLIs are |
| `SCUFRIS_CODEX_HOME`, `SCUFRIS_CLAUDE_HOME` | `~/.codex`, `~/.claude` | Where they keep auth and sessions |
| `SCUFRIS_AGENT_AUTH_MODE` | `chatgpt` | `chatgpt` (subscription) or `api_key` for codex; the key is `SCUFRIS_OPENAI_API_KEY` |
| `SCUFRIS_OPENCODE_URL` | `http://127.0.0.1:4096` | The `opencode serve` daemon the opencode backend drives |
| `SCUFRIS_OPENCODE_MODEL`, `_PROVIDER`, `_PASSWORD` | `gemma-4-26B-A4B-it`, `llamacpp`, unset | That daemon's model, provider id, and HTTP Basic password |

### Telegram

| Variable | Default | What it decides |
|---|---|---|
| `SCUFRIS_TELEGRAM_BOT_TOKEN` | unset | A token from @BotFather starts an in-process long-poll bot. Unset = no bot |
| `SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS` | empty | The chat ids allowed to drive it. This allowlist IS the authentication; empty ignores everyone |
| `SCUFRIS_TELEGRAM_STREAM` | `true` | Stream the turn live (thinking message + one widget per tool call) rather than one final message |

### Scheduled checks and the digest

| Variable | Default | What it decides |
|---|---|---|
| `SCUFRIS_HOST_CHECKS_ENABLED` | `1` | The master switch for the scheduler |
| `SCUFRIS_HOST_WATCH_ENABLED` | `1` | The frequent pass, which delivers ONLY on a warn/crit or a recovery |
| `SCUFRIS_HOST_WATCH_INTERVAL_SECONDS` | `900.0` | How quickly bad news arrives (not how often you are messaged) |
| `SCUFRIS_HOST_DIGEST_ENABLED` | `1` | The daily heartbeat, which always delivers even when it is one line |
| `SCUFRIS_HOST_DIGEST_AT` | `08:00` | Local time it goes out |
| `SCUFRIS_HOST_DIGEST_MUTED_UNTIL` | `0.0` | Unix time until which nothing is DELIVERED. Runs still happen and stay readable on `/host/` |
| `SCUFRIS_CHECK_DISK_WARN_PERCENT` / `_CRIT_PERCENT` | `85.0` / `95.0` | Disk pressure thresholds |
| `SCUFRIS_CHECK_TEMP_WARN_CELSIUS` | `85.0` | Thermal threshold |
| `SCUFRIS_CHECK_STORE_DEAD_PATHS` | `5000` | Dead store paths at which the store check speaks up (and only when its filesystem is also tight) |
| `SCUFRIS_CHECK_FLAKE_AGE_DAYS` | `30` | When the config flake's pins count as stale |
| `SCUFRIS_CHECK_ESCALATE_GC` | `0` | Whether a breached store check may PROPOSE a collection. Off until you trust the digests |

## 4. Enable the optional features

### Authentication

Any non-loopback bind requires an operator session, and the server **refuses to
start** without a credential configured. It does not warn and serve.

```sh
scufris hash-password       # prompts, prints SCUFRIS_AUTH_PASSWORD_HASH=...
```

Put that line where your secrets live - for a nix.dotfiles deployment, `sops
secrets/scufris.env`, the same dotenv the unit already takes as an
`EnvironmentFile` - and restart. The password itself is never stored; what is
kept is a `scrypt` hash.

The session is an opaque id in an `HttpOnly`, `SameSite=Lax` cookie backed by a
revocable server-side record, so signing out (or deleting the session file under
`SCUFRIS_STATE_DIR`) genuinely ends it. State-changing requests additionally
need a CSRF token and a same-origin `Origin`/`Referer`. The app's own MCP tool
subprocesses use a per-process bearer token instead of a cookie, minted at
startup and never persisted - and that token is refused outright on the decision
endpoints, whatever the bind address.

`examples/auth_session.py` drives the whole boundary over a real socket and
prints each refusal with its reason.

### Agents

Agents are on by default but do nothing until a backend CLI is authenticated,
because they drive an LLM CLI under YOUR subscription (a personal-use path, not
a shared or commercial one).

```sh
nix develop                    # provides `codex` and `scufris`
scufris login                  # Sign in with ChatGPT (opens a browser)
scufris chat "what is using my memory?"
```

For claude, run `claude` once to authenticate it. For a self-hosted model, start
the daemon and point scufris at it:

```sh
OPENCODE_CONFIG=examples/opencode/opencode.json opencode serve --port 4096
```

For offline development with no CLI at all, set
`SCUFRIS_ENABLE_MOCK_BACKEND=1` and create a `mock` agent.

Then manage agents from `/agents/`: each picks a backend, a model, and a
permission mode (`manual` / `edit` / `auto`), and keeps one resumable session.
`/agents/host` is the reserved host agent - bound to the machine rather than a
project, read-only on files, and the only agent that can propose host changes.

### Privileged host actions (the root helper)

This is the feature that lets Scufris change the machine, so enabling it is a
deliberate, separate act in your NixOS configuration:

```nix
services.scufris-hostd = {
  enable = true;
  group = "scufris";                               # a DEDICATED group, not `users`
  secretFile = config.sops.secrets."scufris-hostd-secret".path;
};
```

The same secret must reach the app as `SCUFRIS_HOSTD_SECRET`. Generate one with
`openssl rand -base64 32`.

Three consequences worth knowing before you turn it on:

- Without the secret the helper refuses to start and the app answers every
  mutating host endpoint with "not configured". There is no half-enabled state.
- With it, an operator password becomes mandatory **even on loopback**: the app
  refuses to start with host agency and nobody to be the human who approves.
- The helper writes its own audit log at `/var/log/scufris-hostd/audit.jsonl`,
  root-owned and append-only. Nothing outside the helper can delete an entry,
  and no protocol verb can.

Every option, every verb, and the socket language are in
[`scufris/hostd/README.md`](scufris/hostd/README.md). What each surface then
looks like - the `/host/` approval queue, the Telegram buttons, which agent
holds which tool - is in [`scufris/README.md`](scufris/README.md).

`examples/host_action.py` prints the whole contract, including an action with no
undo and one stopped mid-apply.

### Activating a NixOS configuration change

Nothing extra to enable: this rides on the root helper. Set
`SCUFRIS_HOST_CONFIG_REPO` to your flake and `SCUFRIS_HOST_CONFIG_ATTR` if this
machine's `nixosConfigurations` attribute is not its hostname.

Editing that repository is ordinary project work - a worktree, a commit, a
review - and Scufris has no way to write to it. Activating what was committed is
a host action: post a ref, and Scufris resolves it to a commit, builds
`nixosConfigurations.<attr>` from THAT commit as you (never as root), and
proposes activating the exact store path it built with a `nix store
diff-closures` preview. `examples/nixos_change.py` drives the whole flow.

### Telegram

```
SCUFRIS_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS=123456789
```

Both belong in the secret dotenv. The bot long-polls outward, so it opens no
inbound port and needs no public webhook - and the allowlist is the whole
authentication story, so an empty one ignores everyone.

The chat talks to the same orchestrator as the landing page, receives the
scheduled digests, and can **decide host approvals**: a new proposal announces
itself with the same text the dashboard shows plus inline Approve/Deny buttons.
That is a real grant - whoever holds an allowlisted chat is the operator there
and can approve a root action with no password. A one-way action's first tap only
arms it; approving takes a second, differently-worded tap. `/approvals` lists
what is waiting and `/deny <id> <reason>` is the typed form.
`examples/telegram_approval.py` prints exactly what the phone shows.

### Scheduled checks and the digest

On by default, and they need Telegram configured to reach you. `watch` (every 15
minutes) messages you only when a check enters a warn/crit state or recovers;
`daily` (08:00) always sends, even when it is one line - that line is the
heartbeat that makes silence from `watch` mean "nothing is wrong" rather than
"is it even running?".

The checks are code with explicit thresholds, not a model turn. Tune them from
the table above or the `/settings/` page; a mute stops the messages, not the
watching. `examples/host_digest.py` prints the digest in every state, which is
the fastest way to judge the wording before living with it.

## Verify a deployment

```sh
nix flake check                   # ruff + mypy + pytest + file-size guard + records
cd web && npm run ci              # prettier + eslint + vitest + build
nix build .#scufris .#scufris-web # what a release ships
nix build .#scufris-vm-test       # the app on a real NixOS VM (needs KVM)
nix build .#scufris-hostd-vm-test # the root helper on a real socket, real activation
```

The running instance reports what it is - `scufris --version`, the
`scufris_version` field on `/api/agent/health`, and the settings view - so you
can tell what is deployed without reading a nix store path.

## Working on Scufris

Read [`AGENTS.md`](AGENTS.md) first: build and test commands, the harness-first
testing philosophy, conventions, and how work is tracked (the `tatr` CLI, one
folder per task under `tasks/`, driven through the
`/spike -> /plan -> /work -> /review -> /compound` lifecycle). Releases are cut
by pushing a `vX.Y.Z` tag; the procedure is in
[`docs/RELEASING.md`](docs/RELEASING.md).

## License

See [`LICENSE`](LICENSE).
