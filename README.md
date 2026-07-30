# Scufris

[![ci](https://github.com/alexjercan/scufris/actions/workflows/ci.yaml/badge.svg)](https://github.com/alexjercan/scufris/actions/workflows/ci.yaml)

Scufris ("Scuffed Jarvis") is a self-hosted dashboard for a single machine: it
shows live stats about the host, lets you drive local CLI tools through UI
elements, and gives you a chat panel to talk to an LLM-backed agent about the
box. A scuffed, personal Jarvis for one NixOS machine.

> Early stage. The core direction is set; the concrete shape of each pillar
> (dashboard framework, metrics source, LLM harness) is being worked out as
> spikes under [`tasks/`](tasks/) - read those before assuming an approach.

## What it does

- **Monitoring dashboard** - collect and surface host metrics (CPU, memory,
  disk, network, processes, temperatures, ...) in a live dashboard view.
- **CLI control** - expose a curated set of local command-line tools as UI
  actions, so common commands become buttons and forms instead of retyped
  incantations.
- **Chat agent** - an assistant panel backed by an external LLM provider
  (target: GPT-5.5, reached through a Pro/Plus subscription rather than a
  metered API key) that you can ask about the host and hand tasks to.

## Build and run

Python project managed with [`uv`](https://docs.astral.sh/uv/) and built
reproducibly through Nix via
[`uv2nix`](https://github.com/pyproject-nix/uv2nix). On NixOS, `nix develop`
provides the dev shell with the interpreter, the locked virtualenv, and `uv`.

```sh
nix develop            # enter the dev shell (venv activated, uv on PATH)
scufris                # run the app (console entry point)
python -m scufris      # same, via the module

ruff check .           # lint
mypy .                 # type-check
python -m pytest       # tests (use `-m`, not bare `pytest`, in a worktree)
nix flake check        # full QA gate (ruff + mypy + pytest + task records) in Nix

nix run .#scufris      # build and run the packaged app
nix build .#scufris    # build the runtime derivation

cd web
npm ci                 # frontend deps (once per checkout)
npm run ci             # frontend gate (prettier + eslint + vitest + build)
```

Dependencies are managed with uv (`uv add <pkg>`, then `uv lock`); uv2nix reads
`uv.lock`, so re-enter `nix develop` after changing deps. See
[`AGENTS.md`](AGENTS.md) for the full build/test/task workflow.

## Access and authentication

Two supported shapes, and the bind address decides which one you are in:

- **Loopback development** (`SCUFRIS_HOST=127.0.0.1`, the default): no login.
  `pytest`, the examples, and the mock backend need no credentials.
- **Authenticated LAN**: any non-loopback bind requires an operator session, and
  the server **refuses to start** without a credential configured. It does not
  warn and serve.

Set one up once:

```sh
scufris hash-password            # prompts, prints SCUFRIS_AUTH_PASSWORD_HASH=...
```

Put that line wherever your secrets live (for the `nix.dotfiles` deployment,
`sops secrets/scufris.env` - the same dotenv that already carries
`SCUFRIS_TELEGRAM_BOT_TOKEN`), then restart. The password itself is never stored:
what is kept is a `scrypt` hash of it.

The session is an opaque id in an `HttpOnly`, `SameSite=Lax` cookie backed by a
revocable server-side record, so signing out (or deleting the session file under
`SCUFRIS_STATE_DIR`) genuinely ends it. State-changing requests additionally
require a CSRF token and a same-origin `Origin`/`Referer`. The app's own MCP tool
subprocesses authenticate with a per-process bearer token instead of a cookie -
it is minted at startup and never persisted.

Not supported: public internet exposure, an untrusted network, or a shared host.
Traffic is plaintext HTTP; put a TLS-terminating proxy or a VPN in front if the
dashboard needs to leave a trusted LAN. Telegram is unaffected - its
authentication is the chat-id allowlist.

`examples/auth_session.py` drives the whole boundary over a real socket and
prints each refusal with its reason.

## Acting on the host

Reading the machine needs no privilege. CHANGING it goes through one contract,
and there are no exceptions to it:

    propose -> preview -> approve -> apply -> audit -> roll back

The privileged surface is a separate root process, `scufris-hostd`, which
speaks a closed set of typed verbs over a unix socket - start, stop, restart or
reload a service, socket, timer, path or mount unit; trim old system
generations; collect the Nix store; activate a built NixOS configuration or roll
back to an earlier generation - and builds every command itself. Targets,
slices and scopes are refused outright: `emergency.target` and `user.slice` are
how a deny-list of service names gets walked around. The dashboard cannot hand it a
command, only ask for a verb. There is no shell verb at any privilege under any
approval, and the refused class (partitioning, users, key material, the
firewall, scufris itself) has no verb rather than a check that could have a bug.

One agent may PROPOSE a change and will be shown the preview: the **host agent**,
which is bound to the machine rather than to a project and is the only audience
carrying the propose tools (the orchestrator keeps the read-only host tools and
delegates a change to it). Only a human may approve one: the decision endpoints
refuse the machine token the app's own tool subprocesses hold, whatever the bind
address, and an action that cannot be undone is refused unless the approval carries
an explicit acknowledgement rather than the ordinary confirmation.

While a proposal waits, the requesting agent is `blocked` - visible to the
orchestrator, and not answerable by it. When the decision lands, the agent is
resumed with the outcome, or with the denial and its reason so it can adapt
instead of proposing the same thing again.

Enable it deliberately, in the NixOS configuration, with `nixosModules.hostd`:

```nix
services.scufris-hostd = {
  enable = true;
  group = "scufris";                               # a DEDICATED group, not `users`
  secretFile = config.sops.secrets."scufris-hostd-secret".path;
};
```

The same secret must reach the app as `SCUFRIS_HOSTD_SECRET`. Without it the
helper refuses to start and the app answers every mutating host endpoint with
"not configured" - there is no half-enabled state. With it, an operator password
becomes mandatory even on loopback: approving is a human act, and the app refuses
to start with host agency and nobody to be that human. Stated plainly: that secret
raises the bar against the model acting unasked. It is not a boundary against a
compromised operator account, which on this machine is already root-equivalent.

The helper's audit log (`/var/log/scufris-hostd/audit.jsonl`) is root-owned,
append-only and size-rotated. Nothing outside the helper can delete an entry,
and no protocol verb can.

`examples/host_action.py` prints the whole contract - including an action with
no undo, and one stopped mid-apply. `examples/host_agent.py` drives the agent's
round trip against the real app and a real helper: propose, the orchestrator being
refused, the decision, and the turn the agent is resumed with. The design records
are `tasks/20260729-125020/DECISION.md` and `tasks/20260729-125040/DECISION.md`.

### Changing the NixOS configuration

"Add this package" is two things, and Scufris keeps them apart.

**Editing the configuration is ordinary project work.** `~/personal/nix.dotfiles`
is a git repository with its own tests and review, so an agent changes it the way
an agent changes any project: a worktree, a commit on a branch, a review. Scufris
has no configuration editor, no typed "add a package" verb and no way to write to
that repository at all.

**Activating what was committed is a host action.** Post a ref to
`/api/host/config/changes` (or let the host agent do it with
`propose_nixos_change` - that tool is on its server, not the orchestrator's) and
Scufris resolves it to a commit, builds
`nixosConfigurations.<host>` from THAT COMMIT as the operator - never as root,
because a configuration evaluated as root could read a host key into a
derivation - and proposes the activation of the exact store path it built. The
preview is `nix store diff-closures` between the running system and the built
one; approving it points the system profile at that path and switches to it;
rolling back is a proposal of its own that returns the system to a recorded
generation.

Three properties are worth stating because they are easy to assume and are
enforced instead:

- The build takes the tree from the commit, so uncommitted edits are not in it
  and the flow cannot dirty the repository. It says both, rather than leaving you
  to wonder.
- `activate` cannot be proposed directly. Its argument is a store path, and a
  caller who chose that path would be choosing what the machine boots while the
  closure diff faithfully described their choice.
- The preview does NOT list the units that would restart. The only thing that can
  produce that list is the proposed configuration's own
  `switch-to-configuration`, as root - and running unapproved code to preview it
  would defeat the approval. Read the commit's diff for what changed.

What this does NOT do is make an approved activation safe in the abstract: a
configuration can run anything as root once activated. The controls are the
reviewed commit, the diff you read, and a root-written audit record naming the
revision. `examples/nixos_change.py` drives the whole flow, and
`tasks/20260729-125035/DECISION.md` is the design record.

## Releases

Versions are cut by pushing a `vX.Y.Z` tag, which runs a pipeline that re-runs
the full gate on the tagged commit, builds the Python distribution, checks the
built wheel actually runs, and then publishes a
[GitHub Release](https://github.com/alexjercan/scufris/releases) whose notes are
that version's [`CHANGELOG.md`](CHANGELOG.md) section, with the wheel and sdist
attached. Every push and pull request runs the same QA gate (the badge above).

Run Scufris from a released version rather than from whatever `master` is
today by pinning the flake input to a tag:

```nix
{
  inputs.scufris.url = "github:alexjercan/scufris/v0.1.0";
}
```

The running instance reports which version it is - `scufris --version`, the
`scufris_version` field on `/api/agent/health`, and the dashboard's settings
view - so you can tell what is deployed without reading a Nix store path.

The release procedure itself (bumping, cutting the changelog, tagging, what the
guards check, what to do when a release fails halfway, how to yank one) is in
[`AGENTS.md`](AGENTS.md#releasing).

## Agents (optional)

Scufris runs **agents**: project-bound assistants you manage from the `/agents`
page (rendered as cards), each opening a dedicated `/agents/<id>` chat page, plus
a landing orchestrator chat and one reserved **host agent** (`/agents/host`) that
is bound to this MACHINE instead of a project and holds the host toolset. The
orchestrator and the host agent are configured from settings rather than the agents
page: the host agent stays read-only on files by construction, because its power is
proposing host changes an operator approves. Agents are **on by default** (set
`SCUFRIS_AGENT_ENABLED=0` to disable them), but do nothing until the operator
authenticates a backend CLI, since they drive an LLM CLI under your own
subscription - a personal-use path, not for shared/commercial use (see
[`tasks/20260719-153040/SPIKE.md`](tasks/20260719-153040/SPIKE.md)).

Each agent picks:

- a **backend** - **codex** (the `codex` CLI's app-server runner, from the nix dev
  shell `pkgs.codex`), **claude** (the `claude` Claude Code CLI), or **opencode**
  (a self-hosted model: scufris drives a running `opencode serve` daemon aimed at
  a local `llama.cpp` server - see
  [`examples/opencode/`](examples/opencode/) and
  [`tasks/20260722-135404/SPIKE.md`](tasks/20260722-135404/SPIKE.md)). codex and
  claude run natively on NixOS with nothing extra; opencode needs the daemon
  running.
- a **model** - a per-backend default (codex -> gpt-5.5, claude -> claude-opus-4-8,
  opencode -> gemma-4-26B-A4B-it), editable from a dropdown; switching the backend
  re-defaults the model.
- a **permission mode** - `manual` (read-only, default), `edit` (may edit project
  files), or `auto` (edit + run commands), mapped to codex's `--sandbox` /
  claude's `--permission-mode` / opencode's per-request `tools` map.

Each agent keeps one resumable session (its own conversation); switching an
agent's backend starts a fresh session (sessions are backend-specific).

Agents and the orchestrator talk **both ways**: a sub-agent that hits a decision
it cannot safely make signals it (the `request_input` tool), and the orchestrator
either is woken automatically (opt-in `SCUFRIS_AUTO_WAKE`) or polls
(`pending_agents`), then answers by resuming the sub-agent's session - so a stalled
"should I merge?" loop self-heals instead of hanging. A host approval travels the
same way with one difference: that agent is `blocked` rather than `waiting`, and it
is the OPERATOR who answers, so the orchestrator sees it in `pending_agents` and is
refused if it tries to answer it. With multiple orchestrator
chats, each child is stamped with the chat that spawned it, so `pending_agents`
scopes to the calling chat (its own children plus any UI-launched, never another
chat's). See [`examples/comms_loop.py`](examples/comms_loop.py) for a runnable
end-to-end walkthrough against the mock backend.

Any in-flight turn can be **cancelled**: while a run streams, the chat's send
button becomes a square stop button that aborts it (the partial answer is kept,
tagged `(cancelled)`). The orchestrator can also stop a sub-agent on request via
its `cancel_agent(agent_id)` tool, so "cancel that agent" works by instruction or
by opening the agent's chat and hitting stop.

```sh
nix develop                    # provides `codex` and `scufris`

# 1. Authenticate once (Sign in with ChatGPT, opens a browser).
scufris login                  # or run `codex login` directly (claude: `claude`)

# 2. Talk to the landing orchestrator (agents are on by default;
#    set SCUFRIS_AGENT_ENABLED=0 to disable them).
scufris chat "what is using my memory?"
```

Set `SCUFRIS_AGENT_AUTH_MODE=api_key` plus `SCUFRIS_OPENAI_API_KEY` to use a
metered API key instead of the subscription. All agent settings are in
[`.env.example`](.env.example). For offline dev/demo without a CLI, set
`SCUFRIS_ENABLE_MOCK_BACKEND=1` and create a `mock` agent.

## Project layout

| Path | What it is |
| --- | --- |
| `scufris/` | The Python package (entry point in `__main__.py`) |
| `flake.nix` | Nix dev shell + uv2nix wiring; `nix flake check` runs the QA gate |
| `pyproject.toml` | Project metadata, dependencies, ruff/mypy/pytest config |
| `uv.lock` | Locked dependency set (source of truth for uv2nix) |
| `tests/` | Pytest suite (harness/integration first) |
| `examples/` | Small runnable scripts that exercise a component end to end |
| `docs/` | Durable design docs and the lessons ledger |
| `tasks/` | tatr task records (one folder per task) |

## Contributing / working on it

Work is tracked with the `tatr` CLI as markdown under `tasks/`, and driven
through the `/spike -> /plan -> /work -> /review -> /compound` skill lifecycle
(or `/flow` end to end). Read [`AGENTS.md`](AGENTS.md) first - it covers the
build commands, testing philosophy, conventions, and task/tag rules.

## License

See [`LICENSE`](LICENSE).
