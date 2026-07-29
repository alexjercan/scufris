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
pytest                 # tests
nix flake check        # full QA gate (ruff + mypy + pytest) in Nix

nix run .#scufris      # build and run the packaged app
nix build .#scufris    # build the runtime derivation
```

Dependencies are managed with uv (`uv add <pkg>`, then `uv lock`); uv2nix reads
`uv.lock`, so re-enter `nix develop` after changing deps. See
[`AGENTS.md`](AGENTS.md) for the full build/test/task workflow.

## Agents (optional)

Scufris runs **agents**: project-bound assistants you manage from the `/agents`
page (rendered as cards), each opening a dedicated `/agents/<id>` chat page, plus
a landing orchestrator chat. Agents are **on by default** (set
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
"should I merge?" loop self-heals instead of hanging. With multiple orchestrator
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
