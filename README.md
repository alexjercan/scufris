# Scufris

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
