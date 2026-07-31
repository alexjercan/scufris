# AGENTS.md

Scufris: single-host NixOS monitoring dashboard, local CLI controls, and an
LLM-backed assistant.

Start here:

- Run `tatr ls --sort priority`.
- Read the active task and its sibling records under `tasks/<id>/`.
- Grep `LESSONS.md` for the affected area.
- Read the domain README before changing its code.

## Sources of truth

| Path | Covers |
|------|--------|
| `README.md` | Setup, deployment, all environment variables |
| `scufris/README.md` | Architecture, trust boundaries, HTTP and MCP surfaces, module map |
| `scufris/host/README.md` | Read-only host inspection |
| `scufris/hostd/README.md` | Root helper, socket protocol, verbs, audit log |
| `web/README.md` | Frontend pages, conventions, build gate |
| `pyproject.toml`, `uv.lock` | Python metadata and dependencies |
| `flake.nix` | Dev shell, packages, checks |
| `docs/RELEASING.md` | Release, retry, and yank procedure |

## Agent workflow

- Tracker and epics: use `tatr`; task records live under `tasks/`; schedule with one release tag and relative priority, or `backlog` at priority 0.
- Examples and retention: use `examples/` for runnable end-to-end proofs; task artifacts stay in `tasks/<id>/`; loose temporary notes go only in `docs/scratch/`.
- Domain docs: setup in `README.md`; architecture in `scufris/README.md`; package detail in the nearest README.
- Research and network: read existing spikes and decisions first; resolve open architecture or external behavior with a task-local `SPIKE.md`.
- Checks and records: run checks touched by the diff plus `tatr check --ledger LESSONS.md`; keep task-local records with the change.

Task lifecycle:

```text
/spike -> /plan -> /work -> /review -> /compound
```

- `/flow`: drives the full lifecycle.
- `LESSONS.md`: durable ledger. User decides pending promotions.
- `docs/scratch/`: release-blocking until compiled and emptied.
- Task records: append-only history, not live documentation.

## Commands

```sh
nix develop
scufris
python -m scufris

ruff check .
ruff format .
mypy .
python -m pytest

nix flake check
nix build .#scufris .#scufris-web

cd web && npm ci
cd web && npm run ci
```

- Canonical backend gate: `nix flake check`.
- Canonical frontend gate: `cd web && npm run ci`.
- Release packages: explicit `nix build`; flake check only evaluates packages.
- CI decides green: both gates plus both package builds.
- KVM-only release tests: `nix build .#scufris-vm-test` and `nix build .#scufris-hostd-vm-test`.
- Worktree tests: always `python -m pytest`, never bare `pytest`.
- Dependency changes: `uv add <pkg>` or edit `pyproject.toml`, then `uv lock`; re-enter `nix develop`.

## Implementation rules

- Global `~/AGENTS.md` rules apply.
- Python: target 3.13, Ruff line length 88, double quotes, full mypy coverage.
- Missing stubs: add narrow `pyproject.toml` overrides. No blanket source ignores.
- Config: pydantic-settings and `.env`. No scattered raw environment reads.
- Secrets: never hardcode, log, or pass to agent transcripts.
- Async request paths: async I/O only. No blocking sync calls.
- Authenticated frontend API calls: `apiFetch` in `web/src/common.ts`; login bootstrap stays in `web/src/login.ts`.
- New setting: update `README.md` and `.env.example` in the same task.
- New hostd verb or frame: update `scufris/hostd/README.md` in the same task.
- Notable change: update `CHANGELOG.md`.

## Testing

- Prefer integration and end-to-end coverage.
- Bugs: reproduce first with the highest-fidelity practical harness.
- Features: test through the interface the app uses.
- HTTP: exercise FastAPI routes with a client.
- Outbound HTTP: use `respx`.
- Async: use `pytest-asyncio`.
- Substantial component: add a small runnable `examples/` proof when useful.
- Re-read edited artifacts. Passing tooling does not prove correct content.

## Security invariants

- Authentication: loopback may be open; non-loopback requires an operator session. See `tasks/20260729-125015/DECISION.md`.
- HTTP auth: one deny-by-default middleware in `scufris/app.py`; public paths only in `scufris/auth.py`. No per-route auth dependencies.
- MCP callback auth: per-process `SCUFRIS_API_TOKEN`; never trust loopback. Operator-only host decisions reject bearer credentials.
- Identity: derive requester and approver from credentials, never request bodies.
- Child processes: build environments through `agent.agent_subprocess_env`; strip every `config.SECRET_ENV_VARS` entry.
- Host reads: unprivileged `scufris/host/`.
- Host changes: root `scufris/hostd/`; fixed contract: `propose -> preview -> approve -> apply -> audit -> roll back`.
- Host mutation tools: host agent only. Orchestrator delegates; project agents never receive them.
- Host capabilities: typed verbs and helper-built argv only. No shell verb. Refuse unsafe unit types, not selected names.
- Approval: HTTP decisions require an operator session; Telegram derives operator identity from its allowlisted chat credential. No approve MCP tool. `HostApprovalService` owns every decision surface.
- Pending approval: agent state `BLOCKED`, never `WAITING`; only the operator decides.
- Strong confirmation: destructive one-way actions only, not routine service control.
- Proposal truth: helper owns proposals and audit. Apply only the stored proposal; report partial multi-step state accurately.
- Audit: root-owned, append-only, rotated without single-file deletion; coalesce unauthenticated refusal floods.
- Nix CLI: route new CLI calls through `host.run.nix_cli`.
- NixOS config: external project; Scufris never edits it. Build committed refs as the operator; callers cannot supply activation store paths; rollback names a generation number.
- Scheduled escalation: only `checks.ESCALATABLE`; default off; all actions still use `HostApprovalService`.

Architecture and rationale: `scufris/README.md` plus decisions
`20260729-125020`, `20260729-125029`, `20260729-125035`,
`20260729-125040`, and `20260729-125046`.

## Worktrees and commits

- Isolated work: `sprout` worktree only. Never hand-create worktrees or use `.claude/worktrees/`.
- Main checkout: assume shared. Before commit, verify branch and status; stage explicit paths only.
- Sprout worktree: `git add -A` allowed after checking `web/node_modules` is not staged.
- Hook: `hooks/pre-commit`; enable with `git config core.hooksPath hooks` outside `nix develop`.
- Commit only when asked. User authorship only; no AI attribution trailers.

## Releasing

- Procedure: `docs/RELEASING.md`.
- Sources must agree: `pyproject.toml` version, `CHANGELOG.md` section, git tag.
- Guard: `scripts/check-release-ready.sh vX.Y.Z`.
- Release only from the main checkout, on `master`, inside `nix develop`.
- Push master before tagging. Published releases only fix forward.

## Documentation sync

- Code changes and affected live docs land in the same task.
- Live docs: root and package READMEs, `.env.example`, this file.
- Task records: historical evidence. Never rewrite old records after later renames.
- Long-lived cross-task material: `docs/`.
- Task-specific design and fix notes: `tasks/<id>/NOTES.md`.
