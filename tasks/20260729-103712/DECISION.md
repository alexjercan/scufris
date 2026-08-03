# Decision: Extract the remaining routers and reduce create_app to assembly

- DATE: 20260803-080514
- STATUS: ACCEPTED
- TASK: 20260729-103712
- TAGS: refactor, backend, maintainability

## Context

`scufris/app.py` is 2621 lines against the repository's 600-line source cap, and
it is one of two ALLOWLIST ratchet entries in `scripts/check_file_size.py`. The
guard is one-way: an entry that becomes stale fails the check, so the split and
the allowlist removal are one commit. `20260801-100425` established the shape
this follows - one module per domain under `api/`, each exporting a
`build_*_router` factory over a frozen deps dataclass, reaching for nothing
(`tests/test_domain_routers.py::test_domain_router_dependency_isolation`), with
`iter_routes` as the one way to walk the real surface. `20260801-100441` moved
the turn and run LOGIC into `orchestrator/`, so what is left in `app.py` is
already translation-only; this is a move.

Two measurements force the shape of this task. First, the routes alone are not
enough: moving all ~1420 lines of routes and response models still leaves
`create_app` around 710 lines, because the factory also carries the scheduled
host-checks pass (~130 lines), the approval->agent bridge (~148), the Telegram
callback wiring (~194) and the OpenAPI tag map (~85). Second,
`tests/test_domain_routers.py` is 893 lines against the 900-line test cap, so
the new coverage cannot extend it.

## Decision

Split into nine modules, eight of them new, chosen so each one has a single
reason to change and nothing is a wrapper over one caller:

- `api/projects.py`, `api/agents.py`, `api/agent_runs.py`, `api/chat.py`,
  `api/legacy_agent.py` - the five routers, each a `build_*_router(deps)` over a
  frozen deps dataclass, exactly like `api/host.py`;
- `api/openapi.py` - the description, the tag metadata, `route_tags` and
  `apply_route_tags(app)`;
- `env_bridge.py` - `ensure_api_base` / `ensure_den_path`;
- `host_watch.py` (`HostWatchService`) and `host_approval_bridge.py`
  (`HostApprovalBridge`) - the two non-route blocks lifted out of `create_app`;
- `telegram/wiring.py` - `build_approval_ops`, `build_settings_ops`,
  `start_bot`.

The agent surface splits in TWO routers rather than one. `/api/agents/*` is
~490 lines of routes plus ~160 of models; as one module it lands near 670, over
the cap, and the split falls on a real seam - the RECORD (CRUD, the backend
catalog, the pending queue, the project capabilities) versus the RUN (launch,
cancel, status, events, chat, the sub-agent signals, fork, and the per-agent
diagnostics reads). They take different dependency sets: the record router needs
the stores, the run router needs `AgentRunService` and `AgentDiagnostics`.

`/api/agent/*` stays a router of its OWN rather than being folded into the
scoped ones. It is the compatibility surface: same paths, same responses,
delegating to the same services, and keeping it separate is what lets it be
deleted in one move when the frontend stops calling it.

`env_bridge.py` exists as its own module because `ensure_den_path` is called
from `api/legacy_agent.py`, `api/agent_runs.py` and `telegram/wiring.py` as well
as `app.py`; leaving it in `app.py` makes every router import the factory that
imports them.

## Alternatives considered

- **Move only the routes and keep `create_app` otherwise intact.** What this
  task's Steps originally said. It leaves `app.py` around 710 lines, so the
  allowlist entry cannot be removed and the file-size criterion fails. Measured,
  not estimated: the four non-route blocks above are 557 lines together.
- **Raise `SOURCE_CAP`, or leave `scufris/app.py` allowlisted.** The cheapest
  option and the one the guard exists to refuse - its docstring calls the
  allowlist a ratchet, not a config knob, and this task is the task that owns
  the entry.
- **One `api/agents.py` for the whole agent surface.** Fewer modules, and it
  reads as one domain. Rejected on the cap: ~670 lines, and the fix would be to
  split it anyway, later, without the record/run seam being visible.
- **Fold `ensure_api_base` / `ensure_den_path` into `mcp_common.py`.** Avoids a
  new module and the functions are about MCP tool runs. Rejected: that module
  states it deliberately imports nothing heavy so `den_mcp_server` stays usable
  on a box with only the `today`/`macros` CLIs, and both helpers need
  `Settings`.
- **FastAPI `Depends` instead of deps dataclasses.** Already decided against in
  `20260801-100425` DECISION.md 1; re-deciding it here would leave two
  conventions in `api/`.
- **Split `tests/test_app.py` in the same task.** It is 4241 lines and the other
  allowlist entry. Rejected as scope: it is a list of independent cases, its cap
  is separate, and nothing in this task's DoD needs it. It stays allowlisted.

## Consequences

Easier: `create_app` becomes readable as an object graph, and a missing
dependency is a construction error at the `include_router` line rather than a
`NameError` on a live request. Every remaining surface becomes drivable on a
bare `FastAPI()` over fakes, which is what the new isolation test asserts. The
compatibility surface becomes one deletable file.

Harder: `scufris/app.py` lands ~25 lines under the cap, so the next thing added
to `create_app` pushes it over - the named fallback is moving `run_server` and
`ensure_api_base` into `cli.py`, its only caller. Eight new modules is real
navigation cost, mitigated by the `api/` table in `scufris/README.md` section 7.
Four monkeypatch bind sites move with the code (`get_backend`, `TelegramBot`,
`_ensure_den_path`, `AgentConfigUpdate`); `tests/conftest.py` already keeps the
`get_backend` list in one place for this reason, and a listed site that
disappears fails loudly, but a NEW importer of `get_backend` is silent until it
is added there.
