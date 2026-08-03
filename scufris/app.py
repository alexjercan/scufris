"""The Scufris application FACTORY.

`create_app` owns the object graph - the one database handle, the stores, the
services and the supervisors - and nothing else: every route lives on a router
under `api/`, and the factory includes them. Its own body is settings and
collector defaults, that graph, the lifespan, the two middlewares, the
`include_router` calls, the static mount and the OpenAPI tag pass.

The collector, the process collector, the config builder and the host inspector
are injected so tests can supply fakes; production uses the real ones.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator

from fastapi import FastAPI

from scufris_host import (
    Collector,
    HostInspector,
    HostOverviewCache,
    ProcessCollector,
    PsutilCollector,
    PsutilProcessCollector,
)
from scufris_hostd import ActionKind, Requester

from .agent_diagnostics import (
    AgentDiagnostics,
)
from .agent_store import (
    AgentStore,
)
from .api.agent_runs import (
    AgentRunDeps,
    build_agent_run_router,
)
from .api.agents import AgentDeps, build_agent_router
from .api.auth import SessionGate, auth_middleware, build_auth_router
from .api.chat import ChatDeps, build_chat_router
from .api.host import HostDeps, build_host_router
from .api.hostconfig import HostConfigDeps, build_hostconfig_router
from .api.legacy_agent import LegacyAgentDeps, build_legacy_agent_router
from .api.openapi import API_DESCRIPTION, OPENAPI_TAGS, apply_route_tags
from .api.projects import ProjectDeps, build_project_router
from .api.request_log import log_requests
from .api.static import mount_web_dist
from .auth import (
    LoginThrottle,
    SessionStore,
    mint_api_token,
    validate_auth_config,
)
from .auth import (
    now as auth_now,
)
from .config import (
    Settings,
)
from .db import close_state_database, state_database
from .digest import DigestStore
from .host_actions import (
    HostActionStore,
)
from .host_approval_bridge import HostApprovalBridge
from .host_approvals import (
    HostApprovalService,
)
from .host_watch import HostWatchService
from .hostclient import (
    HostdClient,
    HostdError,
    HostdUnavailable,
    host_supervisor,
)
from .hostconfig import (
    ConfigChange,
    ConfigChangeBuilder,
    ConfigChangeService,
    ConfigChangeStore,
    config_supervisor,
)
from .orchestrator import (
    AgentRunService,
    OrchestratorTurnService,
)
from .projects import (
    ProjectStore,
)
from .reasoning_store import ReasoningStore
from .scheduler import HostScheduler, SchedulerStore
from .settings_store import (
    SettingsStore,
)
from .supervisor import agent_supervisor
from .telegram.wiring import (
    build_approval_ops,
    build_settings_ops,
    start_bot,
)
from .version import scufris_version
from .wake import WakeBridge

logger = logging.getLogger(__name__)


SCUFRIS_VERSION = scufris_version()


def create_app(
    collector: Collector | None = None,
    settings: Settings | None = None,
    process_collector: ProcessCollector | None = None,
    config_builder: ConfigChangeBuilder | None = None,
    host_inspector: HostInspector | None = None,
) -> FastAPI:
    """Build the app.

    ``config_builder`` is the seam for the NixOS build: tests inject one whose
    executor is scripted, because the real one spawns `nix build` and there is no
    honest way to fake a system build through a runner.

    ``host_inspector`` is the same seam for READING the host, and the scheduled
    checks are why it exists: a check pass walks the nix store and shells out to
    systemctl, which is tens of seconds against the real machine and depends on the
    machine it runs on. Tests inject one over a `FakeRunner` replaying captured
    output. Default: a real inspector on the configured config repo, as before.
    """
    settings = settings or Settings()
    collector = collector or PsutilCollector()
    process_collector = process_collector or PsutilProcessCollector()
    # The schema comes up, and the operator's legacy JSON comes in, BEFORE the
    # first store: a store never reads a database that is a revision behind the
    # code reading it, nor one that is still missing the records the operator can
    # see in their `projects.json`. Both are no-ops after the first start.
    #
    # Taken from the process-wide accessor rather than opened here, so the app
    # and the leaves that cannot be injected - `CodexBackend.read_transcript`,
    # an in-process MCP tool - hold the SAME handle. Two handles on one file are
    # two pools contending on one write lock (DECISION.md 3 of 20260801-100409).
    # The lifespan below closes AND evicts it.
    db = state_database(settings.state_dir)
    projects = ProjectStore(settings, db)
    # First-class agents: named, project-bound records (A1). Running one is A3.
    # The landing orchestrator is a reserved record in this store (B5bc), so the
    # landing chat + session endpoints run through the same backend path as any
    # other agent - there is no longer a separate injected `Agent` object.
    agents = AgentStore(settings, projects, db)
    # Captures codex "thinking" from the live stream so a hard reload can re-show
    # the spoiler (reasoning is not recoverable from the rollout - see
    # reasoning_store). Written per turn in the turn stream, read at /transcript.
    reasoning_store = ReasoningStore(db)
    # What an agent can report about itself, asked of its own backend adapter.
    # Reads `settings` live (it mutates in place), so a backend switch moves the
    # whole capability set with it.
    diagnostics = AgentDiagnostics(settings)

    # Runtime-mutable settings: env base with persisted overrides layered on.
    # Mutations happen in place, so the closures below read the new value live
    # (the backend is resolved per turn via get_backend(agent.backend), not
    # cached). Switching the orchestrator's backend drops its active session so a
    # stale cross-backend session id is never resumed under the new backend.
    def _on_settings_change(changed: set[str]) -> None:
        if "agent_backend" in changed:
            agents.set_orchestrator_session(None)

    store = SettingsStore(settings, db, on_change=_on_settings_change)
    # Agent turns run as background jobs under the supervisor (ADR-001), not
    # inside the request. A dropped client no longer cancels a turn, and there is
    # no request timeout - a per-run heartbeat guards a genuinely stalled turn.
    supervisor = agent_supervisor(max_concurrent=settings.agent_max_concurrent)
    # The whole run lifecycle - launching a turn, the one-run-per-agent guard,
    # cancel/status/events, the sub-agent signals and the completion fan-out -
    # lives in this service rather than in closures over the factory, so a turn
    # can be driven with no app (scufris/orchestrator/runs.py).
    runs = AgentRunService(
        settings=settings,
        agents=agents,
        projects=projects,
        reasoning_store=reasoning_store,
        supervisor=supervisor,
    )
    # The orchestrator's own turns, once, for the three transports that start
    # them: the landing chat, the Telegram bot and the wake bridge.
    turn = OrchestratorTurnService(
        settings=settings,
        agents=agents,
        supervisor=supervisor,
        runs=runs,
    )

    # Codex sessions are not concurrency-safe, so an agent's turns run one at a
    # time: `AgentRunService.launch` reserves the supervisor's serialize slot keyed
    # on `agent.id`. The orchestrator's session-mutating endpoints (reset/new/switch/
    # delete) reserve the SAME key via `supervisor.serialized(ORCHESTRATOR_ID)`, so
    # they cannot interleave with an in-flight orchestrator turn - and because a
    # turn reserves its slot synchronously in `start()`, a mutation arriving right
    # after cannot slip in front of its own turn. (fork is the exception: it
    # LAUNCHES a turn, so it must NOT hold the lock or it self-deadlocks on the
    # key the launch reserves - see fork_session.)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The Telegram bot (if a token is configured) runs as a background task
        # for the app's lifetime, cancelled cleanly on shutdown. It is started
        # here rather than at create_app time so its poll loop lives on the
        # serving event loop. `_start_telegram_bot` is defined later in
        # create_app; the closure resolves it at call time.
        telegram_task = _start_telegram_bot()
        # The scheduler ticks for the app's lifetime. Started here rather than at
        # create_app time so its loop lives on the SERVING event loop, like the bot's.
        checks_task = asyncio.create_task(scheduler.run_forever())
        app.state.host_checks_task = checks_task
        # Recover the approval queue from the helper before serving. The app's
        # registry is in-memory by design (the helper owns proposals), so without
        # this a restart inside a proposal's ten-minute window leaves a real pending
        # approval unreachable - the operator would see an empty queue while the
        # helper still held an appliable action. A helper that is not configured
        # or not running is not an error here: there is simply nothing to
        # recover, and every host route
        # already answers "not configured" honestly.
        try:
            await approvals.refresh_pending()
        except (HostdUnavailable, HostdError) as exc:
            logger.info("could not recover the host approval queue: %s", exc)
        try:
            yield
        finally:
            checks_task.cancel()
            with suppress(asyncio.CancelledError):
                await checks_task
            if telegram_task is not None:
                telegram_task.cancel()
                with suppress(asyncio.CancelledError):
                    await telegram_task
            await runs.aclose()  # cancel any in-flight runs on shutdown
            # Last: the stores read through this handle, so it outlives anything
            # that might still be finishing above. EVICTED as well as closed - a
            # disposed engine still dials a fresh connection, so leaving it in
            # the accessor would hand the next `create_app` a handle whose
            # lifespan has already ended.
            close_state_database(settings.state_dir)

    app = FastAPI(
        title="Scufris API",
        summary="Scuffed Jarvis: a host dashboard and multi-agent orchestrator.",
        description=API_DESCRIPTION,
        version=SCUFRIS_VERSION,
        lifespan=_lifespan,
        openapi_tags=OPENAPI_TAGS,
    )
    # Exposed for tests and future per-agent endpoints (A3/A4).
    app.state.supervisor = supervisor
    # The run lifecycle, exposed alongside it: the routes translate for this
    # object, and a test that wants to observe a turn without an HTTP round trip
    # reaches it here.
    app.state.runs = runs
    # Exposed so tests can seed the orchestrator's active session directly (the
    # landing session state now lives in the store, not an injected agent).
    app.state.agents = agents
    app.state.projects = projects
    # The ONE handle, exposed so the boundary proof can assert every store holds
    # THIS object rather than one of its own.
    app.state.db = db

    # --- authentication --------------------------------------------------
    #
    # Fail closed FIRST: a network-reachable bind with no credential must not
    # produce an app at all (see auth.validate_auth_config). This raises
    # AuthConfigError, which `scufris serve` reports as a startup failure.
    validate_auth_config(settings)
    # The machine credential for THIS process's own tool subprocesses. It is put
    # on the app's own Settings, deliberately NOT in os.environ: an env var is
    # inherited by the agent CLI subprocess and hence by every shell command the
    # model runs, which would hand a sandboxed sub-agent the operator's full API
    # credential. From here it reaches exactly two places - the MCP servers that
    # call the API (agent.scufris_mcp_servers) and the in-process tool console
    # (which passes it through a ContextVar, since that tool's httpx call loops
    # back to this same server). Review round 1, finding 2.
    app.state.api_token = mint_api_token()
    settings.auth_api_token = app.state.api_token
    sessions = SessionStore(db)
    # Sweep once at startup so a restart clears out sessions that expired while
    # the server was down, rather than carrying them until each id is presented.
    # Not offloaded: `create_app` is synchronous and runs before the loop exists.
    sessions.prune(
        now=auth_now(),
        idle=settings.auth_session_idle_seconds,
        absolute=settings.auth_session_max_seconds,
    )
    app.state.sessions = sessions
    # The shared identity gate: the middleware enforces with it, the host routes
    # stamp the audit with it, and `/api/agents/{id}/chat` asks it whether the
    # caller is an agent. One object, so those three cannot disagree about who
    # is asking.
    gate = SessionGate(settings, sessions)
    app.state.auth_required = gate.required
    throttle = LoginThrottle(
        max_failures=settings.auth_login_max_failures,
        window_seconds=settings.auth_login_window_seconds,
    )
    # Registered BEFORE the request logger below, so the logger stays outermost
    # (Starlette applies middleware in reverse) and a denial is still logged.
    app.middleware("http")(auth_middleware(gate, app.state.api_token))
    app.include_router(build_auth_router(gate, throttle))
    app.middleware("http")(log_requests)

    inspector = host_inspector or HostInspector(config_repo=settings.host_config_repo)
    host_overview_cache = HostOverviewCache(
        inspector,
        settings.host_overview_seconds,
    )

    # --- privileged host actions -----------------------------------------
    #
    # propose -> preview -> approve -> apply -> audit -> roll back. The verbs,
    # the previews, the proposals and the audit log all live in the root helper
    # (scufris_hostd); this builds the object graph the host router serves over
    # it. The routes themselves are in `scufris/api/host.py`.

    hostd = HostdClient(settings.hostd_socket, settings.hostd_secret)
    host_actions = HostActionStore(db)
    # One at a time: two root commands running concurrently on one machine is
    # not something an operator approved.
    host_supervisor_ = host_supervisor(max_concurrent=1)
    # The ONE decision path. The host router is one surface over it; the Telegram
    # bot is the other, and it calls the same methods with a chat-derived actor.
    # Every rule after "who is deciding" lives in the service, so the two cannot
    # drift.
    approvals = HostApprovalService(
        hostd=hostd, actions=host_actions, supervisor=host_supervisor_
    )
    app.state.hostd = hostd
    app.state.host_actions = host_actions
    app.state.host_supervisor = host_supervisor_
    app.state.host_approvals = approvals

    # --- the scheduled host checks and the digest ---------------------------
    #
    # The one thing here that starts without a person. The scheduler owns the clock;
    # this owns what a run DOES: read the checks off the loop, render a digest,
    # deliver it (or not, per the schedule and the mute), and escalate a breach into
    # the ordinary approval queue if the operator has switched that on.

    digests = DigestStore(db)
    scheduler_store = SchedulerStore(db)
    app.state.digests = digests

    watch = HostWatchService(
        settings=settings,
        inspector=inspector,
        agents=agents,
        diagnostics=diagnostics,
        digests=digests,
        approvals=approvals,
        hostd=hostd,
        # Both late-bound on purpose: the scheduler is built around this service
        # on the next line, and the bot starts further down.
        muted=lambda: scheduler.muted(),
        telegram_bot=lambda: getattr(app.state, "telegram_bot", None),
    )

    scheduler = HostScheduler(
        scheduler_store,
        run=watch.run,
        watch_interval=lambda: settings.host_watch_interval_seconds,
        daily_at=lambda: settings.host_digest_at,
        watch_enabled=lambda: (
            settings.host_checks_enabled and settings.host_watch_enabled
        ),
        daily_enabled=lambda: (
            settings.host_checks_enabled and settings.host_digest_enabled
        ),
        muted_until=lambda: settings.host_digest_muted_until,
    )
    app.state.host_scheduler = scheduler

    # Included HERE, after the scheduler above: a router factory binds its
    # dependencies at construction rather than by late closure lookup, so
    # everything `HostDeps` names has to exist by this line. That is the point -
    # the routes used to read `scheduler` and `digests` from a scope that bound
    # them a thousand lines further down, and only worked because no request
    # arrived before `create_app` returned.
    app.include_router(
        build_host_router(
            HostDeps(
                settings=settings,
                gate=gate,
                collector=collector,
                processes=process_collector,
                overview=host_overview_cache,
                hostd=hostd,
                actions=host_actions,
                approvals=approvals,
                runs=host_supervisor_,
                scheduler=scheduler,
                digests=digests,
            )
        )
    )

    # --- NixOS configuration changes (R3) --------------------------------
    #
    # The configuration repository is a PROJECT: an agent edits and commits it
    # through the ordinary project machinery, and none of that happens here. What
    # happens here is the last mile - resolve a ref to a commit, build it as the
    # operator, and hand the resulting store path to the helper as an `activate`
    # proposal. `ConfigChangeService` owns that; this wires it.

    config_changes = ConfigChangeStore(db)
    config_builder = config_builder or ConfigChangeBuilder(
        build_timeout=settings.host_config_build_timeout
    )
    # Its own supervisor: a NixOS build can run for an hour and needs no
    # privilege, so it must not sit in the single slot that serializes approved
    # root commands.
    config_supervisor_ = config_supervisor(max_concurrent=1)
    # Sweep once at startup, for the same class of reason as `sessions.prune`
    # above: a build the last process was running is not running now, and left
    # `building` it would refuse every later build of that repository with a 409
    # that cancelling cannot clear. Not offloaded - `create_app` is synchronous
    # and runs before the loop exists.
    config_changes.abandon_builds()
    app.state.config_changes = config_changes
    app.state.config_supervisor = config_supervisor_

    async def _propose_activation(built: ConfigChange, requester: Requester) -> str:
        """Propose activating what the build produced, and return the proposal id.

        Injected rather than built inside the change service: that package knows
        how to turn a ref into a store path, and this knows how a proposal reaches
        the operator. It goes through the approval service like every other
        proposal, so a configuration activation waiting on the operator marks the
        agent that asked for it as BLOCKED and reaches the operator's surfaces the
        same way a unit restart does.

        Straight to `record_proposal` rather than `approvals.propose`, which
        refuses ACTIVATE on purpose: that refusal is about a CALLER naming a store
        path, and this path is one this server built from a revision it resolved.
        """
        proposal = await hostd.propose(
            ActionKind.ACTIVATE,
            {
                "toplevel": built.toplevel,
                "repo": built.resolved.repo,
                "rev": built.resolved.rev,
            },
            requester,
        )
        await approvals.record_proposal(proposal)
        return proposal.id

    app.include_router(
        build_hostconfig_router(
            HostConfigDeps(
                gate=gate,
                changes=ConfigChangeService(
                    store=config_changes,
                    builder=config_builder,
                    supervisor=config_supervisor_,
                    propose=_propose_activation,
                    settings=settings,
                ),
            )
        )
    )

    # --- projects ---------------------------------------------------------
    #
    # The workspaces an agent runs in. Every rule about what a project is lives in
    # `ProjectStore`; the router translates for it and serves the detail page's
    # SPA shell.

    app.include_router(
        build_project_router(ProjectDeps(settings=settings, projects=projects))
    )

    # --- running an agent -------------------------------------------------
    #
    # The turn surface under /api/agents/{id}/, then the record surface over the
    # rows themselves. The two never collide because no path of one is a prefix
    # of the other, and both are bare (absolute paths on the router).

    app.include_router(
        build_agent_run_router(
            AgentRunDeps(
                settings=settings,
                agents=agents,
                runs=runs,
                diagnostics=diagnostics,
                gate=gate,
                approvals=approvals,
                supervisor=supervisor,
            )
        )
    )

    app.include_router(
        build_agent_router(
            AgentDeps(settings=settings, agents=agents, store=store, runs=runs)
        )
    )

    # --- the console's own agent -------------------------------------------
    #
    # The singular `/api/agent/*` surface, orchestrator-scoped. Mostly aliases for
    # `/api/agents/orchestrator/*`, answered out of the SAME services, plus the
    # settings view, the "try it" tool runner and the session switcher.

    app.include_router(
        build_legacy_agent_router(
            LegacyAgentDeps(
                settings=settings,
                agents=agents,
                store=store,
                diagnostics=diagnostics,
                runs=runs,
                supervisor=supervisor,
                api_token=app.state.api_token,
            )
        )
    )

    # --- a pending approval is a BLOCKED agent -----------------------------

    approval_bridge = HostApprovalBridge(
        agents=agents,
        projects=projects,
        runs=runs,
        approvals=approvals,
        telegram_bot=lambda: getattr(app.state, "telegram_bot", None),
    )
    approval_bridge.connect()

    wake_bridge = WakeBridge(
        agents=agents,
        settings=settings,
        is_orchestrator_busy=turn.busy,
        launch=turn.wake,
    )
    # The completion fan-out, in the order the turn path used to hard-code: a
    # sub-agent that finished needing input wakes the orchestrator (BC4), and the
    # orchestrator's OWN completion drains any deferred wake; then a host-action
    # decision that arrived while the agent was mid-turn is delivered. Both run
    # past the finished run's serialize-key release, which is what lets them
    # launch a turn for the very agent that just finished (lesson
    # `serialize-then-launch-self-deadlocks-on-shared-key`).
    runs.on_complete(wake_bridge.on_run_complete)
    runs.on_complete(approval_bridge.drain_deferred_decision)

    # Built whether or not a bot is running, and exposed: a test can then drive the
    # REAL decision path (and the real allowlist refusal) without starting a poll
    # loop against a stubbed Bot API, and the production wiring can be asserted
    # rather than assumed.
    app.state.telegram_approval_ops = build_approval_ops(settings, approvals)

    def _start_telegram_bot() -> "asyncio.Task[None] | None":
        """Start the bot on the SERVING loop and publish it on `app.state`.

        Returns the poll-loop task for the lifespan to cancel, or None when no
        token is configured.
        """
        bot, task = start_bot(
            settings,
            turn,
            settings_ops=build_settings_ops(settings, agents, diagnostics, collector),
            approval_ops=app.state.telegram_approval_ops,
        )
        app.state.telegram_bot = bot
        app.state.telegram_task = task
        return task

    # --- the orchestrator's own chat --------------------------------------

    app.include_router(build_chat_router(ChatDeps(settings=settings, turn=turn)))

    # Mounted LAST so the /api routes above take precedence; everything else
    # falls through to the static bundle.
    mount_web_dist(app, settings.web_dist)

    # Group the API endpoints under OpenAPI tags so /docs (Swagger) and /redoc
    # render organized, labelled sections. LAST, after every include: the pass
    # walks the app's real surface, so a router added below it goes untagged.
    apply_route_tags(app)

    return app
