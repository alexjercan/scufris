"""The agent run service: one turn's whole lifecycle, for every caller.

`AgentRunService` owns what used to be a set of `create_app` closures over the
supervisor and two dictionaries - launching a turn, the one-run-per-agent guard,
cancelling, reading status, relaying events, the sub-agent signals, and the
completion fan-out. Everything a transport does to a run goes through here, so
the web routes, the Telegram bot and the wake bridge cannot disagree about what
"busy" means or about when a turn is finished.

It knows nothing about HTTP: it refuses with the typed errors in `errors.py`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from scufris_core import EventBus, RunPhase, RunState

from ..agent import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamSessionStarted,
)
from ..agent_store import ORCHESTRATOR_ID, AgentRecord, AgentStore
from ..backends import get_backend
from ..backends.base import BackendStatus
from ..config import Settings, canonical_backend
from ..enums import AgentState
from ..projects import Project, ProjectNotFound, ProjectStore
from ..reasoning_store import ReasoningStore
from ..sessions import format_fork_seed, strip_steering
from ..supervisor import AgentSupervisor
from .errors import (
    AgentProjectMissing,
    NoActiveRun,
    RunAlreadyActive,
    TurnEndedWithoutReply,
    TurnFailed,
)

#: Called with an agent id at the end of a run's completion callback, in
#: registration order, after the terminal outcome is persisted and past the
#: run's serialize-key release.
CompletionHook = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class TurnStatus:
    """A run's live state merged with its backend's read-only session progress.

    A neutral snapshot, not the HTTP response model: the route shapes it into
    `AgentRunStatus`. ``prompt`` is present only while the run is live.
    """

    agent_id: str
    state: str
    session_id: str | None
    progress: BackendStatus | None
    prompt: str | None


class AgentRunService:
    """Launch, supervise and finish agent turns.

    One instance per app. It holds the supervisor, the agent -> current-run-id
    registry and the claim set that closes the launch race, so those three are
    read and written in one place.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        agents: AgentStore,
        projects: ProjectStore,
        reasoning_store: ReasoningStore,
        supervisor: AgentSupervisor,
    ) -> None:
        self._settings = settings
        self._agents = agents
        self._projects = projects
        self._reasoning = reasoning_store
        self._supervisor = supervisor
        # The latest supervisor run id for each agent (a run id is unique per
        # launch; the agent id serializes them). Lets the status/events readers
        # find an agent's current run without colliding on re-runs of the same
        # agent.
        self._agent_runs: dict[str, str] = {}
        # Run ids claimed by `launch` but not yet handed to `supervisor.start`.
        # The supervisor knows nothing about a run until it is started, and
        # starting it now sits behind an `await` (the `mark_running` offload), so
        # without this set the one-run-per-agent guard would read
        # `supervisor.status(...) is None` for a launch already in progress and
        # let a second turn through (review round 1, R1.1).
        self._launching_runs: set[str] = set()
        self._complete_hooks: list[CompletionHook] = []

    # --- completion fan-out ------------------------------------------------

    def on_complete(self, hook: CompletionHook) -> None:
        """Register a hook to run when any agent's turn finishes.

        Hooks fire in registration order at the very end of the completion
        callback - after the terminal outcome is persisted and past the run's
        serialize-key release, so a hook may launch a turn for the agent that
        just finished without deadlocking on its own key (lesson
        `serialize-then-launch-self-deadlocks-on-shared-key`). That ordering is
        load-bearing; do not move the fan-out earlier.
        """
        self._complete_hooks.append(hook)

    # --- the one-run-per-agent predicate -----------------------------------

    def active(self, agent_id: str) -> bool:
        """Whether a turn for ``agent_id`` is claimed, queued or running.

        The one-run-per-agent predicate, shared by the guard in `launch` and by
        the wake bridge's is-busy check so the two can never disagree about what
        "busy" means. Pure in-memory reads: it is called from the synchronous
        step that also claims the slot, and an await in here would reopen the
        window it closes.
        """
        run_id = self._agent_runs.get(agent_id)
        if run_id is None:
            return False
        if run_id in self._launching_runs:
            return True
        state = self._supervisor.status(run_id)
        return state is not None and state.state in (RunPhase.QUEUED, RunPhase.RUNNING)

    def run_id(self, agent_id: str) -> str | None:
        """The agent's current run id, or None if it has never run."""
        return self._agent_runs.get(agent_id)

    # --- resolution --------------------------------------------------------

    def require_agent(self, agent_id: str) -> AgentRecord:
        """The agent record, or `AgentNotFound`."""
        return self._agents.get(agent_id)

    def require_agent_project(self, agent: AgentRecord) -> Project | None:
        """The agent's project, or None for the orchestrator.

        Raises `AgentProjectMissing` when the binding points at a project that
        has been deleted.

        Synchronous, and callers on the event loop must offload it: the store
        reads through `Database.transaction()`, whose begin is immediate - it
        takes SQLite's single write lock and waits up to `busy_timeout` for it.
        On the loop thread that wait stalls every other request, SSE stream and
        probe in the process, which is why
        `packages/core/src/scufris_core/engine.py` tells loop-thread callers to
        offload the whole synchronous unit of work.
        """
        # The reserved orchestrator (and only it) has no project binding: it runs
        # in the server cwd. Everyone else must resolve to a real project.
        if not agent.project_id:
            return None
        try:
            return self._projects.get(agent.project_id)
        except ProjectNotFound as exc:
            raise AgentProjectMissing() from exc

    # --- launching ---------------------------------------------------------

    async def launch(
        self,
        agent: AgentRecord,
        project: Project | None,
        prompt: str,
        *,
        image_paths: list[str] | None = None,
        on_done: Callable[[], None] | None = None,
    ) -> tuple[str, EventBus[StreamEvent]]:
        """Stream one turn of ``prompt`` through the agent's backend (resuming its
        session), on the SAME supervisor + event bus + agent-run registry as a
        goal run. Persists the (possibly new) session id and terminal state.
        Shared by ``run`` (goal), per-agent ``chat`` (message), and the landing
        orchestrator chat (B5bc). Raises `RunAlreadyActive` when a run/chat for
        this agent is already active.

        A coroutine because it writes the agent's RUNNING state, which is a store
        call and so is offloaded with ``asyncio.to_thread``; ``supervisor.start``
        stays on the loop after it, in that order (DECISION.md 2). Everything
        the check-then-claim below needs therefore happens BEFORE the first
        await: the guard and the claim are one synchronous step, and callers
        that must not be interleaved (the orchestrator's session fork) rely on
        that."""
        if self.active(agent.id):
            raise RunAlreadyActive()

        settings = self._settings
        agents = self._agents
        # Resolved before the claim, so a bad backend name raises with no
        # reservation left behind for the agent to be 409ed on forever.
        backend = get_backend(agent.backend)
        is_codex = canonical_backend(agent.backend) == "codex"

        run_id = f"{agent.id}:{uuid.uuid4().hex}"
        # Claim the slot in the SAME synchronous step as the guard above, and
        # hold the claim until the supervisor knows about the run.
        self._agent_runs[agent.id] = run_id
        self._launching_runs.add(run_id)

        captured: dict[str, str] = {}
        # Accumulate the turn's live reasoning ("thinking") deltas so the sidecar
        # can persist them for reload survival (codex-only; reasoning is absent
        # from the rollout). Codex is the only backend that streams these.
        reasoning_parts: list[str] = []

        async def turn_stream() -> AsyncIterator[StreamEvent]:
            async for event in backend.stream(
                settings,
                prompt,
                session_id=agent.session_id,
                cwd=project.cwd if project is not None else None,
                image_paths=image_paths,
                permission_mode=agent.permission_mode,
                is_orchestrator=agent.id == ORCHESTRATOR_ID,
                agent_id=agent.id,
            ):
                if isinstance(event, StreamReasoningDelta):
                    reasoning_parts.append(event.delta)
                if isinstance(event, StreamSessionStarted):
                    # Record ownership the moment the session id is known (turn-start,
                    # before the turn streams), so a client refreshing mid-turn sees
                    # the session instead of nothing until mark_finished. Keyed under
                    # the launch-time backend snapshot; idempotent with the terminal
                    # mark_finished. Also seeds captured[] so an errored turn (no
                    # done frame) still persists the id.
                    captured["session_id"] = event.session_id
                    await asyncio.to_thread(
                        agents.record_running_session,
                        agent.id,
                        agent.backend,
                        event.session_id,
                    )
                if isinstance(event, StreamDone):
                    if event.session_id:
                        captured["session_id"] = event.session_id
                    # The final reply text is the durable outcome's message (BC1),
                    # so the orchestrator can read what a finished agent said/asked
                    # after the per-run bus has closed.
                    captured["message"] = event.reply.text
                    # Persist this turn's reasoning to the sidecar BEFORE yielding
                    # the done frame, so a reload the client triggers on `done`
                    # reads a sidecar that is already written (the on_complete
                    # persist callback runs in the supervisor's finally, AFTER the
                    # frame - too late; lesson out-of-context-review-misses-cross-
                    # layer-timing). One entry per completed codex turn, keyed by
                    # the just-captured session id; keeps the sidecar 1:1 with the
                    # assistant messages read_transcript surfaces (empty reasoning
                    # when the model did not think). Non-codex turns never stream
                    # reasoning, so nothing is written.
                    if is_codex and event.reply.text.strip():
                        await asyncio.to_thread(
                            self._reasoning.append,
                            captured.get("session_id"),
                            "".join(reasoning_parts),
                            answer=event.reply.text,
                        )
                yield event

        async def persist(run_state: RunState) -> None:
            # Best-effort: if the agent was deleted mid-run, mark_finished raises
            # AgentNotFound, which the supervisor swallows and logs - the terminal
            # state just is not persisted for a record that no longer exists.
            #
            # A turn FAILED when the supervisor's RunPhase is not DONE (the
            # except-clause paths: cancelled / stall / budget / crash) OR when a
            # backend yielded a terminal StreamError (RunPhase stays DONE but
            # _drain recorded the detail on run.error). Either way the agent's
            # terminal state is ERROR, and the diagnostic detail is the durable
            # outcome message so agent_status / pending_agents can report WHY. The
            # error detail WINS over any captured reply on a failed turn: a rogue
            # backend that emits both a done frame and a trailing StreamError must
            # still surface the failure, not a stale success reply.
            # A user-initiated cancel is its OWN terminal state, distinct from a
            # crash/stall/backend-error: it must read as a neutral stop (not a red
            # ERROR) and must NOT surface in pending_agents as an agent needing the
            # orchestrator. Keyed off the explicit run.cancelled flag, not the
            # "cancelled" error string, so a real error is never misclassified.
            failed = run_state.state != RunPhase.DONE or bool(run_state.error)
            if run_state.cancelled:
                terminal_state = AgentState.CANCELLED
                message = "cancelled"
            elif failed:
                terminal_state = AgentState.ERROR
                message = run_state.error or captured.get("message", "")
            else:
                terminal_state = AgentState.DONE
                message = captured.get("message", "")
            # Turn-owned cleanup (e.g. an attached image tempdir) runs when the
            # run ends, not when a relay disconnects - and FIRST, before this
            # callback's first await, so it is still synchronous with respect to
            # the relay the way it was before persisting became a coroutine.
            if on_done is not None:
                on_done()
            # Only the STORE call is offloaded. The tail below is loop-bound
            # (it launches turns) and ordered after it, which is why this
            # callback is awaited in place rather than run in a worker.
            await asyncio.to_thread(
                agents.mark_finished,
                agent.id,
                state=terminal_state,
                session_id=captured.get("session_id"),
                # Key the session under the backend this turn RAN on (the
                # launch-time snapshot), not whatever the current config is - a
                # backend switch that raced the turn must not mislabel it.
                backend=agent.backend,
                message=message,
                run_id=run_id,
            )
            # The completion fan-out, in registration order. It runs HERE - after
            # mark_finished, so a hook reads the current outcome, and in the
            # supervisor's finally, past the run's serialize-key release, so a
            # hook that launches a turn for this same agent cannot deadlock on
            # the key this run held. Today that is the wake bridge (BC4) and then
            # the deferred host-decision drain.
            for hook in self._complete_hooks:
                await hook(agent.id)

        try:
            await asyncio.to_thread(agents.mark_running, agent.id)
            bus = self._supervisor.start(
                run_id,
                turn_stream,
                serialize_key=agent.id,
                budget_seconds=None,
                heartbeat_seconds=settings.agent_heartbeat_seconds,
                on_complete=persist,
                # Expose the raw turn prompt on the run's status so a client
                # reattaching mid-turn can render the user bubble before the
                # backend flushes it to its durable log (the steering added
                # downstream is stripped at the status read boundary).
                prompt=prompt,
            )
        finally:
            # Released either way: once started the supervisor answers for the
            # run, and if mark_running raised (the agent was deleted mid-launch)
            # the claim must not outlive the launch that failed.
            self._launching_runs.discard(run_id)
        return run_id, bus

    async def drain(self, bus: EventBus[StreamEvent]) -> StreamDone:
        """Consume a background turn's event bus and return its terminal
        ``StreamDone`` (reply + session id). Used by the non-streaming landing
        chat and fork, which need the whole reply rather than an SSE relay.
        Raises `TurnFailed` on a ``StreamError`` and `TurnEndedWithoutReply` if
        the turn ends without a terminal event. Reading the session id off the
        done event (not the store) avoids racing the on_complete persist
        callback."""
        async for _seq, event in bus.subscribe(after_seq=0):
            if isinstance(event, StreamDone):
                return event
            if isinstance(event, StreamError):
                raise TurnFailed(event.detail)
        raise TurnEndedWithoutReply()

    # --- lifecycle ---------------------------------------------------------

    def cancel(self, agent_id: str) -> None:
        """Cancel the agent's in-flight run (the chat stop button, or the
        orchestrator's ``cancel_agent`` tool). Truly aborts the backend turn -
        the supervisor cancels the run task, whose drain aclose()s the backend
        stream so its cleanup runs (e.g. the Claude subprocess is killed). The
        persist callback then records a CANCELLED terminal outcome. Works for the
        orchestrator too (it is an agent in the registry keyed ORCHESTRATOR_ID).
        Raises `NoActiveRun` when there is nothing live to stop.

        A concurrent turn is refused elsewhere, so at most one run is live per
        agent - the single registry entry is the one to stop.
        """
        run_id = self._agent_runs.get(agent_id)
        if run_id is None or not self._supervisor.cancel(run_id):
            raise NoActiveRun()

    def status(self, agent: AgentRecord) -> TurnStatus:
        """Merge the live supervisor run-state with the backend's read-only
        rollout/session progress for the agent."""
        run_id = self._agent_runs.get(agent.id)
        run_state = self._supervisor.status(run_id) if run_id else None
        state = run_state.state if run_state is not None else agent.state
        progress = get_backend(agent.backend).read_status(
            self._settings, agent.session_id
        )
        # Expose the in-flight prompt only while the run is live, steering stripped
        # the same way read_transcript strips it, so a mid-turn reattach renders a
        # user bubble that matches the post-reload transcript (and its dedup guard).
        prompt: str | None = None
        if (
            run_state is not None
            and run_state.state in (RunPhase.QUEUED, RunPhase.RUNNING)
            and run_state.prompt is not None
        ):
            prompt = strip_steering(run_state.prompt).strip() or None
        return TurnStatus(
            agent_id=agent.id,
            state=state,
            session_id=agent.session_id,
            progress=progress,
            prompt=prompt,
        )

    def bus(self, agent_id: str) -> EventBus[StreamEvent]:
        """The agent's current run event bus, for a live relay. Raises
        `NoActiveRun` when the agent has no live run."""
        run_id = self._agent_runs.get(agent_id)
        bus = self._supervisor.bus(run_id) if run_id else None
        if bus is None:
            raise NoActiveRun()
        return bus

    # --- the sub-agent signals ---------------------------------------------

    def request_input(self, agent: AgentRecord, question: str) -> AgentState:
        """A sub-agent signals it is blocked and needs a decision (BC2). Records a
        WAITING outcome carrying the question, keyed to the agent's CURRENT run so
        the turn-end completion preserves it (see ``AgentStore.request_input`` /
        ``mark_finished``); returns immediately - the agent ends its turn and the
        orchestrator answers later by resuming. Raises `AgentNotFound` for the
        orchestrator, which is not a sub-agent."""
        outcome = self._agents.request_input(
            agent.id,
            question,
            run_id=self._agent_runs.get(agent.id, ""),
            session_id=agent.session_id,
        )
        return outcome.state

    def report_back(self, agent: AgentRecord, summary: str) -> AgentState:
        """A sub-agent signals it has FINISHED its task and hands back a result.
        Records a REPORTED outcome carrying the summary, keyed to the agent's
        CURRENT run so the turn-end completion preserves it (see
        ``AgentStore.report_back`` / ``mark_finished``); returns immediately - the
        agent ends its turn and the orchestrator is woken / sees it in
        `pending_agents`, reads the report and acknowledges (no resume). Raises
        `AgentNotFound` for the orchestrator, which is not a sub-agent."""
        outcome = self._agents.report_back(
            agent.id,
            summary,
            run_id=self._agent_runs.get(agent.id, ""),
            session_id=agent.session_id,
        )
        return outcome.state

    async def acknowledge(self, agent_id: str) -> bool:
        """Mark an agent's pending signal handled (BC3), so it drops out of the
        pending list. Idempotent: False if there was nothing pending (already
        handled, or no outcome) - a cleared or never-seen agent simply acks to
        False rather than being an error."""
        return await asyncio.to_thread(self._agents.acknowledge, agent_id)

    async def awaiting_approval(self, agent: AgentRecord, note: str) -> None:
        """Record the agent as BLOCKED on an operator decision, keyed to its
        current run the way the other signals are."""
        await asyncio.to_thread(
            self._agents.awaiting_approval,
            agent.id,
            note,
            run_id=self._agent_runs.get(agent.id, ""),
            session_id=agent.session_id,
        )

    def fork_seed(
        self,
        agent: AgentRecord,
        session_id: str | None,
        message_index: int,
        text: str,
    ) -> str:
        """The prompt that continues a conversation from an edited message.

        The backend has no native branch, so the turns before the edit are
        pasted as context ahead of the edited text. ``session_id`` is passed
        explicitly because the two forks read different sessions: a project
        agent forks its ONE session, the orchestrator forks whichever of its
        many the operator is looking at.

        Reading the transcript is a store call for a codex session (it pulls the
        reasoning sidecar out of the database), not just file I/O, so loop-thread
        callers offload this.
        """
        messages = get_backend(agent.backend).read_transcript(
            self._settings, session_id
        )
        return format_fork_seed(messages[: max(0, message_index)], text)

    # --- shutdown ----------------------------------------------------------

    async def aclose(self) -> None:
        """Cancel any in-flight runs. Called from the app's lifespan shutdown."""
        await self._supervisor.aclose()


__all__ = ["AgentRunService", "CompletionHook", "TurnStatus"]
