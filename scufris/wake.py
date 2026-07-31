"""The wake bridge: grant the turn-based orchestrator a turn when a sub-agent
needs it.

The orchestrator is a turn-based process; nothing can push an unsolicited message
into a running turn, and `_launch_agent_turn` refuses a second turn while one is
active (409) and reserves the orchestrator's serialize key internally. So this
bridge does NOT hold that key: it observes RUN COMPLETIONS and, when a sub-agent
finished awaiting a decision (a ``WAITING`` outcome from ``request_input``) or
errored, GRANTS the orchestrator a turn via an injected ``launch`` callback with
the question(s) in the prompt.

Deferral + batching: if the orchestrator is mid-turn the wake is kept in a pending
map, not dropped; every completion - including the orchestrator's OWN turn ending -
drains the map, so a deferred wake fires as soon as the orchestrator goes idle, and
several completions that pile up while it is busy fold into ONE wake turn.

Everything here runs on the event loop (the supervisor's ``on_complete`` callback),
serially, so the pending map needs no lock and the is-busy -> launch window cannot
interleave. Config-gated by ``settings.auto_wake`` (off by default); when off the
orchestrator polls ``pending_agents`` (BC3) instead.
"""

from __future__ import annotations

from typing import Callable

from .agent_store import ORCHESTRATOR_ID, AgentStore
from .config import Settings
from .enums import AgentState


def wake_prompt(batch: dict[str, tuple[AgentState, str]]) -> str:
    """The turn prompt injected when waking the orchestrator, listing each
    sub-agent that needs it, its state and its question / result / last message.
    The state tells you what to do: a ``waiting`` agent needs a decision and a
    resume; a ``reported`` agent has FINISHED and only needs its report read; an
    ``error`` agent crashed."""
    lines = ["[wake] One or more sub-agents need your attention:"]
    for agent_id, (state, message) in batch.items():
        lines.append(f"- {agent_id} ({state}): {message}")
    lines.append(
        "Call pending_agents to see the full list. For a 'waiting' or 'error' agent, "
        "answer it with message_agent(agent_id, reply) to resume its session; a "
        "'reported' agent has finished, so just read its report - no resume needed. "
        "Then call acknowledge(agent_id) so it stops pending."
    )
    return "\n".join(lines)


class WakeBridge:
    """Wakes the orchestrator on a sub-agent needs-input/error completion (BC4).

    ``is_orchestrator_busy`` reports whether the orchestrator has a queued/running
    turn; ``launch`` grants it one turn with the given prompt and returns True, or
    False if it turned out to be busy (a 409 race). Neither is allowed to hold the
    orchestrator's serialize key - the bridge fires from the completion callback,
    after the finishing run has released its key.
    """

    def __init__(
        self,
        *,
        agents: AgentStore,
        settings: Settings,
        is_orchestrator_busy: Callable[[], bool],
        launch: Callable[[str], bool],
    ) -> None:
        self._agents = agents
        self._settings = settings
        self._is_busy = is_orchestrator_busy
        self._launch = launch
        # agent_id -> (state, its question / result / last message), awaiting a wake.
        self._pending: dict[str, tuple[AgentState, str]] = {}

    def on_run_complete(self, agent_id: str) -> None:
        """Call after a run's terminal outcome is persisted. Enqueues a sub-agent
        that needs input, reported its result, or errored, then drains - ANY
        completion (the orchestrator's own turn ending included) is a chance to fire
        deferred wakes. A no-op when ``auto_wake`` is off."""
        if not self._settings.auto_wake:
            return
        if agent_id != ORCHESTRATOR_ID:
            outcome = self._agents.outcome(agent_id)
            if (
                outcome is not None
                and not outcome.acknowledged
                and outcome.state
                in (AgentState.WAITING, AgentState.REPORTED, AgentState.ERROR)
            ):
                self._pending[agent_id] = (
                    outcome.state,
                    outcome.message or f"(agent {agent_id} {outcome.state})",
                )
        self._drain()

    def _drain(self) -> None:
        # No await here: on the single event loop this runs to completion before
        # any other completion, so the pending map is race-free and the
        # is-busy -> launch window cannot interleave.
        if not self._pending or self._is_busy():
            return
        batch = dict(self._pending)
        if self._launch(wake_prompt(batch)):
            for agent_id in batch:
                self._pending.pop(agent_id, None)
        # else: the orchestrator became busy (409 race); keep the batch pending and
        # a later completion drains it again.
