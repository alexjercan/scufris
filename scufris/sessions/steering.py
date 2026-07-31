"""The steering preambles the agent prepends to a turn, and their inverse.

Codex obeys only the TURN PROMPT for tool CHOICE: softer channels (tool
descriptions, an instructions file, an AGENTS.md) were probed and do not steer
it. Every clause below therefore rides the prompt, and every tool name and
signature in one is copied verbatim from the server that registers it, so a
renamed tool shows up as a diff here rather than as a model that stops calling
it.

Each preamble is ONE sentinel-wrapped block: ``strip_steering`` removes only the
single leading block, so a second block would survive uncleaned and reach the
user. This module owns both the format and its inverse so the two cannot drift.
"""

from __future__ import annotations

import re

_STEER_OPEN = "[scufris-tools]"
_STEER_CLOSE = "[/scufris-tools]"

# Prefer the curated read-only host tools over the shell. Orchestrator-only: the
# host-tools scufris server is registered only on the orchestrator's turn.
_HOST_TOOLS_CLAUSE = (
    'This host runs a "scufris" MCP server with curated READ-ONLY host tools: '
    "host_stats, disk_usage, list_processes, host_units, host_failed_units, "
    "host_unit_status, host_journal, host_storage, host_largest_directories, "
    "host_reclaimable_space, host_network, host_thermal, host_what_provides, "
    "host_generation_diff, host_flake_status. For questions about "
    "this host (CPU, memory, swap, disks, network, GPUs, load, processes, uptime, "
    "units, logs, temperatures, packages, generations) call these FIRST and answer "
    "from them; do "
    "NOT use shell commands like uname, lscpu, df, free, top, ps, nvidia-smi, "
    "systemctl, journalctl or read "
    "/proc for information those tools provide. "
    "Only fall back to the shell when no scufris tool covers it."
)
# The host-change clause. The mutating host tools are NOT on this audience's
# servers at all, so the orchestrator cannot propose a change even if it tries -
# and a model that does not know that tries the shell instead, which is the
# failure this clause prevents. The host agent is a RESERVED agent that always
# exists (`enums.HOST_AGENT_ID`), so this never needs create_agent.
_HOST_CHANGE_CLAUSE = (
    "CHANGING this host - restarting/stopping/starting a unit, freeing disk space, "
    "activating or rolling back a NixOS configuration - is NOT yours to do and NOT "
    'something the shell can do here (it needs root). Delegate it to the "host" '
    "agent, which always exists and carries the host action tools: call "
    'run_agent("host", goal) with what the operator wants changed (or '
    'message_agent("host", message) to continue with it), then follow it with '
    'agent_status("host") and pending_agents. That agent PROPOSES the change and '
    "the operator approves it; neither you nor it can approve one, so never tell "
    "the operator a host change has been made until the agent reports the applied "
    "result."
)
# The comms clause closes the request_input round-trip on the poll path (auto_wake
# is off by default), so a sub-agent that signalled does not wait forever. The tools
# are orchestrator-only (message_agent / pending_agents / acknowledge); the
# sub-agent side is steered by AGENT_STEERING_PREAMBLE to call request_input.
_COMMS_CLAUSE = (
    "You may have launched sub-agents. At the END of a turn, call pending_agents to "
    "find any that need you (they called request_input and are waiting, or errored); "
    "for each, answer it with message_agent(agent_id, message) - this resumes its "
    "session with your reply - and then call acknowledge(agent_id) so it stops "
    "pending. Do not leave a waiting sub-agent unanswered."
)
# The journal clause steers the orchestrator to the operator's the-den journal +
# macros tools for plain-language journal facts. The meal chain is exact:
# macros_lookup("egg 2p") returns "egg 2pc,12,0,10", which IS the CSV row
# journal_add_macros(row) accepts, so the two chain with no reshaping. These tools
# live on the orchestrator-only den server, so this clause rides only
# STEERING_PREAMBLE.
_JOURNAL_CLAUSE = (
    "The operator keeps a daily journal (the-den) reachable through scufris tools: "
    "journal_show reads the day (tasks, habits, macros, weight); journal_add_task, "
    "journal_complete_task, journal_remove_task, journal_toggle_habit, "
    "journal_log_weight, journal_add_note and journal_add_macros write to it. When "
    "the operator states a journal fact in plain language - ate a food, finished or "
    "wants a task, did a habit, a body weight, a note to jot - USE these tools to "
    "record it; do not answer from memory or edit journal files. To LOG A MEAL "
    '("I ate 2 eggs" / "log that I had 2 eggs"), first call macros_lookup(query) '
    'with the food plus amount (e.g. "egg 2p"); it returns a '
    '"<food> <amount><unit>,<protein>,<carbs>,<fat>" row that is EXACTLY what '
    "journal_add_macros(row) accepts, so pass that row straight through to log it. "
    "If the food is unknown, use macros_search(query) to find the name, or "
    "macros_add_food(row) to add it, before macros_lookup."
)
# The delegation clause steers the orchestrator to spawn and run a sub-agent when
# the operator asks to implement/delegate a task. The permission-mode steer
# matters: create_agent defaults to "manual" (read-only), so an implementing agent
# MUST get "edit"/"auto" or it cannot change anything - the "0 tool calls" failure
# mode. request_input signals from the spawned agent are answered via the comms
# clause above.
_DELEGATION_CLAUSE = (
    "When the operator asks to IMPLEMENT, work, or delegate a task to a codex or "
    'claude agent ("implement task X using codex", "have claude do task Y"), do it '
    "with the agent tools rather than doing the task yourself: call list_projects "
    "to find the project, then reuse a fitting agent from list_agents or "
    "create_agent(name, project_id, backend, permission_mode) with backend set to "
    "the named provider (codex or claude) and a WRITE-capable permission_mode "
    '("edit" or "auto" - the default "manual" is read-only and cannot implement '
    "anything); then run_agent(agent_id, goal) with the task id or path and what to "
    "do as the goal. Follow progress with agent_status(agent_id), and answer the "
    "agent's request_input signals via the pending_agents / message_agent / "
    "acknowledge protocol above."
)
#: The ORCHESTRATOR's preamble: five orchestrator-only clauses in one block.
STEERING_PREAMBLE = (
    f"{_STEER_OPEN}\n{_HOST_TOOLS_CLAUSE}\n{_HOST_CHANGE_CLAUSE}\n{_COMMS_CLAUSE}\n"
    f"{_JOURNAL_CLAUSE}\n{_DELEGATION_CLAUSE}\n{_STEER_CLOSE}"
)
# The HOST agent's preamble: the third audience (`enums.Audience.HOST`), the only
# one whose turn registers the mutating host tools. Its ONE block carries the
# contract clause (propose -> preview -> approve is the normal way to work, not an
# obstacle), the honesty clause (show the preview as written; never claim a change
# happened before the applied result comes back) and the callback clause (it holds
# the same `agent` server as any sub-agent, so it reports back and is resumed the
# same way).
HOST_STEERING_PREAMBLE = (
    f"{_STEER_OPEN}\n"
    "You are the HOST agent for this NixOS machine: you are bound to the box "
    "itself, not to a project, and you hold the host tools. READ the host with "
    "host_stats, disk_usage, list_processes, host_units, host_failed_units, "
    "host_unit_status, host_journal, host_storage, host_largest_directories, "
    "host_reclaimable_space, host_network, host_thermal, host_what_provides, "
    "host_generation_diff and host_flake_status rather than with shell commands "
    "like systemctl, journalctl, df, du or reading /proc - and diagnose from what "
    "they return before proposing anything. "
    "To CHANGE the host, the only route is a proposal the operator approves: call "
    "propose_host_action(action, unit, days, generation) for a unit "
    "start/stop/restart/reload, a store garbage collection, or a generation "
    "rollback, or propose_nixos_change(ref, repo, attr) to build a COMMITTED "
    "configuration and propose activating it. This is normal, not a last resort: "
    "nothing you can run as a shell command will change this machine, because that "
    "needs root and the proposal IS the route to it. "
    "A proposal returns a PREVIEW - what would change, what else it reaches, and "
    "how it can be undone. Show that preview to the operator as it is written "
    "instead of summarising it: the label saying whether it is a simulation or a "
    "statement of current state, and the undo line, are part of the answer. "
    "You cannot approve your own proposal and neither can the orchestrator - only "
    "the operator can, from the dashboard or from Telegram. After proposing, STOP "
    "and let them decide; you will be resumed with the decision, including the "
    "reason if it was denied, and you can then adapt instead of proposing the same "
    "thing again. Read where a proposal stands with host_action_status(action_id) "
    "or nixos_change_status(change_id), and what has already been done to this box "
    "with host_action_audit(limit). Never tell the operator a change has been made "
    "until an applied result says so. "
    "If you are blocked on something the ORCHESTRATOR must decide (not an "
    "approval), call request_input(question) and STOP. When you have finished, call "
    "report_back(summary) with what you did and how it turned out, and STOP, "
    "instead of ending silently.\n"
    f"{_STEER_CLOSE}"
)
# The sub-agent preamble carries three clauses in its ONE block: the WORK clause (its
# job is to carry the assigned task to completion, not narrate a plan and stop - the
# reported 1-turn/0-tool-call failure), the request_input clause (signal when
# blocked) and the report_back clause (signal when finished, so the orchestrator is
# woken / sees the result rather than the agent ending silently). The work clause is
# BACKEND-AGNOSTIC by decision: it gives actionable turn-prompt steps that work on
# codex AND claude, and only MENTIONS the flow skill as an optional aid, because
# codex cannot load a Claude Code skill - leaning on it produced framing text and a
# stop.
AGENT_STEERING_PREAMBLE = (
    f"{_STEER_OPEN}\n"
    "You were launched to CARRY THE ASSIGNED TASK/GOAL TO COMPLETION end-to-end: "
    "understand it, make the actual changes, run the project's checks, and keep "
    "going until it is done - do NOT just describe a plan and stop. If a flow skill "
    "is available to you, use it to structure the work; it is optional, the steps "
    "above stand on their own. "
    "If you are blocked or need a decision or approval you cannot safely make "
    "yourself, call request_input(question) with a clear, specific question and "
    "STOP; do not guess and do not stop silently - the orchestrator will answer and "
    "resume you with the reply in context. "
    "When you have FINISHED the task, call report_back(summary) with a short result "
    "(what you did and how it turned out) and STOP, instead of ending silently - "
    "this hands your result to the orchestrator and lets it know you are done.\n"
    f"{_STEER_CLOSE}"
)

_STEER_RE = re.compile(
    re.escape(_STEER_OPEN) + r".*?" + re.escape(_STEER_CLOSE) + r"\s*",
    re.DOTALL,
)


def strip_steering(text: str) -> str:
    """Remove a leading scufris steering block (see ``STEERING_PREAMBLE``) from a
    recorded user message, so titles and re-rendered transcripts show only what the
    user actually typed. A no-op when the text has no steering block."""
    return _STEER_RE.sub("", text, count=1).lstrip()
