// The orchestrator landing entry. The landing IS the orchestrator's chat page, so
// it mounts the shared chat component (agent-chat-view) on `#agent-chat` and wires
// it to the orchestrator's endpoints, PLUS the orchestrator-only sidebar: the
// session switcher and the context/usage stat boxes (chat-sidebar). A project
// agent's detail page reuses the same component without any of this session UI.
//
// Fork here is the multi-session semantics: editing a past message branches a NEW
// session (POST /api/agent/session/fork) and keeps the original - the injected
// `forkTurn` below. The render helpers stay exported and pure so jsdom drives the
// panel without fetch; `startAgent` does the fetching + wiring.

import { apiFetch, fetchJson, loadConfig } from "./common";
import {
    type AgentInfo,
    type AgentRunStatus,
    type AgentTool,
    type Capability,
    type ChatReply,
    type SessionContext,
    type SessionsResponse,
    type TranscriptMessage,
    type UsageQuota,
} from "./agent-types";
import {
    streamChatTurn,
    subscribeEvents,
    type StreamHandlers,
} from "./chat-stream";
import { parseIso } from "./chat-format";
import { createAgentChat } from "./agent-chat-view";
import { transcriptReply } from "./agent-chat-log";
import type { ChatMsg } from "./agent-chat-types";
import { renderContext, renderSessions, renderUsage } from "./chat-sidebar";

// Example prompts shown in the onboarding empty state; clicking one fills the
// composer (the user still presses send, so nothing fires behind their back).
const EXAMPLE_PROMPTS = [
    "what's using the most CPU right now?",
    "how full are my disks?",
    "which processes are using the most memory?",
];

// The slim chat head (in the sidebar): the model and a compact "N tools" link. The
// tools themselves live on the Settings page, so the head only shows the count.
export function renderAgentPanel(
    info: AgentInfo | null,
    tools: AgentTool[],
): void {
    const modelEl = document.getElementById("agent-model");
    if (modelEl) modelEl.textContent = info ? `model ${info.model}` : "";

    const link = document.getElementById(
        "agent-tools-link",
    ) as HTMLAnchorElement | null;
    if (!link) return;
    if (tools.length === 0) {
        link.hidden = true;
        return;
    }
    link.hidden = false;
    link.textContent = `${tools.length} tool${tools.length === 1 ? "" : "s"}`;
}

export async function startAgent(): Promise<void> {
    const root = document.getElementById("agent-chat");
    if (!root) return;
    const config = await loadConfig();

    // The session forks branch from, kept in sync with the backend's current
    // session (set after the first live turn, a switch, a fork, or a delete).
    let currentSessionId: string | null = null;
    const enc = (id: string): string => encodeURIComponent(id);

    async function loadSessions(): Promise<void> {
        try {
            const data = await fetchJson<SessionsResponse>(
                "/api/agent/sessions",
            );
            currentSessionId = data.current;
            renderSessions(data.sessions, data.current, {
                onOpen: (id) => void switchSession(id),
                onDelete: (id, title) => void deleteSession(id, title),
            });
        } catch (err: unknown) {
            console.error(err);
        }
    }

    // Shared transcript -> ChatMsg mapping (switch, and the mount-time auto-open).
    const toChatMsgs = (messages: TranscriptMessage[]): ChatMsg[] =>
        messages.map((m) => ({
            role: m.role === "assistant" ? "assistant" : "user",
            text: m.text,
            ts: parseIso(m.ts),
            reply: transcriptReply(m),
            // Re-hydrate the reloaded turn's "thinking" spoiler (renderChatLog
            // renders it collapsed, identical to the live/settled turn).
            reasoning: m.reasoning ?? undefined,
        }));

    // The mount-time transcript: open the CURRENT session (set at turn-start by
    // the backend, so a mid-turn refresh has one) instead of the welcome state.
    // Sets `currentSessionId` so fork/switch work without waiting for the sidebar.
    async function loadCurrentTranscript(): Promise<ChatMsg[]> {
        const data = await fetchJson<SessionsResponse>("/api/agent/sessions");
        currentSessionId = data.current;
        if (!data.current) return [];
        const t = await fetchJson<{ messages: TranscriptMessage[] }>(
            `/api/agent/session/${enc(data.current)}`,
        );
        return toChatMsgs(t.messages);
    }

    // Follow an in-flight orchestrator turn (one started in another tab, or before
    // this reload) off its run bus, so a mid-turn refresh keeps streaming instead
    // of freezing on the settled transcript. Mirrors the per-agent reattach
    // (startAgentChat): gate on the run being live, inject the driving prompt
    // (Q1-A) as a user bubble, then subscribe. 404/idle -> no-op.
    async function reattachOrchestrator(
        handlers: StreamHandlers,
    ): Promise<void> {
        let status: AgentRunStatus;
        try {
            status = await fetchJson<AgentRunStatus>(
                "/api/agents/orchestrator/status",
            );
        } catch {
            return;
        }
        if (status.state !== "running" && status.state !== "queued") return;
        if (status.prompt) handlers.onUserPrompt?.(status.prompt);
        await subscribeEvents("/api/agents/orchestrator/events", handlers);
    }

    async function loadContext(): Promise<void> {
        try {
            renderContext(
                await fetchJson<SessionContext | null>("/api/agent/context"),
            );
        } catch (err: unknown) {
            console.error(err);
        }
    }

    async function loadUsage(): Promise<void> {
        try {
            // Capability envelope: unwrapped to the value here, so the meter
            // keeps rendering a plain reading and an unsupported backend renders
            // as absent (renderUsage hides the meter on null) rather than a zero.
            const quota =
                await fetchJson<Capability<UsageQuota>>("/api/agent/usage");
            renderUsage(quota.value);
        } catch (err: unknown) {
            console.error(err);
        }
    }

    // Refresh the whole sidebar (list + current-session context + account usage).
    async function refreshSidebar(): Promise<void> {
        await Promise.all([loadSessions(), loadContext(), loadUsage()]);
    }

    async function switchSession(id: string): Promise<void> {
        try {
            await apiFetch("/api/agent/session", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "switch", session_id: id }),
            });
            currentSessionId = id;
            const data = await fetchJson<{ messages: TranscriptMessage[] }>(
                `/api/agent/session/${enc(id)}`,
            );
            control.setMessages(toChatMsgs(data.messages));
            await refreshSidebar();
        } catch (err: unknown) {
            console.error(err);
        }
    }

    async function newChat(): Promise<void> {
        control.reset();
        try {
            await apiFetch("/api/agent/session", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "new" }),
            });
        } finally {
            await refreshSidebar();
        }
    }

    async function deleteSession(id: string, title: string): Promise<void> {
        if (!window.confirm(`Delete conversation "${title}"?`)) return;
        try {
            const res = await apiFetch(`/api/agent/session/${enc(id)}`, {
                method: "DELETE",
            });
            if (res.ok) {
                const data = (await res.json()) as { current: string | null };
                // If the active conversation was deleted, clear the chat.
                if (data.current === null) control.reset();
            }
            await refreshSidebar();
        } catch (err: unknown) {
            console.error(err);
        }
    }

    const control = createAgentChat(root, {
        exportTitle: "Orchestrator chat",
        exportFilename: "scufris-orchestrator-chat.md",
        enableImage: true,
        forkVerb: "fork",
        forkHint:
            "Editing branches a NEW conversation from here; the original session is kept.",
        welcome: config.agent_enabled
            ? { examples: EXAMPLE_PROMPTS, forkHint: true }
            : undefined,
        disabledReason: config.agent_enabled
            ? undefined
            : "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login`.",
        // Open the current session on mount (so a mid-turn refresh shows the
        // conversation, not the welcome state), then reattach to any in-flight
        // orchestrator turn so it keeps streaming. An empty current -> welcome.
        loadTranscript: loadCurrentTranscript,
        reattach: reattachOrchestrator,
        streamTurn: (message, handlers, image, signal) =>
            streamChatTurn(
                "/api/chat/stream",
                message,
                handlers,
                image,
                signal,
            ),
        // Stop the in-flight orchestrator turn. The landing is the orchestrator's
        // chat, and the orchestrator is an agent in the run registry (id
        // "orchestrator"), so the same per-agent cancel endpoint stops it. Ignore a
        // 404 (nothing running) - the local settle already kept the partial.
        cancelTurn: async () => {
            try {
                await apiFetch("/api/agents/orchestrator/cancel", {
                    method: "POST",
                });
            } catch {
                // best-effort: the local settle already kept the partial
            }
        },
        forkTurn: async (index, text, handlers, signal) => {
            if (!currentSessionId) {
                handlers.onError("no active session to fork");
                return;
            }
            try {
                const res = await apiFetch("/api/agent/session/fork", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        source_id: currentSessionId,
                        message_index: index,
                        text,
                    }),
                    signal,
                });
                if (!res.ok) {
                    handlers.onError(`fork failed (${String(res.status)})`);
                    return;
                }
                const data = (await res.json()) as {
                    current: string | null;
                    reply: ChatReply;
                };
                currentSessionId = data.current;
                handlers.onDone(data.reply);
            } catch (err: unknown) {
                handlers.onError(err instanceof Error ? err.message : "error");
            }
        },
        onAfterTurn: () => void refreshSidebar(),
    });

    control.setSlashCommands([
        {
            name: "new",
            description: "start a new chat",
            run: () => void newChat(),
        },
        {
            name: "settings",
            description: "open the settings page",
            run: () => window.location.assign("/settings/"),
        },
        {
            name: "tasks",
            description: "list open tatr tasks",
            run: () => control.fillComposer("List my open tatr tasks."),
        },
        {
            name: "host",
            description: "summarize this host",
            run: () =>
                control.fillComposer(
                    "Give me a quick summary of this host: CPU, memory, disks.",
                ),
        },
        {
            name: "export",
            description: "download this chat as markdown",
            run: () => control.exportChat(),
        },
        {
            name: "help",
            description: "list the slash commands",
            run: () =>
                control.note(
                    "Commands: /new, /settings, /tasks, /host, /export, /help",
                ),
        },
    ]);

    document
        .getElementById("chat-reset")
        ?.addEventListener("click", () => void newChat());

    if (!config.agent_enabled) return;
    try {
        const [info, tools] = await Promise.all([
            fetchJson<AgentInfo>("/api/agent/info"),
            fetchJson<AgentTool[]>("/api/agent/tools"),
        ]);
        renderAgentPanel(info, tools);
        await refreshSidebar();
    } catch (err: unknown) {
        console.error(err);
    }
}
