// Agent page: the chat panel plus the agent info (model), tools panel, per-turn
// tool-call chips and a running token/context indicator. No import-time side
// effects (the `agent.ts` entry calls `startAgent`); the render helpers are
// exported so the jsdom tests can drive them without fetch.

import {
    el,
    escapeHtml,
    fetchJson,
    loadConfig,
    type AgentInfo,
    type AgentTool,
    type AppConfig,
    type ChatReply,
    type SessionInfo,
    type SessionsResponse,
    type TokenUsage,
    type TranscriptMessage,
} from "./common";

// Session usage, persisted across turns; reset on "new chat".
let _cumulativeOutput = 0;
let _lastContext = 0;

function fmtTokens(n: number): string {
    return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

export function applyUsage(usage: TokenUsage | null): void {
    if (usage) {
        _cumulativeOutput += usage.output_tokens;
        _lastContext = usage.input_tokens;
    }
    const usageEl = document.getElementById("agent-usage");
    if (!usageEl) return;
    usageEl.textContent =
        _cumulativeOutput > 0 || _lastContext > 0
            ? `ctx ${fmtTokens(_lastContext)} · ${fmtTokens(_cumulativeOutput)} out`
            : "";
}

export function _resetAgentState(): void {
    _cumulativeOutput = 0;
    _lastContext = 0;
    applyUsage(null);
}

export function renderAgentPanel(
    info: AgentInfo | null,
    tools: AgentTool[],
): void {
    const modelEl = document.getElementById("agent-model");
    if (modelEl) modelEl.textContent = info ? `model ${info.model}` : "";

    const toggle = document.getElementById(
        "agent-tools-toggle",
    ) as HTMLButtonElement | null;
    const panel = document.getElementById("agent-tools");
    if (!toggle || !panel) return;

    if (tools.length === 0) {
        toggle.hidden = true;
        return;
    }
    toggle.hidden = false;
    toggle.textContent = `tools (${tools.length})`;
    panel.replaceChildren();
    for (const tool of tools) {
        const item = el("div", "agent-tools__item");
        item.innerHTML =
            `<span class="agent-tools__name">${escapeHtml(tool.name)}</span>` +
            `<span class="agent-tools__desc">${escapeHtml(tool.description)}</span>`;
        panel.appendChild(item);
    }
    toggle.onclick = () => {
        panel.hidden = !panel.hidden;
    };
}

export function messageMeta(reply: ChatReply): HTMLElement | null {
    const bits: string[] = [];
    for (const call of reply.tool_calls) {
        bits.push(`<span class="chat__chip">${escapeHtml(call.tool)}</span>`);
    }
    if (reply.usage) {
        bits.push(
            `<span class="chat__tok">${reply.usage.output_tokens} tok</span>`,
        );
    }
    if (bits.length === 0) return null;
    const meta = el("div", "chat__meta");
    meta.innerHTML = bits.join("");
    return meta;
}

function appendMessage(
    log: HTMLElement,
    role: string,
    text: string,
): HTMLElement {
    const msg = el("div", `chat__msg chat__msg--${role}`);
    msg.textContent = text;
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
    return msg;
}

// A coarse "2h ago" label for the session list; empty for an unparseable stamp.
function relativeTime(iso: string | null): string {
    if (!iso) return "";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return "";
    const secs = Math.max(0, (Date.now() - then) / 1000);
    if (secs < 60) return "just now";
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

// Render the sidebar session list, highlighting the current one. Titles come
// from user messages, so they are escaped. Clicking an item switches sessions.
export function renderSessions(
    sessions: SessionInfo[],
    currentId: string | null,
): void {
    const list = document.getElementById("session-list");
    if (!list) return;
    list.replaceChildren();
    if (sessions.length === 0) {
        list.appendChild(el("div", "sidebar__empty", "no sessions yet"));
        return;
    }
    for (const session of sessions) {
        const item = el("button", "session");
        item.setAttribute("type", "button");
        if (session.id === currentId) item.classList.add("is-active");
        item.innerHTML =
            `<span class="session__title">${escapeHtml(session.title)}</span>` +
            `<span class="session__time">${escapeHtml(relativeTime(session.updated_at))}</span>`;
        item.addEventListener("click", () => void switchSession(session.id));
        list.appendChild(item);
    }
}

async function loadSessions(): Promise<void> {
    try {
        const data = await fetchJson<SessionsResponse>("/api/agent/sessions");
        renderSessions(data.sessions, data.current);
    } catch (err: unknown) {
        console.error(err);
    }
}

async function switchSession(id: string): Promise<void> {
    const log = document.getElementById("chat-log");
    try {
        await fetch("/api/agent/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "switch", session_id: id }),
        });
        _resetAgentState();
        if (log) {
            const data = await fetchJson<{ messages: TranscriptMessage[] }>(
                `/api/agent/session/${encodeURIComponent(id)}`,
            );
            log.replaceChildren();
            for (const message of data.messages) {
                appendMessage(
                    log,
                    message.role === "user" ? "user" : "assistant",
                    message.text,
                );
            }
        }
        await loadSessions();
    } catch (err: unknown) {
        console.error(err);
    }
}

async function newChat(): Promise<void> {
    const log = document.getElementById("chat-log");
    _resetAgentState();
    try {
        await fetch("/api/agent/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "new" }),
        });
    } finally {
        if (log) log.replaceChildren();
        await loadSessions();
    }
}

async function sendChat(message: string): Promise<ChatReply> {
    const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
    });
    if (!resp.ok) {
        const detail = (await resp.json().catch(() => null)) as {
            detail?: string;
        } | null;
        throw new Error(
            detail?.detail || `chat failed (${String(resp.status)})`,
        );
    }
    return (await resp.json()) as ChatReply;
}

export function initChat(config: AppConfig): void {
    const form = document.getElementById("chat-form") as HTMLFormElement | null;
    const input = document.getElementById(
        "chat-input",
    ) as HTMLInputElement | null;
    const log = document.getElementById("chat-log");
    const reset = document.getElementById("chat-reset");
    if (!form || !input || !log || !reset) return;

    if (!config.agent_enabled) {
        appendMessage(
            log,
            "system",
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login`.",
        );
        input.disabled = true;
        return;
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const message = input.value.trim();
        if (!message) return;
        appendMessage(log, "user", message);
        input.value = "";
        input.disabled = true;
        const pending = appendMessage(log, "assistant", "...");
        sendChat(message)
            .then((reply) => {
                pending.textContent = reply.text || "(no reply)";
                const meta = messageMeta(reply);
                if (meta) pending.after(meta);
                applyUsage(reply.usage);
                // A first turn creates (and titles) a session; refresh the list
                // so it appears and stays highlighted.
                void loadSessions();
            })
            .catch((err: unknown) => {
                pending.classList.add("chat__msg--error");
                pending.textContent =
                    err instanceof Error ? err.message : "error";
            })
            .finally(() => {
                input.disabled = false;
                input.focus();
                log.scrollTop = log.scrollHeight;
            });
    });

    reset.addEventListener("click", () => void newChat());
}

export async function startAgent(): Promise<void> {
    const config = await loadConfig();
    initChat(config);
    if (!config.agent_enabled) return;
    try {
        const [info, tools] = await Promise.all([
            fetchJson<AgentInfo>("/api/agent/info"),
            fetchJson<AgentTool[]>("/api/agent/tools"),
        ]);
        renderAgentPanel(info, tools);
        await loadSessions();
    } catch (err: unknown) {
        console.error(err);
    }
}
