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
    type SessionContext,
    type SessionInfo,
    type SessionsResponse,
    type TokenUsage,
    type TranscriptMessage,
    type UsageQuota,
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
        const row = el("div", "session");
        if (session.id === currentId) row.classList.add("is-active");

        const open = el("button", "session__open");
        open.setAttribute("type", "button");
        open.innerHTML =
            `<span class="session__title">${escapeHtml(session.title)}</span>` +
            `<span class="session__time">${escapeHtml(relativeTime(session.updated_at))}</span>`;
        open.addEventListener("click", () => void switchSession(session.id));

        const del = el("button", "session__del");
        del.setAttribute("type", "button");
        del.setAttribute("aria-label", "delete conversation");
        del.title = "delete";
        del.textContent = "×";
        del.addEventListener("click", (event) => {
            event.stopPropagation();
            void deleteSession(session.id, session.title);
        });

        row.appendChild(open);
        row.appendChild(del);
        list.appendChild(row);
    }
}

async function deleteSession(id: string, title: string): Promise<void> {
    if (!window.confirm(`Delete conversation "${title}"?`)) return;
    try {
        const res = await fetch(
            `/api/agent/session/${encodeURIComponent(id)}`,
            { method: "DELETE" },
        );
        if (res.ok) {
            const data = (await res.json()) as { current: string | null };
            // If the active conversation was the one deleted, clear the chat.
            if (data.current === null) {
                const log = document.getElementById("chat-log");
                if (log) log.replaceChildren();
                _resetAgentState();
            }
        }
        await refreshSidebar();
    } catch (err: unknown) {
        console.error(err);
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

function usageBar(percent: number): HTMLElement {
    const wrap = el("div", "bar");
    const fill = el("div", "bar__fill");
    fill.style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
    wrap.appendChild(fill);
    return wrap;
}

function usageRow(label: string, value: string): HTMLElement {
    return el(
        "div",
        "usage-block__row",
        `<span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span>`,
    );
}

// A coarse "2d 5h" countdown to a unix reset time; "-" when unknown.
function resetsIn(resetsAt: number | null): string {
    if (!resetsAt) return "-";
    const secs = resetsAt - Date.now() / 1000;
    if (secs <= 0) return "now";
    const days = Math.floor(secs / 86400);
    const hours = Math.floor((secs % 86400) / 3600);
    if (days > 0) return `${days}d ${hours}h`;
    const mins = Math.floor((secs % 3600) / 60);
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
}

// The current session's context usage. NOT a per-component breakdown (codex does
// not expose that) - the real axes it gives: window used, token mix, turn/tool
// counts. Hidden when there is no active session.
export function renderContext(ctx: SessionContext | null): void {
    const panel = document.getElementById("context-panel");
    if (!panel) return;
    if (!ctx || ctx.context_window <= 0) {
        panel.replaceChildren();
        panel.hidden = true;
        return;
    }
    panel.hidden = false;
    const usedPct = (ctx.input_tokens / ctx.context_window) * 100;
    panel.replaceChildren();
    panel.appendChild(el("div", "usage-block__head", "context"));
    panel.appendChild(usageBar(usedPct));
    panel.appendChild(
        usageRow(
            `${fmtTokens(ctx.input_tokens)} / ${fmtTokens(ctx.context_window)}`,
            `${usedPct.toFixed(0)}%`,
        ),
    );
    panel.appendChild(usageRow("cached", fmtTokens(ctx.cached_input_tokens)));
    panel.appendChild(
        usageRow(
            "output",
            fmtTokens(ctx.output_tokens + ctx.reasoning_output_tokens),
        ),
    );
    panel.appendChild(
        usageRow("turns / tools", `${ctx.turn_count} / ${ctx.tool_call_count}`),
    );
}

// The account's subscription usage (the weekly rate-limit window). Hidden when
// codex has not reported a limit yet.
export function renderUsage(usage: UsageQuota | null): void {
    const meter = document.getElementById("usage-meter");
    if (!meter) return;
    const primary = usage?.primary ?? null;
    if (!usage || !primary) {
        meter.replaceChildren();
        meter.hidden = true;
        return;
    }
    meter.hidden = false;
    meter.replaceChildren();
    const head = primary.window_minutes >= 10080 ? "weekly usage" : "usage";
    meter.appendChild(el("div", "usage-block__head", head));
    meter.appendChild(usageBar(primary.used_percent));
    meter.appendChild(usageRow("used", `${primary.used_percent.toFixed(0)}%`));
    meter.appendChild(usageRow("resets", resetsIn(primary.resets_at)));
    if (usage.plan_type) meter.appendChild(usageRow("plan", usage.plan_type));
    if (usage.secondary) {
        meter.appendChild(
            usageRow(
                "secondary",
                `${usage.secondary.used_percent.toFixed(0)}% · ${resetsIn(usage.secondary.resets_at)}`,
            ),
        );
    }
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
        renderUsage(await fetchJson<UsageQuota | null>("/api/agent/usage"));
    } catch (err: unknown) {
        console.error(err);
    }
}

// Refresh the whole sidebar (list + current-session context + account usage).
async function refreshSidebar(): Promise<void> {
    await Promise.all([loadSessions(), loadContext(), loadUsage()]);
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
        await refreshSidebar();
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
        await refreshSidebar();
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
                // A turn creates/updates the session and moves usage; refresh the
                // list, the context block and the weekly meter.
                void refreshSidebar();
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
        await refreshSidebar();
    } catch (err: unknown) {
        console.error(err);
    }
}
