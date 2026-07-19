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
    type StreamEvent,
    type TokenUsage,
    type ToolCall,
    type TranscriptMessage,
    type UsageQuota,
} from "./common";
import { renderMarkdown } from "./markdown";

// Session usage, persisted across turns; reset on "new chat".
let _cumulativeOutput = 0;
let _lastContext = 0;

// The chat log is driven from this array (the source of truth), so a message
// knows its index - which is what forking (edit a past message -> branch) needs.
// `reply` carries the tool/token meta for assistant turns so it survives a
// re-render. `_editingIndex` is the user message currently open in the inline
// editor; `_currentSessionId` is the session forks branch from.
interface LogEntry {
    role: string;
    text: string;
    reply?: ChatReply;
}
let _messages: LogEntry[] = [];
let _editingIndex: number | null = null;
let _currentSessionId: string | null = null;

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

// Reset only the cumulative token/context indicator (a fresh conversation's
// running total). Does NOT touch the message log - callers manage that.
function resetUsage(): void {
    _cumulativeOutput = 0;
    _lastContext = 0;
    applyUsage(null);
}

export function _resetAgentState(): void {
    resetUsage();
    _messages = [];
    _editingIndex = null;
    _currentSessionId = null;
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

// Append a transient bubble NOT tracked in `_messages` (the pending "..." and
// error/system lines). Tracked history goes through `_messages` + `renderLog`.
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

function chatLog(): HTMLElement | null {
    return document.getElementById("chat-log");
}

// Rebuild the chat log from `_messages`. User turns get an "edit" affordance (to
// fork); the one being edited renders an inline editor instead. Assistant turns
// with a stored reply re-render their tool/token meta line.
function renderLog(): void {
    const log = chatLog();
    if (!log) return;
    log.replaceChildren();
    _messages.forEach((entry, index) => {
        if (entry.role === "user" && index === _editingIndex) {
            log.appendChild(editorFor(index, entry.text));
            return;
        }
        const msg = el("div", `chat__msg chat__msg--${entry.role}`);
        if (entry.role === "assistant") {
            // Model output is untrusted; renderMarkdown builds the DOM safely
            // (no innerHTML). The modifier switches off pre-wrap so prose is not
            // double-spaced (the code block's <pre> preserves its own layout).
            msg.classList.add("chat__msg--md");
            msg.appendChild(renderMarkdown(entry.text));
        } else {
            msg.textContent = entry.text;
        }
        log.appendChild(msg);
        if (entry.role === "assistant" && entry.reply) {
            const meta = messageMeta(entry.reply);
            if (meta) log.appendChild(meta);
        }
        if (entry.role === "user") {
            const edit = el("button", "chat__edit");
            edit.setAttribute("type", "button");
            edit.textContent = "edit";
            edit.title = "edit this message and branch a new chat";
            edit.addEventListener("click", () => beginEdit(index));
            log.appendChild(edit);
        }
    });
    log.scrollTop = log.scrollHeight;
}

function editorFor(index: number, text: string): HTMLElement {
    const box = el("div", "chat__editor");
    const area = document.createElement("textarea");
    area.className = "chat__editor-input";
    area.value = text;
    const actions = el("div", "chat__editor-actions");
    const save = el("button", "chat__send");
    save.setAttribute("type", "button");
    save.textContent = "fork";
    save.addEventListener("click", () => void forkFrom(index, area.value));
    const cancel = el("button", "chat__reset");
    cancel.setAttribute("type", "button");
    cancel.textContent = "cancel";
    cancel.addEventListener("click", () => {
        _editingIndex = null;
        renderLog();
    });
    actions.appendChild(save);
    actions.appendChild(cancel);
    box.appendChild(area);
    box.appendChild(actions);
    return box;
}

function beginEdit(index: number): void {
    _editingIndex = index;
    renderLog();
}

// Fork: keep the turns BEFORE the edited message, replace it with the edit, and
// run that as a fresh session's first turn (the backend pastes the prior context
// since codex-exec has no native branch).
async function forkFrom(index: number, text: string): Promise<void> {
    const trimmed = text.trim();
    _editingIndex = null;
    if (!trimmed || !_currentSessionId) {
        renderLog();
        return;
    }
    _messages = _messages
        .slice(0, index)
        .concat([{ role: "user", text: trimmed }]);
    renderLog();
    const log = chatLog();
    const pending = log ? appendMessage(log, "assistant", "...") : null;
    try {
        const res = await fetch("/api/agent/session/fork", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_id: _currentSessionId,
                message_index: index,
                text: trimmed,
            }),
        });
        if (!res.ok) throw new Error(`fork failed (${String(res.status)})`);
        const data = (await res.json()) as {
            current: string | null;
            reply: ChatReply;
        };
        _currentSessionId = data.current;
        resetUsage();
        _messages.push({
            role: "assistant",
            text: data.reply.text || "(no reply)",
            reply: data.reply,
        });
        renderLog();
        applyUsage(data.reply.usage);
        await refreshSidebar();
    } catch (err: unknown) {
        if (pending) {
            pending.classList.add("chat__msg--error");
            pending.textContent = err instanceof Error ? err.message : "error";
        }
    }
}

// Test hook: drive the chat log directly without fetch.
export function _renderChatForTest(
    messages: { role: string; text: string }[],
): void {
    _messages = messages.map((m) => ({ role: m.role, text: m.text }));
    _editingIndex = null;
    renderLog();
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
                _messages = [];
                _editingIndex = null;
                resetUsage();
                renderLog();
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
        // Keep the fork source id in sync with the backend's current session
        // (set after the first live turn, a switch, or a fork).
        _currentSessionId = data.current;
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
    try {
        await fetch("/api/agent/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "switch", session_id: id }),
        });
        resetUsage();
        _editingIndex = null;
        _currentSessionId = id;
        const data = await fetchJson<{ messages: TranscriptMessage[] }>(
            `/api/agent/session/${encodeURIComponent(id)}`,
        );
        _messages = data.messages.map((m) => ({ role: m.role, text: m.text }));
        renderLog();
        await refreshSidebar();
    } catch (err: unknown) {
        console.error(err);
    }
}

async function newChat(): Promise<void> {
    _resetAgentState();
    renderLog();
    try {
        await fetch("/api/agent/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "new" }),
        });
    } finally {
        await refreshSidebar();
    }
}

// Parse whatever complete SSE frames are in `buffer`, returning the events and
// the unconsumed remainder (a partial frame carried to the next chunk). Pure.
export function parseSseFrames(buffer: string): {
    events: StreamEvent[];
    rest: string;
} {
    const events: StreamEvent[] = [];
    let rest = buffer;
    let sep = rest.indexOf("\n\n");
    while (sep !== -1) {
        const frame = rest.slice(0, sep);
        rest = rest.slice(sep + 2);
        const dataLine = frame
            .split("\n")
            .find((line) => line.startsWith("data:"));
        if (dataLine) {
            try {
                events.push(
                    JSON.parse(dataLine.slice(5).trim()) as StreamEvent,
                );
            } catch {
                // ignore a malformed frame
            }
        }
        sep = rest.indexOf("\n\n");
    }
    return { events, rest };
}

interface StreamHandlers {
    onTool: (tool: ToolCall) => void;
    onDone: (reply: ChatReply) => void;
    onError: (detail: string) => void;
    onTextDelta?: (delta: string) => void;
    onReasoningDelta?: (delta: string) => void;
}

// POST a message and consume the SSE turn-progress stream, dispatching events.
export async function sendChatStream(
    message: string,
    handlers: StreamHandlers,
): Promise<void> {
    const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
    });
    if (!resp.ok || !resp.body) {
        handlers.onError(`chat failed (${String(resp.status)})`);
        return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseFrames(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) {
            if (event.kind === "tool") handlers.onTool(event.tool);
            else if (event.kind === "done") handlers.onDone(event.reply);
            else if (event.kind === "error") handlers.onError(event.detail);
            else if (event.kind === "text_delta")
                handlers.onTextDelta?.(event.delta);
            else if (event.kind === "reasoning_delta")
                handlers.onReasoningDelta?.(event.delta);
            // unknown kinds are ignored, not treated as errors
        }
    }
}

// Run one streaming turn. Handles both backends: `exec` (no deltas -> a
// "working... Ns" indicator + tool line, reply on done) and `app_server` (text
// fills in token-by-token, reasoning streams into a collapsible "thinking"
// section). The markdown re-render is throttled to one animation frame so a fast
// token stream does not thrash the DOM.
function runStreamingTurn(
    message: string,
    log: HTMLElement,
    input: HTMLInputElement,
): void {
    const pending = appendMessage(log, "assistant", "");
    pending.classList.add("chat__msg--pending");

    const status = el("div", "chat__status");
    const spinner = el("span", "chat__spinner");
    const label = el("span", "chat__pending-label");
    status.append(spinner, label);
    const thinking = el("details", "chat__thinking");
    const thinkingSummary = el("summary", "", "thinking");
    const thinkingBody = el("div", "chat__thinking-body");
    thinking.append(thinkingSummary, thinkingBody);
    thinking.hidden = true;
    const body = el("div", "chat__stream-body");
    pending.append(status, thinking, body);

    const started = Date.now();
    const tools: string[] = [];
    let streamed = "";
    let reasoning = "";
    let renderQueued = false;

    const paintStatus = (): void => {
        const secs = Math.floor((Date.now() - started) / 1000);
        const ran = tools.length ? ` · ran ${tools.join(", ")}` : "";
        const what = streamed ? "streaming" : "working";
        label.textContent = `${what}... ${secs}s${ran}`;
    };
    const scheduleRender = (): void => {
        if (renderQueued) return;
        renderQueued = true;
        requestAnimationFrame(() => {
            renderQueued = false;
            body.replaceChildren(renderMarkdown(streamed));
            log.scrollTop = log.scrollHeight;
        });
    };
    paintStatus();
    const timer = window.setInterval(paintStatus, 500);
    const stop = (): void => {
        window.clearInterval(timer);
        input.disabled = false;
        input.focus();
        log.scrollTop = log.scrollHeight;
    };
    const fail = (detail: string): void => {
        pending.classList.remove("chat__msg--pending");
        pending.classList.add("chat__msg--error");
        pending.replaceChildren();
        pending.textContent = detail;
        stop();
    };

    void sendChatStream(message, {
        onTextDelta: (delta) => {
            streamed += delta;
            pending.classList.add("chat__msg--md");
            scheduleRender();
        },
        onReasoningDelta: (delta) => {
            reasoning += delta;
            thinking.hidden = false;
            thinkingBody.textContent = reasoning;
        },
        onTool: (tool) => {
            tools.push(tool.tool);
            paintStatus();
        },
        onDone: (reply) => {
            _messages.push({
                role: "assistant",
                text: reply.text || streamed || "(no reply)",
                reply,
            });
            renderLog();
            applyUsage(reply.usage);
            void refreshSidebar();
            stop();
        },
        onError: fail,
    }).catch((err: unknown) => {
        fail(err instanceof Error ? err.message : "error");
    });
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
        _editingIndex = null;
        _messages.push({ role: "user", text: message });
        renderLog();
        input.value = "";
        input.disabled = true;
        runStreamingTurn(message, log, input);
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
