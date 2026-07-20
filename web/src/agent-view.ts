// Agent page: the chat panel plus the agent info (model), tools panel, per-turn
// tool-call chips, and the sidebar's three labeled stat boxes (Sessions / this
// session / account). No import-time side effects (the `agent.ts` entry calls
// `startAgent`); the render helpers are exported so the jsdom tests can drive
// them without fetch.

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
    type ToolCall,
    type TranscriptMessage,
    type UsageQuota,
} from "./common";
import { renderMarkdown } from "./markdown";

// The chat log is driven from this array (the source of truth), so a message
// knows its index - which is what forking (edit a past message -> branch) needs.
// `reply` carries the tool/token meta for assistant turns so it survives a
// re-render. `_editingIndex` is the user message currently open in the inline
// editor; `_currentSessionId` is the session forks branch from.
interface LogEntry {
    role: string;
    text: string;
    reply?: ChatReply;
    ts?: number; // epoch ms when the turn was shown/recorded (for a timestamp)
}
let _messages: LogEntry[] = [];
let _editingIndex: number | null = null;
let _currentSessionId: string | null = null;
// Whether the agent is enabled (gates the onboarding empty state) and whether the
// log is pinned to the bottom (drives the no-yank auto-scroll + "new messages"
// pill). Module state so a re-render can consult them.
let _agentEnabled = false;
let _stickToBottom = true;
// Set while renderLog rebuilds the DOM, so the log's own scroll listener ignores
// the scroll events that clearing/repopulating children emits (they would
// otherwise mis-read the follow state off a transiently-short log).
let _rendering = false;
// How many messages arrived while the user was scrolled up (shown on the "new
// messages" pill), and the message count at the last render (to detect growth).
let _unreadCount = 0;
let _prevMsgCount = 0;

function fmtTokens(n: number): string {
    return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

// Parse an ISO timestamp (from the transcript API) to epoch ms, or undefined.
function parseIso(iso: string | null): number | undefined {
    if (!iso) return undefined;
    const ms = Date.parse(iso);
    return Number.isNaN(ms) ? undefined : ms;
}

// A short clock label for a message ("14:39" same day, "Jul 19, 14:39" older).
// Empty for a missing/unparseable stamp. Also used as the element's title (full).
export function formatTimestamp(ms: number | undefined): string {
    if (ms === undefined || Number.isNaN(ms)) return "";
    const d = new Date(ms);
    const hh = `${d.getHours()}`.padStart(2, "0");
    const mm = `${d.getMinutes()}`.padStart(2, "0");
    const now = new Date();
    const sameDay =
        d.getFullYear() === now.getFullYear() &&
        d.getMonth() === now.getMonth() &&
        d.getDate() === now.getDate();
    if (sameDay) return `${hh}:${mm}`;
    const month = d.toLocaleString(undefined, { month: "short" });
    return `${month} ${d.getDate()}, ${hh}:${mm}`;
}

// A clipboard-copy button that flips its label to "copied" briefly. Guarded:
// `navigator.clipboard` is absent in jsdom and on insecure origins, so it no-ops
// there rather than throwing. `getText` is read lazily at click time.
function copyButton(getText: () => string, cls: string): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = cls;
    btn.textContent = "copy";
    btn.title = "copy to clipboard";
    btn.addEventListener("click", () => {
        const clip = navigator.clipboard;
        if (!clip) return;
        void clip.writeText(getText()).then(
            () => {
                btn.textContent = "copied";
                setTimeout(() => (btn.textContent = "copy"), 1200);
            },
            () => undefined,
        );
    });
    return btn;
}

// Is the log scrolled to (near) its bottom? Within a small slop so a sub-pixel
// gap still counts as "at bottom". In jsdom all three metrics are 0, so this is
// true - fine, the pill is a real-layout concern verified in the browser.
export function isNearBottom(log: HTMLElement): boolean {
    return log.scrollHeight - log.scrollTop - log.clientHeight < 48;
}

function jumpPill(): HTMLElement | null {
    return document.getElementById("chat-jump");
}

// Auto-scroll ONLY when the user is following the bottom; otherwise reveal the
// "new messages" pill instead of yanking them away from scrolled-up history.
// `restoreTop` (the scrollTop captured before a full rebuild) is put back when
// not following, so a renderLog() replaceChildren - which resets scrollTop to 0 -
// does not fling a scrolled-up reader to the TOP of the history.
// Show/hide the "new messages" pill and label it with the unread count (or a
// plain "jump to latest" when the user scrolled up with nothing new since).
function refreshPill(): void {
    const pill = jumpPill();
    if (!pill) return;
    pill.hidden = _stickToBottom;
    pill.textContent =
        _unreadCount > 0
            ? `${_unreadCount} new message${_unreadCount === 1 ? "" : "s"}`
            : "jump to latest";
}

function maybeScroll(log: HTMLElement, restoreTop?: number): void {
    if (_stickToBottom) {
        log.scrollTop = log.scrollHeight;
        _unreadCount = 0;
    } else if (restoreTop !== undefined) {
        log.scrollTop = restoreTop;
    }
    refreshPill();
}

export function _resetAgentState(): void {
    _messages = [];
    _editingIndex = null;
    _currentSessionId = null;
    _stickToBottom = true;
    _unreadCount = 0;
    _prevMsgCount = 0;
}

// The slim chat head: just the model and a compact "N tools" link. The tools
// themselves live on the Settings page (rendered as cards there), so the head no
// longer inlines the raw list - it only shows the count and links across.
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

export function messageMeta(reply: ChatReply): HTMLElement | null {
    const bits: string[] = [];
    if (reply.tool_calls.length > 0) {
        // A clear "ran" label in front of prominent tool chips - tool execution is
        // the point of the agent, so surface it rather than a faint badge.
        bits.push(`<span class="chat__ran">ran</span>`);
        for (const call of reply.tool_calls) {
            bits.push(
                `<span class="chat__chip">${escapeHtml(call.tool)}</span>`,
            );
        }
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

// Rebuild an assistant message's reply meta (tool chips + token count) from a
// reloaded transcript, so switching to a past session shows what it ran - not just
// on the live turn. Returns undefined when there is nothing to show (user turns,
// or an assistant turn with no tools/usage) so `messageMeta` renders no line.
export function transcriptReply(m: TranscriptMessage): ChatReply | undefined {
    if (m.tool_calls.length === 0 && !m.usage) return undefined;
    return { text: m.text, tool_calls: m.tool_calls, usage: m.usage };
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
    maybeScroll(log);
    return msg;
}

function chatLog(): HTMLElement | null {
    return document.getElementById("chat-log");
}

// Example prompts shown in the onboarding empty state; clicking one fills the
// composer (the user still presses send, so nothing fires behind their back).
const EXAMPLE_PROMPTS = [
    "what's using the most CPU right now?",
    "how full are my disks?",
    "which processes are using the most memory?",
];

function fillComposer(text: string): void {
    const input = document.getElementById(
        "chat-input",
    ) as HTMLTextAreaElement | null;
    if (!input || input.disabled) return;
    input.value = text;
    autosizeComposer(input);
    input.focus();
}

// The onboarding empty state: a short welcome + a few clickable example prompts,
// shown in place of a blank log for a fresh conversation.
function renderWelcome(): HTMLElement {
    const wrap = el("div", "chat__welcome");
    wrap.appendChild(
        el("div", "chat__welcome-title", "Ask your scuffed Jarvis"),
    );
    wrap.appendChild(
        el(
            "div",
            "chat__welcome-sub",
            "It can inspect this host and run tools. Try one of these:",
        ),
    );
    const chips = el("div", "chat__examples");
    for (const prompt of EXAMPLE_PROMPTS) {
        const chip = el("button", "chat__example");
        chip.setAttribute("type", "button");
        chip.textContent = prompt; // textContent: fixed strings, but keep it safe
        chip.addEventListener("click", () => fillComposer(prompt));
        chips.appendChild(chip);
    }
    wrap.appendChild(chips);
    // Surface the otherwise-undiscoverable fork feature.
    wrap.appendChild(
        el(
            "div",
            "chat__welcome-hint",
            "Tip: edit one of your messages to branch the conversation from that point.",
        ),
    );
    return wrap;
}

// The per-message footer: a timestamp plus the message's action (copy for an
// assistant reply, edit-to-fork for a user turn), aligned to the message's side.
function messageFoot(entry: LogEntry, index: number): HTMLElement {
    const foot = el("div", `chat__foot chat__foot--${entry.role}`);
    const label = formatTimestamp(entry.ts);
    if (label && entry.ts !== undefined) {
        const time = document.createElement("time");
        time.className = "chat__time";
        time.textContent = label;
        time.dateTime = new Date(entry.ts).toISOString();
        time.title = new Date(entry.ts).toLocaleString();
        foot.appendChild(time);
    }
    if (entry.role === "assistant") {
        foot.appendChild(copyButton(() => entry.text, "chat__copy"));
    } else if (entry.role === "user") {
        const edit = el("button", "chat__edit");
        edit.setAttribute("type", "button");
        edit.textContent = "edit";
        edit.title = "edit this message and branch a new chat";
        edit.addEventListener("click", () => beginEdit(index));
        foot.appendChild(edit);
    }
    return foot;
}

// Rebuild the chat log from `_messages`. A fresh conversation shows an onboarding
// empty state. Each turn gets a footer (timestamp + copy/edit). The rebuild keeps
// the user's scroll position (see maybeScroll): it only sticks to the bottom when
// they were already following it.
function renderLog(): void {
    const log = chatLog();
    if (!log) return;
    // Count messages that arrived while scrolled up (for the pill). `_messages` is
    // stable during a render, so compute the growth once up front; maybeScroll
    // zeroes it again when the user is following the bottom.
    const grew = _messages.length - _prevMsgCount;
    if (grew > 0 && !_stickToBottom) _unreadCount += grew;
    _prevMsgCount = _messages.length;
    // Capture the scroll position BEFORE replaceChildren wipes it, so a rebuild
    // that is not following the bottom can restore the reader's place.
    const prevTop = log.scrollTop;
    _rendering = true;
    log.replaceChildren();

    if (_agentEnabled && _messages.length === 0 && _editingIndex === null) {
        log.appendChild(renderWelcome());
        _rendering = false;
        maybeScroll(log, prevTop);
        return;
    }

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
        log.appendChild(messageFoot(entry, index));
    });
    _rendering = false;
    maybeScroll(log, prevTop);
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
        .concat([{ role: "user", text: trimmed, ts: Date.now() }]);
    _stickToBottom = true;
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
        _messages.push({
            role: "assistant",
            text: data.reply.text || "(no reply)",
            reply: data.reply,
            ts: Date.now(),
        });
        renderLog();
        // The sidebar (context box + account) re-renders from the API, which is
        // the authoritative token source - no separate client-side counter.
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
    messages: { role: string; text: string; ts?: number; reply?: ChatReply }[],
): void {
    _messages = messages.map((m) => ({
        role: m.role,
        text: m.text,
        ts: m.ts,
        reply: m.reply,
    }));
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
        // The title truncates with an ellipsis; a native tooltip reveals the full
        // text on hover. Set as a property (no attribute escaping needed).
        open.title = session.title;
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

function usageRow(label: string, value: string, tip?: string): HTMLElement {
    const row = el(
        "div",
        "usage-block__row",
        `<span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span>`,
    );
    // A one-line explanation on hover (native title tooltip) - the stats are
    // jargon otherwise. Escaping is not needed for the title property (set as a
    // string, not innerHTML), but the values here are all fixed literals anyway.
    if (tip) row.title = tip;
    return row;
}

// A labeled box heading with an optional hover explanation.
function blockHead(text: string, tip?: string): HTMLElement {
    const head = el("div", "usage-block__head", escapeHtml(text));
    if (tip) head.title = tip;
    return head;
}

// A small muted footnote inside a stat box (e.g. data freshness).
function blockHint(text: string): HTMLElement {
    return el("div", "usage-block__hint", escapeHtml(text));
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
    panel.appendChild(
        blockHead(
            "this session",
            "The active conversation's token footprint and activity.",
        ),
    );
    panel.appendChild(usageBar(usedPct));
    panel.appendChild(
        usageRow(
            `${fmtTokens(ctx.input_tokens)} / ${fmtTokens(ctx.context_window)}`,
            `${usedPct.toFixed(0)}%`,
            "How full the model's context window is: last turn's input tokens vs the window size.",
        ),
    );
    panel.appendChild(
        usageRow(
            "cached",
            fmtTokens(ctx.cached_input_tokens),
            "Input tokens served from the prompt cache (cheaper, faster).",
        ),
    );
    panel.appendChild(
        usageRow(
            "output",
            fmtTokens(ctx.output_tokens + ctx.reasoning_output_tokens),
            "Total tokens the model generated this session (reply + reasoning).",
        ),
    );
    panel.appendChild(
        usageRow(
            "turns / tools",
            `${ctx.turn_count} / ${ctx.tool_call_count}`,
            "Number of exchanges and tool calls in this conversation.",
        ),
    );
    panel.appendChild(blockHint("as of last turn"));
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
    const windowLabel =
        primary.window_minutes >= 10080 ? "weekly" : "rate limit";
    meter.appendChild(
        blockHead(
            "account",
            "Your ChatGPT subscription's usage against its rate-limit window.",
        ),
    );
    meter.appendChild(usageBar(primary.used_percent));
    meter.appendChild(
        usageRow(
            `used (${windowLabel})`,
            `${primary.used_percent.toFixed(0)}%`,
            "Percentage of the subscription quota consumed in the current window.",
        ),
    );
    meter.appendChild(
        usageRow(
            "resets",
            resetsIn(primary.resets_at),
            "Time until the quota window rolls over.",
        ),
    );
    if (usage.plan_type)
        meter.appendChild(
            usageRow(
                "plan",
                usage.plan_type,
                "Your ChatGPT subscription tier.",
            ),
        );
    if (usage.secondary) {
        meter.appendChild(
            usageRow(
                "secondary",
                `${usage.secondary.used_percent.toFixed(0)}% · ${resetsIn(usage.secondary.resets_at)}`,
                "A second rate-limit window (e.g. a shorter burst limit).",
            ),
        );
    }
    meter.appendChild(blockHint("as of last turn"));
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
        _editingIndex = null;
        _currentSessionId = id;
        const data = await fetchJson<{ messages: TranscriptMessage[] }>(
            `/api/agent/session/${encodeURIComponent(id)}`,
        );
        _messages = data.messages.map((m) => ({
            role: m.role,
            text: m.text,
            ts: parseIso(m.ts),
            reply: transcriptReply(m),
        }));
        _stickToBottom = true;
        renderLog();
        await refreshSidebar();
    } catch (err: unknown) {
        console.error(err);
    }
}

async function newChat(): Promise<void> {
    _resetAgentState();
    renderLog();
    // A11y: move focus to the composer so a keyboard user can start typing.
    (
        document.getElementById("chat-input") as HTMLTextAreaElement | null
    )?.focus();
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

// A composer slash-command. codex itself has no slash commands, so this is a
// client-side palette: `run` either performs an action, navigates, or fills the
// composer with a prompt the user can then send.
export interface SlashCommand {
    name: string;
    description: string;
    run: () => void;
}

// Render the tracked conversation as markdown, for `/export` download.
export function chatMarkdown(
    messages: { role: string; text: string }[],
): string {
    return messages
        .map((m) => `**${m.role}**\n\n${m.text}`)
        .join("\n\n---\n\n");
}

function exportChat(): void {
    const text = chatMarkdown(_messages);
    if (!text) return;
    try {
        const blob = new Blob([text], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "scufris-chat.md";
        a.click();
        URL.revokeObjectURL(url);
    } catch {
        // Blob/URL are absent in some environments (e.g. jsdom) - no-op there.
    }
}

function showCommandHelp(): void {
    const log = chatLog();
    if (!log) return;
    const lines = SLASH_COMMANDS.map((c) => `/${c.name} - ${c.description}`);
    appendMessage(log, "system", `Commands:\n${lines.join("\n")}`);
}

export const SLASH_COMMANDS: SlashCommand[] = [
    { name: "new", description: "start a new chat", run: () => void newChat() },
    {
        name: "settings",
        description: "open the settings page",
        run: () => window.location.assign("/settings/"),
    },
    {
        name: "tasks",
        description: "list open tatr tasks",
        run: () => fillComposer("List my open tatr tasks."),
    },
    {
        name: "host",
        description: "summarize this host",
        run: () =>
            fillComposer(
                "Give me a quick summary of this host: CPU, memory, disks.",
            ),
    },
    {
        name: "export",
        description: "download this chat as markdown",
        run: () => exportChat(),
    },
    {
        name: "help",
        description: "list the slash commands",
        run: () => showCommandHelp(),
    },
];

// Commands matching what the user has typed: a lone `/token` at the very start of
// the composer (no space/newline yet - once they type an arg, it is a real prompt).
export function matchSlashCommands(
    value: string,
    commands: SlashCommand[] = SLASH_COMMANDS,
): SlashCommand[] {
    if (!value.startsWith("/")) return [];
    const query = value.slice(1);
    if (/\s/.test(query)) return [];
    return commands.filter((c) => c.name.startsWith(query));
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

// Grow the composer textarea to fit its content up to a max, then let it
// scroll. Reset to `auto` first so it can also SHRINK when text is deleted or
// cleared. jsdom has no layout engine (scrollHeight is 0), so this is a no-op
// under tests and is verified by eye in the served bundle.
const COMPOSER_MAX_HEIGHT = 200;
function autosizeComposer(input: HTMLTextAreaElement): void {
    input.style.height = "auto";
    const next = Math.min(input.scrollHeight, COMPOSER_MAX_HEIGHT);
    input.style.height = `${next}px`;
    input.style.overflowY =
        input.scrollHeight > COMPOSER_MAX_HEIGHT ? "auto" : "hidden";
}

// Run one streaming turn. Handles both backends: `exec` (no deltas -> a
// "working... Ns" indicator + tool line, reply on done) and `app_server` (text
// fills in token-by-token, reasoning streams into a collapsible "thinking"
// section). The markdown re-render is throttled to one animation frame so a fast
// token stream does not thrash the DOM.
function runStreamingTurn(
    message: string,
    log: HTMLElement,
    input: HTMLTextAreaElement,
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
    let lastRender = 0;
    let flushTimer = 0;

    const paintStatus = (): void => {
        const secs = Math.floor((Date.now() - started) / 1000);
        const ran = tools.length ? ` · ran ${tools.join(", ")}` : "";
        const what = streamed ? "streaming" : "working";
        label.textContent = `${what}... ${secs}s${ran}`;
    };
    // Paint the growing answer EAGERLY and synchronously (the first token shows
    // immediately), throttled to ~every 50ms so a fast stream does not thrash the
    // DOM. Deliberately NOT requestAnimationFrame: a single queued rAF can be
    // clobbered by the onDone renderLog (which detaches this bubble) before it
    // ever paints, so a buffered burst would show nothing until the end.
    const renderNow = (): void => {
        window.clearTimeout(flushTimer);
        flushTimer = 0;
        lastRender = Date.now();
        body.replaceChildren(renderMarkdown(streamed));
        maybeScroll(log);
    };
    const scheduleRender = (): void => {
        const since = Date.now() - lastRender;
        if (since >= 50) renderNow();
        else if (!flushTimer)
            flushTimer = window.setTimeout(renderNow, 50 - since);
    };
    paintStatus();
    const timer = window.setInterval(paintStatus, 500);
    const stop = (): void => {
        window.clearInterval(timer);
        window.clearTimeout(flushTimer);
        input.disabled = false;
        autosizeComposer(input);
        input.focus();
        maybeScroll(log);
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
                ts: Date.now(),
            });
            renderLog();
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
    ) as HTMLTextAreaElement | null;
    const log = document.getElementById("chat-log");
    const reset = document.getElementById("chat-reset");
    if (!form || !input || !log || !reset) return;

    if (!config.agent_enabled) {
        _agentEnabled = false;
        appendMessage(
            log,
            "system",
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login`.",
        );
        input.disabled = true;
        return;
    }
    _agentEnabled = true;

    // --- Slash-command palette ---
    const palette = document.getElementById("chat-palette");
    let paletteItems: SlashCommand[] = [];
    let paletteIdx = 0;
    const paletteOpen = (): boolean => palette !== null && !palette.hidden;

    const closePalette = (): void => {
        paletteIdx = 0;
        if (palette) {
            palette.hidden = true;
            palette.replaceChildren();
        }
    };

    const renderPalette = (): void => {
        if (!palette) return;
        paletteItems = matchSlashCommands(input.value);
        if (paletteItems.length === 0) {
            closePalette();
            return;
        }
        if (paletteIdx >= paletteItems.length) paletteIdx = 0;
        palette.replaceChildren();
        paletteItems.forEach((cmd, i) => {
            const item = el(
                "div",
                `chat__palette-item${i === paletteIdx ? " is-active" : ""}`,
                `<span class="chat__palette-name">/${escapeHtml(cmd.name)}</span>` +
                    `<span class="chat__palette-desc">${escapeHtml(cmd.description)}</span>`,
            );
            item.setAttribute("role", "option");
            item.setAttribute(
                "aria-selected",
                i === paletteIdx ? "true" : "false",
            );
            // mousedown (not click) so it fires before the textarea blurs.
            item.addEventListener("mousedown", (event) => {
                event.preventDefault();
                runCommand(cmd);
            });
            palette.appendChild(item);
        });
        palette.hidden = false;
    };

    const runCommand = (cmd: SlashCommand): void => {
        input.value = "";
        autosizeComposer(input);
        closePalette();
        cmd.run();
    };

    const submit = (): void => {
        if (input.disabled) return;
        const message = input.value.trim();
        if (!message) return;
        closePalette();
        _editingIndex = null;
        _messages.push({ role: "user", text: message, ts: Date.now() });
        _stickToBottom = true; // the user just acted; follow their new turn down
        renderLog();
        input.value = "";
        autosizeComposer(input);
        input.disabled = true;
        runStreamingTurn(message, log, input);
    };

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        submit();
    });

    input.addEventListener("keydown", (event) => {
        // When the command palette is open, the arrow keys, Enter/Tab (accept) and
        // Escape (dismiss) drive it instead of the composer.
        if (paletteOpen()) {
            if (event.key === "ArrowDown") {
                event.preventDefault();
                paletteIdx = (paletteIdx + 1) % paletteItems.length;
                renderPalette();
                return;
            }
            if (event.key === "ArrowUp") {
                event.preventDefault();
                paletteIdx =
                    (paletteIdx - 1 + paletteItems.length) %
                    paletteItems.length;
                renderPalette();
                return;
            }
            if (event.key === "Enter" || event.key === "Tab") {
                event.preventDefault();
                runCommand(paletteItems[paletteIdx]);
                return;
            }
            if (event.key === "Escape") {
                event.preventDefault();
                closePalette();
                return;
            }
        }
        // Enter sends; Shift+Enter inserts a newline (the textarea's default). Guard
        // `isComposing` so committing an IME candidate with Enter does not fire a
        // half-typed message.
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            submit();
        }
    });
    input.addEventListener("input", () => {
        autosizeComposer(input);
        renderPalette();
    });
    input.addEventListener("blur", () => closePalette());

    // No-yank scrolling: track whether the user is following the bottom, and let
    // the "new messages" pill jump them back down on demand.
    log.addEventListener("scroll", () => {
        if (_rendering) return;
        _stickToBottom = isNearBottom(log);
        if (_stickToBottom) _unreadCount = 0;
        refreshPill();
    });
    const pill = jumpPill();
    if (pill)
        pill.addEventListener("click", () => {
            _stickToBottom = true;
            _unreadCount = 0;
            log.scrollTop = log.scrollHeight;
            refreshPill();
        });

    reset.addEventListener("click", () => void newChat());

    // Paint the onboarding empty state on load and focus the composer (a11y).
    renderLog();
    input.focus();
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
