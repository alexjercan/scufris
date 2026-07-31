// The orchestrator landing sidebar renders: the session switcher, the current
// session's context-window usage, and the account's subscription quota. These are
// ORCHESTRATOR-ONLY (a project agent is single-session, so its detail page shows a
// live status box instead - see agent-detail-view). Pure renders that query their
// mount points by id and take action callbacks, so jsdom tests drive them without
// fetch and the wiring (switch/delete) lives in the entry.

import { el, escapeHtml } from "./common";
import type { SessionContext, SessionInfo, UsageQuota } from "./agent-types";
import { fmtTokens, relativeTime } from "./chat-format";

// What clicking a session row does. Wired by the orchestrator entry.
export interface SessionActions {
    onOpen(id: string): void;
    onDelete(id: string, title: string): void;
}

// Render the sidebar session list, highlighting the current one. Titles come
// from user messages, so they are escaped. Clicking an item switches sessions.
export function renderSessions(
    sessions: SessionInfo[],
    currentId: string | null,
    actions: SessionActions,
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
        open.addEventListener("click", () => actions.onOpen(session.id));

        const del = el("button", "session__del");
        del.setAttribute("type", "button");
        del.setAttribute("aria-label", "delete conversation");
        del.title = "delete";
        del.textContent = "×";
        del.addEventListener("click", (event) => {
            event.stopPropagation();
            actions.onDelete(session.id, session.title);
        });

        row.appendChild(open);
        row.appendChild(del);
        list.appendChild(row);
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
