// The read-only CARDS the agent settings page composes: the key/value `panel`
// primitive and one builder per card (session/status, account, usage, memory,
// sessions, the project capability lists). Split out of `agent-settings-view`
// so that module is the PAGE - fetch, form, layout - and this one is what the
// operator reads. Every function here is pure: data in, a detached element out.

import { authLabel, el, escapeHtml, formatBytes } from "./common";
import type {
    AccountInfo,
    AgentRunStatus,
    Capability,
    MemoryFootprint,
    ProjectCapabilities,
    SessionsResponse,
    UsageQuota,
} from "./agent-types";
import { fmtTokens } from "./chat-format";

// A read-only key/value panel; a null value shows a dash so a panel never looks
// broken. `title` and string values are escaped.
export function panel(
    title: string,
    rows: [string, string | null][],
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", escapeHtml(title)));
    for (const [key, value] of rows) {
        card.appendChild(
            el(
                "div",
                "settings__row",
                `<span class="settings__key">${escapeHtml(key)}</span>` +
                    `<span class="settings__val">${escapeHtml(value ?? "-")}</span>`,
            ),
        );
    }
    return card;
}

// A read-only list card for a project's discovered capabilities (skills or
// custom tools): each row is name + a detail line, an empty list becomes an
// explicit "none" note (via `panel`) so the surface is always transparent.
// `meta`, when present, is appended after the description (used for a tool's
// transport kind). All values are escaped.
export function capabilityPanel(
    title: string,
    emptyNote: string,
    rows: { name: string; description: string; meta?: string }[],
): HTMLElement {
    if (rows.length === 0) {
        return panel(title.toLowerCase(), [["available", emptyNote]]);
    }
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", `${title} (${rows.length})`));
    for (const row of rows) {
        const detail = row.meta
            ? `${escapeHtml(row.description)} <span class="settings__key">${escapeHtml(row.meta)}</span>`
            : escapeHtml(row.description);
        card.appendChild(
            el(
                "div",
                "settings__row",
                `<span class="settings__key">${escapeHtml(row.name)}</span>` +
                    `<span class="settings__val">${detail}</span>`,
            ),
        );
    }
    return card;
}

// The two read-only cards for what an agent's PROJECT defines: its skills and
// its custom tools/MCP servers. Rendered only for a project agent (capabilities
// non-null); the orchestrator / a project-less agent passes null and gets
// neither card. Each list shows an explicit empty state rather than vanishing.
export function projectCapabilityCards(
    caps: ProjectCapabilities,
): HTMLElement[] {
    return [
        capabilityPanel(
            "Project skills",
            "none (this project defines no skills)",
            caps.skills.map((s) => ({
                name: s.name,
                description: s.description,
            })),
        ),
        capabilityPanel(
            "Project tools",
            "none (this project defines no tools)",
            caps.tools.map((t) => ({
                name: t.name,
                description: t.description,
                meta: t.kind || undefined,
            })),
        ),
    ];
}

// A coarse "2d 5h" countdown to a unix reset time; "-" when unknown.
export function resetsIn(resetsAt: number | null): string {
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

// The live status + context-window panel, from the agent's /status. A never-run
// agent (idle + no session) shows "not started" rather than a bare idle/0/0.
export function statusPanel(status: AgentRunStatus | null): HTMLElement {
    if (status === null) return panel("this session", [["state", null]]);
    if (status.state === "idle" && !status.session_id) {
        return panel("this session", [["state", "not started"]]);
    }
    const rows: [string, string | null][] = [
        ["state", status.state],
        ["turns / tools", `${status.turns} / ${status.tool_calls}`],
    ];
    if (status.context_window > 0) {
        const usedPct = (status.input_tokens / status.context_window) * 100;
        rows.push([
            `${fmtTokens(status.input_tokens)} / ${fmtTokens(status.context_window)}`,
            `${usedPct.toFixed(0)}%`,
        ]);
        rows.push(["output", fmtTokens(status.output_tokens)]);
    }
    return panel("this session", rows);
}

// What a panel prints in place of a measurement it does not have: a backend with
// no such reader is not a reader that found nothing, so the two must not read
// alike. Null - a failed fetch, with no envelope at all - is neither, and stays
// the bare dash. This is the web half of the one three-state vocabulary;
// `scufris/telegram/text.py` carries the Python half and `scufris/README.md`
// states it once for both.
export function capabilityText<T>(
    cap: Capability<T> | null,
    backend: string,
): string | null {
    if (!cap) return null;
    return cap.supported
        ? "nothing reported yet"
        : `not reported by the ${backend} backend`;
}

export function usagePanel(
    usage: Capability<UsageQuota> | null,
    backend: string,
): HTMLElement {
    const quota = usage?.value ?? null;
    const primary = quota?.primary ?? null;
    if (!quota || !primary) {
        return panel("account usage", [
            ["quota", capabilityText(usage, backend)],
        ]);
    }
    const windowLabel =
        primary.window_minutes >= 10080 ? "weekly" : "rate limit";
    const rows: [string, string | null][] = [
        [`used (${windowLabel})`, `${primary.used_percent.toFixed(0)}%`],
        ["resets", resetsIn(primary.resets_at)],
    ];
    if (quota.plan_type) rows.push(["plan", quota.plan_type]);
    if (quota.secondary) {
        rows.push([
            "secondary",
            `${quota.secondary.used_percent.toFixed(0)}% · ${resetsIn(quota.secondary.resets_at)}`,
        ]);
    }
    return panel("account usage", rows);
}

export function memoryPanel(
    memory: Capability<MemoryFootprint> | null,
    backend: string,
): HTMLElement {
    const footprint = memory?.value ?? null;
    if (!footprint) {
        return panel("on-disk memory", [
            ["sessions", capabilityText(memory, backend)],
        ]);
    }
    const rows: [string, string | null][] = [
        ["sessions", String(footprint.session_count)],
        ["size", formatBytes(footprint.total_bytes)],
    ];
    return panel("on-disk memory", rows);
}

// The orchestrator's multi-session overview (it alone runs several chats). A
// read-only count + the current session's title, with a link to the landing chat
// where the switcher lives - the settings page does not switch sessions itself.
export function sessionsPanel(sessions: SessionsResponse): HTMLElement {
    const current =
        sessions.sessions.find((s) => s.id === sessions.current) ?? null;
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Sessions"));
    for (const [key, value] of [
        ["count", String(sessions.sessions.length)],
        ["current", current ? current.title : "-"],
    ] as [string, string][]) {
        card.appendChild(
            el(
                "div",
                "settings__row",
                `<span class="settings__key">${escapeHtml(key)}</span>` +
                    `<span class="settings__val">${escapeHtml(value)}</span>`,
            ),
        );
    }
    const link = document.createElement("a");
    link.href = "/";
    link.className = "settings__note settings__notelink";
    link.textContent = "switch or start sessions on the chat ->";
    card.appendChild(link);
    return card;
}

export function accountPanel(account: AccountInfo | null): HTMLElement {
    if (!account) return panel("account", [["model", null]]);
    return panel("account", [
        ["model", account.model],
        ["auth", authLabel(account.auth_mode)],
        ["enabled", account.enabled ? "yes" : "no"],
    ]);
}
