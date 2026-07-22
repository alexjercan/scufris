// The ONE per-agent settings PAGE (served for /agents/<id>/settings by the shared
// detail shell). It renders any agent's settings the same way - orchestrator or
// project agent - composing the shared editable field controls (agent-fields),
// the Health card (reused from settings-view), and the detailed panels
// (status/context + usage/memory/account) fed by the per-agent endpoints. This
// replaces the old per-agent settings MODAL. `renderAgentSettings` is PURE and
// `createAgentSettings` takes INJECTED deps, so jsdom drives it without fetch.

import { el, escapeHtml, fetchJson, formatBytes, sendJson } from "./common";
import type {
    AccountInfo,
    Agent,
    AgentHealth,
    AgentRunStatus,
    BackendOption,
    MemoryFootprint,
    Project,
    UsageQuota,
} from "./common";
import { agentFields } from "./agent-fields";
import type { AgentFieldValues } from "./agent-fields";
import { renderHealthCard } from "./settings-view";
import { fmtTokens } from "./chat-format";
import { agentIdFromPath } from "./agent-detail-view";

// Everything the page renders, fetched up front so the render is pure. `agent`
// null means no such agent; the panel fields are best-effort (null when a source
// failed or the backend has no such data - e.g. a claude agent has no usage).
export interface AgentSettingsData {
    agent: Agent | null;
    project: Project | null;
    backends: BackendOption[];
    health: AgentHealth | null;
    status: AgentRunStatus | null;
    usage: UsageQuota | null;
    memory: MemoryFootprint | null;
    account: AccountInfo | null;
}

// The API seam. `startAgentSettings` wires these to the real endpoints; tests
// pass fakes. `writable` gates the editable form vs a read-only view.
export interface AgentSettingsDeps {
    load: () => Promise<AgentSettingsData>;
    save: (fields: AgentFieldValues) => Promise<void>;
    writable: boolean;
}

function backLink(agentId: string): HTMLElement {
    const a = document.createElement("a");
    a.href = `/agents/${encodeURIComponent(agentId)}`;
    a.className = "agents__back";
    a.textContent = "← back to chat";
    return a;
}

// A read-only key/value panel; a null value shows a dash so a panel never looks
// broken. `title` and string values are escaped.
function panel(title: string, rows: [string, string | null][]): HTMLElement {
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

// The live status + context-window panel, from the agent's /status. A never-run
// agent (idle + no session) shows "not started" rather than a bare idle/0/0.
function statusPanel(status: AgentRunStatus | null): HTMLElement {
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

function usagePanel(usage: UsageQuota | null): HTMLElement {
    const primary = usage?.primary ?? null;
    if (!usage || !primary) {
        return panel("account usage", [["quota", null]]);
    }
    const windowLabel =
        primary.window_minutes >= 10080 ? "weekly" : "rate limit";
    const rows: [string, string | null][] = [
        [`used (${windowLabel})`, `${primary.used_percent.toFixed(0)}%`],
        ["resets", resetsIn(primary.resets_at)],
    ];
    if (usage.plan_type) rows.push(["plan", usage.plan_type]);
    if (usage.secondary) {
        rows.push([
            "secondary",
            `${usage.secondary.used_percent.toFixed(0)}% · ${resetsIn(usage.secondary.resets_at)}`,
        ]);
    }
    return panel("account usage", rows);
}

function memoryPanel(memory: MemoryFootprint | null): HTMLElement {
    if (!memory) return panel("on-disk memory", [["sessions", null]]);
    const rows: [string, string | null][] = [
        ["sessions", String(memory.session_count)],
        ["size", formatBytes(memory.total_bytes)],
    ];
    return panel("on-disk memory", rows);
}

function accountPanel(account: AccountInfo | null): HTMLElement {
    if (!account) return panel("account", [["model", null]]);
    return panel("account", [
        ["model", account.model],
        ["auth", account.auth_mode],
        ["enabled", account.enabled ? "yes" : "no"],
    ]);
}

// The editable field form (agent-fields), prefilled with the agent's values, that
// PATCHes on submit. The project is fixed after creation, so it is a read-only row.
function settingsForm(
    agent: Agent,
    project: Project | null,
    backends: BackendOption[],
    deps: AgentSettingsDeps,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Settings"));
    card.appendChild(
        el(
            "div",
            "settings__row",
            `<span class="settings__key">project</span>` +
                `<span class="settings__val">${escapeHtml(project ? project.name : agent.project_id || "server dir")}</span>`,
        ),
    );

    const form = document.createElement("form");
    form.className = "settings__addserver";
    const fields = agentFields("agent settings", backends, {
        name: agent.name,
        backend: agent.backend,
        model: agent.model,
        description: agent.description,
        permission_mode: agent.permission_mode,
    });
    form.append(
        fields.name,
        fields.backend,
        fields.model,
        fields.modelList,
        fields.description,
        fields.mode,
    );
    const save = document.createElement("button");
    save.type = "submit";
    save.className = "settings__btn";
    save.textContent = "save settings";
    form.appendChild(save);

    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const values = fields.read();
        if (!values.name) return;
        void (async () => {
            try {
                await deps.save(values);
            } catch (err: unknown) {
                window.alert(err instanceof Error ? err.message : String(err));
            }
        })();
    });
    card.appendChild(form);
    return card;
}

// A read-only view of the editable fields (a non-writable server).
function readonlySettings(agent: Agent, project: Project | null): HTMLElement {
    return panel("Settings", [
        ["project", project ? project.name : agent.project_id || "server dir"],
        ["backend", agent.backend],
        ["model", agent.model],
        ["permission mode", agent.permission_mode],
        ["description", agent.description || "-"],
    ]);
}

// Render the whole page (PURE). `deps.writable` chooses the editable form vs a
// read-only view; the health + panels always show.
export function renderAgentSettings(
    root: HTMLElement,
    data: AgentSettingsData,
    deps: AgentSettingsDeps,
): void {
    root.replaceChildren();
    if (data.agent === null) {
        root.appendChild(el("div", "settings__empty", "no such agent."));
        return;
    }
    const agent = data.agent;
    root.appendChild(backLink(agent.id));
    root.appendChild(el("h1", "settings__title", escapeHtml(agent.name)));

    root.appendChild(
        deps.writable
            ? settingsForm(agent, data.project, data.backends, deps)
            : readonlySettings(agent, data.project),
    );
    if (data.health) root.appendChild(renderHealthCard(data.health));
    root.appendChild(statusPanel(data.status));
    root.appendChild(accountPanel(data.account));
    root.appendChild(usagePanel(data.usage));
    root.appendChild(memoryPanel(data.memory));
}

// Load the data, render, and re-render after a save.
export function createAgentSettings(
    root: HTMLElement,
    deps: AgentSettingsDeps,
): void {
    const wrapped: AgentSettingsDeps = {
        writable: deps.writable,
        load: deps.load,
        save: async (fields) => {
            await deps.save(fields);
            await refresh();
        },
    };
    const refresh = async (): Promise<void> => {
        const data = await deps.load();
        renderAgentSettings(root, data, wrapped);
    };
    void refresh();
}

// Best-effort fetch: a panel's data failing must not blank the whole page.
function maybe<T>(url: string): Promise<T | null> {
    return fetchJson<T>(url).catch(() => null);
}

export function startAgentSettings(): void {
    const root = document.getElementById("agent-settings");
    if (!root) return;
    const id = agentIdFromPath(window.location.pathname);
    if (!id) return;
    const enc = encodeURIComponent(id);

    createAgentSettings(root, {
        // A read-only server (no writable config) is reported by the backends
        // fetch failing on write, but the simplest signal is the agent config's
        // own writability - the per-agent PATCH 403s. We render the form
        // optimistically and surface a 403 on save.
        writable: true,
        load: async (): Promise<AgentSettingsData> => {
            const agent = await maybe<Agent>(`/api/agents/${enc}`);
            const [project, backends, health, status, usage, memory, account] =
                await Promise.all([
                    (async (): Promise<Project | null> => {
                        if (!agent?.project_id) return null;
                        const projects =
                            (await maybe<Project[]>("/api/projects")) ?? [];
                        return (
                            projects.find((p) => p.id === agent.project_id) ??
                            null
                        );
                    })(),
                    maybe<BackendOption[]>("/api/agents/backends"),
                    maybe<AgentHealth>("/api/agent/health"),
                    maybe<AgentRunStatus>(`/api/agents/${enc}/status`),
                    maybe<UsageQuota>(`/api/agents/${enc}/usage`),
                    maybe<MemoryFootprint>(`/api/agents/${enc}/memory`),
                    maybe<AccountInfo>(`/api/agents/${enc}/account`),
                ]);
            return {
                agent,
                project,
                backends: backends ?? [],
                health,
                status,
                usage,
                memory,
                account,
            };
        },
        save: (fields) =>
            sendJson<Agent>(`/api/agents/${enc}`, "PATCH", fields).then(
                () => undefined,
            ),
    });
}
