// The ONE per-agent settings PAGE (served for /agents/<id>/settings by the shared
// detail shell). It renders any agent's settings the same way - orchestrator or
// project agent - composing the shared editable field controls (agent-fields),
// the Health card (reused from settings-view), and the detailed panels
// (status/context + usage/memory/account) fed by the per-agent endpoints. This
// replaces the old per-agent settings MODAL. `renderAgentSettings` is PURE and
// `createAgentSettings` takes INJECTED deps, so jsdom drives it without fetch.

import {
    ORCHESTRATOR_ID,
    el,
    escapeHtml,
    fetchJson,
    formatBytes,
    sendJson,
} from "./common";
import type {
    AccountInfo,
    Agent,
    AgentConfig,
    AgentHealth,
    AgentRunStatus,
    AgentTool,
    BackendOption,
    MemoryFootprint,
    ProfilesResponse,
    Project,
    SessionsResponse,
    UsageQuota,
} from "./common";
import { agentFields } from "./agent-fields";
import type { AgentFieldValues } from "./agent-fields";
import {
    renderGlobalConfig,
    renderHealthCard,
    renderProfileSwitcher,
    renderServerControls,
    renderToolControls,
    type SettingsActions,
} from "./settings-view";
import { fmtTokens } from "./chat-format";
import { agentIdFromPath } from "./agent-detail-view";

// The GLOBAL agent config sections (system toggles + MCP servers + tools +
// profiles), shown ONLY on the orchestrator's settings - it is the "system" agent
// and these are shared across all agents. A project agent's `global` is null.
export interface AgentSettingsGlobal {
    config: AgentConfig;
    tools: AgentTool[];
    profiles: ProfilesResponse | null;
    actions: SettingsActions;
}

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
    // Present only for the orchestrator (it alone is multi-session): its list of
    // chat sessions + which is current. A read-only overview here; the actual
    // switch/new/delete lives on the landing chat sidebar. Null for a project
    // agent (single session).
    sessions: SessionsResponse | null;
    // Present only for the orchestrator (the shared/global config sections).
    global: AgentSettingsGlobal | null;
    // Whether the server accepts config writes (SCUFRIS_SETTINGS_WRITABLE). Read
    // from the fetched agent config; false -> the editable form + the global write
    // controls become a read-only view, so a click can never 403.
    writable: boolean;
}

// The API seam. `startAgentSettings` wires these to the real endpoints; tests
// pass fakes. `load` receives the component's `reload` so the global sections'
// actions can re-render after a change.
export interface AgentSettingsDeps {
    load: (reload: () => void) => Promise<AgentSettingsData>;
    save: (fields: AgentFieldValues) => Promise<void>;
}

function backLink(agentId: string): HTMLElement {
    const a = document.createElement("a");
    // The orchestrator is the hidden default; its chat lives at `/`, not under
    // /agents. Link back there so its settings never expose the /agents/... URL.
    a.href =
        agentId === ORCHESTRATOR_ID
            ? "/"
            : `/agents/${encodeURIComponent(agentId)}`;
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

// The orchestrator's multi-session overview (it alone runs several chats). A
// read-only count + the current session's title, with a link to the landing chat
// where the switcher lives - the settings page does not switch sessions itself.
function sessionsPanel(sessions: SessionsResponse): HTMLElement {
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

    if (!data.writable) {
        root.appendChild(
            el(
                "p",
                "settings__note",
                "Read-only server (SCUFRIS_SETTINGS_WRITABLE=0); set it to change config here.",
            ),
        );
    }
    root.appendChild(
        data.writable
            ? settingsForm(agent, data.project, data.backends, deps)
            : readonlySettings(agent, data.project),
    );
    if (data.health) root.appendChild(renderHealthCard(data.health));
    // The orchestrator's multi-session overview, above its per-session panels.
    if (data.sessions) root.appendChild(sessionsPanel(data.sessions));
    root.appendChild(statusPanel(data.status));
    root.appendChild(accountPanel(data.account));
    root.appendChild(usagePanel(data.usage));
    root.appendChild(memoryPanel(data.memory));

    // The orchestrator's shared/global config sections. These are WRITE controls,
    // so they render only on a writable server (else a click 403s). backend/model/
    // permission_mode are NOT here; they are the agent's own record fields above.
    if (data.global && data.writable) {
        const g = data.global;
        root.appendChild(renderGlobalConfig(g.config, g.actions));
        root.appendChild(renderServerControls(g.config, g.actions));
        if (g.config.tools_enabled && g.tools.length > 0) {
            const tools = el("section", "settings__card");
            tools.appendChild(
                el("h2", "settings__title", `Tools (${g.tools.length})`),
            );
            tools.appendChild(renderToolControls(g.tools, g.actions));
            root.appendChild(tools);
        }
        if (g.profiles) {
            root.appendChild(renderProfileSwitcher(g.profiles, g.actions));
        }
    }
}

// Load the data, render, and re-render after a save.
export function createAgentSettings(
    root: HTMLElement,
    deps: AgentSettingsDeps,
): void {
    const refresh = (): void => void run();
    const wrapped: AgentSettingsDeps = {
        load: deps.load,
        save: async (fields) => {
            await deps.save(fields);
            refresh();
        },
    };
    const run = async (): Promise<void> => {
        const data = await deps.load(refresh);
        renderAgentSettings(root, data, wrapped);
    };
    refresh();
}

// Best-effort fetch: a panel's data failing must not blank the whole page.
function maybe<T>(url: string): Promise<T | null> {
    return fetchJson<T>(url).catch(() => null);
}

// Build the SettingsActions for the orchestrator's global config sections, wired
// to the singular /api/agent/* config endpoints; `reload` re-renders the page.
function orchestratorGlobalActions(reload: () => void): SettingsActions {
    const patchTo = (url: string, method: string, body?: unknown) =>
        sendJson<unknown>(url, method, body).then(() => undefined);
    return {
        patch: (u) => patchTo("/api/agent/config", "PATCH", u),
        addServer: (s) => patchTo("/api/agent/mcp_servers", "POST", s),
        removeServer: (sid) =>
            patchTo(
                `/api/agent/mcp_servers/${encodeURIComponent(sid)}`,
                "DELETE",
            ),
        createProfile: (n) =>
            patchTo("/api/agent/profiles", "POST", { name: n }),
        activateProfile: (n) =>
            patchTo("/api/agent/profiles/activate", "POST", { name: n }),
        deleteProfile: (n) =>
            patchTo(`/api/agent/profiles/${encodeURIComponent(n)}`, "DELETE"),
        reload: () => {
            reload();
            return Promise.resolve();
        },
    };
}

// The real deps for an agent id: per-agent fields + panels, plus (orchestrator
// only) the shared/global config sections. Used by BOTH the /settings root page
// and the /agents/<id>/settings page, so they render identically.
export function agentSettingsDeps(agentId: string): AgentSettingsDeps {
    const enc = encodeURIComponent(agentId);
    const isOrchestrator = agentId === ORCHESTRATOR_ID;
    return {
        load: async (reload): Promise<AgentSettingsData> => {
            const agent = await maybe<Agent>(`/api/agents/${enc}`);
            const [
                project,
                backends,
                health,
                statusData,
                usage,
                memory,
                account,
                config,
                tools,
                profiles,
                sessions,
            ] = await Promise.all([
                (async (): Promise<Project | null> => {
                    if (!agent?.project_id) return null;
                    const projects =
                        (await maybe<Project[]>("/api/projects")) ?? [];
                    return (
                        projects.find((p) => p.id === agent.project_id) ?? null
                    );
                })(),
                maybe<BackendOption[]>("/api/agents/backends"),
                maybe<AgentHealth>("/api/agent/health"),
                maybe<AgentRunStatus>(`/api/agents/${enc}/status`),
                maybe<UsageQuota>(`/api/agents/${enc}/usage`),
                maybe<MemoryFootprint>(`/api/agents/${enc}/memory`),
                maybe<AccountInfo>(`/api/agents/${enc}/account`),
                // The global config is fetched for EVERY agent (it carries the
                // server's writability); it also feeds the orchestrator sections.
                maybe<AgentConfig>("/api/agent/config"),
                isOrchestrator
                    ? maybe<AgentTool[]>("/api/agent/tools")
                    : Promise.resolve(null),
                isOrchestrator
                    ? maybe<ProfilesResponse>("/api/agent/profiles")
                    : Promise.resolve(null),
                // Sessions are the orchestrator's alone (it is multi-session).
                isOrchestrator
                    ? maybe<SessionsResponse>("/api/agent/sessions")
                    : Promise.resolve(null),
            ]);
            // The global config sections are the orchestrator's alone.
            const global: AgentSettingsGlobal | null =
                isOrchestrator && config
                    ? {
                          config,
                          tools: tools ?? [],
                          profiles,
                          actions: orchestratorGlobalActions(reload),
                      }
                    : null;
            return {
                agent,
                project,
                backends: backends ?? [],
                health,
                status: statusData,
                usage,
                memory,
                account,
                sessions,
                global,
                // Read-only server -> a read-only view (no live-but-403 controls).
                writable: config?.writable ?? true,
            };
        },
        save: (fields) =>
            sendJson<Agent>(`/api/agents/${enc}`, "PATCH", fields).then(
                () => undefined,
            ),
    };
}

// Entry for /agents/<id>/settings (mounted by the agent-detail shell).
export function startAgentSettings(): void {
    const root = document.getElementById("agent-settings");
    if (!root) return;
    const id = agentIdFromPath(window.location.pathname);
    if (!id) return;
    createAgentSettings(root, agentSettingsDeps(id));
}

// Entry for the /settings root page: the ORCHESTRATOR's settings, rendered by the
// SAME component as /agents/orchestrator/settings (that is the whole point).
export function startSettings(): void {
    const root = document.getElementById("settings");
    if (!root) return;
    createAgentSettings(root, agentSettingsDeps(ORCHESTRATOR_ID));
}
