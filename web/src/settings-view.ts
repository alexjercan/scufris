// Settings / operator console: the agent's effective config plus live health
// checks and the curated tool list. INTERACTIVE when the server is writable
// (config.writable) and actions are wired - toggles/inputs dispatch to the
// config endpoints and the page re-renders from the server; otherwise a
// read-only view. No import-time side effects (the `settings.ts` entry calls
// `startSettings`); `renderSettings` is pure so jsdom tests drive it fetch-free.

import { el, escapeHtml, fetchJson, sendJson } from "./common";
import type {
    AgentConfig,
    AgentConfigUpdate,
    AgentHealth,
    AgentTool,
    HealthCheck,
    McpServerSpec,
} from "./common";

// Actions the writable controls dispatch. `startSettings` wires these to the
// real endpoints; jsdom tests pass fakes. Each resolves after the server has
// applied the change so the caller can re-render from fresh data.
export interface SettingsActions {
    patch(update: AgentConfigUpdate): Promise<void>;
    addServer(spec: McpServerSpec): Promise<void>;
    removeServer(id: string): Promise<void>;
    reload(): Promise<void>;
}

// The env var that sets each Agent config row - so the operator knows what to edit
// (config is env-var only). Sandbox is always read-only (no knob).
const ENV_VARS: Record<string, string> = {
    status: "SCUFRIS_AGENT_ENABLED",
    backend: "SCUFRIS_AGENT_BACKEND",
    model: "SCUFRIS_AGENT_MODEL",
    "auth mode": "SCUFRIS_AGENT_AUTH_MODE",
    tools: "SCUFRIS_AGENT_TOOLS_ENABLED",
};

const STATUSES = ["ok", "warn", "error"];

function configRow(label: string, value: string): HTMLElement {
    const env = ENV_VARS[label];
    const envChip = env
        ? `<span class="settings__env">${escapeHtml(env)}</span>`
        : "";
    return el(
        "div",
        "settings__row",
        `<span class="settings__key">${escapeHtml(label)}${envChip}</span>` +
            `<span class="settings__val">${escapeHtml(value)}</span>`,
    );
}

function toolCard(tool: AgentTool): HTMLElement {
    const card = el("div", "tool-card");
    const head = el("div", "tool-card__head");
    head.appendChild(el("span", "tool-card__name", escapeHtml(tool.name)));
    head.appendChild(el("span", "tool-card__server", escapeHtml(tool.server)));
    card.appendChild(head);
    card.appendChild(
        el("div", "tool-card__desc", escapeHtml(tool.description)),
    );
    if (tool.args.length > 0) {
        card.appendChild(
            el(
                "div",
                "tool-card__args",
                `args: ${tool.args.map((a) => escapeHtml(a)).join(", ")}`,
            ),
        );
    }
    return card;
}

// Run an action, surfacing any error inline, then reload the page from fresh
// server data (single authoritative render - no parallel client copy).
async function dispatch(
    actions: SettingsActions,
    run: () => Promise<void>,
): Promise<void> {
    try {
        await run();
        await actions.reload();
    } catch (err: unknown) {
        window.alert(err instanceof Error ? err.message : String(err));
    }
}

function toggleRow(
    label: string,
    on: boolean,
    onChange: (next: boolean) => void,
    confirmOff?: string,
): HTMLElement {
    const row = el("div", "settings__row settings__row--control");
    row.appendChild(el("span", "settings__key", escapeHtml(label)));
    const input = document.createElement("input");
    input.type = "checkbox";
    input.className = "settings__toggle";
    input.checked = on;
    input.setAttribute("aria-label", label);
    input.addEventListener("change", () => {
        const next = input.checked;
        // Guard a high-impact turn-OFF behind a confirm; a stray click must not
        // silently disable the agent or its tools.
        if (!next && confirmOff && !window.confirm(confirmOff)) {
            input.checked = true;
            return;
        }
        onChange(next);
    });
    row.appendChild(input);
    return row;
}

function selectRow(
    label: string,
    value: string,
    options: string[],
    onChange: (next: string) => void,
): HTMLElement {
    const row = el("div", "settings__row settings__row--control");
    row.appendChild(el("span", "settings__key", escapeHtml(label)));
    const select = document.createElement("select");
    select.className = "settings__select";
    select.setAttribute("aria-label", label);
    for (const opt of options) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        if (opt === value) o.selected = true;
        select.appendChild(o);
    }
    select.addEventListener("change", () => {
        onChange(select.value);
    });
    row.appendChild(select);
    return row;
}

function textRow(
    label: string,
    value: string,
    onSave: (next: string) => void,
): HTMLElement {
    const row = el("div", "settings__row settings__row--control");
    row.appendChild(el("span", "settings__key", escapeHtml(label)));
    const input = document.createElement("input");
    input.type = "text";
    input.className = "settings__input";
    input.value = value;
    input.setAttribute("aria-label", label);
    const save = () => {
        if (input.value !== value) onSave(input.value);
    };
    input.addEventListener("change", save);
    row.appendChild(input);
    return row;
}

const BACKENDS = ["app_server", "exec", "mock"];

function renderAgentControls(
    config: AgentConfig,
    actions: SettingsActions,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Agent"));
    card.appendChild(
        toggleRow(
            "enabled",
            config.enabled,
            (next) => {
                void dispatch(actions, () =>
                    actions.patch({ agent_enabled: next }),
                );
            },
            "Disable the agent? Chat will stop working until re-enabled.",
        ),
    );
    card.appendChild(
        selectRow("backend", config.backend, BACKENDS, (next) => {
            void dispatch(actions, () =>
                actions.patch({ agent_backend: next }),
            );
        }),
    );
    card.appendChild(
        textRow("model", config.model, (next) => {
            void dispatch(actions, () => actions.patch({ agent_model: next }));
        }),
    );
    card.appendChild(configRow("auth mode", config.auth_mode));
    card.appendChild(configRow("sandbox", config.sandbox));
    card.appendChild(
        toggleRow(
            "tools",
            config.tools_enabled,
            (next) => {
                void dispatch(actions, () =>
                    actions.patch({ agent_tools_enabled: next }),
                );
            },
            "Disable all tools? The agent will not be able to call any.",
        ),
    );
    return card;
}

function renderServerControls(
    config: AgentConfig,
    actions: SettingsActions,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "MCP servers"));
    for (const server of config.mcp_servers) {
        const row = el(
            "div",
            "settings__row",
            `<span class="settings__key">${escapeHtml(server.id)}</span>` +
                `<span class="settings__badge">${escapeHtml(server.source)}</span>`,
        );
        // The built-in scufris server is not removable (it is derived from the
        // tools toggle, not the configured list).
        if (server.source === "configured") {
            const rm = document.createElement("button");
            rm.type = "button";
            rm.className = "settings__btn settings__btn--danger";
            rm.textContent = "remove";
            rm.addEventListener("click", () => {
                if (!window.confirm(`Remove MCP server "${server.id}"?`))
                    return;
                void dispatch(actions, () => actions.removeServer(server.id));
            });
            row.appendChild(rm);
        }
        card.appendChild(row);
    }
    card.appendChild(renderAddServerForm(actions));
    return card;
}

function renderAddServerForm(actions: SettingsActions): HTMLElement {
    const form = document.createElement("form");
    form.className = "settings__addserver";
    const idIn = document.createElement("input");
    idIn.type = "text";
    idIn.placeholder = "id";
    idIn.className = "settings__input";
    idIn.setAttribute("aria-label", "new MCP server id");
    const cmdIn = document.createElement("input");
    cmdIn.type = "text";
    cmdIn.placeholder = "command";
    cmdIn.className = "settings__input";
    cmdIn.setAttribute("aria-label", "new MCP server command");
    const argsIn = document.createElement("input");
    argsIn.type = "text";
    argsIn.placeholder = "args (space-separated)";
    argsIn.className = "settings__input";
    argsIn.setAttribute("aria-label", "new MCP server args");
    const add = document.createElement("button");
    add.type = "submit";
    add.className = "settings__btn";
    add.textContent = "add server";
    form.append(idIn, cmdIn, argsIn, add);
    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const id = idIn.value.trim();
        const command = cmdIn.value.trim();
        if (!id || !command) return;
        const args = argsIn.value.trim()
            ? argsIn.value.trim().split(/\s+/)
            : [];
        void dispatch(actions, async () => {
            await actions.addServer({ id, command, args });
            idIn.value = cmdIn.value = argsIn.value = "";
        });
    });
    return form;
}

function toolControlCard(
    tool: AgentTool,
    onToggle: (nowEnabled: boolean) => void,
): HTMLElement {
    const card = toolCard(tool);
    if (!tool.enabled) card.classList.add("tool-card--disabled");
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.className = "settings__toggle tool-card__toggle";
    toggle.checked = tool.enabled;
    toggle.setAttribute("aria-label", `enable ${tool.name}`);
    toggle.addEventListener("change", () => {
        onToggle(toggle.checked);
    });
    card.querySelector(".tool-card__head")?.appendChild(toggle);
    return card;
}

function renderToolControls(
    tools: AgentTool[],
    actions: SettingsActions,
): HTMLElement {
    const grid = el("div", "tool-grid");
    for (const tool of tools) {
        grid.appendChild(
            toolControlCard(tool, (nowEnabled) => {
                // Rebuild the whole disabled set (every tool currently off),
                // then flip this one, so the server gets the full list.
                const disabled = new Set(
                    tools.filter((t) => !t.enabled).map((t) => t.name),
                );
                if (nowEnabled) disabled.delete(tool.name);
                else disabled.add(tool.name);
                void dispatch(actions, () =>
                    actions.patch({ disabled_tools: [...disabled] }),
                );
            }),
        );
    }
    return grid;
}

function healthRow(check: HealthCheck): HTMLElement {
    const row = el("div", "health__row");
    const status = STATUSES.includes(check.status) ? check.status : "warn";
    row.appendChild(el("span", `health__dot health__dot--${status}`));
    const body = el("div", "health__body");
    body.appendChild(
        el(
            "div",
            "health__line",
            `<span class="health__name">${escapeHtml(check.name)}</span>` +
                `<span class="health__detail">${escapeHtml(check.detail)}</span>`,
        ),
    );
    if (check.hint) {
        body.appendChild(el("div", "health__hint", escapeHtml(check.hint)));
    }
    row.appendChild(body);
    return row;
}

function renderHealthCard(health: AgentHealth): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Health"));

    const bits = [`scufris ${health.scufris_version}`];
    if (health.codex_version) bits.push(health.codex_version);
    bits.push(
        `${health.session_count} session${health.session_count === 1 ? "" : "s"}`,
    );
    if (health.last_session) {
        const when = new Date(health.last_session);
        if (!Number.isNaN(when.getTime())) {
            bits.push(`last active ${when.toLocaleDateString()}`);
        }
    }
    card.appendChild(el("p", "settings__note", bits.join(" · ")));

    for (const check of health.checks) card.appendChild(healthRow(check));
    return card;
}

export function renderSettings(
    root: HTMLElement,
    config: AgentConfig | null,
    tools: AgentTool[],
    health: AgentHealth | null = null,
    actions: SettingsActions | null = null,
): void {
    root.replaceChildren();
    if (!config) {
        root.appendChild(
            el(
                "div",
                "settings__empty",
                "could not load the agent configuration.",
            ),
        );
        return;
    }

    // Interactive only when the server accepts writes AND actions were wired
    // (jsdom display tests pass none). Otherwise render the read-only view.
    const live = config.writable && actions !== null;

    if (health) root.appendChild(renderHealthCard(health));

    if (live) {
        root.appendChild(renderAgentControls(config, actions));
    } else {
        const agent = el("section", "settings__card");
        agent.appendChild(el("h2", "settings__title", "Agent"));
        agent.appendChild(
            el(
                "p",
                "settings__note",
                config.writable
                    ? "Read-only view."
                    : "Read-only server (SCUFRIS_SETTINGS_WRITABLE=0); set it to change config here.",
            ),
        );
        agent.appendChild(
            configRow("status", config.enabled ? "enabled" : "disabled"),
        );
        agent.appendChild(configRow("backend", config.backend));
        agent.appendChild(configRow("model", config.model));
        agent.appendChild(configRow("auth mode", config.auth_mode));
        agent.appendChild(configRow("sandbox", config.sandbox));
        agent.appendChild(
            configRow("tools", config.tools_enabled ? "enabled" : "disabled"),
        );
        root.appendChild(agent);
    }

    if (live) {
        root.appendChild(renderServerControls(config, actions));
    } else {
        const servers = el("section", "settings__card");
        servers.appendChild(el("h2", "settings__title", "MCP servers"));
        if (config.mcp_servers.length === 0) {
            servers.appendChild(
                el("div", "settings__empty", "none registered."),
            );
        } else {
            for (const server of config.mcp_servers) {
                servers.appendChild(
                    el(
                        "div",
                        "settings__row",
                        `<span class="settings__key">${escapeHtml(server.id)}</span>` +
                            `<span class="settings__badge">${escapeHtml(server.source)}</span>`,
                    ),
                );
            }
        }
        root.appendChild(servers);
    }

    const toolSection = el("section", "settings__card");
    // When tools are disabled the agent cannot call any, so do not imply a live
    // catalog - say so plainly (this page exists to answer "why won't it use a
    // tool?"). Otherwise show the tools as cards.
    if (!config.tools_enabled) {
        toolSection.appendChild(el("h2", "settings__title", "Tools"));
        toolSection.appendChild(
            el(
                "div",
                "settings__empty",
                "tools are disabled - toggle 'tools' above to enable them.",
            ),
        );
    } else if (tools.length === 0) {
        toolSection.appendChild(el("h2", "settings__title", "Tools"));
        toolSection.appendChild(
            el("div", "settings__empty", "no tools available."),
        );
    } else {
        toolSection.appendChild(
            el("h2", "settings__title", `Tools (${tools.length})`),
        );
        toolSection.appendChild(
            live
                ? renderToolControls(tools, actions)
                : (() => {
                      const grid = el("div", "tool-grid");
                      for (const tool of tools)
                          grid.appendChild(toolCard(tool));
                      return grid;
                  })(),
        );
    }
    root.appendChild(toolSection);
}

export async function startSettings(): Promise<void> {
    const root = document.getElementById("settings");
    if (!root) return;
    const load = async (): Promise<void> => {
        const [config, tools, health] = await Promise.all([
            fetchJson<AgentConfig>("/api/agent/config"),
            fetchJson<AgentTool[]>("/api/agent/tools"),
            // Health is best-effort - a failure should not blank the whole page.
            fetchJson<AgentHealth>("/api/agent/health").catch(() => null),
        ]);
        renderSettings(root, config, tools, health, actions);
    };
    const actions: SettingsActions = {
        patch: (update) =>
            sendJson<AgentConfig>("/api/agent/config", "PATCH", update).then(
                () => undefined,
            ),
        addServer: (spec) =>
            sendJson<AgentConfig>("/api/agent/mcp_servers", "POST", spec).then(
                () => undefined,
            ),
        removeServer: (id) =>
            sendJson<AgentConfig>(
                `/api/agent/mcp_servers/${encodeURIComponent(id)}`,
                "DELETE",
            ).then(() => undefined),
        reload: load,
    };
    try {
        await load();
    } catch (err: unknown) {
        console.error(err);
        renderSettings(root, null, []);
    }
}
