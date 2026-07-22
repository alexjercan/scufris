// Settings section renders for the operator console. These composable, PURE
// render helpers (the Health card, the global System toggles, MCP servers, the
// tool controls, the Profiles switcher) are reused by the unified per-agent
// settings page (agent-settings-view) - which owns the page composition and the
// entry now. Side-effect-free so jsdom tests drive each render fetch-free.

import { authLabel, el, escapeHtml } from "./common";
import type {
    AgentConfig,
    AgentConfigUpdate,
    AgentHealth,
    AgentTool,
    HealthCheck,
    McpServerSpec,
    ProfilesResponse,
    ToolRunResult,
} from "./common";

// Actions the writable controls dispatch. The agent-settings entry wires these to
// the real endpoints; jsdom tests pass fakes. Each resolves after the server has
// applied the change so the caller can re-render from fresh data.
export interface SettingsActions {
    patch(update: AgentConfigUpdate): Promise<void>;
    addServer(spec: McpServerSpec): Promise<void>;
    removeServer(id: string): Promise<void>;
    createProfile(name: string): Promise<void>;
    activateProfile(name: string): Promise<void>;
    deleteProfile(name: string): Promise<void>;
    // Run ONE MCP tool by name (the "try it" runner); resolves with its result or
    // throws with the server's error detail on a 4xx.
    runTool(
        name: string,
        args: Record<string, unknown>,
    ): Promise<ToolRunResult>;
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

export function renderGlobalConfig(
    config: AgentConfig,
    actions: SettingsActions,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "System"));
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
    card.appendChild(configRow("auth mode", authLabel(config.auth_mode)));
    card.appendChild(configRow("sandbox", config.sandbox));
    return card;
}

export function renderServerControls(
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

export function renderToolControls(
    tools: AgentTool[],
    actions: SettingsActions,
): HTMLElement {
    const grid = el("div", "tool-grid");
    for (const tool of tools) {
        const card = toolControlCard(tool, (nowEnabled) => {
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
        });
        // Only enabled tools get a runner: a disabled tool 403s at the endpoint.
        if (tool.enabled) card.appendChild(toolRunner(tool, actions));
        grid.appendChild(card);
    }
    return grid;
}

// An interactive "try it" runner appended to a tool card: a form generated from
// the tool's parameter schema, gated by a confirm step, that runs ONE tool and
// renders its result inline - no chat turn, no agent involved.
function toolRunner(tool: AgentTool, actions: SettingsActions): HTMLElement {
    const runner = el("div", "tool-runner");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "tool-runner__toggle";
    toggle.textContent = "Try it";

    const body = el("div", "tool-runner__body");
    body.hidden = true;
    toggle.addEventListener("click", () => {
        body.hidden = !body.hidden;
    });

    const form = document.createElement("form");
    form.className = "tool-runner__form";
    const inputs = new Map<string, HTMLInputElement>();
    for (const p of tool.parameters) {
        const field = el("label", "tool-runner__field");
        field.appendChild(
            el(
                "span",
                "tool-runner__label",
                `${escapeHtml(p.name)}${p.required ? " *" : ""}`,
            ),
        );
        const input = document.createElement("input");
        input.name = p.name;
        input.setAttribute("aria-label", p.name);
        if (p.type === "boolean") input.type = "checkbox";
        else if (p.type === "integer" || p.type === "number")
            input.type = "number";
        else input.type = "text";
        if (p.description) input.title = p.description;
        field.appendChild(input);
        inputs.set(p.name, input);
        form.appendChild(field);
    }

    const run = document.createElement("button");
    run.type = "submit";
    run.className = "tool-runner__run";
    run.textContent = "Run";
    form.appendChild(run);

    const result = el("div", "tool-runner__result");
    result.hidden = true;

    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        // Running a host tool is a real capability - confirm before every run.
        if (!window.confirm(`Run ${tool.name} now?`)) return;
        void runAndRender(
            actions,
            tool.name,
            collectArgs(tool, inputs),
            result,
        );
    });

    body.appendChild(form);
    body.appendChild(result);
    runner.appendChild(toggle);
    runner.appendChild(body);
    return runner;
}

// Collect the form values into an args object, coercing by declared type. Empty
// optional fields are omitted so the tool applies its own default; a required
// field left blank is sent absent and the endpoint returns a 422 we surface.
function collectArgs(
    tool: AgentTool,
    inputs: Map<string, HTMLInputElement>,
): Record<string, unknown> {
    const args: Record<string, unknown> = {};
    for (const p of tool.parameters) {
        const input = inputs.get(p.name);
        if (!input) continue;
        if (p.type === "boolean") {
            args[p.name] = input.checked;
        } else if (input.value !== "") {
            args[p.name] =
                p.type === "integer" || p.type === "number"
                    ? Number(input.value)
                    : input.value;
        }
    }
    return args;
}

async function runAndRender(
    actions: SettingsActions,
    name: string,
    args: Record<string, unknown>,
    target: HTMLElement,
): Promise<void> {
    target.hidden = false;
    target.className = "tool-runner__result";
    target.textContent = "Running...";
    try {
        const res = await actions.runTool(name, args);
        // Prefer the structured block when present, else the text output. An
        // empty-object `structured` is intentionally treated as "no structured"
        // and falls back to the text (every scufris tool emits one or the other).
        const pretty =
            Object.keys(res.structured).length > 0
                ? JSON.stringify(res.structured, null, 2)
                : res.text;
        // Escape everything - a tool result is untrusted host output.
        target.innerHTML = `<pre class="tool-runner__output">${escapeHtml(pretty)}</pre>`;
    } catch (err: unknown) {
        target.className = "tool-runner__result tool-runner__result--error";
        target.textContent = err instanceof Error ? err.message : String(err);
    }
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

// Exported so the unified per-agent settings page (agent-settings-view) renders
// the SAME Health card as the /settings page - one health render, no drift.
export function renderHealthCard(health: AgentHealth): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Health"));

    const bits = [`scufris ${health.scufris_version}`];
    // The backend CLI version, whichever backend this agent runs (codex/claude);
    // the version string self-labels (e.g. "codex 0.144" / "claude 1.x").
    if (health.backend_version) bits.push(health.backend_version);
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

// A read-only key/value panel. `rows` values are already display strings; a
// null value shows a dash so a panel never collapses or looks broken.
export function renderProfileSwitcher(
    profiles: ProfilesResponse,
    actions: SettingsActions,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Profiles"));
    const list = el("div", "profiles");
    for (const name of profiles.profiles) {
        const active = name === profiles.active;
        const item = el(
            "div",
            `profiles__item${active ? " profiles__item--active" : ""}`,
        );
        const pick = document.createElement("button");
        pick.type = "button";
        pick.className = "profiles__name";
        pick.textContent = name;
        pick.disabled = active;
        pick.setAttribute("aria-label", `activate ${name}`);
        pick.addEventListener("click", () => {
            void dispatch(actions, () => actions.activateProfile(name));
        });
        item.appendChild(pick);
        if (active) item.appendChild(el("span", "profiles__badge", "active"));
        else {
            const del = document.createElement("button");
            del.type = "button";
            del.className = "settings__btn settings__btn--danger";
            del.textContent = "delete";
            del.setAttribute("aria-label", `delete ${name}`);
            del.addEventListener("click", () => {
                if (!window.confirm(`Delete profile "${name}"?`)) return;
                void dispatch(actions, () => actions.deleteProfile(name));
            });
            item.appendChild(del);
        }
        list.appendChild(item);
    }
    card.appendChild(list);

    const form = document.createElement("form");
    form.className = "settings__addserver";
    const nameIn = document.createElement("input");
    nameIn.type = "text";
    nameIn.placeholder = "new profile name";
    nameIn.className = "settings__input";
    nameIn.setAttribute("aria-label", "new profile name");
    const add = document.createElement("button");
    add.type = "submit";
    add.className = "settings__btn";
    add.textContent = "save as";
    form.append(nameIn, add);
    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const name = nameIn.value.trim();
        if (!name) return;
        void dispatch(actions, async () => {
            await actions.createProfile(name);
            nameIn.value = "";
        });
    });
    card.appendChild(form);
    return card;
}
