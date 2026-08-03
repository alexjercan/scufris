// Settings section renders for the operator console. These composable, PURE
// render helpers (the Health card and the tool controls) are reused by the
// unified per-agent settings page (agent-settings-view) - which owns the page
// composition and the entry now. Side-effect-free so jsdom tests drive each
// render fetch-free.

import { el, escapeHtml } from "./common";
import type {
    AgentConfigUpdate,
    AgentHealth,
    AgentTool,
    HealthCheck,
    McpServerHealth,
    ToolRunResult,
} from "./agent-types";

// Actions the writable controls dispatch. The agent-settings entry wires these to
// the real endpoints; jsdom tests pass fakes. Each resolves after the server has
// applied the change so the caller can re-render from fresh data.
export interface SettingsActions {
    patch(update: AgentConfigUpdate): Promise<void>;
    // Run ONE MCP tool by name (the "try it" runner); resolves with its result or
    // throws with the server's error detail on a 4xx.
    runTool(
        name: string,
        args: Record<string, unknown>,
    ): Promise<ToolRunResult>;
    reload(): Promise<void>;
}

const STATUSES = ["ok", "warn", "error"];

// The read-only presentation of one tool (name + server badge + description +
// args). Both the orchestrator's writable console (via `toolControlCard`) and a
// sub-agent's read-only Tools grid render this SAME card, so the two surfaces
// never drift apart; the sub-agent path uses it bare (no toggle/runner).
export function toolCard(tool: AgentTool): HTMLElement {
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

// One tool card inside a server block. When `actions` is set (the orchestrator
// console) the card is writable: the enable toggle rebuilds the FULL disabled set
// across every server's tools (`universe`), and an enabled tool gets the "try it"
// runner. When `actions` is null (a sub-agent's read-only view) the card is bare.
// A disabled tool is dimmed (`tool-card--disabled`) + its toggle off; there is no
// health bulb here - server health lives in the Health card.
function mcpToolCard(
    tool: AgentTool,
    universe: AgentTool[],
    actions: SettingsActions | null,
): HTMLElement {
    const card = actions
        ? toolControlCard(tool, (nowEnabled) => {
              const disabled = new Set(
                  universe.filter((t) => !t.enabled).map((t) => t.name),
              );
              if (nowEnabled) disabled.delete(tool.name);
              else disabled.add(tool.name);
              void dispatch(actions, () =>
                  actions.patch({ disabled_tools: [...disabled] }),
              );
          })
        : toolCard(tool);
    if (!tool.enabled) card.classList.add("tool-card--disabled");
    // Only writable + enabled tools get a runner: a disabled tool 403s.
    if (actions && tool.enabled) card.appendChild(toolRunner(tool, actions));
    return card;
}

// One collapsible server block: a summary line (id + tool count) over a grid of
// its tool cards. Organizational only - the per-server green/amber/red status
// lives in the Health card, not here.
function mcpServerBlock(
    server: McpServerHealth,
    universe: AgentTool[],
    actions: SettingsActions | null,
): HTMLElement {
    const details = document.createElement("details");
    details.className = "mcp-server";
    const summary = document.createElement("summary");
    summary.className = "mcp-server__summary";
    summary.appendChild(el("span", "mcp-server__id", escapeHtml(server.id)));
    summary.appendChild(
        el("span", "mcp-server__count", `${server.tools.length} tools`),
    );
    details.appendChild(summary);
    const grid = el("div", "tool-grid");
    for (const tool of server.tools) {
        grid.appendChild(mcpToolCard(tool, universe, actions));
    }
    details.appendChild(grid);
    return details;
}

// The "MCP tools" section: one collapsible block per server (scufris + den for
// the orchestrator; the callback server for a sub-agent) grouping its tools.
// Organizational only - server health is shown as rows in the Health card. Pass
// `actions` for the writable orchestrator console (toggles + runners); pass null
// for a sub-agent's read-only view. Reuses the shared `toolCard` / `tool-grid`.
export function renderMcpServers(
    servers: McpServerHealth[],
    actions: SettingsActions | null,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "MCP tools"));
    // The full tool universe (across all servers) so a toggle rebuilds the whole
    // disabled_tools set, not just one server's slice.
    const universe = servers.flatMap((s) => s.tools);
    for (const server of servers) {
        card.appendChild(mcpServerBlock(server, universe, actions));
    }
    return card;
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
    if (health.session_count !== null) {
        bits.push(
            `${health.session_count} session${health.session_count === 1 ? "" : "s"}`,
        );
    }
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
