// Settings / operator console: a READ-ONLY view of the agent's effective config
// (env-var driven) plus live health checks and the curated tool list as cards. No
// import-time side effects (the `settings.ts` entry calls `startSettings`);
// `renderSettings` is exported and pure so jsdom tests can drive it without fetch.

import { el, escapeHtml, fetchJson } from "./common";
import type {
    AgentConfig,
    AgentHealth,
    AgentTool,
    HealthCheck,
} from "./common";

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

    if (health) root.appendChild(renderHealthCard(health));

    const agent = el("section", "settings__card");
    agent.appendChild(el("h2", "settings__title", "Agent"));
    agent.appendChild(
        el(
            "p",
            "settings__note",
            "Read-only. Everything here is set via environment variables; restart to change.",
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

    const servers = el("section", "settings__card");
    servers.appendChild(el("h2", "settings__title", "MCP servers"));
    if (config.mcp_servers.length === 0) {
        servers.appendChild(el("div", "settings__empty", "none registered."));
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
                "tools are disabled (SCUFRIS_AGENT_TOOLS_ENABLED=0).",
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
        const grid = el("div", "tool-grid");
        for (const tool of tools) grid.appendChild(toolCard(tool));
        toolSection.appendChild(grid);
    }
    root.appendChild(toolSection);
}

export async function startSettings(): Promise<void> {
    const root = document.getElementById("settings");
    if (!root) return;
    try {
        const [config, tools, health] = await Promise.all([
            fetchJson<AgentConfig>("/api/agent/config"),
            fetchJson<AgentTool[]>("/api/agent/tools"),
            // Health is best-effort - a failure should not blank the whole page.
            fetchJson<AgentHealth>("/api/agent/health").catch(() => null),
        ]);
        renderSettings(root, config, tools, health);
    } catch (err: unknown) {
        console.error(err);
        renderSettings(root, null, []);
    }
}
