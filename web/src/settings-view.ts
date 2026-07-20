// Settings page: a READ-ONLY view of the agent's effective configuration (set via
// environment variables) plus the curated tool list as cards. No import-time side
// effects (the `settings.ts` entry calls `startSettings`); `renderSettings` is
// exported and pure so jsdom tests can drive it without fetch.

import { el, escapeHtml, fetchJson } from "./common";
import type { AgentConfig, AgentTool } from "./common";

function configRow(label: string, value: string): HTMLElement {
    return el(
        "div",
        "settings__row",
        `<span class="settings__key">${escapeHtml(label)}</span>` +
            `<span class="settings__val">${escapeHtml(value)}</span>`,
    );
}

function toolCard(tool: AgentTool): HTMLElement {
    const card = el("div", "tool-card");
    card.appendChild(el("div", "tool-card__name", escapeHtml(tool.name)));
    card.appendChild(
        el("div", "tool-card__desc", escapeHtml(tool.description)),
    );
    return card;
}

export function renderSettings(
    root: HTMLElement,
    config: AgentConfig | null,
    tools: AgentTool[],
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
        const [config, tools] = await Promise.all([
            fetchJson<AgentConfig>("/api/agent/config"),
            fetchJson<AgentTool[]>("/api/agent/tools"),
        ]);
        renderSettings(root, config, tools);
    } catch (err: unknown) {
        console.error(err);
        renderSettings(root, null, []);
    }
}
