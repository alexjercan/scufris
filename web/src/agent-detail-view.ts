// The per-agent detail page (served for /agents/<id> by the backend SPA shell),
// reshaped chat-first: the chat (agent-chat-view) is the primary right pane; this
// module renders the LEFT sidebar (agent header + a Settings LINK + live
// status/context stat boxes). The Settings link goes to the shared per-agent
// settings PAGE (/agents/<id>/settings, agent-settings-view) - the old settings
// modal is retired. `renderSidebar` is PURE; `startAgentDetail` fetches + polls.

import { DEFAULT_POLL_SECONDS, el, escapeHtml, fetchJson } from "./common";
import type { Agent, AgentRunStatus, Project } from "./agent-types";

// Parse the agent id out of `/agents/<id>` or `/agents/<id>/settings`.
export function agentIdFromPath(pathname: string): string | null {
    const m = /^\/agents\/([^/]+)/.exec(pathname);
    return m ? decodeURIComponent(m[1]) : null;
}

function backLink(): HTMLElement {
    const a = document.createElement("a");
    a.href = "/agents/";
    a.className = "agents__back";
    a.textContent = "← all agents";
    return a;
}

function stateBadge(state: string): HTMLElement {
    const badge = el(
        "span",
        `agents__badge agents__badge--${escapeHtml(state)}`,
    );
    badge.textContent = state;
    return badge;
}

// A key/value line inside a stat box (reuses the card `.row` styling).
function kvRow(key: string, value: string): HTMLElement {
    const row = el("div", "row");
    const k = el("span");
    k.textContent = key;
    const v = el("span");
    v.textContent = value;
    row.append(k, v);
    return row;
}

function statBox(title: string): HTMLElement {
    const box = el("div", "usage-block");
    box.appendChild(el("div", "usage-block__head", escapeHtml(title)));
    return box;
}

// A never-run agent has no session and an idle state.
function notStarted(status: AgentRunStatus | null): boolean {
    return status !== null && status.state === "idle" && !status.session_id;
}

function statusBox(status: AgentRunStatus | null): HTMLElement {
    const box = statBox("status");
    if (status === null) {
        box.appendChild(el("div", "settings__empty", "loading..."));
        return box;
    }
    if (notStarted(status)) {
        box.appendChild(el("div", "settings__empty", "not started"));
        return box;
    }
    box.appendChild(kvRow("state", status.state));
    box.appendChild(kvRow("turns", String(status.turns)));
    box.appendChild(kvRow("tools", String(status.tool_calls)));
    return box;
}

function contextBox(status: AgentRunStatus | null): HTMLElement {
    const box = statBox("context");
    if (status === null || notStarted(status) || status.context_window <= 0) {
        box.appendChild(el("div", "settings__empty", "no context yet"));
        return box;
    }
    const pct = Math.min(
        100,
        (status.input_tokens / status.context_window) * 100,
    );
    const bar = el("div", "bar");
    const fill = el("div", "bar__fill");
    fill.style.width = `${pct.toFixed(1)}%`;
    bar.appendChild(fill);
    box.appendChild(bar);
    box.appendChild(
        kvRow(
            "context",
            `${String(status.input_tokens)}/${String(status.context_window)}`,
        ),
    );
    box.appendChild(kvRow("output", String(status.output_tokens)));
    return box;
}

// The LEFT sidebar: agent header + a Settings LINK + live stat boxes. Pure. The
// Settings link navigates to the per-agent settings PAGE (/agents/<id>/settings,
// the shared agent-settings-view) for EVERY agent, including the orchestrator,
// which is now editable there (U1); the old per-agent settings modal is retired.
export function renderSidebar(
    root: HTMLElement,
    agent: Agent | null,
    project: Project | null,
    status: AgentRunStatus | null,
): void {
    root.replaceChildren();
    root.appendChild(backLink());
    if (agent === null) {
        root.appendChild(el("div", "settings__empty", "no such agent."));
        return;
    }

    const head = el("div", "sidebar__agenthead");
    head.appendChild(el("h2", "settings__title", escapeHtml(agent.name)));
    head.appendChild(stateBadge(status ? status.state : agent.state));
    root.appendChild(head);
    root.appendChild(
        el(
            "div",
            "sidebar__project settings__empty",
            escapeHtml(
                project ? project.name : agent.project_id || "server dir",
            ),
        ),
    );

    const settingsLink = document.createElement("a");
    settingsLink.className = "sidebar__new";
    settingsLink.href = `/agents/${encodeURIComponent(agent.id)}/settings`;
    settingsLink.textContent = "Settings";
    settingsLink.setAttribute("aria-label", "open settings");
    root.appendChild(settingsLink);

    root.appendChild(statusBox(status));
    root.appendChild(contextBox(status));
}

export async function startAgentDetail(): Promise<void> {
    const sidebar = document.getElementById("agent-sidebar");
    if (!sidebar) return;
    const id = agentIdFromPath(window.location.pathname);

    let agent: Agent | null = null;
    let project: Project | null = null;
    let status: AgentRunStatus | null = null;

    const render = (): void => {
        renderSidebar(sidebar, agent, project, status);
    };

    const load = async (): Promise<void> => {
        if (!id) {
            render();
            return;
        }
        try {
            agent = await fetchJson<Agent>(
                `/api/agents/${encodeURIComponent(id)}`,
            );
        } catch {
            agent = null;
            render();
            return;
        }
        try {
            const projects = await fetchJson<Project[]>("/api/projects");
            project = projects.find((p) => p.id === agent?.project_id) ?? null;
        } catch {
            project = null;
        }
        render();
    };

    const pollStatus = (): void => {
        if (!id) return;
        void fetchJson<AgentRunStatus>(
            `/api/agents/${encodeURIComponent(id)}/status`,
        )
            .then((s) => {
                status = s;
                // The sidebar has no inputs (its Settings is a LINK to the
                // separate settings page), so re-rendering it never wipes an edit.
                render();
            })
            .catch(() => {
                /* leave prior status */
            });
    };

    await load();
    pollStatus();
    window.setInterval(pollStatus, DEFAULT_POLL_SECONDS * 1000);
}
