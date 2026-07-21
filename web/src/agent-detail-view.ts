// The per-agent detail page (served for /agents/<id> by the backend SPA shell).
// F1 delivers the routing + a read-only detail with live status; the editable
// settings form is F3 and the chat is F4. `renderAgentDetail` is PURE (no fetch)
// so jsdom tests drive it; `startAgentDetail` reads the id from the URL, fetches,
// and polls status.

import { DEFAULT_POLL_SECONDS, el, escapeHtml, fetchJson } from "./common";
import type { Agent, AgentRunStatus, Project } from "./common";

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

export function renderAgentDetail(
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

    const card = el("section", "settings__card");
    const head = el("div", "settings__row settings__row--control");
    head.appendChild(el("h2", "settings__title", escapeHtml(agent.name)));
    head.appendChild(stateBadge(status ? status.state : agent.state));
    card.appendChild(head);

    for (const [key, value] of [
        ["project", project ? project.name : agent.project_id],
        ["backend", agent.backend],
        ["model", agent.model || "-"],
        ["description", agent.description || "-"],
        ["mode", agent.permission_mode],
    ]) {
        card.appendChild(
            el(
                "div",
                "settings__row",
                `<span class="settings__key">${escapeHtml(key)}</span>` +
                    `<span class="settings__val">${escapeHtml(value)}</span>`,
            ),
        );
    }

    card.appendChild(el("h3", "settings__subhead", "Status"));
    if (status === null) {
        card.appendChild(el("div", "settings__empty", "loading status..."));
    } else if (status.state === "idle" && !status.session_id) {
        card.appendChild(
            el(
                "div",
                "settings__empty",
                "not started - run this agent to begin.",
            ),
        );
    } else {
        const rows: [string, string][] = [
            ["turns", String(status.turns)],
            ["tools", String(status.tool_calls)],
            ["input tokens", String(status.input_tokens)],
            ["output tokens", String(status.output_tokens)],
        ];
        for (const [key, value] of rows) {
            card.appendChild(
                el(
                    "div",
                    "settings__row",
                    `<span class="settings__key">${escapeHtml(key)}</span>` +
                        `<span class="settings__val">${escapeHtml(value)}</span>`,
                ),
            );
        }
        if (status.last_message) {
            card.appendChild(el("h3", "settings__subhead", "Last message"));
            card.appendChild(
                el("div", "agents__lastmsg", escapeHtml(status.last_message)),
            );
        }
    }
    root.appendChild(card);
}

export async function startAgentDetail(): Promise<void> {
    const root = document.getElementById("agent-detail");
    if (!root) return;
    const id = agentIdFromPath(window.location.pathname);
    if (!id) {
        renderAgentDetail(root, null, null, null);
        return;
    }

    let agent: Agent | null = null;
    let project: Project | null = null;
    let status: AgentRunStatus | null = null;

    const load = async (): Promise<void> => {
        try {
            agent = await fetchJson<Agent>(
                `/api/agents/${encodeURIComponent(id)}`,
            );
        } catch {
            renderAgentDetail(root, null, null, null);
            return;
        }
        try {
            const projects = await fetchJson<Project[]>("/api/projects");
            project = projects.find((p) => p.id === agent?.project_id) ?? null;
        } catch {
            project = null;
        }
        renderAgentDetail(root, agent, project, status);
    };

    const pollStatus = (): void => {
        void fetchJson<AgentRunStatus>(
            `/api/agents/${encodeURIComponent(id)}/status`,
        )
            .then((s) => {
                status = s;
                renderAgentDetail(root, agent, project, status);
            })
            .catch(() => {
                /* leave prior status */
            });
    };

    await load();
    pollStatus();
    window.setInterval(pollStatus, DEFAULT_POLL_SECONDS * 1000);
}
