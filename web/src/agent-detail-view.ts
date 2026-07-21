// The per-agent detail page (served for /agents/<id> by the backend SPA shell).
// It reads the id from the path, shows the agent's read-only facts (project) +
// live run status, and a per-agent SETTINGS-edit form (name, backend, model,
// description, permission mode) that PATCHes /api/agents/{id}. The chat is F4.
// `renderAgentDetail` is PURE (no fetch) so jsdom tests drive it; the injected
// `save` action is wired to the API by `startAgentDetail`, which also fetches
// and polls status.

import {
    DEFAULT_POLL_SECONDS,
    el,
    escapeHtml,
    fetchJson,
    sendJson,
} from "./common";
import type { Agent, AgentRunStatus, BackendOption, Project } from "./common";
import { agentFields } from "./agent-fields";
import type { AgentFieldValues } from "./agent-fields";

// Actions the page dispatches. `startAgentDetail` wires these to the API; jsdom
// tests pass fakes. `save` PATCHes the agent and resolves after the server
// applied it.
export interface AgentDetailActions {
    save(fields: AgentFieldValues): Promise<void>;
}

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

function readonlyRow(key: string, value: string): HTMLElement {
    return el(
        "div",
        "settings__row",
        `<span class="settings__key">${escapeHtml(key)}</span>` +
            `<span class="settings__val">${escapeHtml(value)}</span>`,
    );
}

// The editable settings form, prefilled with the agent's current values.
function settingsForm(
    agent: Agent,
    backends: BackendOption[],
    actions: AgentDetailActions,
): HTMLElement {
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
                await actions.save(values);
            } catch (err: unknown) {
                window.alert(err instanceof Error ? err.message : String(err));
            }
        })();
    });
    return form;
}

function statusSection(status: AgentRunStatus | null): HTMLElement[] {
    const nodes: HTMLElement[] = [el("h3", "settings__subhead", "Status")];
    if (status === null) {
        nodes.push(el("div", "settings__empty", "loading status..."));
        return nodes;
    }
    if (status.state === "idle" && !status.session_id) {
        nodes.push(
            el(
                "div",
                "settings__empty",
                "not started - run this agent to begin.",
            ),
        );
        return nodes;
    }
    const rows: [string, string][] = [
        ["turns", String(status.turns)],
        ["tools", String(status.tool_calls)],
        ["input tokens", String(status.input_tokens)],
        ["output tokens", String(status.output_tokens)],
    ];
    for (const [key, value] of rows) nodes.push(readonlyRow(key, value));
    if (status.last_message) {
        nodes.push(el("h3", "settings__subhead", "Last message"));
        nodes.push(
            el("div", "agents__lastmsg", escapeHtml(status.last_message)),
        );
    }
    return nodes;
}

export function renderAgentDetail(
    root: HTMLElement,
    agent: Agent | null,
    project: Project | null,
    backends: BackendOption[],
    status: AgentRunStatus | null,
    actions: AgentDetailActions,
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

    // The project binding is fixed after creation, so it stays read-only; the
    // model is now an editable settings field (it follows the backend).
    card.appendChild(
        readonlyRow("project", project ? project.name : agent.project_id),
    );

    card.appendChild(el("h3", "settings__subhead", "Settings"));
    card.appendChild(settingsForm(agent, backends, actions));

    for (const node of statusSection(status)) card.appendChild(node);
    root.appendChild(card);
}

export async function startAgentDetail(): Promise<void> {
    const root = document.getElementById("agent-detail");
    if (!root) return;
    const id = agentIdFromPath(window.location.pathname);

    let agent: Agent | null = null;
    let project: Project | null = null;
    let backends: BackendOption[] = [];
    let status: AgentRunStatus | null = null;

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
        try {
            backends = await fetchJson<BackendOption[]>("/api/agents/backends");
        } catch {
            backends = [];
        }
        render();
    };

    const actions: AgentDetailActions = {
        save: async (fields) => {
            await sendJson<Agent>(
                `/api/agents/${encodeURIComponent(id ?? "")}`,
                "PATCH",
                fields,
            );
            await load();
        },
    };

    const render = (): void => {
        renderAgentDetail(root, agent, project, backends, status, actions);
    };

    // Don't clobber the settings form (or wipe an in-progress edit) on a poll.
    const editingSettings = (): boolean => {
        const active = document.activeElement;
        return (
            active !== null &&
            root.contains(active) &&
            ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)
        );
    };

    const pollStatus = (): void => {
        if (!id) return;
        void fetchJson<AgentRunStatus>(
            `/api/agents/${encodeURIComponent(id)}/status`,
        )
            .then((s) => {
                status = s;
                if (!editingSettings()) render();
            })
            .catch(() => {
                /* leave prior status */
            });
    };

    await load();
    pollStatus();
    window.setInterval(pollStatus, DEFAULT_POLL_SECONDS * 1000);
}
