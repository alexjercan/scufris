// Agents dashboard: the cockpit for managing agents that run on projects.
// Agents render as Stats-style cards (name, friendly backend label, state badge,
// project, live turns/tokens); clicking a card opens the dedicated `/agents/<id>`
// page (its detail + chat live there, not here). A create form binds a new agent
// to a project. `renderAgents` is PURE (no fetch) so jsdom tests drive it
// directly; `startAgents` does the fetch orchestration and status polling and
// wires the actions.

import {
    DEFAULT_POLL_SECONDS,
    backendLabel,
    el,
    escapeHtml,
    fetchJson,
    sendJson,
} from "./common";
import type { Agent, AgentRunStatus, BackendOption, Project } from "./common";
import { agentFields } from "./agent-fields";
import type { AgentFieldValues } from "./agent-fields";

// A create request: the shared agent fields plus the project the agent binds to.
export interface AgentCreateFields extends AgentFieldValues {
    project_id: string;
}

// Actions the page dispatches. `startAgents` wires these to the API; jsdom tests
// pass fakes. `open` navigates to the per-agent page. Each mutating action
// resolves after the server applied it.
export interface AgentActions {
    create(fields: AgentCreateFields): Promise<void>;
    remove(id: string): Promise<void>;
    open(id: string): void;
    reload(): Promise<void>;
}

async function dispatch(
    actions: AgentActions,
    run: () => Promise<void>,
): Promise<void> {
    try {
        await run();
        await actions.reload();
    } catch (err: unknown) {
        window.alert(err instanceof Error ? err.message : String(err));
    }
}

function stateBadge(state: string): HTMLElement {
    const badge = el(
        "span",
        `agents__badge agents__badge--${escapeHtml(state)}`,
    );
    badge.textContent = state;
    return badge;
}

// A key/value line inside a card (mirrors the Stats `.row` shape).
function row(key: string, value: string): HTMLElement {
    const line = el("div", "row");
    const k = el("span");
    k.textContent = key;
    const v = el("span");
    v.textContent = value;
    line.appendChild(k);
    line.appendChild(v);
    return line;
}

// A never-run agent has no session and an idle state; distinguish that from a
// real run so a card shows "not started" instead of a meaningless 0/0.
function hasStarted(status: AgentRunStatus | undefined): boolean {
    return (
        status !== undefined && !(status.state === "idle" && !status.session_id)
    );
}

function agentCard(
    agent: Agent,
    projects: Project[],
    status: AgentRunStatus | undefined,
    actions: AgentActions,
): HTMLElement {
    const state = status ? status.state : agent.state;
    const card = el("section", "card agents__card");
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `open ${agent.name}`);

    const title = el("h2", "card__title");
    const name = el("span");
    name.textContent = agent.name;
    title.appendChild(name);
    title.appendChild(stateBadge(state));
    card.appendChild(title);

    const project = projects.find((p) => p.id === agent.project_id);
    const rows = el("div", "card__rows");
    rows.appendChild(row("backend", backendLabel(agent.backend)));
    rows.appendChild(row("project", project ? project.name : agent.project_id));
    rows.appendChild(row("mode", agent.permission_mode));
    if (hasStarted(status) && status) {
        rows.appendChild(row("turns", String(status.turns)));
        rows.appendChild(
            row(
                "tokens",
                `${String(status.input_tokens)} in / ${String(status.output_tokens)} out`,
            ),
        );
    } else {
        rows.appendChild(row("status", "not started"));
    }
    card.appendChild(rows);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "settings__btn settings__btn--danger agents__card-del";
    del.textContent = "delete";
    del.setAttribute("aria-label", `delete ${agent.name}`);
    del.addEventListener("click", (ev) => {
        // The card itself is clickable; keep a delete from also navigating.
        ev.stopPropagation();
        if (!window.confirm(`Delete agent "${agent.name}"?`)) return;
        void dispatch(actions, () => actions.remove(agent.id));
    });
    card.appendChild(del);

    const open = (): void => {
        actions.open(agent.id);
    };
    card.addEventListener("click", open);
    card.addEventListener("keydown", (ev: KeyboardEvent) => {
        // Only the card itself opens on Enter/Space; a keydown bubbling up from
        // the focused delete button must not also navigate (its click already
        // stops propagation, but keydown is a separate event).
        if (ev.target !== card) return;
        if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            open();
        }
    });
    return card;
}

function createForm(
    projects: Project[],
    backends: BackendOption[],
    actions: AgentActions,
): HTMLElement {
    const form = document.createElement("form");
    form.className = "settings__addserver";

    // Shared name/backend/model/description/mode controls, plus a create-only
    // project picker inserted right after the name (its historic position).
    const fields = agentFields("new agent", backends);

    const project = document.createElement("select");
    project.className = "settings__input";
    project.setAttribute("aria-label", "new agent project");
    for (const p of projects) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.name;
        project.appendChild(opt);
    }

    form.append(
        fields.name,
        project,
        fields.backend,
        fields.model,
        fields.modelList,
        fields.description,
        fields.mode,
    );

    const add = document.createElement("button");
    add.type = "submit";
    add.className = "settings__btn";
    add.textContent = "create agent";
    form.appendChild(add);

    // An agent must bind to a project; guide the operator when there are none.
    if (projects.length === 0) {
        add.disabled = true;
        form.appendChild(
            el(
                "div",
                "settings__empty",
                "create a project first (Projects page) to bind an agent to it.",
            ),
        );
    }

    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const values = fields.read();
        const projectValue = project.value;
        if (!values.name || !projectValue) return;
        void dispatch(actions, async () => {
            await actions.create({ ...values, project_id: projectValue });
            fields.name.value = "";
            fields.description.value = "";
            fields.mode.value = "manual";
        });
    });
    return form;
}

export function renderAgents(
    root: HTMLElement,
    agents: Agent[] | null,
    projects: Project[],
    backends: BackendOption[],
    statuses: Record<string, AgentRunStatus>,
    actions: AgentActions,
): void {
    root.replaceChildren();
    if (agents === null) {
        root.appendChild(
            el("div", "settings__empty", "could not load agents."),
        );
        return;
    }

    const head = el("section", "settings__card");
    head.appendChild(el("h2", "settings__title", "Agents"));
    if (agents.length === 0) {
        head.appendChild(el("div", "settings__empty", "no agents yet."));
    }
    head.appendChild(createForm(projects, backends, actions));
    root.appendChild(head);

    if (agents.length > 0) {
        const grid = el("div", "cards");
        for (const agent of agents) {
            grid.appendChild(
                agentCard(agent, projects, statuses[agent.id], actions),
            );
        }
        root.appendChild(grid);
    }
}

export async function startAgents(): Promise<void> {
    const root = document.getElementById("agents");
    if (!root) return;
    let agents: Agent[] | null = null;
    let projects: Project[] = [];
    let backends: BackendOption[] = [];
    let statuses: Record<string, AgentRunStatus> = {};

    const enc = encodeURIComponent;

    const actions: AgentActions = {
        create: (fields) =>
            sendJson<Agent>("/api/agents", "POST", fields).then(
                () => undefined,
            ),
        remove: (id) =>
            sendJson<unknown>(`/api/agents/${enc(id)}`, "DELETE").then(
                () => undefined,
            ),
        open: (id) => {
            window.location.assign(`/agents/${enc(id)}`);
        },
        reload: () => load(),
    };

    const render = (): void => {
        renderAgents(root, agents, projects, backends, statuses, actions);
    };

    // Re-rendering wipes the create-form inputs, so a poll skips the render
    // while the operator is typing there (the next tick catches up).
    const typingInForm = (): boolean => {
        const active = document.activeElement;
        return (
            active !== null &&
            root.contains(active) &&
            ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)
        );
    };

    // Poll each agent's live run status so the cards show fresh turns/tokens.
    // A per-agent status fetch is cheap; the fleet is small.
    const refreshStatuses = async (): Promise<void> => {
        if (!agents) return;
        const entries = await Promise.all(
            agents.map(async (a) => {
                try {
                    const s = await fetchJson<AgentRunStatus>(
                        `/api/agents/${enc(a.id)}/status`,
                    );
                    return [a.id, s] as const;
                } catch {
                    return null;
                }
            }),
        );
        const next: Record<string, AgentRunStatus> = {};
        for (const e of entries) if (e) next[e[0]] = e[1];
        statuses = next;
        if (!typingInForm()) render();
    };

    const load = async (): Promise<void> => {
        try {
            agents = await fetchJson<Agent[]>("/api/agents");
        } catch {
            agents = null;
            render();
            return;
        }
        try {
            projects = await fetchJson<Project[]>("/api/projects");
        } catch {
            projects = [];
        }
        try {
            backends = await fetchJson<BackendOption[]>("/api/agents/backends");
        } catch {
            backends = [];
        }
        render();
        await refreshStatuses();
    };

    window.setInterval(() => {
        if (!typingInForm()) void load();
    }, DEFAULT_POLL_SECONDS * 1000);

    await load();
}
