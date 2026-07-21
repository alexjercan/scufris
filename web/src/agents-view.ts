// Agents dashboard: the cockpit for managing agents that run on projects. Lists
// agents with a live state badge, creates one (bound to a project), and shows a
// selected agent's config + live run status, a Run button, and a live events
// log. `renderAgents` is PURE (no fetch) so jsdom tests drive it directly;
// `startAgents` does the fetch orchestration, status polling and the SSE events
// stream, and wires the actions.

import {
    AGENT_BACKENDS,
    DEFAULT_POLL_SECONDS,
    el,
    escapeHtml,
    fetchJson,
    sendJson,
} from "./common";
import type { Agent, AgentRunStatus, Project } from "./common";

export interface AgentCreateFields {
    name: string;
    project_id: string;
    backend: string;
    goal: string;
    write_enabled: boolean;
}

// Actions the page dispatches. `startAgents` wires these to the API; jsdom tests
// pass fakes. Each mutating action resolves after the server applied it.
export interface AgentActions {
    create(fields: AgentCreateFields): Promise<void>;
    remove(id: string): Promise<void>;
    run(id: string): Promise<void>;
    select(id: string | null): void;
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

function agentList(
    agents: Agent[],
    projects: Project[],
    selectedId: string | null,
    actions: AgentActions,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Agents"));
    if (agents.length === 0) {
        card.appendChild(el("div", "settings__empty", "no agents yet."));
    }
    const list = el("div", "agents");
    for (const agent of agents) {
        const active = agent.id === selectedId;
        const item = el(
            "div",
            `agents__item${active ? " agents__item--active" : ""}`,
        );
        const open = document.createElement("button");
        open.type = "button";
        open.className = "agents__name";
        open.textContent = agent.name;
        open.setAttribute("aria-label", `open ${agent.name}`);
        open.addEventListener("click", () => {
            actions.select(active ? null : agent.id);
        });
        item.appendChild(open);
        item.appendChild(stateBadge(agent.state));
        item.appendChild(el("span", "agents__meta", escapeHtml(agent.backend)));
        list.appendChild(item);
    }
    card.appendChild(list);
    card.appendChild(createForm(projects, actions));
    return card;
}

function createForm(projects: Project[], actions: AgentActions): HTMLElement {
    const form = document.createElement("form");
    form.className = "settings__addserver";

    const name = document.createElement("input");
    name.type = "text";
    name.placeholder = "name";
    name.className = "settings__input";
    name.setAttribute("aria-label", "new agent name");
    form.appendChild(name);

    const project = document.createElement("select");
    project.className = "settings__input";
    project.setAttribute("aria-label", "new agent project");
    for (const p of projects) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.name;
        project.appendChild(opt);
    }
    form.appendChild(project);

    const backend = document.createElement("select");
    backend.className = "settings__input";
    backend.setAttribute("aria-label", "new agent backend");
    for (const b of AGENT_BACKENDS) {
        const opt = document.createElement("option");
        opt.value = b;
        opt.textContent = b;
        backend.appendChild(opt);
    }
    form.appendChild(backend);

    const goal = document.createElement("textarea");
    goal.placeholder = "goal (what should this agent do?)";
    goal.className = "settings__input";
    goal.setAttribute("aria-label", "new agent goal");
    form.appendChild(goal);

    const writeLabel = el("label", "agents__check");
    const write = document.createElement("input");
    write.type = "checkbox";
    write.setAttribute("aria-label", "new agent write enabled");
    writeLabel.appendChild(write);
    writeLabel.appendChild(document.createTextNode(" allow writes"));
    form.appendChild(writeLabel);

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
        const nameValue = name.value.trim();
        const projectValue = project.value;
        if (!nameValue || !projectValue) return;
        void dispatch(actions, async () => {
            await actions.create({
                name: nameValue,
                project_id: projectValue,
                backend: backend.value,
                goal: goal.value.trim(),
                write_enabled: write.checked,
            });
            name.value = "";
            goal.value = "";
            write.checked = false;
        });
    });
    return form;
}

function statusRows(status: AgentRunStatus): HTMLElement {
    const wrap = el("div", "agents__status");
    // A never-run agent has no session and an idle state - show that, not 0s
    // that read like real (but meaningless) counters.
    if (status.state === "idle" && !status.session_id) {
        wrap.appendChild(
            el(
                "div",
                "settings__empty",
                "not started - run this agent to begin.",
            ),
        );
        return wrap;
    }
    const rows: [string, string][] = [
        ["turns", String(status.turns)],
        ["tools", String(status.tool_calls)],
        ["input tokens", String(status.input_tokens)],
        ["output tokens", String(status.output_tokens)],
    ];
    for (const [key, value] of rows) {
        wrap.appendChild(
            el(
                "div",
                "settings__row",
                `<span class="settings__key">${escapeHtml(key)}</span>` +
                    `<span class="settings__val">${escapeHtml(value)}</span>`,
            ),
        );
    }
    if (status.last_message) {
        wrap.appendChild(el("h3", "settings__subhead", "Last message"));
        wrap.appendChild(
            el("div", "agents__lastmsg", escapeHtml(status.last_message)),
        );
    }
    return wrap;
}

function detailPanel(
    agent: Agent,
    projects: Project[],
    status: AgentRunStatus | null,
    actions: AgentActions,
): HTMLElement {
    const card = el("section", "settings__card");
    const head = el("div", "settings__row settings__row--control");
    const title = el("h2", "settings__title", escapeHtml(agent.name));
    head.appendChild(title);
    head.appendChild(stateBadge(status ? status.state : agent.state));
    card.appendChild(head);

    const project = projects.find((p) => p.id === agent.project_id);
    for (const [key, value] of [
        ["project", project ? project.name : agent.project_id],
        ["backend", agent.backend],
        ["model", agent.model || "-"],
        ["goal", agent.goal || "-"],
        ["writes", agent.write_enabled ? "enabled" : "read-only"],
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

    const controls = el("div", "settings__row settings__row--control");
    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.className = "settings__btn";
    runBtn.textContent = "run";
    runBtn.setAttribute("aria-label", `run ${agent.name}`);
    runBtn.addEventListener("click", () => {
        void dispatch(actions, () => actions.run(agent.id));
    });
    controls.appendChild(runBtn);
    const del = document.createElement("button");
    del.type = "button";
    del.className = "settings__btn settings__btn--danger";
    del.textContent = "delete";
    del.setAttribute("aria-label", `delete ${agent.name}`);
    del.addEventListener("click", () => {
        if (!window.confirm(`Delete agent "${agent.name}"?`)) return;
        void dispatch(actions, async () => {
            await actions.remove(agent.id);
            actions.select(null);
        });
    });
    controls.appendChild(del);
    card.appendChild(controls);

    card.appendChild(el("h3", "settings__subhead", "Status"));
    if (status === null) {
        card.appendChild(el("div", "settings__empty", "loading status..."));
    } else {
        card.appendChild(statusRows(status));
    }

    card.appendChild(el("h3", "settings__subhead", "Live events"));
    const log = el("div", "agents__events");
    log.id = "agent-events";
    card.appendChild(log);
    return card;
}

export function renderAgents(
    root: HTMLElement,
    agents: Agent[] | null,
    projects: Project[],
    selectedId: string | null,
    status: AgentRunStatus | null,
    actions: AgentActions,
): void {
    root.replaceChildren();
    if (agents === null) {
        root.appendChild(
            el("div", "settings__empty", "could not load agents."),
        );
        return;
    }
    root.appendChild(agentList(agents, projects, selectedId, actions));
    const selected = agents.find((a) => a.id === selectedId);
    if (selected) {
        root.appendChild(detailPanel(selected, projects, status, actions));
    }
}

export async function startAgents(): Promise<void> {
    const root = document.getElementById("agents");
    if (!root) return;
    let selectedId: string | null = null;
    let status: AgentRunStatus | null = null;
    let projects: Project[] = [];
    let events: EventSource | null = null;

    const closeEvents = (): void => {
        if (events) {
            events.close();
            events = null;
        }
    };

    const load = async (): Promise<void> => {
        let agents: Agent[] | null;
        try {
            agents = await fetchJson<Agent[]>("/api/agents");
        } catch {
            renderAgents(root, null, projects, null, null, actions);
            return;
        }
        try {
            projects = await fetchJson<Project[]>("/api/projects");
        } catch {
            projects = [];
        }
        if (selectedId && !agents.some((a) => a.id === selectedId)) {
            selectedId = null;
            status = null;
            closeEvents();
        }
        renderAgents(root, agents, projects, selectedId, status, actions);
    };

    const isActive = (s: AgentRunStatus | null): boolean =>
        s !== null && (s.state === "running" || s.state === "queued");

    const pollStatus = (id: string): void => {
        void fetchJson<AgentRunStatus>(
            `/api/agents/${encodeURIComponent(id)}/status`,
        )
            .then((s) => {
                if (selectedId !== id) return;
                status = s;
                // Reattach the live event stream for an already-running agent
                // (e.g. selected mid-run). Only when active, so we never open an
                // EventSource on an idle agent (a 404 that would auto-reconnect).
                if (isActive(s) && events === null) openEvents(id);
            })
            .catch(() => {
                /* leave prior status */
            })
            .finally(() => {
                if (selectedId === id) void load();
            });
    };

    const openEvents = (id: string): void => {
        closeEvents();
        const source = new EventSource(
            `/api/agents/${encodeURIComponent(id)}/events`,
        );
        events = source;
        source.onmessage = (ev: MessageEvent<string>) => {
            if (selectedId !== id) return;
            const log = document.getElementById("agent-events");
            if (!log) return;
            const line = el("div", "agents__eventline");
            line.textContent = ev.data;
            log.appendChild(line);
            // Refresh the polled status as events arrive.
            pollStatus(id);
        };
        // A 404 (no active run) or a closed stream just ends quietly. Guard on
        // identity so a stale source's late onerror cannot close a newer stream.
        source.onerror = () => {
            if (events === source) closeEvents();
        };
    };

    const actions: AgentActions = {
        create: (fields) =>
            sendJson<Agent>("/api/agents", "POST", fields).then(
                () => undefined,
            ),
        remove: (id) =>
            sendJson<unknown>(
                `/api/agents/${encodeURIComponent(id)}`,
                "DELETE",
            ).then(() => undefined),
        run: (id) =>
            sendJson<unknown>(
                `/api/agents/${encodeURIComponent(id)}/run`,
                "POST",
                {},
            ).then(() => {
                openEvents(id);
                pollStatus(id);
            }),
        select: (id) => {
            selectedId = id;
            status = null;
            closeEvents();
            void load();
            if (id) pollStatus(id);
        },
        reload: load,
    };

    // Keep a running/queued agent's status fresh between SSE events. Bounded to
    // an active selected agent, and skipped while the operator is typing in the
    // create form (a full re-render would wipe the input).
    const typingInForm = (): boolean => {
        const active = document.activeElement;
        return (
            active !== null &&
            root.contains(active) &&
            ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)
        );
    };
    window.setInterval(() => {
        if (selectedId && isActive(status) && !typingInForm()) {
            pollStatus(selectedId);
        }
    }, DEFAULT_POLL_SECONDS * 1000);

    await load();
}
