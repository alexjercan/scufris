// The per-agent detail page (served for /agents/<id> by the backend SPA shell),
// reshaped chat-first: the chat (agent-chat-view) is the primary right pane; this
// module renders the LEFT sidebar (agent header + live status/context stat boxes
// + a "Settings" button) and the Settings MODAL (the editable F3 form). The
// render functions are PURE so jsdom tests drive them; `startAgentDetail` fetches,
// polls `/status`, and wires the modal toggle + the save action.

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

// The API seam. `startAgentDetail` wires `save` to PATCH; jsdom tests pass fakes.
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

function readonlyRow(key: string, value: string): HTMLElement {
    return el(
        "div",
        "settings__row",
        `<span class="settings__key">${escapeHtml(key)}</span>` +
            `<span class="settings__val">${escapeHtml(value)}</span>`,
    );
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

// The LEFT sidebar: agent header + Settings button + live stat boxes. Pure.
// `onOpenSettings` is the UI callback the Settings button fires (open the modal).
export function renderSidebar(
    root: HTMLElement,
    agent: Agent | null,
    project: Project | null,
    status: AgentRunStatus | null,
    onOpenSettings: () => void,
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
            escapeHtml(project ? project.name : agent.project_id),
        ),
    );

    const settingsBtn = document.createElement("button");
    settingsBtn.type = "button";
    settingsBtn.className = "sidebar__new";
    settingsBtn.textContent = "Settings";
    settingsBtn.setAttribute("aria-label", "open settings");
    settingsBtn.addEventListener("click", onOpenSettings);
    root.appendChild(settingsBtn);

    root.appendChild(statusBox(status));
    root.appendChild(contextBox(status));
}

// The editable settings form (F3), prefilled with the agent's current values.
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
        fields.modelList,
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

// The Settings MODAL content (the F3 form inside an overlay card). Pure.
// `onClose` hides the modal. The project is read-only (fixed after creation).
export function renderSettingsModal(
    root: HTMLElement,
    agent: Agent,
    project: Project | null,
    backends: BackendOption[],
    actions: AgentDetailActions,
    onClose: () => void,
): void {
    root.replaceChildren();
    const card = el("section", "agent-modal__card settings__card");

    const head = el("div", "settings__row settings__row--control");
    head.appendChild(el("h2", "settings__title", "Settings"));
    const close = document.createElement("button");
    close.type = "button";
    close.className = "settings__btn";
    close.textContent = "close";
    close.setAttribute("aria-label", "close settings");
    close.addEventListener("click", onClose);
    head.appendChild(close);
    card.appendChild(head);

    card.appendChild(
        readonlyRow("project", project ? project.name : agent.project_id),
    );
    card.appendChild(settingsForm(agent, backends, actions));
    root.appendChild(card);

    // Clicking the backdrop (the overlay itself, not the card) closes it.
    // Use `onclick` (property, not addEventListener) so a re-render REPLACES the
    // handler instead of stacking a new one on the persistent modal root.
    root.onclick = (ev): void => {
        if (ev.target === root) onClose();
    };
}

export async function startAgentDetail(): Promise<void> {
    const sidebar = document.getElementById("agent-sidebar");
    const modal = document.getElementById("agent-settings-modal");
    if (!sidebar || !modal) return;
    const id = agentIdFromPath(window.location.pathname);

    let agent: Agent | null = null;
    let project: Project | null = null;
    let backends: BackendOption[] = [];
    let status: AgentRunStatus | null = null;

    const closeSettings = (): void => {
        modal.hidden = true;
        modal.replaceChildren();
    };
    const openSettings = (): void => {
        if (!agent) return;
        renderSettingsModal(
            modal,
            agent,
            project,
            backends,
            actions,
            closeSettings,
        );
        modal.hidden = false;
    };

    const render = (): void => {
        renderSidebar(sidebar, agent, project, status, openSettings);
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
            closeSettings();
            await load();
        },
    };

    const pollStatus = (): void => {
        if (!id) return;
        void fetchJson<AgentRunStatus>(
            `/api/agents/${encodeURIComponent(id)}/status`,
        )
            .then((s) => {
                status = s;
                // The sidebar has no inputs; re-rendering it never wipes an edit
                // (the settings form lives in the separate modal root).
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
