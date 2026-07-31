// The per-project detail PAGE (served for /projects/<id> by the backend shell).
// It shows a project's metadata, the agents registered to it (each linking to its
// own page), and its tatr tasks - the drill-in that the /projects list navigates
// to. `renderProjectDetail` is PURE (no fetch) so jsdom drives it directly;
// `startProjectDetail` does the fetch orchestration.

import { el, escapeHtml, fetchJson, sendJson } from "./common";
import type { Agent, Project, ProjectTask } from "./agent-types";

// The project id embedded in the path (/projects/<id>[/...]); null off-route.
export function projectIdFromPath(pathname: string): string | null {
    const m = /^\/projects\/([^/]+)/.exec(pathname);
    return m ? decodeURIComponent(m[1]) : null;
}

// Everything the page renders, fetched up front so the render is pure. `project`
// null means no such project; `tasks` null means still loading.
export interface ProjectDetailData {
    project: Project | null;
    agents: Agent[];
    tasks: ProjectTask[] | null;
}

// The API seam. `startProjectDetail` wires this to the real endpoints; tests pass
// a fake. `remove` deletes the project (the caller navigates back on success).
export interface ProjectDetailActions {
    remove: (id: string) => Promise<void>;
}

function backLink(): HTMLElement {
    const a = document.createElement("a");
    a.href = "/projects/";
    a.className = "agents__back";
    a.textContent = "<- back to projects";
    return a;
}

// A read-only key/value card; string values are escaped.
function metaCard(project: Project): HTMLElement {
    const card = el("section", "settings__card");
    for (const [key, value] of [
        ["id", project.id],
        ["cwd", project.cwd],
        ["language", project.language || "-"],
        ["description", project.description || "-"],
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
    return card;
}

// The agents registered to this project, each a link to its own detail page. An
// empty state when the project has none.
function agentsCard(agents: Agent[]): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(
        el("h2", "settings__title", `Agents (${String(agents.length)})`),
    );
    if (agents.length === 0) {
        card.appendChild(
            el("div", "settings__empty", "no agents on this project."),
        );
        return card;
    }
    const list = el("div", "projects");
    for (const agent of agents) {
        const row = el("div", "projects__item");
        const link = document.createElement("a");
        link.className = "projects__name";
        link.href = `/agents/${encodeURIComponent(agent.id)}`;
        link.textContent = agent.name;
        row.appendChild(link);
        row.appendChild(el("span", "projects__badge", escapeHtml(agent.state)));
        list.appendChild(row);
    }
    card.appendChild(list);
    return card;
}

// The project's tatr tasks (its specs). Loading vs empty vs the list, reusing the
// `projtasks` markup from the projects list.
function tasksCard(tasks: ProjectTask[] | null): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Tasks"));
    if (tasks === null) {
        card.appendChild(el("div", "settings__empty", "loading tasks..."));
        return card;
    }
    if (tasks.length === 0) {
        card.appendChild(el("div", "settings__empty", "no tatr tasks here."));
        return card;
    }
    const list = el("div", "projtasks");
    for (const task of tasks) {
        const tags = task.tags.map((t) => escapeHtml(t)).join(", ");
        list.appendChild(
            el(
                "div",
                "projtasks__row",
                `<span class="projtasks__pri">p${String(task.priority)}</span>` +
                    `<span class="projtasks__title">${escapeHtml(task.title)}</span>` +
                    `<span class="projtasks__tags">${tags}</span>`,
            ),
        );
    }
    card.appendChild(list);
    return card;
}

// Render the whole page (PURE).
export function renderProjectDetail(
    root: HTMLElement,
    data: ProjectDetailData,
    actions: ProjectDetailActions,
): void {
    root.replaceChildren();
    root.appendChild(backLink());
    if (data.project === null) {
        root.appendChild(el("div", "settings__empty", "no such project."));
        return;
    }
    const project = data.project;

    const head = el("div", "settings__row settings__row--control");
    head.appendChild(el("h1", "settings__title", escapeHtml(project.name)));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "settings__btn settings__btn--danger";
    del.textContent = "delete";
    del.setAttribute("aria-label", `delete ${project.name}`);
    del.addEventListener("click", () => {
        if (!window.confirm(`Delete project "${project.name}"?`)) return;
        void (async () => {
            try {
                await actions.remove(project.id);
            } catch (err: unknown) {
                window.alert(err instanceof Error ? err.message : String(err));
            }
        })();
    });
    head.appendChild(del);
    root.appendChild(head);

    root.appendChild(metaCard(project));
    root.appendChild(agentsCard(data.agents));
    root.appendChild(tasksCard(data.tasks));
}

// Best-effort fetch: a panel's data failing must not blank the whole page.
function maybe<T>(url: string): Promise<T | null> {
    return fetchJson<T>(url).catch(() => null);
}

// Load the data, render, and re-render as the async pieces arrive.
export async function startProjectDetail(): Promise<void> {
    const root = document.getElementById("project-detail");
    if (!root) return;
    const id = projectIdFromPath(window.location.pathname);

    const actions: ProjectDetailActions = {
        remove: (pid) =>
            sendJson<unknown>(
                `/api/projects/${encodeURIComponent(pid)}`,
                "DELETE",
            ).then(() => {
                // Back to the list once the project is gone.
                window.location.assign("/projects/");
            }),
    };

    if (!id) {
        renderProjectDetail(
            root,
            { project: null, agents: [], tasks: null },
            actions,
        );
        return;
    }
    const enc = encodeURIComponent(id);

    // The project record decides the page; without it there is nothing to show.
    const project = await maybe<Project>(`/api/projects/${enc}`);
    if (!project) {
        renderProjectDetail(
            root,
            { project: null, agents: [], tasks: null },
            actions,
        );
        return;
    }

    // Render metadata immediately; agents + tasks fill in as they resolve.
    let agents: Agent[] = [];
    let tasks: ProjectTask[] | null = null;
    const render = (): void =>
        renderProjectDetail(root, { project, agents, tasks }, actions);
    render();

    void maybe<Agent[]>("/api/agents").then((all) => {
        agents = (all ?? []).filter((a) => a.project_id === id);
        render();
    });
    void maybe<ProjectTask[]>(`/api/projects/${enc}/tasks`).then((t) => {
        tasks = t ?? [];
        render();
    });
}
