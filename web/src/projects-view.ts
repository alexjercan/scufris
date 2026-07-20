// Projects page: the visible home of the projects-orchestrator concept. Lists
// projects, creates one, and shows a selected project's metadata + its tatr
// tasks. `renderProjects` is PURE (no fetch) so jsdom tests drive it directly;
// `startProjects` does the fetch orchestration and wires the actions.

import { el, escapeHtml, fetchJson, sendJson } from "./common";
import type { Project, ProjectTask } from "./common";

// Actions the page dispatches. `startProjects` wires these to the API; jsdom
// tests pass fakes. Each resolves after the server applied the change.
export interface ProjectActions {
    create(fields: {
        name: string;
        cwd: string;
        language: string;
        description: string;
    }): Promise<void>;
    remove(id: string): Promise<void>;
    select(id: string | null): void;
    reload(): Promise<void>;
}

async function dispatch(
    actions: ProjectActions,
    run: () => Promise<void>,
): Promise<void> {
    try {
        await run();
        await actions.reload();
    } catch (err: unknown) {
        window.alert(err instanceof Error ? err.message : String(err));
    }
}

function projectList(
    projects: Project[],
    selectedId: string | null,
    actions: ProjectActions,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Projects"));
    if (projects.length === 0) {
        card.appendChild(el("div", "settings__empty", "no projects yet."));
    }
    const list = el("div", "projects");
    for (const project of projects) {
        const active = project.id === selectedId;
        const item = el(
            "div",
            `projects__item${active ? " projects__item--active" : ""}`,
        );
        const open = document.createElement("button");
        open.type = "button";
        open.className = "projects__name";
        open.textContent = project.name;
        open.setAttribute("aria-label", `open ${project.name}`);
        open.addEventListener("click", () => {
            actions.select(active ? null : project.id);
        });
        item.appendChild(open);
        if (project.language) {
            item.appendChild(
                el("span", "projects__badge", escapeHtml(project.language)),
            );
        }
        list.appendChild(item);
    }
    card.appendChild(list);
    card.appendChild(createForm(actions));
    return card;
}

function createForm(actions: ProjectActions): HTMLElement {
    const form = document.createElement("form");
    form.className = "settings__addserver";
    const inputs: Record<string, HTMLInputElement> = {};
    for (const [key, placeholder] of [
        ["name", "name"],
        ["cwd", "cwd (an existing directory)"],
        ["language", "language (optional)"],
        ["description", "description (optional)"],
    ]) {
        const input = document.createElement("input");
        input.type = "text";
        input.placeholder = placeholder;
        input.className = "settings__input";
        input.setAttribute("aria-label", `new project ${key}`);
        inputs[key] = input;
        form.appendChild(input);
    }
    const add = document.createElement("button");
    add.type = "submit";
    add.className = "settings__btn";
    add.textContent = "create project";
    form.appendChild(add);
    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const name = inputs.name.value.trim();
        const cwd = inputs.cwd.value.trim();
        if (!name || !cwd) return;
        void dispatch(actions, async () => {
            await actions.create({
                name,
                cwd,
                language: inputs.language.value.trim(),
                description: inputs.description.value.trim(),
            });
            for (const input of Object.values(inputs)) input.value = "";
        });
    });
    return form;
}

function detailPanel(
    project: Project,
    tasks: ProjectTask[] | null,
    actions: ProjectActions,
): HTMLElement {
    const card = el("section", "settings__card");
    const head = el("div", "settings__row settings__row--control");
    head.appendChild(el("h2", "settings__title", escapeHtml(project.name)));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "settings__btn settings__btn--danger";
    del.textContent = "delete";
    del.setAttribute("aria-label", `delete ${project.name}`);
    del.addEventListener("click", () => {
        if (!window.confirm(`Delete project "${project.name}"?`)) return;
        void dispatch(actions, async () => {
            await actions.remove(project.id);
            actions.select(null);
        });
    });
    head.appendChild(del);
    card.appendChild(head);

    for (const [key, value] of [
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

    card.appendChild(el("h3", "settings__subhead", "Tasks"));
    if (tasks === null) {
        card.appendChild(el("div", "settings__empty", "loading tasks..."));
    } else if (tasks.length === 0) {
        card.appendChild(el("div", "settings__empty", "no tatr tasks here."));
    } else {
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
    }
    return card;
}

export function renderProjects(
    root: HTMLElement,
    projects: Project[] | null,
    selectedId: string | null,
    tasks: ProjectTask[] | null,
    actions: ProjectActions,
): void {
    root.replaceChildren();
    if (projects === null) {
        root.appendChild(
            el("div", "settings__empty", "could not load projects."),
        );
        return;
    }
    root.appendChild(projectList(projects, selectedId, actions));
    const selected = projects.find((p) => p.id === selectedId);
    if (selected) root.appendChild(detailPanel(selected, tasks, actions));
}

export async function startProjects(): Promise<void> {
    const root = document.getElementById("projects");
    if (!root) return;
    let selectedId: string | null = null;
    let tasks: ProjectTask[] | null = null;

    const load = async (): Promise<void> => {
        let projects: Project[] | null;
        try {
            projects = await fetchJson<Project[]>("/api/projects");
        } catch {
            renderProjects(root, null, null, null, actions);
            return;
        }
        // Drop a selection that no longer exists (e.g. after delete).
        if (selectedId && !projects.some((p) => p.id === selectedId)) {
            selectedId = null;
            tasks = null;
        }
        renderProjects(root, projects, selectedId, tasks, actions);
    };

    const actions: ProjectActions = {
        create: (fields) =>
            sendJson<Project>("/api/projects", "POST", fields).then(
                () => undefined,
            ),
        remove: (id) =>
            sendJson<unknown>(
                `/api/projects/${encodeURIComponent(id)}`,
                "DELETE",
            ).then(() => undefined),
        select: (id) => {
            selectedId = id;
            tasks = null;
            void load();
            if (id) {
                // Guard against a race: a slow tasks response for a
                // previously-selected project must not overwrite the current
                // selection's tasks. Only apply/render if still selected.
                void fetchJson<ProjectTask[]>(
                    `/api/projects/${encodeURIComponent(id)}/tasks`,
                )
                    .then((t) => {
                        if (selectedId === id) tasks = t;
                    })
                    .catch(() => {
                        if (selectedId === id) tasks = [];
                    })
                    .finally(() => {
                        if (selectedId === id) void load();
                    });
            }
        },
        reload: load,
    };

    await load();
}
