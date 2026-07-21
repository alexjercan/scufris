// Projects page: the visible home of the projects-orchestrator concept. Lists
// DISCOVERED directories (scanned one level under the base dirs) UNIONed with the
// registered projects - marking which are registered - registers a discovered
// dir, creates a brand-new one under a base dir, and shows a selected registered
// project's metadata + its tatr tasks. `renderProjects` is PURE (no fetch) so
// jsdom tests drive it directly; `startProjects` does the fetch orchestration.

import { el, escapeHtml, fetchJson, sendJson } from "./common";
import type {
    DiscoveredProject,
    DiscoveredProjects,
    Project,
    ProjectTask,
} from "./common";

// Actions the page dispatches. `startProjects` wires these to the API; jsdom
// tests pass fakes. Each resolves after the server applied the change.
export interface ProjectActions {
    // Register an EXISTING (discovered) directory as a project.
    register(fields: {
        name: string;
        cwd: string;
        language: string;
    }): Promise<void>;
    // Create a BRAND-NEW project directory under a base dir, then register it.
    createNew(fields: { name: string; base: string }): Promise<void>;
    remove(id: string): Promise<void>;
    // Open a registered project by its id (null to close); only registered dirs
    // have a detail view.
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

function discoveredRow(
    project: DiscoveredProject,
    selectedId: string | null,
    actions: ProjectActions,
): HTMLElement {
    const active = project.registered && project.project_id === selectedId;
    const item = el(
        "div",
        `projects__item${active ? " projects__item--active" : ""}`,
    );

    // A registered dir opens its detail; an unregistered one is a plain label
    // (its action is "register", not "open").
    if (project.registered && project.project_id) {
        const pid = project.project_id;
        const open = document.createElement("button");
        open.type = "button";
        open.className = "projects__name";
        open.textContent = project.name;
        open.setAttribute("aria-label", `open ${project.name}`);
        open.addEventListener("click", () =>
            actions.select(active ? null : pid),
        );
        item.appendChild(open);
    } else {
        const label = el("span", "projects__name", escapeHtml(project.name));
        item.appendChild(label);
    }

    if (project.language) {
        item.appendChild(
            el("span", "projects__badge", escapeHtml(project.language)),
        );
    }

    if (project.registered) {
        item.appendChild(
            el("span", "projects__tag projects__tag--registered", "registered"),
        );
    } else {
        const reg = document.createElement("button");
        reg.type = "button";
        reg.className = "settings__btn projects__register";
        reg.textContent = "register";
        reg.setAttribute("aria-label", `register ${project.name}`);
        reg.addEventListener("click", () => {
            void dispatch(actions, () =>
                actions.register({
                    name: project.name,
                    cwd: project.path,
                    language: project.language,
                }),
            );
        });
        item.appendChild(reg);
    }
    return item;
}

function projectList(
    data: DiscoveredProjects,
    selectedId: string | null,
    actions: ProjectActions,
): HTMLElement {
    const card = el("section", "settings__card");
    card.appendChild(el("h2", "settings__title", "Projects"));
    if (data.projects.length === 0) {
        card.appendChild(
            el("div", "settings__empty", "no projects or discovered dirs."),
        );
    }
    const list = el("div", "projects");
    for (const project of data.projects) {
        list.appendChild(discoveredRow(project, selectedId, actions));
    }
    card.appendChild(list);
    card.appendChild(createForm(data.base_dirs, actions));
    return card;
}

// Create a NEW directory under a base dir and register it. The base is chosen
// from the server's configured base dirs (no free-typed path here - that path
// would have to already exist; use "register" for an existing dir instead).
function createForm(baseDirs: string[], actions: ProjectActions): HTMLElement {
    const form = document.createElement("form");
    form.className = "settings__addserver";

    const name = document.createElement("input");
    name.type = "text";
    name.placeholder = "new project name";
    name.className = "settings__input";
    name.setAttribute("aria-label", "new project name");
    form.appendChild(name);

    const base = document.createElement("select");
    base.className = "settings__select";
    base.setAttribute("aria-label", "new project base dir");
    for (const dir of baseDirs) {
        const opt = document.createElement("option");
        opt.value = dir;
        opt.textContent = dir;
        base.appendChild(opt);
    }
    form.appendChild(base);

    const add = document.createElement("button");
    add.type = "submit";
    add.className = "settings__btn";
    add.textContent = "create project";
    add.disabled = baseDirs.length === 0; // nowhere to create without a base dir
    form.appendChild(add);

    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const nm = name.value.trim();
        if (!nm || !base.value) return;
        void dispatch(actions, async () => {
            await actions.createNew({ name: nm, base: base.value });
            name.value = "";
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
    data: DiscoveredProjects | null,
    selectedId: string | null,
    selectedProject: Project | null,
    tasks: ProjectTask[] | null,
    actions: ProjectActions,
): void {
    root.replaceChildren();
    if (data === null) {
        root.appendChild(
            el("div", "settings__empty", "could not load projects."),
        );
        return;
    }
    root.appendChild(projectList(data, selectedId, actions));
    if (selectedProject) {
        root.appendChild(detailPanel(selectedProject, tasks, actions));
    }
}

export async function startProjects(): Promise<void> {
    const root = document.getElementById("projects");
    if (!root) return;
    let selectedId: string | null = null;
    let selectedProject: Project | null = null;
    let tasks: ProjectTask[] | null = null;

    const load = async (): Promise<void> => {
        let data: DiscoveredProjects | null;
        try {
            data = await fetchJson<DiscoveredProjects>(
                "/api/projects/discovered",
            );
        } catch {
            renderProjects(root, null, null, null, null, actions);
            return;
        }
        // Drop a selection that no longer exists (e.g. after delete/unregister).
        const stillThere = data.projects.some(
            (p) => p.registered && p.project_id === selectedId,
        );
        if (selectedId && !stillThere) {
            selectedId = null;
            selectedProject = null;
            tasks = null;
        }
        renderProjects(root, data, selectedId, selectedProject, tasks, actions);
    };

    const actions: ProjectActions = {
        register: (fields) =>
            sendJson<Project>("/api/projects", "POST", fields).then(
                () => undefined,
            ),
        createNew: (fields) =>
            sendJson<Project>("/api/projects/new", "POST", fields).then(
                () => undefined,
            ),
        remove: (id) =>
            sendJson<unknown>(
                `/api/projects/${encodeURIComponent(id)}`,
                "DELETE",
            ).then(() => undefined),
        select: (id) => {
            selectedId = id;
            selectedProject = null;
            tasks = null;
            void load();
            if (id) {
                // Load the full project record + its tasks. Guard against a race:
                // a slow response for a previously-selected project must not
                // overwrite the current selection.
                void fetchJson<Project>(
                    `/api/projects/${encodeURIComponent(id)}`,
                )
                    .then((p) => {
                        if (selectedId === id) selectedProject = p;
                    })
                    .catch(() => undefined)
                    .finally(() => {
                        if (selectedId === id) void load();
                    });
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
