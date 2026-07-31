// Projects page: the visible home of the projects-orchestrator concept. Lists
// DISCOVERED directories (scanned one level under the base dirs) UNIONed with the
// registered projects - marking which are registered - registers a discovered dir
// and creates a brand-new one under a base dir. A registered project's name links
// to its detail page (/projects/<id>), where its metadata, agents and tatr tasks
// live. `renderProjects` is PURE (no fetch) so jsdom tests drive it directly;
// `startProjects` does the fetch orchestration.

import { el, escapeHtml, fetchJson, sendJson } from "./common";
import type {
    DiscoveredProject,
    DiscoveredProjects,
    Project,
} from "./agent-types";

type ProjectRegistrationFilter = "all" | "registered" | "unregistered";

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
    actions: ProjectActions,
): HTMLElement {
    const item = el("div", "projects__item");
    item.dataset.registration = project.registered
        ? "registered"
        : "unregistered";

    // A registered dir's name links to its detail page; an unregistered one is a
    // plain label (its action is "register", not "open").
    if (project.registered && project.project_id) {
        const open = document.createElement("a");
        open.className = "projects__name";
        open.href = `/projects/${encodeURIComponent(project.project_id)}`;
        open.textContent = project.name;
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

function filterLabel(filter: ProjectRegistrationFilter): string {
    if (filter === "registered") return "registered";
    if (filter === "unregistered") return "unregistered";
    return "all";
}

function applyProjectFilter(
    list: HTMLElement,
    empty: HTMLElement,
    filter: ProjectRegistrationFilter,
): void {
    let visible = 0;
    for (const row of list.querySelectorAll<HTMLElement>(".projects__item")) {
        const matches = filter === "all" || row.dataset.registration === filter;
        row.hidden = !matches;
        if (matches) visible += 1;
    }
    empty.hidden = visible > 0;
    empty.textContent = `no ${filterLabel(filter)} projects.`;
}

function filterControl(
    list: HTMLElement,
    empty: HTMLElement,
): HTMLSelectElement {
    const select = document.createElement("select");
    select.className = "settings__select projects__filter";
    select.setAttribute("aria-label", "project registration filter");

    const options: { value: ProjectRegistrationFilter; label: string }[] = [
        { value: "all", label: "all projects" },
        { value: "registered", label: "registered" },
        { value: "unregistered", label: "unregistered" },
    ];
    for (const option of options) {
        const opt = document.createElement("option");
        opt.value = option.value;
        opt.textContent = option.label;
        select.appendChild(opt);
    }

    select.addEventListener("change", () => {
        applyProjectFilter(
            list,
            empty,
            select.value as ProjectRegistrationFilter,
        );
    });
    return select;
}

function projectList(
    data: DiscoveredProjects,
    actions: ProjectActions,
): HTMLElement {
    const card = el("section", "settings__card");
    const heading = document.createElement("div");
    heading.className = "projects__header";
    heading.appendChild(el("h2", "settings__title", "Projects"));
    card.appendChild(heading);
    if (data.projects.length === 0) {
        card.appendChild(
            el("div", "settings__empty", "no projects or discovered dirs."),
        );
    }
    const list = el("div", "projects");
    for (const project of data.projects) {
        list.appendChild(discoveredRow(project, actions));
    }
    const filterEmpty = el("div", "settings__empty projects__filter-empty");
    filterEmpty.hidden = true;
    if (data.projects.length > 0) {
        heading.appendChild(filterControl(list, filterEmpty));
    }
    card.appendChild(list);
    card.appendChild(filterEmpty);
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

export function renderProjects(
    root: HTMLElement,
    data: DiscoveredProjects | null,
    actions: ProjectActions,
): void {
    root.replaceChildren();
    if (data === null) {
        root.appendChild(
            el("div", "settings__empty", "could not load projects."),
        );
        return;
    }
    root.appendChild(projectList(data, actions));
}

export async function startProjects(): Promise<void> {
    const root = document.getElementById("projects");
    if (!root) return;

    const load = async (): Promise<void> => {
        let data: DiscoveredProjects | null;
        try {
            data = await fetchJson<DiscoveredProjects>(
                "/api/projects/discovered",
            );
        } catch {
            renderProjects(root, null, actions);
            return;
        }
        renderProjects(root, data, actions);
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
        reload: load,
    };

    await load();
}
