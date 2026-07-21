import { beforeEach, describe, expect, it } from "vitest";

import type {
    DiscoveredProject,
    DiscoveredProjects,
    Project,
    ProjectTask,
} from "./common";
import { renderProjects } from "./projects-view";
import type { ProjectActions } from "./projects-view";

function disco(over: Partial<DiscoveredProject> = {}): DiscoveredProject {
    return {
        path: "/home/alex/personal/my-app",
        name: "my-app",
        language: "python",
        registered: false,
        project_id: null,
        ...over,
    };
}

function data(
    projects: DiscoveredProject[],
    baseDirs: string[] = ["/home/alex/personal"],
): DiscoveredProjects {
    return { projects, base_dirs: baseDirs };
}

function project(over: Partial<Project> = {}): Project {
    return {
        id: "my-app",
        cwd: "/home/alex/personal/my-app",
        name: "My App",
        language: "python",
        description: "a thing",
        ...over,
    };
}

function task(over: Partial<ProjectTask> = {}): ProjectTask {
    return {
        id: "20260720-120000",
        title: "spec one",
        priority: 20,
        tags: ["feature"],
        ...over,
    };
}

function fakeActions(over: Partial<ProjectActions> = {}): ProjectActions {
    return {
        register: () => Promise.resolve(),
        createNew: () => Promise.resolve(),
        remove: () => Promise.resolve(),
        select: () => undefined,
        reload: () => Promise.resolve(),
        ...over,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="projects"></main>';
    root = document.getElementById("projects") as HTMLElement;
});

describe("renderProjects", () => {
    it("lists discovered dirs and shows a create form", () => {
        renderProjects(root, data([disco()]), null, null, null, fakeActions());
        expect(root.textContent).toContain("Projects");
        expect(root.textContent).toContain("my-app");
        expect(root.textContent).toContain("python"); // language badge
        expect(root.querySelector(".settings__addserver")).not.toBeNull();
        // No detail panel until a registered project is selected.
        expect(root.textContent).not.toContain("cwd");
    });

    it("shows an empty state when there are no dirs", () => {
        renderProjects(root, data([]), null, null, null, fakeActions());
        expect(root.textContent).toContain("no projects or discovered dirs.");
    });

    it("shows a fallback when the page could not load", () => {
        renderProjects(root, null, null, null, null, fakeActions());
        expect(root.textContent).toContain("could not load projects.");
    });

    it("marks a registered dir and opens its detail on click", () => {
        const selected: (string | null)[] = [];
        renderProjects(
            root,
            data([disco({ registered: true, project_id: "my-app" })]),
            null,
            null,
            null,
            fakeActions({ select: (id) => selected.push(id) }),
        );
        expect(root.textContent).toContain("registered");
        const open = root.querySelector(
            '.projects__name[aria-label="open my-app"]',
        ) as HTMLButtonElement;
        open.dispatchEvent(new Event("click"));
        expect(selected).toEqual(["my-app"]);
    });

    it("registers a discovered (unregistered) dir via its register button", async () => {
        const registered: unknown[] = [];
        renderProjects(
            root,
            data([
                disco({
                    name: "fresh",
                    path: "/home/alex/personal/fresh",
                    language: "",
                }),
            ]),
            null,
            null,
            null,
            fakeActions({
                register: (f) => {
                    registered.push(f);
                    return Promise.resolve();
                },
            }),
        );
        // An unregistered dir has no "open" button, just a register action.
        expect(
            root.querySelector('.projects__name[aria-label="open fresh"]'),
        ).toBeNull();
        const reg = root.querySelector(
            '.projects__register[aria-label="register fresh"]',
        ) as HTMLButtonElement;
        reg.dispatchEvent(new Event("click"));
        await flush();
        expect(registered).toEqual([
            { name: "fresh", cwd: "/home/alex/personal/fresh", language: "" },
        ]);
    });

    it("renders a selected registered project's detail + tasks", () => {
        renderProjects(
            root,
            data([disco({ registered: true, project_id: "my-app" })]),
            "my-app",
            project(),
            [task(), task({ title: "spec two", priority: 5, tags: ["bug"] })],
            fakeActions(),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("/home/alex/personal/my-app"); // cwd
        expect(text).toContain("a thing"); // description
        const rows = root.querySelectorAll(".projtasks__row");
        expect(rows.length).toBe(2);
        expect(text).toContain("spec one");
        expect(text).toContain("p20");
    });

    it("creates a new project from the form (name + base picker)", async () => {
        const created: unknown[] = [];
        renderProjects(
            root,
            data([], ["/home/alex/personal", "/home/alex/work"]),
            null,
            null,
            null,
            fakeActions({
                createNew: (f) => {
                    created.push(f);
                    return Promise.resolve();
                },
            }),
        );
        const nameIn = root.querySelector(
            'input[aria-label="new project name"]',
        ) as HTMLInputElement;
        const baseSel = root.querySelector(
            'select[aria-label="new project base dir"]',
        ) as HTMLSelectElement;
        expect([...baseSel.options].map((o) => o.value)).toEqual([
            "/home/alex/personal",
            "/home/alex/work",
        ]);
        nameIn.value = "New";
        baseSel.value = "/home/alex/work";
        nameIn.form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(created).toEqual([{ name: "New", base: "/home/alex/work" }]);
    });

    it("disables create and does not submit without a name", async () => {
        const created: unknown[] = [];
        renderProjects(
            root,
            data([], []), // no base dirs -> create disabled
            null,
            null,
            null,
            fakeActions({
                createNew: (f) => {
                    created.push(f);
                    return Promise.resolve();
                },
            }),
        );
        const add = root.querySelector(
            ".settings__addserver .settings__btn",
        ) as HTMLButtonElement;
        expect(add.disabled).toBe(true); // nowhere to create
        // Even with base dirs, an empty name does not submit.
        renderProjects(
            root,
            data([], ["/home/alex/personal"]),
            null,
            null,
            null,
            fakeActions({
                createNew: (f) => {
                    created.push(f);
                    return Promise.resolve();
                },
            }),
        );
        root.querySelector(".settings__addserver")?.dispatchEvent(
            new Event("submit"),
        );
        await flush();
        expect(created).toEqual([]);
    });

    it("escapes a hostile discovered name (no injection)", () => {
        renderProjects(
            root,
            data([disco({ name: "<img src=x onerror=alert(1)>" })]),
            null,
            null,
            null,
            fakeActions(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });

    it("escapes hostile tatr task titles/tags in the detail", () => {
        renderProjects(
            root,
            data([disco({ registered: true, project_id: "my-app" })]),
            "my-app",
            project(),
            [task({ title: "<img src=x onerror=alert(3)>", tags: ["<b>"] })],
            fakeActions(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(3)>");
    });
});
