import { beforeEach, describe, expect, it } from "vitest";

import type { Project, ProjectTask } from "./common";
import { renderProjects } from "./projects-view";
import type { ProjectActions } from "./projects-view";

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
        create: () => Promise.resolve(),
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
    it("lists projects and shows a create form", () => {
        renderProjects(root, [project()], null, null, fakeActions());
        expect(root.textContent).toContain("Projects");
        expect(root.textContent).toContain("My App");
        expect(root.querySelector(".settings__addserver")).not.toBeNull();
        // No detail panel until one is selected.
        expect(root.textContent).not.toContain("cwd");
    });

    it("shows an empty state when there are no projects", () => {
        renderProjects(root, [], null, null, fakeActions());
        expect(root.textContent).toContain("no projects yet.");
    });

    it("shows a fallback when projects could not load", () => {
        renderProjects(root, null, null, null, fakeActions());
        expect(root.textContent).toContain("could not load projects.");
    });

    it("renders a selected project's detail with metadata", () => {
        renderProjects(root, [project()], "my-app", null, fakeActions());
        const text = root.textContent ?? "";
        expect(text).toContain("/home/alex/personal/my-app"); // cwd
        expect(text).toContain("python"); // language
        expect(text).toContain("a thing"); // description
        expect(text).toContain("loading tasks..."); // tasks null
    });

    it("renders the selected project's tatr tasks", () => {
        renderProjects(
            root,
            [project()],
            "my-app",
            [task(), task({ title: "spec two", priority: 5, tags: ["bug"] })],
            fakeActions(),
        );
        const rows = root.querySelectorAll(".projtasks__row");
        expect(rows.length).toBe(2);
        expect(root.textContent).toContain("spec one");
        expect(root.textContent).toContain("p20");
        expect(root.textContent).toContain("spec two");
    });

    it("shows an empty-tasks message when the project has none", () => {
        renderProjects(root, [project()], "my-app", [], fakeActions());
        expect(root.textContent).toContain("no tatr tasks here.");
    });

    it("selects a project when its name is clicked", () => {
        const selected: (string | null)[] = [];
        renderProjects(
            root,
            [project()],
            null,
            null,
            fakeActions({ select: (id) => selected.push(id) }),
        );
        const open = root.querySelector(
            '.projects__name[aria-label="open My App"]',
        ) as HTMLButtonElement;
        open.dispatchEvent(new Event("click"));
        expect(selected).toEqual(["my-app"]);
    });

    it("creates a project from the form", async () => {
        const created: unknown[] = [];
        renderProjects(
            root,
            [],
            null,
            null,
            fakeActions({
                create: (fields) => {
                    created.push(fields);
                    return Promise.resolve();
                },
            }),
        );
        const nameIn = root.querySelector(
            'input[aria-label="new project name"]',
        ) as HTMLInputElement;
        const cwdIn = root.querySelector(
            'input[aria-label="new project cwd"]',
        ) as HTMLInputElement;
        nameIn.value = "New";
        cwdIn.value = "/tmp/new";
        nameIn.form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(created).toEqual([
            { name: "New", cwd: "/tmp/new", language: "", description: "" },
        ]);
    });

    it("does not submit create without a name and cwd", async () => {
        const created: unknown[] = [];
        renderProjects(
            root,
            [],
            null,
            null,
            fakeActions({
                create: (fields) => {
                    created.push(fields);
                    return Promise.resolve();
                },
            }),
        );
        const nameIn = root.querySelector(
            'input[aria-label="new project name"]',
        ) as HTMLInputElement;
        nameIn.value = "OnlyName";
        nameIn.form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(created).toEqual([]); // no cwd -> no create
    });

    it("escapes a hostile project name/description (no injection)", () => {
        renderProjects(
            root,
            [
                project({
                    id: "x",
                    name: "<img src=x onerror=alert(1)>",
                    description: "<script>alert(2)</script>",
                }),
            ],
            "x",
            [],
            fakeActions(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.querySelector("script")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });

    it("escapes hostile tatr task titles/tags", () => {
        renderProjects(
            root,
            [project()],
            "my-app",
            [task({ title: "<img src=x onerror=alert(3)>", tags: ["<b>"] })],
            fakeActions(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(3)>");
    });
});
