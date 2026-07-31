import { beforeEach, describe, expect, it } from "vitest";

import type { DiscoveredProject, DiscoveredProjects } from "./agent-types";
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

function fakeActions(over: Partial<ProjectActions> = {}): ProjectActions {
    return {
        register: () => Promise.resolve(),
        createNew: () => Promise.resolve(),
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
        renderProjects(root, data([disco()]), fakeActions());
        expect(root.textContent).toContain("Projects");
        expect(root.textContent).toContain("my-app");
        expect(root.textContent).toContain("python"); // language badge
        expect(root.querySelector(".settings__addserver")).not.toBeNull();
        // The list has no inline detail panel (that lives on /projects/<id>).
        expect(root.textContent).not.toContain("cwd");
    });

    it("shows an empty state when there are no dirs", () => {
        renderProjects(root, data([]), fakeActions());
        expect(root.textContent).toContain("no projects or discovered dirs.");
    });

    it("shows a fallback when the page could not load", () => {
        renderProjects(root, null, fakeActions());
        expect(root.textContent).toContain("could not load projects.");
    });

    it("links a registered dir's name to its detail page", () => {
        renderProjects(
            root,
            data([disco({ registered: true, project_id: "my-app" })]),
            fakeActions(),
        );
        expect(root.textContent).toContain("registered");
        const link = root.querySelector<HTMLAnchorElement>("a.projects__name");
        expect(link?.getAttribute("href")).toBe("/projects/my-app");
        expect(link?.textContent).toBe("my-app");
    });

    it("filters projects by registration status", () => {
        renderProjects(
            root,
            data([
                disco({ name: "tracked", registered: true, project_id: "p1" }),
                disco({ name: "fresh", registered: false, project_id: null }),
            ]),
            fakeActions(),
        );

        const filter = root.querySelector(
            'select[aria-label="project registration filter"]',
        ) as HTMLSelectElement;
        const rows = [...root.querySelectorAll<HTMLElement>(".projects__item")];

        expect(filter).not.toBeNull();
        expect(rows.map((row) => row.hidden)).toEqual([false, false]);

        filter.value = "registered";
        filter.dispatchEvent(new Event("change"));
        expect(rows.map((row) => row.hidden)).toEqual([false, true]);
        expect(root.textContent).toContain("tracked");

        filter.value = "unregistered";
        filter.dispatchEvent(new Event("change"));
        expect(rows.map((row) => row.hidden)).toEqual([true, false]);
        expect(root.textContent).toContain("fresh");
    });

    it("shows an empty state when a registration filter has no matches", () => {
        renderProjects(
            root,
            data([
                disco({ name: "tracked", registered: true, project_id: "p1" }),
            ]),
            fakeActions(),
        );

        const filter = root.querySelector(
            'select[aria-label="project registration filter"]',
        ) as HTMLSelectElement;
        const empty = root.querySelector<HTMLElement>(
            ".projects__filter-empty",
        );

        filter.value = "unregistered";
        filter.dispatchEvent(new Event("change"));

        expect(empty?.hidden).toBe(false);
        expect(empty?.textContent).toBe("no unregistered projects.");
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
            fakeActions({
                register: (f) => {
                    registered.push(f);
                    return Promise.resolve();
                },
            }),
        );
        // An unregistered dir has no link (name is a plain label), just register.
        expect(root.querySelector("a.projects__name")).toBeNull();
        const reg = root.querySelector(
            '.projects__register[aria-label="register fresh"]',
        ) as HTMLButtonElement;
        reg.dispatchEvent(new Event("click"));
        await flush();
        expect(registered).toEqual([
            { name: "fresh", cwd: "/home/alex/personal/fresh", language: "" },
        ]);
    });

    it("creates a new project from the form (name + base picker)", async () => {
        const created: unknown[] = [];
        renderProjects(
            root,
            data([], ["/home/alex/personal", "/home/alex/work"]),
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
            fakeActions(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });
});
