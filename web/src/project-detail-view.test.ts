import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Agent, Project, ProjectTask } from "./common";
import {
    projectIdFromPath,
    renderProjectDetail,
    type ProjectDetailActions,
    type ProjectDetailData,
} from "./project-detail-view";

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

function agent(over: Partial<Agent> = {}): Agent {
    return {
        id: "builder",
        name: "Builder",
        project_id: "my-app",
        backend: "codex",
        model: "gpt-5.5",
        description: "",
        goal: "",
        task_id: "",
        session_id: null,
        state: "idle",
        permission_mode: "manual",
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

function data(over: Partial<ProjectDetailData> = {}): ProjectDetailData {
    return {
        project: project(),
        agents: [agent()],
        tasks: [task()],
        ...over,
    };
}

function fakeActions(
    over: Partial<ProjectDetailActions> = {},
): ProjectDetailActions {
    return { remove: () => Promise.resolve(), ...over };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="root"></main>';
    root = document.getElementById("root") as HTMLElement;
});

describe("projectIdFromPath", () => {
    it("extracts the id from /projects/<id>[/...]", () => {
        expect(projectIdFromPath("/projects/my-app")).toBe("my-app");
        expect(projectIdFromPath("/projects/my-app/anything")).toBe("my-app");
        expect(projectIdFromPath("/projects/")).toBeNull();
        expect(projectIdFromPath("/agents/x")).toBeNull();
    });
});

describe("renderProjectDetail", () => {
    it("renders metadata, the project's agents (linked), and its tasks", () => {
        renderProjectDetail(root, data(), fakeActions());
        const text = root.textContent ?? "";
        // Metadata.
        expect(text).toContain("My App");
        expect(text).toContain("my-app"); // id
        expect(text).toContain("/home/alex/personal/my-app"); // cwd
        expect(text).toContain("a thing"); // description
        // The agent links to its own page.
        const agentLink = root.querySelector<HTMLAnchorElement>(
            'a.projects__name[href="/agents/builder"]',
        );
        expect(agentLink?.textContent).toBe("Builder");
        expect(text).toContain("Agents (1)");
        // The tasks.
        expect(root.querySelectorAll(".projtasks__row").length).toBe(1);
        expect(text).toContain("spec one");
        expect(text).toContain("p20");
        // A back link to the list.
        expect(
            root
                .querySelector<HTMLAnchorElement>(".agents__back")
                ?.getAttribute("href"),
        ).toBe("/projects/");
    });

    it("shows empty states for no agents and no tasks", () => {
        renderProjectDetail(
            root,
            data({ agents: [], tasks: [] }),
            fakeActions(),
        );
        expect(root.textContent).toContain("no agents on this project.");
        expect(root.textContent).toContain("no tatr tasks here.");
        expect(root.textContent).toContain("Agents (0)");
    });

    it("shows a loading state while tasks are null", () => {
        renderProjectDetail(root, data({ tasks: null }), fakeActions());
        expect(root.textContent).toContain("loading tasks...");
    });

    it("shows a fallback for an unknown project", () => {
        renderProjectDetail(root, data({ project: null }), fakeActions());
        expect(root.textContent).toContain("no such project.");
        // Still offers the way back.
        expect(root.querySelector(".agents__back")).not.toBeNull();
    });

    it("deletes (with confirm) via the injected action", async () => {
        const removed: string[] = [];
        vi.spyOn(window, "confirm").mockReturnValue(true);
        renderProjectDetail(
            root,
            data(),
            fakeActions({
                remove: (id) => {
                    removed.push(id);
                    return Promise.resolve();
                },
            }),
        );
        root.querySelector<HTMLButtonElement>(
            ".settings__btn--danger",
        )?.dispatchEvent(new Event("click"));
        await flush();
        expect(removed).toEqual(["my-app"]);
        vi.restoreAllMocks();
    });

    it("does not delete when the confirm is declined", async () => {
        const removed: string[] = [];
        vi.spyOn(window, "confirm").mockReturnValue(false);
        renderProjectDetail(
            root,
            data(),
            fakeActions({
                remove: (id) => {
                    removed.push(id);
                    return Promise.resolve();
                },
            }),
        );
        root.querySelector<HTMLButtonElement>(
            ".settings__btn--danger",
        )?.dispatchEvent(new Event("click"));
        await flush();
        expect(removed).toEqual([]);
        vi.restoreAllMocks();
    });

    it("escapes a hostile project name + task title (no injection)", () => {
        renderProjectDetail(
            root,
            data({
                project: project({ name: "<img src=x onerror=alert(1)>" }),
                tasks: [task({ title: "<b>spec</b>", tags: ["<i>t</i>"] })],
            }),
            fakeActions(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.querySelector(".projtasks__title b")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });
});
