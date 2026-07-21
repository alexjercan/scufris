import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Agent, AgentRunStatus, Project } from "./common";
import { renderAgents } from "./agents-view";
import type { AgentActions } from "./agents-view";

function agent(over: Partial<Agent> = {}): Agent {
    return {
        id: "builder",
        name: "Builder",
        project_id: "my-app",
        backend: "mock",
        model: "gpt-5.5",
        goal: "do the thing",
        task_id: "",
        session_id: null,
        state: "idle",
        write_enabled: false,
        ...over,
    };
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

function status(over: Partial<AgentRunStatus> = {}): AgentRunStatus {
    return {
        agent_id: "builder",
        state: "running",
        session_id: "sess-1",
        turns: 2,
        tool_calls: 1,
        input_tokens: 100,
        output_tokens: 20,
        context_window: 258400,
        last_message: "working on it",
        updated_at: 1785074524,
        ...over,
    };
}

function fakeActions(over: Partial<AgentActions> = {}): AgentActions {
    return {
        create: () => Promise.resolve(),
        remove: () => Promise.resolve(),
        run: () => Promise.resolve(),
        select: () => undefined,
        reload: () => Promise.resolve(),
        ...over,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="agents"></main>';
    root = document.getElementById("agents") as HTMLElement;
});

describe("renderAgents", () => {
    it("lists agents with a state badge and a create form", () => {
        renderAgents(root, [agent()], [project()], null, null, fakeActions());
        expect(root.textContent).toContain("Agents");
        expect(root.textContent).toContain("Builder");
        // A state badge is rendered.
        expect(root.querySelector(".agents__badge")).not.toBeNull();
        // The create form has a project picker with the project as an option.
        const select = root.querySelector<HTMLSelectElement>(
            'select[aria-label="new agent project"]',
        );
        expect(select).not.toBeNull();
        expect(select?.textContent).toContain("My App");
        // No detail panel until one is selected.
        expect(root.textContent).not.toContain("read-only");
    });

    it("shows an empty state when there are no agents", () => {
        renderAgents(root, [], [project()], null, null, fakeActions());
        expect(root.textContent).toContain("no agents yet.");
    });

    it("shows a fallback when agents could not load", () => {
        renderAgents(root, null, [], null, null, fakeActions());
        expect(root.textContent).toContain("could not load agents.");
    });

    it("renders a selected agent's detail with config + status", () => {
        renderAgents(
            root,
            [agent()],
            [project()],
            "builder",
            status(),
            fakeActions(),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("do the thing"); // goal
        expect(text).toContain("read-only"); // write posture
        expect(text).toContain("My App"); // resolved project name
        expect(text).toContain("working on it"); // last message from status
        // A live events log container exists.
        expect(root.querySelector("#agent-events")).not.toBeNull();
    });

    it("shows a loading status placeholder before status arrives", () => {
        renderAgents(
            root,
            [agent()],
            [project()],
            "builder",
            null,
            fakeActions(),
        );
        expect(root.textContent).toContain("loading status...");
    });

    it("dispatches the run action when Run is clicked", async () => {
        const run = vi.fn(() => Promise.resolve());
        renderAgents(
            root,
            [agent()],
            [project()],
            "builder",
            status(),
            fakeActions({ run }),
        );
        const btn = root.querySelector<HTMLButtonElement>(
            'button[aria-label="run Builder"]',
        );
        btn?.click();
        await flush();
        expect(run).toHaveBeenCalledWith("builder");
    });

    it("submits the create form values", async () => {
        const create = vi.fn(() => Promise.resolve());
        renderAgents(
            root,
            [],
            [project()],
            null,
            null,
            fakeActions({ create }),
        );
        const form = root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        );
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="new agent name"]',
        );
        const goal = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="new agent goal"]',
        );
        name!.value = "Reviewer";
        goal!.value = "review the diff";
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(create).toHaveBeenCalledWith({
            name: "Reviewer",
            project_id: "my-app",
            backend: "codex",
            goal: "review the diff",
            write_enabled: false,
        });
    });

    it("escapes hostile agent name/goal so no markup is injected", () => {
        const hostile = agent({
            name: '<img src=x onerror="alert(1)">',
            goal: "<script>alert(2)</script>",
        });
        renderAgents(
            root,
            [hostile],
            [project()],
            "builder",
            status(),
            fakeActions(),
        );
        // No real elements created from the hostile strings.
        expect(root.querySelector("img")).toBeNull();
        expect(root.querySelector("script")).toBeNull();
        expect(root.innerHTML).toContain("&lt;script&gt;");
    });

    it("disables the create form and guides when there are no projects", () => {
        renderAgents(root, [], [], null, null, fakeActions());
        const submit = root.querySelector<HTMLButtonElement>(
            'button[type="submit"]',
        );
        expect(submit).not.toBeNull();
        expect(submit?.disabled).toBe(true);
        expect(root.textContent).toContain("create a project first");
    });

    it("shows 'not started' instead of 0s for a never-run agent", () => {
        const idle = status({ state: "idle", session_id: null, turns: 0 });
        renderAgents(
            root,
            [agent()],
            [project()],
            "builder",
            idle,
            fakeActions(),
        );
        expect(root.textContent).toContain("not started");
        // The 0-count status rows are not shown for a never-run agent.
        expect(root.textContent).not.toContain("input tokens");
    });
});
