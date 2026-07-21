import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Agent, AgentRunStatus, Project } from "./common";
import { renderAgents } from "./agents-view";
import type { AgentActions } from "./agents-view";

function agent(over: Partial<Agent> = {}): Agent {
    return {
        id: "builder",
        name: "Builder",
        project_id: "my-app",
        backend: "codex",
        model: "gpt-5.5",
        description: "does helpful things",
        goal: "do the thing",
        task_id: "",
        session_id: null,
        state: "idle",
        permission_mode: "manual",
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
        open: () => undefined,
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
    it("renders agents as cards with a state badge and a create form", () => {
        renderAgents(root, [agent()], [project()], {}, fakeActions());
        expect(root.textContent).toContain("Agents");
        // An agent card exists (Stats-style .card grid).
        const card = root.querySelector(".cards .agents__card");
        expect(card).not.toBeNull();
        expect(card?.textContent).toContain("Builder");
        // A state badge is rendered on the card.
        expect(card?.querySelector(".agents__badge")).not.toBeNull();
        // The create form has a project picker with the project as an option.
        const select = root.querySelector<HTMLSelectElement>(
            'select[aria-label="new agent project"]',
        );
        expect(select).not.toBeNull();
        expect(select?.textContent).toContain("My App");
    });

    it("shows the friendly backend label, not the raw id", () => {
        renderAgents(
            root,
            [agent({ backend: "claude" })],
            [project()],
            {},
            fakeActions(),
        );
        const card = root.querySelector(".agents__card") as HTMLElement;
        expect(card.textContent).toContain("Claude");
        expect(card.textContent).not.toContain("claude");
        // The create picker offers friendly labels too.
        const backend = root.querySelector<HTMLSelectElement>(
            'select[aria-label="new agent backend"]',
        );
        expect(backend?.textContent).toContain("Codex");
        expect(backend?.textContent).toContain("Claude");
    });

    it("shows an empty state when there are no agents", () => {
        renderAgents(root, [], [project()], {}, fakeActions());
        expect(root.textContent).toContain("no agents yet.");
        expect(root.querySelector(".cards")).toBeNull();
    });

    it("shows a fallback when agents could not load", () => {
        renderAgents(root, null, [], {}, fakeActions());
        expect(root.textContent).toContain("could not load agents.");
    });

    it("shows live turns/tokens from a running status", () => {
        renderAgents(
            root,
            [agent()],
            [project()],
            { builder: status({ turns: 7 }) },
            fakeActions(),
        );
        const card = root.querySelector(".agents__card") as HTMLElement;
        expect(card.textContent).toContain("My App"); // resolved project name
        // Assert the turns row's value explicitly, not a stray substring.
        const rows = [...card.querySelectorAll(".row")];
        const turnsRow = rows.find(
            (r) => r.querySelector("span")?.textContent === "turns",
        );
        expect(turnsRow?.querySelector("span:last-child")?.textContent).toBe(
            "7",
        );
        expect(card.textContent).toContain("100 in / 20 out"); // tokens
    });

    it("shows 'not started' when the agent has no run status", () => {
        renderAgents(root, [agent()], [project()], {}, fakeActions());
        const card = root.querySelector(".agents__card") as HTMLElement;
        expect(card.textContent).toContain("not started");
    });

    it("shows 'not started' for an idle, never-run status", () => {
        const idle = status({ state: "idle", session_id: null, turns: 0 });
        renderAgents(
            root,
            [agent()],
            [project()],
            { builder: idle },
            fakeActions(),
        );
        const card = root.querySelector(".agents__card") as HTMLElement;
        expect(card.textContent).toContain("not started");
    });

    it("opens the agent page when the card is clicked", () => {
        const open = vi.fn();
        renderAgents(root, [agent()], [project()], {}, fakeActions({ open }));
        const card = root.querySelector(".agents__card") as HTMLElement;
        card.click();
        expect(open).toHaveBeenCalledWith("builder");
    });

    it("deletes an agent (with confirm) without navigating", async () => {
        const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
        const remove = vi.fn(() => Promise.resolve());
        const open = vi.fn();
        renderAgents(
            root,
            [agent()],
            [project()],
            {},
            fakeActions({ remove, open }),
        );
        const del = root.querySelector<HTMLButtonElement>(
            'button[aria-label="delete Builder"]',
        );
        del?.click();
        await flush();
        expect(remove).toHaveBeenCalledWith("builder");
        // The delete click must not bubble to the card's open handler.
        expect(open).not.toHaveBeenCalled();
        confirm.mockRestore();
    });

    it("submits the create form values", async () => {
        const create = vi.fn(() => Promise.resolve());
        renderAgents(root, [], [project()], {}, fakeActions({ create }));
        const form = root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        );
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="new agent name"]',
        );
        const description = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="new agent description"]',
        );
        name!.value = "Reviewer";
        description!.value = "reviews diffs";
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(create).toHaveBeenCalledWith({
            name: "Reviewer",
            project_id: "my-app",
            backend: "codex",
            description: "reviews diffs",
            permission_mode: "manual",
        });
    });

    it("escapes a hostile agent name so no markup is injected", () => {
        const hostile = agent({ name: '<img src=x onerror="alert(1)">' });
        renderAgents(root, [hostile], [project()], {}, fakeActions());
        // The name is rendered via textContent, so no real element is created.
        const card = root.querySelector(".agents__card") as HTMLElement;
        expect(card.querySelector("img")).toBeNull();
        expect(card.textContent).toContain('<img src=x onerror="alert(1)">');
    });

    it("disables the create form and guides when there are no projects", () => {
        renderAgents(root, [], [], {}, fakeActions());
        const submit = root.querySelector<HTMLButtonElement>(
            'button[type="submit"]',
        );
        expect(submit).not.toBeNull();
        expect(submit?.disabled).toBe(true);
        expect(root.textContent).toContain("create a project first");
    });
});
