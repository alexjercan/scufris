import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Agent, AgentRunStatus, BackendOption, Project } from "./common";
import { agentIdFromPath, renderAgentDetail } from "./agent-detail-view";
import type { AgentDetailActions } from "./agent-detail-view";

function agent(over: Partial<Agent> = {}): Agent {
    return {
        id: "builder",
        name: "Builder",
        project_id: "my-app",
        backend: "codex",
        model: "gpt-5.5",
        description: "does helpful things",
        goal: "",
        task_id: "",
        session_id: null,
        state: "idle",
        permission_mode: "manual",
        ...over,
    };
}

function project(): Project {
    return {
        id: "my-app",
        cwd: "/home/alex/personal/my-app",
        name: "My App",
        language: "python",
        description: "a thing",
    };
}

function backends(): BackendOption[] {
    return [
        { id: "codex", label: "Codex", default_model: "gpt-5.5" },
        { id: "claude", label: "Claude", default_model: "claude-opus-4-8" },
    ];
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

function fakeActions(
    over: Partial<AgentDetailActions> = {},
): AgentDetailActions {
    return {
        save: () => Promise.resolve(),
        ...over,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="agent-detail"></main>';
    root = document.getElementById("agent-detail") as HTMLElement;
});

describe("agentIdFromPath", () => {
    it("parses /agents/<id> and /agents/<id>/settings", () => {
        expect(agentIdFromPath("/agents/builder")).toBe("builder");
        expect(agentIdFromPath("/agents/builder/")).toBe("builder");
        expect(agentIdFromPath("/agents/builder/settings")).toBe("builder");
        expect(agentIdFromPath("/agents/my%20agent")).toBe("my agent");
    });

    it("returns null for the list or non-agent paths", () => {
        expect(agentIdFromPath("/agents/")).toBeNull();
        expect(agentIdFromPath("/agents")).toBeNull();
        expect(agentIdFromPath("/projects/x")).toBeNull();
    });
});

describe("renderAgentDetail", () => {
    it("renders read-only facts + a back link + live status", () => {
        renderAgentDetail(
            root,
            agent(),
            project(),
            backends(),
            status(),
            fakeActions(),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("Builder"); // title
        expect(text).toContain("My App"); // resolved project name
        expect(text).toContain("working on it"); // last message
        const back = root.querySelector<HTMLAnchorElement>(".agents__back");
        expect(back?.getAttribute("href")).toBe("/agents/");
    });

    it("renders a settings form prefilled with the agent's values incl. model", () => {
        renderAgentDetail(
            root,
            agent(),
            project(),
            backends(),
            status(),
            fakeActions(),
        );
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        const backend = root.querySelector<HTMLSelectElement>(
            'select[aria-label="agent settings backend"]',
        );
        const model = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings model"]',
        );
        const description = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="agent settings description"]',
        );
        const mode = root.querySelector<HTMLSelectElement>(
            'select[aria-label="agent settings permission mode"]',
        );
        expect(name?.value).toBe("Builder");
        expect(backend?.value).toBe("codex");
        expect(model?.value).toBe("gpt-5.5");
        expect(description?.value).toBe("does helpful things");
        expect(mode?.value).toBe("manual");
    });

    it("re-defaults the model when the settings backend changes", () => {
        renderAgentDetail(
            root,
            agent(),
            project(),
            backends(),
            status(),
            fakeActions(),
        );
        const backend = root.querySelector<HTMLSelectElement>(
            'select[aria-label="agent settings backend"]',
        );
        const model = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings model"]',
        );
        backend!.value = "claude";
        backend!.dispatchEvent(new Event("change"));
        expect(model?.value).toBe("claude-opus-4-8");
    });

    it("saves edited settings (incl. model) on submit", async () => {
        const save = vi.fn(() => Promise.resolve());
        renderAgentDetail(
            root,
            agent(),
            project(),
            backends(),
            status(),
            fakeActions({ save }),
        );
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        const model = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings model"]',
        );
        const description = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="agent settings description"]',
        );
        const mode = root.querySelector<HTMLSelectElement>(
            'select[aria-label="agent settings permission mode"]',
        );
        name!.value = "Renamed";
        model!.value = "gpt-5.6";
        description!.value = "new description";
        mode!.value = "edit";
        const form = root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        );
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(save).toHaveBeenCalledWith({
            name: "Renamed",
            backend: "codex",
            model: "gpt-5.6",
            description: "new description",
            permission_mode: "edit",
        });
    });

    it("does not save when the name is blanked", async () => {
        const save = vi.fn(() => Promise.resolve());
        renderAgentDetail(
            root,
            agent(),
            project(),
            backends(),
            status(),
            fakeActions({ save }),
        );
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        name!.value = "   ";
        const form = root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        );
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(save).not.toHaveBeenCalled();
    });

    it("shows a fallback for an unknown agent", () => {
        renderAgentDetail(root, null, null, backends(), null, fakeActions());
        expect(root.textContent).toContain("no such agent.");
    });

    it("shows 'not started' for a never-run agent", () => {
        renderAgentDetail(
            root,
            agent(),
            project(),
            backends(),
            status({ state: "idle", session_id: null, turns: 0 }),
            fakeActions(),
        );
        expect(root.textContent).toContain("not started");
    });

    it("escapes a hostile name and holds a hostile description as text", () => {
        renderAgentDetail(
            root,
            agent({
                name: '<img src=x onerror="alert(1)">',
                description: "<script>alert(2)</script>",
            }),
            project(),
            backends(),
            status(),
            fakeActions(),
        );
        // Title escapes the name; the description sits inertly in a textarea.
        expect(root.querySelector("img")).toBeNull();
        expect(root.querySelector("script")).toBeNull();
        const description = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="agent settings description"]',
        );
        expect(description?.value).toBe("<script>alert(2)</script>");
    });
});
