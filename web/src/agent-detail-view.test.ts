import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Agent, AgentRunStatus, BackendOption, Project } from "./common";
import {
    agentIdFromPath,
    renderSettingsModal,
    renderSidebar,
} from "./agent-detail-view";
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
        turns: 3,
        tool_calls: 2,
        input_tokens: 1000,
        output_tokens: 40,
        context_window: 200000,
        last_message: "working on it",
        updated_at: 1785074524,
        ...over,
    };
}

function fakeActions(
    over: Partial<AgentDetailActions> = {},
): AgentDetailActions {
    return { save: () => Promise.resolve(), ...over };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<div id="root"></div>';
    root = document.getElementById("root") as HTMLElement;
});

describe("agentIdFromPath", () => {
    it("parses /agents/<id> and /agents/<id>/settings", () => {
        expect(agentIdFromPath("/agents/builder")).toBe("builder");
        expect(agentIdFromPath("/agents/builder/settings")).toBe("builder");
        expect(agentIdFromPath("/agents/my%20agent")).toBe("my agent");
    });
    it("returns null for the list or non-agent paths", () => {
        expect(agentIdFromPath("/agents/")).toBeNull();
        expect(agentIdFromPath("/projects/x")).toBeNull();
    });
});

describe("renderSidebar", () => {
    it("renders header + Settings button + status/context boxes, no form or sessions", () => {
        renderSidebar(root, agent(), project(), status(), () => undefined);
        const text = root.textContent ?? "";
        expect(text).toContain("Builder"); // agent name
        expect(text).toContain("My App"); // project
        expect(root.querySelector(".agents__back")).not.toBeNull();
        expect(root.querySelector(".agents__badge")).not.toBeNull();
        // Chat-first: the sidebar has NO settings form (it lives in the modal).
        expect(root.querySelector("form")).toBeNull();
        // Stat boxes present.
        const heads = [...root.querySelectorAll(".usage-block__head")].map(
            (h) => h.textContent,
        );
        expect(heads).toContain("status");
        expect(heads).toContain("context");
        // A Settings button, and no Sessions box.
        expect(
            root.querySelector('button[aria-label="open settings"]'),
        ).not.toBeNull();
        expect(text.toLowerCase()).not.toContain("session");
    });

    it("shows the running turns/tools in the status box", () => {
        renderSidebar(root, agent(), project(), status(), () => undefined);
        const rows = [...root.querySelectorAll(".usage-block .row")];
        const byKey = (k: string) =>
            rows.find((r) => r.querySelector("span")?.textContent === k);
        expect(
            byKey("turns")?.querySelector("span:last-child")?.textContent,
        ).toBe("3");
        expect(
            byKey("tools")?.querySelector("span:last-child")?.textContent,
        ).toBe("2");
    });

    it("shows 'not started' for a never-run agent", () => {
        renderSidebar(
            root,
            agent(),
            project(),
            status({ state: "idle", session_id: null, turns: 0 }),
            () => undefined,
        );
        expect(root.textContent).toContain("not started");
    });

    it("shows a fallback for an unknown agent", () => {
        renderSidebar(root, null, null, null, () => undefined);
        expect(root.textContent).toContain("no such agent.");
    });

    it("fires onOpenSettings when the Settings button is clicked", () => {
        const open = vi.fn();
        renderSidebar(root, agent(), project(), status(), open);
        root.querySelector<HTMLButtonElement>(
            'button[aria-label="open settings"]',
        )?.click();
        expect(open).toHaveBeenCalledOnce();
    });

    it("escapes a hostile agent name", () => {
        renderSidebar(
            root,
            agent({ name: '<img src=x onerror="alert(1)">' }),
            project(),
            status(),
            () => undefined,
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img");
    });
});

describe("renderSettingsModal", () => {
    it("renders the settings form prefilled and saves on submit", async () => {
        const save = vi.fn(() => Promise.resolve());
        renderSettingsModal(
            root,
            agent(),
            project(),
            backends(),
            fakeActions({ save }),
            () => undefined,
        );
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        const model = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings model"]',
        );
        expect(name?.value).toBe("Builder");
        expect(model?.value).toBe("gpt-5.5");
        name!.value = "Renamed";
        root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        )?.dispatchEvent(new Event("submit"));
        await flush();
        expect(save).toHaveBeenCalledWith(
            expect.objectContaining({ name: "Renamed", backend: "codex" }),
        );
    });

    it("closes via the close button and via a backdrop click", () => {
        const onClose = vi.fn();
        renderSettingsModal(
            root,
            agent(),
            project(),
            backends(),
            fakeActions(),
            onClose,
        );
        root.querySelector<HTMLButtonElement>(
            'button[aria-label="close settings"]',
        )?.click();
        expect(onClose).toHaveBeenCalledTimes(1);
        // A click on the overlay backdrop (root itself) also closes.
        root.dispatchEvent(new MouseEvent("click", { bubbles: true }));
        expect(onClose).toHaveBeenCalledTimes(2);
    });

    it("does not save when the name is blanked", async () => {
        const save = vi.fn(() => Promise.resolve());
        renderSettingsModal(
            root,
            agent(),
            project(),
            backends(),
            fakeActions({ save }),
            () => undefined,
        );
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        name!.value = "   ";
        root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        )?.dispatchEvent(new Event("submit"));
        await flush();
        expect(save).not.toHaveBeenCalled();
    });

    it("holds a hostile description inertly in the textarea", () => {
        renderSettingsModal(
            root,
            agent({ description: "<script>alert(2)</script>" }),
            project(),
            backends(),
            fakeActions(),
            () => undefined,
        );
        expect(root.querySelector("script")).toBeNull();
        const desc = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="agent settings description"]',
        );
        expect(desc?.value).toBe("<script>alert(2)</script>");
    });
});
