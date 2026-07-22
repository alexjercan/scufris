import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AgentInfo, AgentTool } from "./common";
import { renderAgentPanel, startAgent } from "./agent-view";

const info: AgentInfo = {
    model: "gpt-5.5",
    auth_mode: "chatgpt",
    enabled: true,
};

function tool(name: string): AgentTool {
    return {
        name,
        description: "does a thing",
        server: "scufris",
        args: [],
        parameters: [],
        enabled: true,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("renderAgentPanel", () => {
    beforeEach(() => {
        document.body.innerHTML =
            '<span id="agent-model"></span>' +
            '<a id="agent-tools-link" href="/settings/" hidden></a>';
    });

    it("shows the model and a tools-count link to /settings/", () => {
        renderAgentPanel(info, [tool("host_stats"), tool("tatr_ls")]);
        expect(document.getElementById("agent-model")?.textContent).toContain(
            "gpt-5.5",
        );
        const link = document.getElementById(
            "agent-tools-link",
        ) as HTMLAnchorElement;
        expect(link.hasAttribute("hidden")).toBe(false);
        expect(link.textContent).toBe("2 tools");
        expect(link.getAttribute("href")).toBe("/settings/");
    });

    it("singularizes and hides for one / zero tools", () => {
        renderAgentPanel(info, [tool("host_stats")]);
        expect(document.getElementById("agent-tools-link")?.textContent).toBe(
            "1 tool",
        );
        renderAgentPanel(info, []);
        expect(
            document.getElementById("agent-tools-link")?.hasAttribute("hidden"),
        ).toBe(true);
    });
});

describe("orchestrator fork wiring (startAgent)", () => {
    afterEach(() => vi.unstubAllGlobals());

    function sse(text: string): ReadableStream<Uint8Array> {
        return new ReadableStream({
            start(c) {
                c.enqueue(new TextEncoder().encode(text));
                c.close();
            },
        });
    }

    it("edit-to-fork on the landing calls the multi-session fork endpoint", async () => {
        const calls: string[] = [];
        const doneFrame =
            'data: {"kind":"done","reply":{"text":"ok","tool_calls":[],"usage":null},"session_id":"s1"}\n\n';
        vi.stubGlobal(
            "fetch",
            vi.fn((url: string, opts?: { body?: string }) => {
                calls.push(url);
                if (url === "/api/config")
                    return Promise.resolve({
                        ok: true,
                        json: () => Promise.resolve({ agent_enabled: true }),
                    });
                if (url === "/api/chat/stream")
                    return Promise.resolve({ ok: true, body: sse(doneFrame) });
                if (url === "/api/agent/sessions")
                    return Promise.resolve({
                        ok: true,
                        json: () =>
                            Promise.resolve({ sessions: [], current: "s1" }),
                    });
                if (url === "/api/agent/session/fork")
                    return Promise.resolve({
                        ok: true,
                        json: () =>
                            Promise.resolve({
                                current: "s2",
                                reply: {
                                    text: "branched",
                                    tool_calls: [],
                                    usage: null,
                                },
                            }),
                    });
                void opts;
                // info/tools/context/usage
                return Promise.resolve({
                    ok: true,
                    json: () =>
                        Promise.resolve(url.endsWith("/tools") ? [] : {}),
                });
            }),
        );

        document.body.innerHTML = '<section id="agent-chat"></section>';
        await startAgent();
        await flush();

        // Send a turn so there is a user message to edit, and currentSessionId
        // gets set from the sessions refresh.
        const root = document.getElementById("agent-chat") as HTMLElement;
        const input = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        )!;
        input.value = "first question";
        root.querySelector<HTMLFormElement>(".chat__form")?.dispatchEvent(
            new Event("submit"),
        );
        await flush();

        // Edit the user message and confirm the fork.
        root.querySelector<HTMLButtonElement>(".chat__edit")?.click();
        const area = root.querySelector<HTMLTextAreaElement>(
            ".chat__editor-input",
        )!;
        area.value = "edited question";
        root.querySelector<HTMLButtonElement>(
            ".chat__editor .chat__send",
        )?.click();
        await flush();

        expect(calls).toContain("/api/agent/session/fork");
        expect(root.textContent).toContain("branched");
    });
});
