import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AgentInfo, AgentTool, TranscriptMessage } from "./common";
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
        renderAgentPanel(info, [tool("host_stats"), tool("disk_usage")]);
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

// A stand-in for the browser EventSource (jsdom has none), mirroring the one in
// agent-chat-view.test.ts, so the landing's reattach wiring is exercised.
const ES_CLOSED = 2;
const openedSources: FakeEventSource[] = [];

class FakeEventSource {
    static readonly CLOSED = ES_CLOSED;
    url: string;
    readyState = 0;
    onmessage: ((ev: MessageEvent<string>) => void) | null = null;
    onerror: (() => void) | null = null;
    onopen: (() => void) | null = null;
    constructor(url: string) {
        this.url = url;
        openedSources.push(this);
    }
    close(): void {
        this.readyState = ES_CLOSED;
    }
}

function lastOpenedSource(): FakeEventSource | undefined {
    return openedSources[openedSources.length - 1];
}

function emitFrame(source: FakeEventSource, data: string): void {
    source.onmessage?.({ data } as MessageEvent<string>);
}

interface LandingStubOpts {
    current: string | null;
    transcript?: TranscriptMessage[];
    runState?: string;
    prompt?: string | null;
}

function tmsg(role: string, text: string): TranscriptMessage {
    return {
        role,
        text,
        ts: null,
        tool_calls: [],
        usage: null,
        reasoning: null,
    };
}

// Route the landing's endpoints so startAgent can auto-open the current session
// and reattach to a live orchestrator run.
function stubLandingFetch(opts: LandingStubOpts): { calls: string[] } {
    const calls: string[] = [];
    vi.stubGlobal(
        "fetch",
        vi.fn((url: string) => {
            calls.push(url);
            if (url === "/api/config")
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ agent_enabled: true }),
                });
            if (url === "/api/agent/sessions")
                return Promise.resolve({
                    ok: true,
                    json: () =>
                        Promise.resolve({
                            sessions: opts.current
                                ? [
                                      {
                                          id: opts.current,
                                          title: "a chat",
                                          started_at: null,
                                          updated_at: null,
                                      },
                                  ]
                                : [],
                            current: opts.current,
                        }),
                });
            if (opts.current && url === `/api/agent/session/${opts.current}`)
                return Promise.resolve({
                    ok: true,
                    json: () =>
                        Promise.resolve({ messages: opts.transcript ?? [] }),
                });
            if (url === "/api/agents/orchestrator/status")
                return Promise.resolve({
                    ok: true,
                    json: () =>
                        Promise.resolve({
                            agent_id: "orchestrator",
                            state: opts.runState ?? "idle",
                            session_id: opts.current,
                            turns: 0,
                            tool_calls: 0,
                            input_tokens: 0,
                            output_tokens: 0,
                            context_window: 0,
                            last_message: null,
                            updated_at: null,
                            prompt: opts.prompt ?? null,
                        }),
                });
            // info/tools/context/usage catch-all.
            return Promise.resolve({
                ok: true,
                json: () => Promise.resolve(url.endsWith("/tools") ? [] : {}),
            });
        }),
    );
    return { calls };
}

describe("orchestrator landing reflects the current session (startAgent)", () => {
    beforeEach(() => {
        openedSources.length = 0;
        document.body.innerHTML = '<section id="agent-chat"></section>';
    });
    afterEach(() => vi.unstubAllGlobals());

    it("auto-opens the current session's transcript on mount (idle: no live bubble)", async () => {
        vi.stubGlobal("EventSource", FakeEventSource);
        stubLandingFetch({
            current: "s1",
            transcript: [
                tmsg("user", "earlier q"),
                tmsg("assistant", "earlier a"),
            ],
            runState: "idle",
        });
        await startAgent();
        await flush();
        await flush();

        const root = document.getElementById("agent-chat") as HTMLElement;
        // The current session is opened, not the welcome empty state.
        expect(root.textContent).toContain("earlier q");
        expect(root.textContent).toContain("earlier a");
        // An idle run opens no live stream and shows no phantom pending bubble.
        expect(lastOpenedSource()).toBeUndefined();
        expect(root.querySelector(".chat__msg--pending")).toBeNull();
    });

    it("reattaches to a live orchestrator turn and renders its prompt + stream", async () => {
        vi.stubGlobal("EventSource", FakeEventSource);
        const { calls } = stubLandingFetch({
            current: "s1",
            transcript: [
                tmsg("user", "earlier q"),
                tmsg("assistant", "earlier a"),
            ],
            runState: "running",
            prompt: "what is using the cpu?",
        });
        await startAgent();
        await flush();
        await flush(); // sessions -> transcript -> status -> open EventSource

        const root = document.getElementById("agent-chat") as HTMLElement;
        // The driving prompt (Q1-A injection) shows before the reply streams.
        expect(root.textContent).toContain("what is using the cpu?");
        const es = lastOpenedSource();
        if (!es)
            throw new Error("expected an EventSource on the orchestrator run");
        expect(es.url).toContain("/api/agents/orchestrator/events");

        emitFrame(es, '{"kind":"text_delta","delta":"the browser"}');
        expect(
            root.querySelector(".chat__msg--pending")?.textContent,
        ).toContain("the browser");

        emitFrame(
            es,
            '{"kind":"done","reply":{"text":"the browser","tool_calls":[],"usage":null},"session_id":"s1"}',
        );
        await flush();
        expect(root.querySelector(".chat__msg--pending")).toBeNull();
        expect(root.textContent).toContain("the browser");
        // The mount-time transcript was loaded once; settle did NOT re-fetch it.
        expect(calls.filter((u) => u === "/api/agent/session/s1").length).toBe(
            1,
        );
    });
});
