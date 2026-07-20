import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
    AgentInfo,
    AgentTool,
    ChatReply,
    SessionContext,
    SessionInfo,
    TokenUsage,
    ToolCall,
    UsageQuota,
} from "./common";
import {
    applyUsage,
    messageMeta,
    parseSseFrames,
    renderAgentPanel,
    renderContext,
    renderSessions,
    renderUsage,
    sendChatStream,
    _renderChatForTest,
    _resetAgentState,
} from "./agent-view";

const info: AgentInfo = {
    model: "gpt-5.5",
    auth_mode: "chatgpt",
    enabled: true,
};

function tool(name: string, description = "does a thing"): AgentTool {
    return { name, description };
}

function usage(input: number, output: number): TokenUsage {
    return {
        input_tokens: input,
        cached_input_tokens: 0,
        output_tokens: output,
        reasoning_output_tokens: 0,
    };
}

function reply(over: Partial<ChatReply> = {}): ChatReply {
    return { text: "hi", tool_calls: [], usage: null, ...over };
}

function session(over: Partial<SessionInfo> = {}): SessionInfo {
    return {
        id: "s1",
        title: "a session",
        started_at: null,
        updated_at: null,
        git_branch: null,
        cwd: null,
        ...over,
    };
}

function ctx(over: Partial<SessionContext> = {}): SessionContext {
    return {
        session_id: "s1",
        context_window: 258400,
        input_tokens: 14612,
        cached_input_tokens: 9984,
        output_tokens: 74,
        reasoning_output_tokens: 43,
        total_tokens: 14700,
        turn_count: 3,
        tool_call_count: 2,
        ...over,
    };
}

function quota(over: Partial<UsageQuota> = {}): UsageQuota {
    return {
        plan_type: "plus",
        primary: {
            used_percent: 34,
            window_minutes: 10080,
            resets_at: null,
        },
        secondary: null,
        ...over,
    };
}

beforeEach(() => {
    document.body.innerHTML =
        '<span id="agent-model"></span><span id="agent-usage"></span>' +
        '<button id="agent-tools-toggle" hidden></button>' +
        '<div id="agent-tools" hidden></div>' +
        '<div id="session-list"></div>' +
        '<div id="context-panel"></div><div id="usage-meter"></div>' +
        '<div id="chat-log"></div>';
    _resetAgentState();
});

describe("parseSseFrames", () => {
    it("parses complete frames and carries a partial remainder", () => {
        const buf =
            'data: {"kind":"tool","tool":{"server":"s","tool":"host_stats","status":"completed"}}\n\n' +
            'data: {"kind":"do';
        const { events, rest } = parseSseFrames(buf);
        expect(events.length).toBe(1);
        expect(events[0].kind).toBe("tool");
        expect(rest).toContain('"kind":"do'); // partial frame carried over
    });

    it("ignores a malformed data frame", () => {
        const { events } = parseSseFrames("data: not json\n\n");
        expect(events.length).toBe(0);
    });

    it("ignores a leading comment/padding frame (no data line)", () => {
        const buf =
            `:${" ".repeat(2048)}\n\n` +
            'data: {"kind":"text_delta","delta":"hi"}\n\n';
        const { events } = parseSseFrames(buf);
        expect(events.length).toBe(1);
        expect(events[0].kind).toBe("text_delta");
    });
});

describe("sendChatStream", () => {
    afterEach(() => vi.unstubAllGlobals());

    function streamOf(text: string): ReadableStream<Uint8Array> {
        return new ReadableStream({
            start(controller) {
                controller.enqueue(new TextEncoder().encode(text));
                controller.close();
            },
        });
    }

    it("dispatches tool events then the done reply", async () => {
        const sse =
            'data: {"kind":"tool","tool":{"server":"scufris","tool":"host_stats","status":"completed"}}\n\n' +
            'data: {"kind":"done","reply":{"text":"all good","tool_calls":[],"usage":null},"session_id":"s1"}\n\n';
        vi.stubGlobal(
            "fetch",
            vi.fn(() => Promise.resolve({ ok: true, body: streamOf(sse) })),
        );
        const tools: ToolCall[] = [];
        let reply: ChatReply | null = null;
        await sendChatStream("hi", {
            onTool: (t) => tools.push(t),
            onDone: (r) => {
                reply = r;
            },
            onError: () => undefined,
        });
        expect(tools.map((t) => t.tool)).toEqual(["host_stats"]);
        expect(reply).not.toBeNull();
        expect((reply as unknown as ChatReply).text).toBe("all good");
    });

    it("dispatches token text + reasoning deltas (app_server backend)", async () => {
        const sse =
            'data: {"kind":"reasoning_delta","delta":"let me think"}\n\n' +
            'data: {"kind":"text_delta","delta":"He"}\n\n' +
            'data: {"kind":"text_delta","delta":"llo"}\n\n' +
            'data: {"kind":"done","reply":{"text":"Hello","tool_calls":[],"usage":null},"session_id":"s1"}\n\n';
        vi.stubGlobal(
            "fetch",
            vi.fn(() => Promise.resolve({ ok: true, body: streamOf(sse) })),
        );
        let text = "";
        let think = "";
        let done = false;
        await sendChatStream("hi", {
            onTool: () => undefined,
            onDone: () => {
                done = true;
            },
            onError: () => undefined,
            onTextDelta: (d) => {
                text += d;
            },
            onReasoningDelta: (d) => {
                think += d;
            },
        });
        expect(text).toBe("Hello"); // token-by-token assembled
        expect(think).toBe("let me think");
        expect(done).toBe(true);
    });

    it("renders token deltas into the DOM INCREMENTALLY (before done)", async () => {
        // Drive the real submit path with a stream we release chunk-by-chunk, and
        // assert the pending bubble grows BEFORE the done frame - i.e. the UI
        // paints as tokens arrive, not once at the end. The render is eager (the
        // first token paints immediately) then throttled (~50ms), NOT rAF-gated.
        const { initChat } = await import("./agent-view");
        document.body.innerHTML =
            '<form id="chat-form"><textarea id="chat-input"></textarea></form>' +
            '<div id="chat-log"></div><button id="chat-reset"></button>';

        let controller: ReadableStreamDefaultController<Uint8Array>;
        const enc = new TextEncoder();
        const body = new ReadableStream<Uint8Array>({
            start(c) {
                controller = c;
            },
        });
        vi.stubGlobal(
            "fetch",
            vi.fn((url: string) =>
                url.endsWith("/api/chat/stream")
                    ? Promise.resolve({ ok: true, body })
                    : Promise.resolve({
                          ok: true,
                          json: () => Promise.resolve({}),
                      }),
            ),
        );

        initChat({ agent_enabled: true } as unknown as Parameters<
            typeof initChat
        >[0]);
        const input = document.getElementById(
            "chat-input",
        ) as HTMLTextAreaElement;
        input.value = "hi";
        document
            .getElementById("chat-form")
            ?.dispatchEvent(new Event("submit"));

        const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));
        await tick();
        controller!.enqueue(
            enc.encode('data: {"kind":"text_delta","delta":"He"}\n\n'),
        );
        await tick();

        // The FIRST token paints immediately (eager), no need to wait a frame.
        const streamBody = () => document.querySelector(".chat__stream-body");
        expect(streamBody()?.textContent).toBe("He");

        controller!.enqueue(
            enc.encode('data: {"kind":"text_delta","delta":"llo"}\n\n'),
        );
        // The second token is within the throttle window; it flushes after ~50ms.
        await tick(70);
        expect(streamBody()?.textContent).toBe("Hello");

        controller!.enqueue(
            enc.encode(
                'data: {"kind":"done","reply":{"text":"Hello","tool_calls":[],"usage":null},"session_id":"s1"}\n\n',
            ),
        );
        controller!.close();
        await tick();
    });

    it("calls onError when the response is not ok", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() =>
                Promise.resolve({ ok: false, status: 503, body: null }),
            ),
        );
        let detail = "";
        await sendChatStream("hi", {
            onTool: () => undefined,
            onDone: () => undefined,
            onError: (d) => {
                detail = d;
            },
        });
        expect(detail).toContain("503");
    });
});

describe("chat log edit-to-fork", () => {
    it("puts an edit affordance on user messages, not assistant ones", () => {
        _renderChatForTest([
            { role: "user", text: "hello there" },
            { role: "assistant", text: "hi back" },
        ]);
        const edits = document.querySelectorAll("#chat-log .chat__edit");
        expect(edits.length).toBe(1); // only the user message
        expect(
            document.querySelectorAll("#chat-log .chat__msg--user").length,
        ).toBe(1);
    });

    it("opens an inline editor prefilled with the message when edit is clicked", () => {
        _renderChatForTest([{ role: "user", text: "original question" }]);
        const edit = document.querySelector<HTMLButtonElement>(
            "#chat-log .chat__edit",
        );
        edit?.click();
        const area = document.querySelector<HTMLTextAreaElement>(
            "#chat-log .chat__editor-input",
        );
        expect(area).not.toBeNull();
        expect(area?.value).toBe("original question");
    });

    it("does not inject markup from a hostile message", () => {
        _renderChatForTest([
            { role: "user", text: "<img src=x onerror=alert(1)>" },
        ]);
        expect(document.querySelector("#chat-log img")).toBeNull();
        expect(document.getElementById("chat-log")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });

    it("renders assistant replies as markdown (code fence -> pre), user plain", () => {
        _renderChatForTest([
            { role: "user", text: "```not code, i typed this```" },
            { role: "assistant", text: "run:\n\n```sh\nls -la\n```" },
        ]);
        // The assistant bubble is markdown-rendered with a code block...
        const assistant = document.querySelector(
            "#chat-log .chat__msg--assistant.chat__msg--md",
        );
        expect(assistant?.querySelector(".md__code code")?.textContent).toBe(
            "ls -la",
        );
        // ...while the user message stays plain text (no markdown pre).
        const user = document.querySelector("#chat-log .chat__msg--user");
        expect(user?.querySelector("pre")).toBeNull();
        expect(user?.textContent).toContain("```not code");
    });
});

describe("renderContext", () => {
    it("shows window usage, token mix and turn/tool counts", () => {
        renderContext(ctx());
        const panel = document.getElementById("context-panel");
        expect(panel?.hidden).toBe(false);
        const text = panel?.textContent ?? "";
        expect(text).toContain("context");
        expect(text).toContain("6%"); // 14612 / 258400 ~ 5.66 -> 6%
        expect(text).toContain("3 / 2"); // turns / tools
        expect(panel?.querySelector(".bar__fill")).not.toBeNull();
    });

    it("hides when there is no active session", () => {
        renderContext(null);
        expect(document.getElementById("context-panel")?.hidden).toBe(true);
        renderContext(ctx({ context_window: 0 }));
        expect(document.getElementById("context-panel")?.hidden).toBe(true);
    });
});

describe("renderUsage", () => {
    it("shows the weekly window, percent and plan", () => {
        renderUsage(quota());
        const meter = document.getElementById("usage-meter");
        expect(meter?.hidden).toBe(false);
        const text = meter?.textContent ?? "";
        expect(text).toContain("weekly usage");
        expect(text).toContain("34%");
        expect(text).toContain("plus");
    });

    it("hides when there is no reported limit", () => {
        renderUsage(null);
        expect(document.getElementById("usage-meter")?.hidden).toBe(true);
        renderUsage(quota({ primary: null }));
        expect(document.getElementById("usage-meter")?.hidden).toBe(true);
    });
});

describe("renderSessions", () => {
    it("lists sessions and highlights the current one", () => {
        renderSessions(
            [
                session({ id: "s1", title: "first" }),
                session({ id: "s2", title: "second" }),
            ],
            "s2",
        );
        const items = document.querySelectorAll("#session-list .session");
        expect(items.length).toBe(2);
        expect(items[0].textContent).toContain("first");
        expect(items[1].classList.contains("is-active")).toBe(true);
        expect(items[0].classList.contains("is-active")).toBe(false);
    });

    it("shows an empty state when there are no sessions", () => {
        renderSessions([], null);
        expect(document.querySelector("#session-list .session")).toBeNull();
        expect(document.getElementById("session-list")?.textContent).toContain(
            "no sessions",
        );
    });

    it("does not inject markup from a hostile session title", () => {
        renderSessions(
            [session({ title: "<img src=x onerror=alert(1)>" })],
            null,
        );
        expect(document.querySelector("#session-list img")).toBeNull();
        expect(document.getElementById("session-list")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });

    it("gives each session an open control and a delete button", () => {
        renderSessions([session({ id: "s1", title: "first" })], null);
        const row = document.querySelector("#session-list .session");
        expect(row?.querySelector(".session__open")).not.toBeNull();
        const del = row?.querySelector(".session__del");
        expect(del).not.toBeNull();
        expect(del?.getAttribute("aria-label")).toBe("delete conversation");
    });
});

describe("renderAgentPanel", () => {
    it("shows the model and lists the tools", () => {
        renderAgentPanel(info, [tool("host_stats"), tool("tatr_ls")]);
        expect(document.getElementById("agent-model")?.textContent).toContain(
            "gpt-5.5",
        );
        const toggle = document.getElementById("agent-tools-toggle");
        expect(toggle?.hasAttribute("hidden")).toBe(false);
        expect(toggle?.textContent).toBe("tools (2)");
        expect(
            document.querySelectorAll("#agent-tools .agent-tools__item").length,
        ).toBe(2);
        expect(document.getElementById("agent-tools")?.textContent).toContain(
            "host_stats",
        );
    });

    it("hides the toggle when there are no tools", () => {
        renderAgentPanel(info, []);
        expect(
            document
                .getElementById("agent-tools-toggle")
                ?.hasAttribute("hidden"),
        ).toBe(true);
    });

    it("does not inject markup from a hostile tool name", () => {
        renderAgentPanel(info, [tool("<img src=x onerror=alert(1)>")]);
        expect(document.querySelector("#agent-tools img")).toBeNull();
        expect(document.getElementById("agent-tools")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });
});

describe("messageMeta", () => {
    it("renders tool chips and the token count", () => {
        const meta = messageMeta(
            reply({
                tool_calls: [
                    {
                        server: "scufris",
                        tool: "host_stats",
                        status: "completed",
                    },
                ],
                usage: usage(47000, 87),
            }),
        );
        expect(meta).not.toBeNull();
        expect(meta?.textContent).toContain("host_stats");
        expect(meta?.textContent).toContain("87 tok");
    });

    it("returns null when there are no tools or usage", () => {
        expect(messageMeta(reply())).toBeNull();
    });
});

describe("applyUsage", () => {
    it("accumulates output tokens and shows context, and resets", () => {
        applyUsage(usage(47000, 87));
        let text = document.getElementById("agent-usage")?.textContent ?? "";
        expect(text).toContain("ctx 47.0k");
        expect(text).toContain("87 out");

        applyUsage(usage(48000, 100));
        text = document.getElementById("agent-usage")?.textContent ?? "";
        expect(text).toContain("187 out"); // cumulative

        _resetAgentState();
        expect(document.getElementById("agent-usage")?.textContent).toBe("");
    });
});

describe("multi-line composer (initChat)", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
        _resetAgentState();
    });

    // A stream that stays open so runStreamingTurn never completes during the
    // test - we only assert what the composer does on submit, not the turn.
    function openStream(): ReadableStream<Uint8Array> {
        return new ReadableStream({ start() {} });
    }

    async function mountComposer(): Promise<HTMLTextAreaElement> {
        vi.stubGlobal(
            "fetch",
            vi.fn(() => Promise.resolve({ ok: true, body: openStream() })),
        );
        const { initChat } = await import("./agent-view");
        _resetAgentState();
        document.body.innerHTML =
            '<form id="chat-form"><textarea id="chat-input"></textarea>' +
            '<button id="chat-send"></button></form>' +
            '<div id="chat-log"></div><button id="chat-reset"></button>';
        initChat({ agent_enabled: true } as unknown as Parameters<
            typeof initChat
        >[0]);
        return document.getElementById("chat-input") as HTMLTextAreaElement;
    }

    function pressEnter(input: HTMLTextAreaElement, shift: boolean): boolean {
        return input.dispatchEvent(
            new KeyboardEvent("keydown", {
                key: "Enter",
                shiftKey: shift,
                cancelable: true,
                bubbles: true,
            }),
        );
    }

    it("sends on Enter (no shift): clears the textarea and posts the message", async () => {
        const input = await mountComposer();
        input.value = "how full is my disk?";
        const notPrevented = pressEnter(input, false);

        expect(notPrevented).toBe(false); // preventDefault() was called
        expect(input.value).toBe(""); // composer cleared on send
        expect(input.disabled).toBe(true); // sending state
        const users = document.querySelectorAll("#chat-log .chat__msg--user");
        expect(users.length).toBe(1);
        expect(users[0].textContent).toContain("how full is my disk?");
    });

    it("does NOT send on Shift+Enter: keeps the value for a newline", async () => {
        const input = await mountComposer();
        input.value = "line one";
        const notPrevented = pressEnter(input, true);

        expect(notPrevented).toBe(true); // default (insert newline) allowed
        expect(input.value).toBe("line one"); // untouched, no send
        expect(input.disabled).toBe(false);
        expect(
            document.querySelectorAll("#chat-log .chat__msg--user").length,
        ).toBe(0);
    });

    it("ignores Enter while a turn is in flight (disabled composer)", async () => {
        const input = await mountComposer();
        input.value = "first";
        pressEnter(input, false); // sends, disables the composer
        expect(input.disabled).toBe(true);

        input.value = "second while busy";
        pressEnter(input, false); // should be a no-op
        expect(
            document.querySelectorAll("#chat-log .chat__msg--user").length,
        ).toBe(1); // still only the first message
    });

    it("does not send an empty / whitespace-only composer", async () => {
        const input = await mountComposer();
        input.value = "   ";
        pressEnter(input, false);
        expect(input.disabled).toBe(false);
        expect(
            document.querySelectorAll("#chat-log .chat__msg--user").length,
        ).toBe(0);
    });
});
