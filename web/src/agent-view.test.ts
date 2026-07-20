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
    formatTimestamp,
    isNearBottom,
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
        '<span id="agent-model"></span>' +
        '<a id="agent-tools-link" href="/settings/" hidden></a>' +
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
    it("shows window usage, token mix and turn/tool counts under a labeled box", () => {
        renderContext(ctx());
        const panel = document.getElementById("context-panel");
        expect(panel?.hidden).toBe(false);
        const text = panel?.textContent ?? "";
        expect(text).toContain("this session"); // box label
        expect(text).toContain("6%"); // 14612 / 258400 ~ 5.66 -> 6%
        expect(text).toContain("3 / 2"); // turns / tools
        expect(text).toContain("as of last turn"); // freshness hint
        expect(panel?.querySelector(".bar__fill")).not.toBeNull();
    });

    it("gives each stat a hover tooltip (title) so the jargon is explained", () => {
        renderContext(ctx());
        const panel = document.getElementById("context-panel");
        const head = panel?.querySelector<HTMLElement>(".usage-block__head");
        expect(head?.title.length ?? 0).toBeGreaterThan(0);
        const rows = panel?.querySelectorAll<HTMLElement>(".usage-block__row");
        expect(rows && rows.length).toBeGreaterThan(0);
        for (const row of rows ?? []) {
            expect(row.title.length).toBeGreaterThan(0);
        }
    });

    it("hides when there is no active session", () => {
        renderContext(null);
        expect(document.getElementById("context-panel")?.hidden).toBe(true);
        renderContext(ctx({ context_window: 0 }));
        expect(document.getElementById("context-panel")?.hidden).toBe(true);
    });
});

describe("renderUsage", () => {
    it("shows the account box: window, percent, plan and freshness", () => {
        renderUsage(quota());
        const meter = document.getElementById("usage-meter");
        expect(meter?.hidden).toBe(false);
        const text = meter?.textContent ?? "";
        expect(text).toContain("account"); // box label
        expect(text).toContain("weekly"); // window descriptor on the used row
        expect(text).toContain("34%");
        expect(text).toContain("plus");
        expect(text).toContain("as of last turn"); // freshness hint
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

    it("gives each row a full-title tooltip (titles truncate with an ellipsis)", () => {
        const long =
            "a very long session title that will not fit in the sidebar";
        renderSessions([session({ id: "s1", title: long })], null);
        const open = document.querySelector<HTMLButtonElement>(
            "#session-list .session__open",
        );
        expect(open?.title).toBe(long);
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
        // The head no longer inlines the tool list; it links to the settings page.
        expect(link.getAttribute("href")).toBe("/settings/");
    });

    it("singularizes the tools-count label for a single tool", () => {
        renderAgentPanel(info, [tool("host_stats")]);
        expect(document.getElementById("agent-tools-link")?.textContent).toBe(
            "1 tool",
        );
    });

    it("hides the tools link when there are no tools", () => {
        renderAgentPanel(info, []);
        expect(
            document.getElementById("agent-tools-link")?.hasAttribute("hidden"),
        ).toBe(true);
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
        expect(meta?.querySelector(".chat__ran")?.textContent).toBe("ran");
        expect(meta?.querySelector(".chat__chip")?.textContent).toBe(
            "host_stats",
        );
        expect(meta?.textContent).toContain("87 tok");
    });

    it("has no 'ran' label when there are no tool calls (usage only)", () => {
        const meta = messageMeta(reply({ usage: usage(1000, 12) }));
        expect(meta?.querySelector(".chat__ran")).toBeNull();
        expect(meta?.textContent).toContain("12 tok");
    });

    it("returns null when there are no tools or usage", () => {
        expect(messageMeta(reply())).toBeNull();
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

describe("formatTimestamp", () => {
    it("shows HH:MM for a same-day time and month+day for older", () => {
        const now = new Date();
        const today = new Date(
            now.getFullYear(),
            now.getMonth(),
            now.getDate(),
            9,
            5,
        );
        expect(formatTimestamp(today.getTime())).toBe("09:05");

        const old = new Date(2020, 0, 15, 9, 5); // Jan 15 2020, 09:05
        expect(formatTimestamp(old.getTime())).toBe("Jan 15, 09:05");
    });

    it("is empty for a missing stamp", () => {
        expect(formatTimestamp(undefined)).toBe("");
    });
});

describe("isNearBottom", () => {
    function logWith(
        scrollHeight: number,
        clientHeight: number,
        scrollTop: number,
    ): HTMLElement {
        const el = document.createElement("div");
        Object.defineProperty(el, "scrollHeight", { value: scrollHeight });
        Object.defineProperty(el, "clientHeight", { value: clientHeight });
        Object.defineProperty(el, "scrollTop", {
            value: scrollTop,
            writable: true,
        });
        return el;
    }

    it("is true at the bottom and false when scrolled up", () => {
        expect(isNearBottom(logWith(1000, 500, 500))).toBe(true); // pinned
        expect(isNearBottom(logWith(1000, 500, 480))).toBe(true); // within slop
        expect(isNearBottom(logWith(1000, 500, 100))).toBe(false); // scrolled up
    });
});

describe("chat message affordances", () => {
    afterEach(() => vi.unstubAllGlobals());

    it("puts a copy button on an assistant reply that copies its raw text", () => {
        const writeText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, "clipboard", {
            value: { writeText },
            configurable: true,
        });
        _renderChatForTest([
            { role: "assistant", text: "run `df -h` to **check** disks" },
        ]);
        const foot = document.querySelector("#chat-log .chat__foot--assistant");
        const copy = foot?.querySelector<HTMLButtonElement>(".chat__copy");
        expect(copy).not.toBeNull();
        copy?.click();
        expect(writeText).toHaveBeenCalledWith(
            "run `df -h` to **check** disks",
        );
    });

    it("renders a timestamp element for a message that has one", () => {
        const now = new Date();
        const ts = new Date(
            now.getFullYear(),
            now.getMonth(),
            now.getDate(),
            14,
            39,
        ).getTime();
        _renderChatForTest([{ role: "user", text: "hi", ts }]);
        const time = document.querySelector<HTMLTimeElement>(
            "#chat-log .chat__foot--user .chat__time",
        );
        expect(time?.textContent).toBe("14:39");
        expect(time?.getAttribute("datetime")).toBeTruthy();
        expect(time?.title.length ?? 0).toBeGreaterThan(0);
    });

    it("omits the timestamp element when a message has no stamp", () => {
        _renderChatForTest([{ role: "assistant", text: "no stamp" }]);
        expect(document.querySelector("#chat-log .chat__time")).toBeNull();
    });

    it("renders copy/edit affordances by default (not JS-gated on hover)", () => {
        // The buttons must exist in the DOM without any hover/interaction so they
        // are visible at rest (a dimmed resting opacity is CSS, eyeball-verified);
        // this guards against re-introducing a JS/`hidden` gate. See the round-2
        // UX spike (tasks/20260720-102348) and frontend-verify-needs-e2e-serve.
        _renderChatForTest([
            { role: "user", text: "a question" },
            { role: "assistant", text: "an answer" },
        ]);
        const copy = document.querySelector<HTMLButtonElement>(
            "#chat-log .chat__foot--assistant .chat__copy",
        );
        const edit = document.querySelector<HTMLButtonElement>(
            "#chat-log .chat__foot--user .chat__edit",
        );
        expect(copy).not.toBeNull();
        expect(edit).not.toBeNull();
        expect(copy?.hidden).toBe(false);
        expect(edit?.hidden).toBe(false);
    });
});

describe("chat onboarding + scroll + a11y (initChat)", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
        _resetAgentState();
    });

    async function mount(): Promise<HTMLTextAreaElement> {
        const { initChat } = await import("./agent-view");
        _resetAgentState();
        document.body.innerHTML =
            '<div class="chat__log-wrap">' +
            '<div id="chat-log" role="log" aria-live="polite"></div>' +
            '<button id="chat-jump" hidden></button></div>' +
            '<form id="chat-form"><textarea id="chat-input"></textarea>' +
            '<button id="chat-send"></button></form>' +
            '<button id="chat-reset"></button>';
        initChat({ agent_enabled: true } as unknown as Parameters<
            typeof initChat
        >[0]);
        return document.getElementById("chat-input") as HTMLTextAreaElement;
    }

    it("shows an onboarding welcome with example prompts and a fork hint", async () => {
        const input = await mount();
        const welcome = document.querySelector("#chat-log .chat__welcome");
        expect(welcome).not.toBeNull();
        const examples =
            document.querySelectorAll<HTMLButtonElement>(".chat__example");
        expect(examples.length).toBeGreaterThan(0);
        examples[0].click();
        expect(input.value).toBe(examples[0].textContent);
        // The otherwise-hidden fork feature is surfaced as a tip.
        expect(
            document.querySelector("#chat-log .chat__welcome-hint")
                ?.textContent,
        ).toContain("branch");
    });

    it("focuses the composer on load (a11y)", async () => {
        const input = await mount();
        expect(document.activeElement).toBe(input);
    });

    it("reveals the 'new messages' pill when the user scrolls up", async () => {
        await mount();
        const log = document.getElementById("chat-log") as HTMLElement;
        const pill = document.getElementById("chat-jump") as HTMLButtonElement;
        expect(pill.hidden).toBe(true);

        // Simulate a log scrolled up from the bottom, then a user scroll event.
        Object.defineProperty(log, "scrollHeight", {
            value: 1000,
            configurable: true,
        });
        Object.defineProperty(log, "clientHeight", {
            value: 300,
            configurable: true,
        });
        Object.defineProperty(log, "scrollTop", {
            value: 100,
            configurable: true,
            writable: true,
        });
        log.dispatchEvent(new Event("scroll"));
        expect(pill.hidden).toBe(false);

        // Clicking the pill jumps back down and hides it.
        pill.click();
        expect(pill.hidden).toBe(true);
    });

    it("counts messages that arrive while scrolled up on the pill", async () => {
        await mount();
        const log = document.getElementById("chat-log") as HTMLElement;
        const pill = document.getElementById("chat-jump") as HTMLButtonElement;
        Object.defineProperty(log, "scrollHeight", {
            value: 1000,
            configurable: true,
        });
        Object.defineProperty(log, "clientHeight", {
            value: 300,
            configurable: true,
        });
        Object.defineProperty(log, "scrollTop", {
            value: 100,
            configurable: true,
            writable: true,
        });
        log.dispatchEvent(new Event("scroll")); // user scrolls up

        _renderChatForTest([
            { role: "user", text: "hi" },
            { role: "assistant", text: "hello" },
        ]);
        expect(pill.hidden).toBe(false);
        expect(pill.textContent).toBe("2 new messages");

        // Jumping to the latest clears the count.
        pill.click();
        expect(pill.hidden).toBe(true);
    });

    it("keeps a scrolled-up reader's position across a rebuild (no yank to top)", async () => {
        const { _renderChatForTest } = await import("./agent-view");
        await mount();
        const log = document.getElementById("chat-log") as HTMLElement;
        Object.defineProperty(log, "scrollHeight", {
            value: 1000,
            configurable: true,
        });
        Object.defineProperty(log, "clientHeight", {
            value: 300,
            configurable: true,
        });
        Object.defineProperty(log, "scrollTop", {
            value: 220,
            configurable: true,
            writable: true,
        });
        log.dispatchEvent(new Event("scroll")); // user is now scrolled up

        // A rebuild (replaceChildren resets scrollTop to 0) must restore the spot.
        _renderChatForTest([
            { role: "assistant", text: "a new reply arrives" },
        ]);
        expect(log.scrollTop).toBe(220); // not 0 (top), not scrollHeight (bottom)
    });
});
