import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
    AgentRunStatus,
    ChatReply,
    ImageAttachment,
    ToolCall,
    TranscriptMessage,
} from "./agent-types";
import type { StreamHandlers } from "./chat-stream";
import { createAgentChat, startAgentChat } from "./agent-chat-view";
import type { AgentChatConfig, ChatMsg } from "./agent-chat-types";

function tool(name: string): ToolCall {
    return { server: "scufris", tool: name, status: "completed" };
}

function reply(over: Partial<ChatReply> = {}): ChatReply {
    return { text: "hi", tool_calls: [], usage: null, ...over };
}

function config(over: Partial<AgentChatConfig> = {}): AgentChatConfig {
    return {
        streamTurn: () => Promise.resolve(),
        loadTranscript: () => Promise.resolve([]),
        ...over,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

function blobText(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            resolve(typeof reader.result === "string" ? reader.result : "");
        };
        reader.onerror = () =>
            reject(reader.error ?? new Error("failed to read blob"));
        reader.readAsText(blob);
    });
}

function mount(over: Partial<AgentChatConfig> = {}) {
    const root = document.createElement("div");
    document.body.appendChild(root);
    const control = createAgentChat(root, config(over));
    return { root, control };
}

function composer(root: HTMLElement) {
    return {
        input: root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        )!,
        form: root.querySelector<HTMLFormElement>(".chat__form")!,
    };
}

// A stand-in for the browser EventSource (jsdom has none), so the real reattach
// wiring - status gate + /events subscription + close-on-terminal - is exercised
// end to end. Each constructed instance is recorded in openedSources; a test
// grabs the latest via lastOpenedSource() and pushes frames at it with emitFrame.
const ES_CLOSED = 2;
const openedSources: FakeEventSource[] = [];

class FakeEventSource {
    // subscribeEvents reads EventSource.CLOSED off the global in its onerror
    // guard, so the stub mirrors the browser constant (CLOSED === 2).
    static readonly CLOSED = ES_CLOSED;
    url: string;
    readyState = 1; // OPEN
    // subscribeEvents only reads ev.data, so a structural stand-in avoids
    // constructing a DOM MessageEvent (whose type the lint service cannot resolve).
    onmessage: ((ev: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;
    constructor(url: string) {
        this.url = url;
        openedSources.push(this); // so a test can grab it and push frames
    }
    close(): void {
        this.readyState = ES_CLOSED;
    }
}

function lastOpenedSource(): FakeEventSource | undefined {
    return openedSources[openedSources.length - 1];
}

// Push an SSE frame at a fake source's handler (a free function, so the typed
// lint service resolves the call cleanly).
function emitFrame(source: FakeEventSource, data: string): void {
    source.onmessage?.({ data });
}

describe("createAgentChat", () => {
    beforeEach(() => {
        document.body.replaceChildren();
    });

    // The export test stubs the global `URL` with a bare
    // {createObjectURL, revokeObjectURL} object. Without this restore it LEAKS
    // into the later describes (whose own afterEach only runs after their first
    // test has already run), leaving `URL` non-constructible for whatever runs
    // next. Restore globals here rather than in the leak's victim.
    afterEach(() => {
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

    it("rebuilds the conversation from the transcript on mount", async () => {
        const history: ChatMsg[] = [
            { role: "user", text: "prev question" },
            { role: "assistant", text: "prev answer" },
        ];
        const { root } = mount({
            loadTranscript: () => Promise.resolve(history),
        });
        await flush();
        expect(root.querySelector(".chat__msg--user")?.textContent).toBe(
            "prev question",
        );
        expect(root.textContent).toContain("prev answer");
    });

    it("sends a message, streams the reply, then re-enables the composer", async () => {
        const streamTurn = vi.fn((_m: string, h: StreamHandlers) => {
            h.onTextDelta?.("hel");
            h.onTextDelta?.("lo");
            h.onDone(reply({ text: "hello" }));
            return Promise.resolve();
        });
        const { root } = mount({ streamTurn });
        await flush();
        const { input, form } = composer(root);
        input.value = "hi agent";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(streamTurn).toHaveBeenCalledWith(
            "hi agent",
            expect.anything(),
            undefined,
            expect.any(AbortSignal),
        );
        expect(root.querySelector(".chat__msg--user")?.textContent).toContain(
            "hi agent",
        );
        expect(
            root.querySelector(".chat__msg--assistant")?.textContent,
        ).toContain("hello");
        expect(input.disabled).toBe(false);
        expect(input.value).toBe("");
    });

    it("keeps the streamed reasoning as a collapsed spoiler after the turn settles", async () => {
        const streamTurn = vi.fn((_m: string, h: StreamHandlers) => {
            h.onReasoningDelta?.("first ");
            h.onReasoningDelta?.("thought");
            h.onTextDelta?.("answer");
            h.onDone(reply({ text: "answer" }));
            return Promise.resolve();
        });
        const { root } = mount({ streamTurn });
        await flush();
        const { input, form } = composer(root);
        input.value = "think about it";
        form.dispatchEvent(new Event("submit"));
        await flush();
        // After settle (which re-renders from msgs, dropping the live bubble),
        // the reasoning still shows as a collapsed thinking spoiler.
        const thinking = root.querySelector<HTMLDetailsElement>(
            "details.chat__thinking",
        );
        expect(thinking).not.toBeNull();
        expect(thinking?.open).toBe(false);
        expect(
            thinking?.querySelector(".chat__thinking-body")?.textContent,
        ).toBe("first thought");
    });

    it("shows a placeholder for a genuinely empty reply", async () => {
        const streamTurn = vi.fn((_m: string, h: StreamHandlers) => {
            h.onDone(reply({ text: "" }));
            return Promise.resolve();
        });
        const { root } = mount({ streamTurn });
        await flush();
        const { input, form } = composer(root);
        input.value = "hi";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(
            root.querySelector(".chat__msg--assistant")?.textContent,
        ).toContain("(no reply)");
    });

    it("disables the composer while a turn streams", async () => {
        let resolveTurn: () => void = () => undefined;
        const pending = new Promise<void>((r) => (resolveTurn = r));
        const streamTurn = vi.fn((_m: string, h: StreamHandlers) =>
            pending.then(() => h.onDone(reply({ text: "done" }))),
        );
        const { root } = mount({ streamTurn });
        await flush();
        const { input, form } = composer(root);
        input.value = "x";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(input.disabled).toBe(true);
        resolveTurn();
        await flush();
        expect(input.disabled).toBe(false);
    });

    it("shows the welcome onboarding with example prompts + fork hint", async () => {
        const { root } = mount({
            welcome: {
                examples: ["what's using the most CPU?"],
                forkHint: true,
            },
        });
        await flush();
        expect(root.querySelector(".chat__welcome")).not.toBeNull();
        const example = root.querySelector<HTMLButtonElement>(".chat__example");
        expect(example?.textContent).toBe("what's using the most CPU?");
        example?.click();
        expect(composer(root).input.value).toBe("what's using the most CPU?");
        expect(
            root.querySelector(".chat__welcome-hint")?.textContent,
        ).toContain("branch");
    });

    it("is inert when disabled: shows the notice and disables the composer", async () => {
        const { root } = mount({ disabledReason: "agent is disabled." });
        await flush();
        expect(root.textContent).toContain("agent is disabled.");
        expect(composer(root).input.disabled).toBe(true);
    });

    it("exports the loaded transcript from the visible export button", async () => {
        let exportedBlob: Blob | undefined;
        const createObjectURL = vi.fn((blob: Blob) => {
            exportedBlob = blob;
            return "blob:chat";
        });
        const revokeObjectURL = vi.fn();
        let downloaded = "";
        vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
            function recordDownload(this: HTMLAnchorElement) {
                downloaded = this.download;
            },
        );
        vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
        const { root } = mount({
            exportTitle: "Agent a1 chat",
            exportFilename: "agent-a1-chat.md",
            loadTranscript: () =>
                Promise.resolve([
                    { role: "user", text: "keep this", ts: 1000 },
                    { role: "assistant", text: "done", ts: 2000 },
                ]),
        });
        await flush();
        root.querySelector<HTMLButtonElement>(".chat__export")?.click();
        expect(exportedBlob).toBeInstanceOf(Blob);
        const text = await blobText(exportedBlob!);
        expect(text).toContain("# Agent a1 chat");
        expect(text).toContain("keep this");
        expect(downloaded).toBe("agent-a1-chat.md");
        expect(revokeObjectURL).toHaveBeenCalledWith("blob:chat");
    });
});

describe("cancel / stop button (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    // A streamTurn that stays open (resolves only when its AbortSignal fires), so
    // the turn keeps "streaming" until the test hits stop. Exposes the live
    // handlers so the test can push a partial delta before cancelling.
    function openTurn() {
        let handlers: StreamHandlers | undefined;
        let resolve: () => void = () => undefined;
        const promise = new Promise<void>((r) => (resolve = r));
        const streamTurn = vi.fn(
            (
                _m: string,
                h: StreamHandlers,
                _img?: ImageAttachment,
                signal?: AbortSignal,
            ) => {
                handlers = h;
                // Mirror the real streamPost: an aborted fetch resolves cleanly.
                signal?.addEventListener("abort", () => resolve());
                return promise;
            },
        );
        return { streamTurn, getHandlers: () => handlers };
    }

    it("stop button cancels a streaming run", async () => {
        const { streamTurn, getHandlers } = openTurn();
        const cancelTurn = vi.fn(() => Promise.resolve());
        const { root } = mount({ streamTurn, cancelTurn });
        const { input, form } = composer(root);
        const send = root.querySelector<HTMLButtonElement>(".chat__send")!;

        input.value = "explain X";
        form.dispatchEvent(new Event("submit"));
        await flush();
        getHandlers()?.onTextDelta?.("partial answer");
        await flush();

        // While streaming the button is the square STOP control, still clickable.
        expect(send.classList.contains("is-stopping")).toBe(true);
        expect(send.getAttribute("aria-label")).toBe("stop");
        expect(send.disabled).toBe(false);

        // Hitting it cancels the backend run and restores the composer.
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(cancelTurn).toHaveBeenCalledTimes(1);
        expect(send.classList.contains("is-stopping")).toBe(false);
        expect(send.textContent).toBe("send");
        expect(input.disabled).toBe(false);
        // A second stop is a no-op (nothing is streaming anymore).
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(cancelTurn).toHaveBeenCalledTimes(1);
    });

    it("partial output kept and marked on cancel", async () => {
        const { streamTurn, getHandlers } = openTurn();
        const cancelTurn = vi.fn(() => Promise.resolve());
        const { root } = mount({ streamTurn, cancelTurn });
        const { input, form } = composer(root);

        input.value = "explain X";
        form.dispatchEvent(new Event("submit"));
        await flush();
        getHandlers()?.onTextDelta?.("half an answer");
        await flush();

        form.dispatchEvent(new Event("submit")); // stop
        await flush();

        const assistant = root.querySelector(".chat__msg--assistant");
        expect(assistant).not.toBeNull();
        // The streamed partial is retained, tagged as interrupted (not an error).
        expect(assistant?.textContent).toContain("half an answer");
        expect(root.querySelector(".chat__cancelled")?.textContent).toBe(
            "(cancelled)",
        );
        expect(root.querySelector(".chat__msg--error")).toBeNull();
    });

    it("no stop affordance when cancelTurn is not wired", async () => {
        const { streamTurn, getHandlers } = openTurn();
        const { root } = mount({ streamTurn }); // no cancelTurn
        const { input, form } = composer(root);
        const send = root.querySelector<HTMLButtonElement>(".chat__send")!;

        input.value = "explain X";
        form.dispatchEvent(new Event("submit"));
        await flush();
        getHandlers()?.onTextDelta?.("partial");
        await flush();

        // Without a cancel path the button stays a disabled "send", not a stop.
        expect(send.classList.contains("is-stopping")).toBe(false);
        expect(send.disabled).toBe(true);
    });
});

describe("reattach on mount (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    it("continues an in-flight turn: live bubble streams, then settles the reply into the log", async () => {
        // The mount transcript already carries the in-flight turn's user/prompt
        // side (the backend writes it at turn start); reattach streams the
        // assistant side and settles it in - no re-fetch (which could race the
        // backend's post-turn session-id persist and drop the turn).
        const loadTranscript = vi.fn(() =>
            Promise.resolve([
                { role: "user", text: "earlier q" },
                { role: "assistant", text: "earlier a" },
                { role: "user", text: "orchestrator prompt" },
            ] as ChatMsg[]),
        );
        // The reattach driver stays pending (like a live EventSource) until we
        // deliver the terminal frame, so we can observe the streaming state.
        let deliverDone: (r: ChatReply) => void = () => undefined;
        let resolveRun: () => void = () => undefined;
        const reattach = vi.fn((h: StreamHandlers) => {
            // First delta paints at once (the throttle only debounces later ones).
            h.onTextDelta?.("live token");
            deliverDone = h.onDone;
            return new Promise<void>((r) => (resolveRun = r));
        });

        const { root } = mount({ loadTranscript, reattach });
        await flush();
        // The in-flight turn renders live and the composer is busy.
        expect(reattach).toHaveBeenCalledTimes(1);
        expect(
            root.querySelector(".chat__msg--pending")?.textContent,
        ).toContain("live token");
        expect(composer(root).input.disabled).toBe(true);

        // The turn settles: the streamed reply lands in the log (once), the
        // transcript is NOT re-fetched, and the composer frees up.
        deliverDone(reply({ text: "settled answer" }));
        resolveRun();
        await flush();
        expect(loadTranscript).toHaveBeenCalledTimes(1); // no reconcile re-fetch
        expect(root.querySelector(".chat__msg--pending")).toBeNull();
        expect(root.textContent).toContain("orchestrator prompt");
        expect(root.textContent).toContain("settled answer");
        expect(composer(root).input.disabled).toBe(false);
        // Exactly one new assistant bubble (earlier a + the settled reply).
        expect(root.querySelectorAll(".chat__msg--assistant").length).toBe(2);
    });

    it("is a no-op when no run is active (reattach resolves with no frames)", async () => {
        const reattach = vi.fn(() => Promise.resolve());
        const { root } = mount({
            loadTranscript: () =>
                Promise.resolve([{ role: "user", text: "only q" }]),
            reattach,
        });
        await flush();
        expect(reattach).toHaveBeenCalledTimes(1);
        // No phantom live bubble, composer stays usable, transcript intact.
        expect(root.querySelector(".chat__msg--pending")).toBeNull();
        expect(composer(root).input.disabled).toBe(false);
        expect(root.textContent).toContain("only q");
    });

    it("does not double-render a locally-sent turn when reattach is configured", async () => {
        const reattach = vi.fn(() => Promise.resolve()); // no active run at mount
        const streamTurn = vi.fn((_m: string, h: StreamHandlers) => {
            h.onDone(reply({ text: "local answer" }));
            return Promise.resolve();
        });
        const { root } = mount({ reattach, streamTurn });
        await flush();
        const { input, form } = composer(root);
        input.value = "hi";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(root.querySelectorAll(".chat__msg--assistant").length).toBe(1);
        expect(root.textContent).toContain("local answer");
    });

    it("replays tool-call chips and a distinct per-turn usage count across reload", async () => {
        // The already-shared transcriptReply -> messageMeta path: a reloaded
        // assistant turn shows its tools + a DISTINCT token count (not a default).
        const { root } = mount({
            loadTranscript: () =>
                Promise.resolve([
                    { role: "user", text: "q" },
                    {
                        role: "assistant",
                        text: "a",
                        reply: reply({
                            tool_calls: [tool("host_stats")],
                            usage: {
                                input_tokens: 10,
                                cached_input_tokens: 0,
                                output_tokens: 137,
                                reasoning_output_tokens: 0,
                            },
                        }),
                    },
                ]),
        });
        await flush();
        const meta = root.querySelector(".chat__meta");
        expect(meta?.textContent).toContain("ran");
        expect(meta?.textContent).toContain("host_stats");
        expect(meta?.textContent).toContain("137 tok");
    });
});

describe("edit-to-fork (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    it("renders no edit affordance without forkTurn", async () => {
        const { root } = mount({
            loadTranscript: () =>
                Promise.resolve([{ role: "user", text: "q" }]),
        });
        await flush();
        expect(root.querySelector(".chat__edit")).toBeNull();
    });

    it("opens an inline editor and forks from the edited message", async () => {
        const forkTurn = vi.fn((_i: number, _t: string, h: StreamHandlers) => {
            h.onDone(reply({ text: "branched" }));
            return Promise.resolve();
        });
        const { root } = mount({
            forkTurn,
            forkVerb: "revert",
            loadTranscript: () =>
                Promise.resolve([
                    { role: "user", text: "original" },
                    { role: "assistant", text: "answer" },
                ]),
        });
        await flush();
        root.querySelector<HTMLButtonElement>(".chat__edit")?.click();
        const area = root.querySelector<HTMLTextAreaElement>(
            ".chat__editor-input",
        );
        expect(area?.value).toBe("original");
        area!.value = "edited question";
        // The confirm button uses the injected verb.
        const save = root.querySelector<HTMLButtonElement>(
            ".chat__editor .chat__send",
        );
        expect(save?.textContent).toBe("revert");
        save?.click();
        await flush();
        expect(forkTurn).toHaveBeenCalledWith(
            0,
            "edited question",
            expect.anything(),
            expect.any(AbortSignal),
        );
        // The tail after the fork point is dropped; the edited turn + reply remain.
        expect(root.querySelectorAll(".chat__msg--user").length).toBe(1);
        expect(root.querySelector(".chat__msg--user")?.textContent).toContain(
            "edited question",
        );
        expect(root.textContent).toContain("branched");
    });
});

describe("startAgentChat (per-agent wiring)", () => {
    afterEach(() => vi.unstubAllGlobals());

    function sse(text: string): ReadableStream<Uint8Array> {
        return new ReadableStream({
            start(c) {
                c.enqueue(new TextEncoder().encode(text));
                c.close();
            },
        });
    }

    function status(state: string, prompt?: string | null): AgentRunStatus {
        return {
            agent_id: "a",
            state,
            session_id: "s",
            turns: 1,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            context_window: 0,
            last_message: null,
            updated_at: null,
            prompt: prompt ?? null,
        };
    }

    // fetch stub routing the three per-agent endpoints startAgentChat calls.
    // `transcripts` is the ordered list of message arrays returned by successive
    // /transcript loads (mount, then the post-turn reconcile).
    function stubAgentFetch(
        runState: string,
        transcripts: TranscriptMessage[][],
        prompt?: string | null,
    ): void {
        let loads = 0;
        vi.stubGlobal(
            "fetch",
            vi.fn((url: string) => {
                if (url.endsWith("/transcript")) {
                    const messages =
                        transcripts[Math.min(loads, transcripts.length - 1)];
                    loads += 1;
                    return Promise.resolve({
                        ok: true,
                        json: () => Promise.resolve({ messages }),
                    });
                }
                if (url.endsWith("/status"))
                    return Promise.resolve({
                        ok: true,
                        json: () => Promise.resolve(status(runState, prompt)),
                    });
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({}),
                });
            }),
        );
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

    it("edit-to-fork on a project agent calls the per-agent revert endpoint", async () => {
        window.history.pushState({}, "", "/agents/a1");
        const calls: string[] = [];
        const doneFrame =
            'data: {"kind":"done","reply":{"text":"reverted","tool_calls":[],"usage":null},"session_id":"s"}\n\n';
        vi.stubGlobal(
            "fetch",
            vi.fn((url: string) => {
                calls.push(url);
                if (url.endsWith("/transcript"))
                    return Promise.resolve({
                        ok: true,
                        json: () =>
                            Promise.resolve({
                                messages: [
                                    {
                                        role: "user",
                                        text: "original",
                                        ts: null,
                                        tool_calls: [],
                                        usage: null,
                                    },
                                    {
                                        role: "assistant",
                                        text: "answer",
                                        ts: null,
                                        tool_calls: [],
                                        usage: null,
                                    },
                                ],
                            }),
                    });
                // the fork endpoint streams SSE
                return Promise.resolve({ ok: true, body: sse(doneFrame) });
            }),
        );

        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        const root = document.getElementById("agent-chat") as HTMLElement;

        root.querySelector<HTMLButtonElement>(".chat__edit")?.click();
        const area = root.querySelector<HTMLTextAreaElement>(
            ".chat__editor-input",
        )!;
        area.value = "edited";
        // The per-agent editor confirm button reads "revert".
        const save = root.querySelector<HTMLButtonElement>(
            ".chat__editor .chat__send",
        );
        expect(save?.textContent).toBe("revert");
        save?.click();
        await flush();

        expect(calls.some((u) => u.endsWith("/agents/a1/fork"))).toBe(true);
        expect(root.textContent).toContain("reverted");
    });

    it("re-hydrates a reloaded turn's thinking spoiler from the transcript", async () => {
        window.history.pushState({}, "", "/agents/a-think");
        // A finished run so mount does not reattach/stream - just the reload path.
        stubAgentFetch("done", [
            [
                tmsg("user", "q"),
                {
                    role: "assistant",
                    text: "answer",
                    ts: null,
                    tool_calls: [],
                    usage: null,
                    reasoning: "the thinking that streamed",
                },
            ],
        ]);
        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        const root = document.getElementById("agent-chat") as HTMLElement;

        const thinking = root.querySelector<HTMLDetailsElement>(
            "details.chat__thinking",
        );
        expect(thinking).not.toBeNull();
        // Collapsed by default on reload (a <details> with no `open`).
        expect(thinking?.open).toBe(false);
        expect(
            thinking?.querySelector(".chat__thinking-body")?.textContent,
        ).toBe("the thinking that streamed");
    });

    it("reattaches to an in-flight run on mount and streams it via the event source", async () => {
        window.history.pushState({}, "", "/agents/a2");
        openedSources.length = 0;
        vi.stubGlobal("EventSource", FakeEventSource);
        // The in-flight turn's prompt is already in the transcript at mount.
        stubAgentFetch("running", [[tmsg("user", "q1")]]);

        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        await flush(); // loadTranscript -> status -> open EventSource

        const es = lastOpenedSource();
        if (!es) throw new Error("expected an EventSource to be opened");
        expect(es.url).toContain("/api/agents/a2/events");
        const root = document.getElementById("agent-chat") as HTMLElement;

        emitFrame(es, '{"kind":"text_delta","delta":"streaming now"}');
        expect(
            root.querySelector(".chat__msg--pending")?.textContent,
        ).toContain("streaming now");

        emitFrame(
            es,
            '{"kind":"done","reply":{"text":"final answer","tool_calls":[],"usage":null},"session_id":"s"}',
        );
        // Closed on the terminal frame so the now-closed run bus never triggers
        // the EventSource auto-reconnect loop.
        expect(es.readyState).toBe(ES_CLOSED);
        await flush();
        expect(root.querySelector(".chat__msg--pending")).toBeNull();
        expect(root.textContent).toContain("q1");
        expect(root.textContent).toContain("final answer");
    });

    it("injects the driving prompt as a user bubble when the transcript lacks it", async () => {
        // Mid-turn the backend has NOT yet flushed the orchestrator's prompt to
        // the rollout, so the mount transcript is empty; /status carries the
        // in-flight prompt and reattach renders it before the reply streams.
        window.history.pushState({}, "", "/agents/a5");
        openedSources.length = 0;
        vi.stubGlobal("EventSource", FakeEventSource);
        stubAgentFetch("running", [[]], "what is using the most memory?");

        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        await flush(); // loadTranscript -> status -> open EventSource

        const root = document.getElementById("agent-chat") as HTMLElement;
        // The prompt bubble is present before any assistant frame arrives.
        expect(root.textContent).toContain("what is using the most memory?");
        expect(root.querySelectorAll(".chat__msg--user").length).toBe(1);

        const es = lastOpenedSource();
        if (!es) throw new Error("expected an EventSource to be opened");
        emitFrame(
            es,
            '{"kind":"done","reply":{"text":"chrome","tool_calls":[],"usage":null},"session_id":"s"}',
        );
        await flush();
        // Prompt and reply both present, still exactly one user bubble.
        expect(root.textContent).toContain("what is using the most memory?");
        expect(root.textContent).toContain("chrome");
        expect(root.querySelectorAll(".chat__msg--user").length).toBe(1);
    });

    it("does not duplicate the driving prompt already in the transcript", async () => {
        // The transcript DID catch up (the prompt is its last message); the same
        // prompt on /status must not render a second user bubble.
        window.history.pushState({}, "", "/agents/a6");
        openedSources.length = 0;
        vi.stubGlobal("EventSource", FakeEventSource);
        stubAgentFetch(
            "running",
            [[tmsg("user", "orchestrator prompt")]],
            "orchestrator prompt",
        );

        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        await flush();

        const root = document.getElementById("agent-chat") as HTMLElement;
        expect(root.textContent).toContain("orchestrator prompt");
        // Exactly one user bubble - the injection deduped against the transcript.
        expect(root.querySelectorAll(".chat__msg--user").length).toBe(1);
    });

    it("gives up (resolves, frees the composer) when the event stream errors closed", async () => {
        window.history.pushState({}, "", "/agents/a4");
        openedSources.length = 0;
        vi.stubGlobal("EventSource", FakeEventSource);
        stubAgentFetch("running", [[tmsg("user", "q1")]]);

        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        await flush();

        const es = lastOpenedSource();
        if (!es) throw new Error("expected an EventSource to be opened");
        const root = document.getElementById("agent-chat") as HTMLElement;

        // A permanently-closed stream (e.g. the run cleared, a 404): readyState
        // CLOSED + onerror -> subscribeEvents resolves, no phantom bubble hangs.
        es.readyState = ES_CLOSED;
        es.onerror?.();
        await flush();
        expect(root.querySelector(".chat__msg--pending")).toBeNull();
        expect(composer(root).input.disabled).toBe(false);
    });

    it("does not open the event stream for an idle run (no phantom reattach)", async () => {
        window.history.pushState({}, "", "/agents/a3");
        openedSources.length = 0;
        vi.stubGlobal("EventSource", FakeEventSource);
        stubAgentFetch("idle", [[]]);

        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        await flush();

        expect(lastOpenedSource()).toBeUndefined();
        const root = document.getElementById("agent-chat") as HTMLElement;
        expect(root.querySelector(".chat__msg--pending")).toBeNull();
        expect(composer(root).input.disabled).toBe(false);
    });

    it("uses an agent-specific export label", async () => {
        window.history.pushState({}, "", "/agents/build_agent");
        let exportedBlob: Blob | undefined;
        const createObjectURL = vi.fn((blob: Blob) => {
            exportedBlob = blob;
            return "blob:agent-chat";
        });
        let downloaded = "";
        vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
            function recordDownload(this: HTMLAnchorElement) {
                downloaded = this.download;
            },
        );
        vi.stubGlobal("URL", {
            createObjectURL,
            revokeObjectURL: vi.fn(),
        });
        stubAgentFetch("idle", [[tmsg("user", "agent work")]]);

        document.body.innerHTML = '<section id="agent-chat"></section>';
        startAgentChat();
        await flush();
        await flush();

        const root = document.getElementById("agent-chat") as HTMLElement;
        root.querySelector<HTMLButtonElement>(".chat__export")?.click();
        expect(await blobText(exportedBlob!)).toContain(
            "# Agent build_agent chat",
        );
        expect(downloaded).toBe("scufris-agent-build_agent-chat.md");
    });
});
