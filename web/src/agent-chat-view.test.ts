import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
    ChatReply,
    ImageAttachment,
    ToolCall,
    TranscriptMessage,
} from "./common";
import type { StreamHandlers } from "./chat-stream";
import {
    createAgentChat,
    messageMeta,
    renderChatLog,
    startAgentChat,
    transcriptReply,
    type AgentChatConfig,
    type ChatMsg,
} from "./agent-chat-view";

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

describe("renderChatLog (pure)", () => {
    let log: HTMLElement;
    beforeEach(() => {
        log = document.createElement("div");
    });

    it("shows an empty state with no messages", () => {
        renderChatLog(log, []);
        expect(log.textContent).toContain("no messages yet");
    });

    it("shows a custom empty state when provided", () => {
        const welcome = document.createElement("div");
        welcome.textContent = "welcome!";
        renderChatLog(log, [], { emptyState: welcome });
        expect(log.textContent).toContain("welcome!");
    });

    it("renders user text plain and assistant markdown with tool chips", () => {
        const msgs: ChatMsg[] = [
            { role: "user", text: "hello there" },
            {
                role: "assistant",
                text: "**done**",
                reply: reply({ tool_calls: [tool("Bash")] }),
            },
        ];
        renderChatLog(log, msgs);
        expect(log.querySelector(".chat__msg--user")?.textContent).toBe(
            "hello there",
        );
        const asst = log.querySelector(".chat__msg--assistant");
        expect(asst?.textContent).toContain("done");
        expect(asst?.textContent).not.toContain("**done**");
        expect(log.querySelector(".chat__chip")?.textContent).toBe("Bash");
    });

    it("escapes hostile user text and assistant tool names", () => {
        const msgs: ChatMsg[] = [
            { role: "user", text: '<img src=x onerror="alert(1)">' },
            {
                role: "assistant",
                text: "ok",
                reply: reply({ tool_calls: [tool("<script>x</script>")] }),
            },
        ];
        renderChatLog(log, msgs);
        expect(log.querySelector("img")).toBeNull();
        expect(log.querySelector("script")).toBeNull();
        expect(log.querySelector(".chat__msg--user")?.textContent).toContain(
            "<img",
        );
        expect(log.querySelector(".chat__chip")?.textContent).toBe(
            "<script>x</script>",
        );
    });

    it("puts an edit affordance on user messages only when onEdit is given", () => {
        const msgs: ChatMsg[] = [
            { role: "user", text: "q" },
            { role: "assistant", text: "a" },
        ];
        renderChatLog(log, msgs);
        expect(log.querySelectorAll(".chat__edit").length).toBe(0);
        renderChatLog(log, msgs, { onEdit: () => undefined });
        expect(log.querySelectorAll(".chat__edit").length).toBe(1);
        // The assistant turn always gets a copy button.
        expect(
            log.querySelector(".chat__foot--assistant .chat__copy"),
        ).not.toBeNull();
    });
});

describe("messageMeta / transcriptReply", () => {
    it("renders the ran label, tool chips and token count", () => {
        const meta = messageMeta(
            reply({
                tool_calls: [tool("host_stats")],
                usage: {
                    input_tokens: 100,
                    cached_input_tokens: 0,
                    output_tokens: 87,
                    reasoning_output_tokens: 0,
                },
            }),
        );
        expect(meta?.querySelector(".chat__ran")?.textContent).toBe("ran");
        expect(meta?.querySelector(".chat__chip")?.textContent).toBe(
            "host_stats",
        );
        expect(meta?.textContent).toContain("87 tok");
    });

    it("is null with no tools and no usage", () => {
        expect(messageMeta(reply())).toBeNull();
    });

    it("rebuilds a reply from a transcript message, undefined when nothing to show", () => {
        const base: TranscriptMessage = {
            role: "assistant",
            text: "a",
            ts: null,
            tool_calls: [],
            usage: null,
        };
        expect(transcriptReply(base)).toBeUndefined();
        const r = transcriptReply({
            ...base,
            tool_calls: [tool("disk_usage")],
        });
        expect(r?.tool_calls.map((t) => t.tool)).toEqual(["disk_usage"]);
    });
});

describe("createAgentChat", () => {
    beforeEach(() => {
        document.body.replaceChildren();
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

    it("does not send an empty message", async () => {
        const streamTurn = vi.fn(() => Promise.resolve());
        const { root } = mount({ streamTurn });
        await flush();
        const { input, form } = composer(root);
        input.value = "   ";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(streamTurn).not.toHaveBeenCalled();
    });

    it("sends on Enter but not on Shift+Enter", async () => {
        const streamTurn = vi.fn((_m: string, h: StreamHandlers) => {
            h.onDone(reply({ text: "ok" }));
            return Promise.resolve();
        });
        const { root } = mount({ streamTurn });
        await flush();
        const { input } = composer(root);
        input.value = "with shift";
        input.dispatchEvent(
            new KeyboardEvent("keydown", { key: "Enter", shiftKey: true }),
        );
        await flush();
        expect(streamTurn).not.toHaveBeenCalled();
        input.value = "plain enter";
        input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
        await flush();
        expect(streamTurn).toHaveBeenCalledWith(
            "plain enter",
            expect.anything(),
            undefined,
        );
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
        );
        // The tail after the fork point is dropped; the edited turn + reply remain.
        expect(root.querySelectorAll(".chat__msg--user").length).toBe(1);
        expect(root.querySelector(".chat__msg--user")?.textContent).toContain(
            "edited question",
        );
        expect(root.textContent).toContain("branched");
    });
});

describe("slash-command palette (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    function type(input: HTMLTextAreaElement, value: string): void {
        input.value = value;
        input.dispatchEvent(new Event("input"));
    }

    it("opens and filters the palette from installed commands", async () => {
        const { root, control } = mount();
        control.setSlashCommands([
            { name: "new", description: "new chat", run: () => undefined },
            { name: "host", description: "host", run: () => undefined },
        ]);
        await flush();
        const { input } = composer(root);
        const palette = root.querySelector<HTMLElement>(".chat__palette")!;
        expect(palette.hidden).toBe(true);
        type(input, "/");
        expect(palette.hidden).toBe(false);
        expect(palette.querySelectorAll(".chat__palette-item").length).toBe(2);
        type(input, "/ho");
        const names = palette.querySelectorAll(".chat__palette-name");
        expect(names.length).toBe(1);
        expect(names[0].textContent).toBe("/host");
    });

    it("Enter runs the highlighted command instead of sending", async () => {
        const run = vi.fn();
        const streamTurn = vi.fn(() => Promise.resolve());
        const { root, control } = mount({ streamTurn });
        control.setSlashCommands([
            { name: "tasks", description: "tasks", run },
        ]);
        await flush();
        const { input } = composer(root);
        type(input, "/tasks");
        input.dispatchEvent(
            new KeyboardEvent("keydown", { key: "Enter", cancelable: true }),
        );
        expect(run).toHaveBeenCalled();
        expect(streamTurn).not.toHaveBeenCalled();
    });
});

describe("image attachments (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    it("attaches a picked image, previews it, and sends it in the bubble", async () => {
        const streamTurn = vi.fn(
            (_m: string, h: StreamHandlers, _img?: ImageAttachment) => {
                h.onDone(reply({ text: "ok" }));
                return Promise.resolve();
            },
        );
        const { root } = mount({ enableImage: true, streamTurn });
        await flush();
        const file = new File([new Uint8Array([1, 2, 3])], "x.png", {
            type: "image/png",
        });
        const fileInput = root.querySelector<HTMLInputElement>(".chat__file")!;
        Object.defineProperty(fileInput, "files", {
            value: [file],
            configurable: true,
        });
        fileInput.dispatchEvent(new Event("change"));
        await new Promise((r) => setTimeout(r, 30)); // FileReader is async
        const attach = root.querySelector<HTMLElement>(".chat__attach")!;
        expect(attach.hidden).toBe(false);
        expect(attach.querySelector(".chat__attach-thumb")).not.toBeNull();

        const { input, form } = composer(root);
        input.value = "what is this?";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(
            root.querySelector(".chat__msg--user .chat__attach-img"),
        ).not.toBeNull();
        // The image payload reaches streamTurn and the preview clears.
        const arg = streamTurn.mock.calls[0][2];
        expect(arg?.mime).toBe("image/png");
        expect(attach.hidden).toBe(true);
    });

    it("has no attach button when image is disabled", async () => {
        const { root } = mount();
        await flush();
        expect(root.querySelector(".chat__attach-btn")).toBeNull();
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
});
