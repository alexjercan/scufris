import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatReply, ToolCall } from "./common";
import type { StreamHandlers } from "./chat-stream";
import {
    createAgentChat,
    renderChatLog,
    type AgentChatDeps,
    type ChatMsg,
} from "./agent-chat-view";

function tool(name: string): ToolCall {
    return { server: "claude", tool: name, status: "completed" };
}

function reply(text: string): ChatReply {
    return { text, tool_calls: [], usage: null };
}

function deps(over: Partial<AgentChatDeps> = {}): AgentChatDeps {
    return {
        streamTurn: () => Promise.resolve(),
        loadTranscript: () => Promise.resolve([]),
        ...over,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("renderChatLog", () => {
    let log: HTMLElement;
    beforeEach(() => {
        log = document.createElement("div");
    });

    it("shows an empty state with no messages", () => {
        renderChatLog(log, []);
        expect(log.textContent).toContain("no messages yet");
    });

    it("renders user text and assistant markdown with tool chips", () => {
        const msgs: ChatMsg[] = [
            { role: "user", text: "hello there", tools: [] },
            { role: "assistant", text: "**done**", tools: [tool("Bash")] },
        ];
        renderChatLog(log, msgs);
        const user = log.querySelector(".chat__msg--user");
        expect(user?.textContent).toBe("hello there");
        const asst = log.querySelector(".chat__msg--assistant");
        // Markdown is built (the ** markers are consumed, not shown literally).
        expect(asst?.textContent).toContain("done");
        expect(asst?.textContent).not.toContain("**done**");
        expect(asst?.querySelector(".chat__tool")?.textContent).toBe("Bash");
    });

    it("escapes hostile user text and assistant tool names", () => {
        const msgs: ChatMsg[] = [
            { role: "user", text: '<img src=x onerror="alert(1)">', tools: [] },
            {
                role: "assistant",
                text: "ok",
                tools: [tool("<script>x</script>")],
            },
        ];
        renderChatLog(log, msgs);
        expect(log.querySelector("img")).toBeNull();
        expect(log.querySelector("script")).toBeNull();
        expect(log.querySelector(".chat__msg--user")?.textContent).toContain(
            "<img",
        );
        expect(log.querySelector(".chat__tool")?.textContent).toBe(
            "<script>x</script>",
        );
    });
});

describe("createAgentChat", () => {
    let root: HTMLElement;
    beforeEach(() => {
        root = document.createElement("div");
        document.body.appendChild(root);
    });

    it("rebuilds the conversation from the transcript on mount", async () => {
        const history: ChatMsg[] = [
            { role: "user", text: "prev question", tools: [] },
            { role: "assistant", text: "prev answer", tools: [] },
        ];
        createAgentChat(
            root,
            deps({ loadTranscript: () => Promise.resolve(history) }),
        );
        await flush();
        expect(root.querySelector(".chat__msg--user")?.textContent).toBe(
            "prev question",
        );
        expect(root.textContent).toContain("prev answer");
    });

    it("sends a message and streams the reply, then re-enables the composer", async () => {
        const streamTurn = vi.fn((_msg: string, h: StreamHandlers) => {
            h.onTextDelta?.("hel");
            h.onTextDelta?.("lo");
            h.onDone(reply("hello"));
            return Promise.resolve();
        });
        createAgentChat(root, deps({ streamTurn }));
        await flush();
        const input = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        );
        const form = root.querySelector<HTMLFormElement>(".chat__composer");
        input!.value = "hi agent";
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(streamTurn).toHaveBeenCalledWith("hi agent", expect.anything());
        expect(root.querySelector(".chat__msg--user")?.textContent).toBe(
            "hi agent",
        );
        expect(root.textContent).toContain("hello");
        expect(input!.disabled).toBe(false); // re-enabled after done
        expect(input!.value).toBe(""); // composer cleared
    });

    it("shows a placeholder for a genuinely empty reply", async () => {
        const streamTurn = vi.fn((_msg: string, h: StreamHandlers) => {
            h.onDone(reply("")); // no deltas, empty authoritative reply
            return Promise.resolve();
        });
        createAgentChat(root, deps({ streamTurn }));
        await flush();
        const input = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        );
        const form = root.querySelector<HTMLFormElement>(".chat__composer");
        input!.value = "hi";
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(
            root.querySelector(".chat__msg--assistant")?.textContent,
        ).toContain("(no reply)");
    });

    it("disables the composer while a turn streams", async () => {
        let resolveTurn: () => void = () => undefined;
        const pending = new Promise<void>((r) => {
            resolveTurn = r;
        });
        const streamTurn = vi.fn((_msg: string, h: StreamHandlers) =>
            pending.then(() => h.onDone(reply("done"))),
        );
        createAgentChat(root, deps({ streamTurn }));
        await flush();
        const input = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        );
        const form = root.querySelector<HTMLFormElement>(".chat__composer");
        input!.value = "x";
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(input!.disabled).toBe(true); // in flight
        resolveTurn();
        await flush();
        expect(input!.disabled).toBe(false); // finished
    });

    it("does not send an empty message", async () => {
        const streamTurn = vi.fn(() => Promise.resolve());
        createAgentChat(root, deps({ streamTurn }));
        await flush();
        const input = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        );
        const form = root.querySelector<HTMLFormElement>(".chat__composer");
        input!.value = "   ";
        form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(streamTurn).not.toHaveBeenCalled();
    });

    it("sends on Enter but not on Shift+Enter", async () => {
        const streamTurn = vi.fn((_msg: string, h: StreamHandlers) => {
            h.onDone(reply("ok"));
            return Promise.resolve();
        });
        createAgentChat(root, deps({ streamTurn }));
        await flush();
        const input = root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        );
        input!.value = "with shift";
        input!.dispatchEvent(
            new KeyboardEvent("keydown", { key: "Enter", shiftKey: true }),
        );
        await flush();
        expect(streamTurn).not.toHaveBeenCalled();

        input!.value = "plain enter";
        input!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
        await flush();
        expect(streamTurn).toHaveBeenCalledWith(
            "plain enter",
            expect.anything(),
        );
    });
});
