import { beforeEach, describe, expect, it } from "vitest";

import type { ChatReply, ToolCall, TranscriptMessage } from "./agent-types";
import {
    distinctTools,
    messageMeta,
    renderChatLog,
    transcriptReply,
} from "./agent-chat-log";
import type { ChatMsg } from "./agent-chat-types";

function tool(name: string): ToolCall {
    return { server: "scufris", tool: name, status: "completed" };
}

function reply(over: Partial<ChatReply> = {}): ChatReply {
    return { text: "hi", tool_calls: [], usage: null, ...over };
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

    it("renders a settled assistant reasoning as a collapsed thinking spoiler", () => {
        const msgs: ChatMsg[] = [
            {
                role: "assistant",
                text: "the answer",
                reasoning: "let me think step by step",
            },
        ];
        renderChatLog(log, msgs);
        const thinking = log.querySelector<HTMLDetailsElement>(
            "details.chat__thinking",
        );
        expect(thinking).not.toBeNull();
        // Collapsed by default: the details is visible but not open.
        expect(thinking?.open).toBe(false);
        expect(thinking?.hidden).toBe(false);
        expect(
            thinking?.querySelector(".chat__thinking-body")?.textContent,
        ).toBe("let me think step by step");
        expect(thinking?.querySelector("summary")?.textContent).toBe(
            "thinking",
        );
    });

    it("renders no thinking spoiler when an assistant message has no reasoning", () => {
        const msgs: ChatMsg[] = [{ role: "assistant", text: "just an answer" }];
        renderChatLog(log, msgs);
        expect(log.querySelector(".chat__thinking")).toBeNull();
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

    it("collapses a polling turn to one chip per distinct tool, in order", () => {
        const meta = messageMeta(
            reply({
                tool_calls: [
                    tool("list_projects"),
                    tool("list_agents"),
                    tool("create_agent"),
                    tool("run_agent"),
                    tool("agent_status"),
                    tool("agent_status"),
                    tool("agent_status"),
                    tool("pending_agents"),
                    tool("agent_status"),
                    tool("pending_agents"),
                ],
            }),
        );
        const chips = [...(meta?.querySelectorAll(".chat__chip") ?? [])].map(
            (c) => c.textContent,
        );
        expect(chips).toEqual([
            "list_projects",
            "list_agents",
            "create_agent",
            "run_agent",
            "agent_status",
            "pending_agents",
        ]);
    });

    it("distinctTools keeps first-occurrence order", () => {
        expect(distinctTools(["a", "b", "a", "c", "b", "a"])).toEqual([
            "a",
            "b",
            "c",
        ]);
    });

    it("rebuilds a reply from a transcript message, undefined when nothing to show", () => {
        const base: TranscriptMessage = {
            role: "assistant",
            text: "a",
            ts: null,
            tool_calls: [],
            usage: null,
            reasoning: null,
        };
        expect(transcriptReply(base)).toBeUndefined();
        const r = transcriptReply({
            ...base,
            tool_calls: [tool("disk_usage")],
        });
        expect(r?.tool_calls.map((t) => t.tool)).toEqual(["disk_usage"]);
    });
});
