import { afterEach, describe, expect, it, vi } from "vitest";

import type { ChatReply, ToolCall } from "./common";
import { parseSseFrames, streamChatTurn, streamPost } from "./chat-stream";

describe("parseSseFrames", () => {
    it("parses complete frames and carries a partial remainder", () => {
        const buf =
            'data: {"kind":"tool","tool":{"server":"s","tool":"host_stats","status":"completed"}}\n\n' +
            'data: {"kind":"do';
        const { events, rest } = parseSseFrames(buf);
        expect(events.length).toBe(1);
        expect(events[0].kind).toBe("tool");
        expect(rest).toContain('"kind":"do');
    });

    it("ignores a malformed data frame", () => {
        expect(parseSseFrames("data: not json\n\n").events.length).toBe(0);
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

describe("streamChatTurn / streamPost", () => {
    afterEach(() => vi.unstubAllGlobals());

    function streamOf(text: string): ReadableStream<Uint8Array> {
        return new ReadableStream({
            start(controller) {
                controller.enqueue(new TextEncoder().encode(text));
                controller.close();
            },
        });
    }

    it("dispatches tool events, text/reasoning deltas, then the done reply", async () => {
        const sse =
            'data: {"kind":"reasoning_delta","delta":"think"}\n\n' +
            'data: {"kind":"tool","tool":{"server":"scufris","tool":"host_stats","status":"completed"}}\n\n' +
            'data: {"kind":"text_delta","delta":"He"}\n\n' +
            'data: {"kind":"text_delta","delta":"llo"}\n\n' +
            'data: {"kind":"done","reply":{"text":"Hello","tool_calls":[],"usage":null},"session_id":"s1"}\n\n';
        vi.stubGlobal(
            "fetch",
            vi.fn(() => Promise.resolve({ ok: true, body: streamOf(sse) })),
        );
        const tools: ToolCall[] = [];
        let text = "";
        let think = "";
        let reply: ChatReply | null = null;
        await streamChatTurn("/api/chat/stream", "hi", {
            onTool: (t) => tools.push(t),
            onDone: (r) => (reply = r),
            onError: () => undefined,
            onTextDelta: (d) => (text += d),
            onReasoningDelta: (d) => (think += d),
        });
        expect(tools.map((t) => t.tool)).toEqual(["host_stats"]);
        expect(text).toBe("Hello");
        expect(think).toBe("think");
        expect((reply as unknown as ChatReply).text).toBe("Hello");
    });

    it("includes the image in the chat body only when attached", async () => {
        const bodies: string[] = [];
        vi.stubGlobal(
            "fetch",
            vi.fn((_url: string, opts: { body: string }) => {
                bodies.push(opts.body);
                return Promise.resolve({
                    ok: true,
                    body: streamOf(
                        'data: {"kind":"done","reply":{"text":"ok","tool_calls":[],"usage":null},"session_id":"s"}\n\n',
                    ),
                });
            }),
        );
        const noop = {
            onTool: () => undefined,
            onDone: () => undefined,
            onError: () => undefined,
        };
        await streamChatTurn("/api/chat/stream", "with", noop, {
            data_base64: "QUJD",
            mime: "image/png",
        });
        await streamChatTurn("/api/chat/stream", "without", noop);
        const withImg = JSON.parse(bodies[0]) as { image?: unknown };
        expect(withImg.image).toEqual({
            data_base64: "QUJD",
            mime: "image/png",
        });
        expect(
            (JSON.parse(bodies[1]) as { image?: unknown }).image,
        ).toBeUndefined();
    });

    it("streamPost sends an arbitrary body shape (the fork endpoint)", async () => {
        let sentBody = "";
        vi.stubGlobal(
            "fetch",
            vi.fn((_url: string, opts: { body: string }) => {
                sentBody = opts.body;
                return Promise.resolve({
                    ok: true,
                    body: streamOf(
                        'data: {"kind":"done","reply":{"text":"ok","tool_calls":[],"usage":null},"session_id":"s"}\n\n',
                    ),
                });
            }),
        );
        await streamPost(
            "/api/agents/a1/fork",
            { message_index: 2, text: "edited" },
            {
                onTool: () => undefined,
                onDone: () => undefined,
                onError: () => undefined,
            },
        );
        expect(JSON.parse(sentBody)).toEqual({
            message_index: 2,
            text: "edited",
        });
    });

    it("calls onError when the response is not ok", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() =>
                Promise.resolve({ ok: false, status: 503, body: null }),
            ),
        );
        let detail = "";
        await streamChatTurn("/api/chat/stream", "hi", {
            onTool: () => undefined,
            onDone: () => undefined,
            onError: (d) => (detail = d),
        });
        expect(detail).toContain("503");
    });
});
