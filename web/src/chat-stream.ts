// Shared SSE chat-turn streaming, used by BOTH the landing chat (agent-view,
// POSTing /api/chat/stream) and the per-agent chat (agent-chat-view, POSTing
// /api/agents/<id>/chat). Pure/side-effect-free so jsdom tests drive it. The
// backend streams the same StreamEvent frames for both endpoints, so only the
// URL differs - hence the URL parameter.

import type {
    ChatReply,
    ImageAttachment,
    StreamEvent,
    ToolCall,
} from "./common";

// Split an SSE buffer into complete events, returning any trailing partial frame
// so the caller can carry it across chunk boundaries. A malformed data line is
// skipped, not fatal.
export function parseSseFrames(buffer: string): {
    events: StreamEvent[];
    rest: string;
} {
    const events: StreamEvent[] = [];
    let rest = buffer;
    let sep = rest.indexOf("\n\n");
    while (sep !== -1) {
        const frame = rest.slice(0, sep);
        rest = rest.slice(sep + 2);
        const dataLine = frame
            .split("\n")
            .find((line) => line.startsWith("data:"));
        if (dataLine) {
            try {
                events.push(
                    JSON.parse(dataLine.slice(5).trim()) as StreamEvent,
                );
            } catch {
                // ignore a malformed frame
            }
        }
        sep = rest.indexOf("\n\n");
    }
    return { events, rest };
}

export interface StreamHandlers {
    onTool: (tool: ToolCall) => void;
    onDone: (reply: ChatReply) => void;
    onError: (detail: string) => void;
    onTextDelta?: (delta: string) => void;
    onReasoningDelta?: (delta: string) => void;
}

// POST a message to `url` and consume the SSE turn-progress stream, dispatching
// each frame to the handlers. Unknown event kinds are ignored (additive), so a
// new backend event kind never routes to onError.
export async function streamChatTurn(
    url: string,
    message: string,
    handlers: StreamHandlers,
    image?: ImageAttachment,
): Promise<void> {
    const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(image ? { message, image } : { message }),
    });
    if (!resp.ok || !resp.body) {
        handlers.onError(`chat failed (${String(resp.status)})`);
        return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseFrames(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) {
            if (event.kind === "tool") handlers.onTool(event.tool);
            else if (event.kind === "done") handlers.onDone(event.reply);
            else if (event.kind === "error") handlers.onError(event.detail);
            else if (event.kind === "text_delta")
                handlers.onTextDelta?.(event.delta);
            else if (event.kind === "reasoning_delta")
                handlers.onReasoningDelta?.(event.delta);
            // unknown kinds are ignored, not treated as errors
        }
    }
}
