// Shared SSE chat-turn streaming, used by BOTH the landing chat (agent-view,
// POSTing /api/chat/stream) and the per-agent chat (agent-chat-view, POSTing
// /api/agents/<id>/chat). Pure/side-effect-free so jsdom tests drive it. The
// backend streams the same StreamEvent frames for both endpoints, so only the
// URL differs - hence the URL parameter.

import { apiFetch } from "./common";
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
    // A local injection hook (not a wire event): the reattach path calls this
    // with the in-flight turn's prompt so runTurn can render the user bubble the
    // mount-time transcript did not yet carry. dispatchStreamEvent never fires it.
    onUserPrompt?: (text: string) => void;
    // The in-flight turn's session id, emitted at turn-start (codex). Lets a tab
    // that started a turn pin the new session id before `done`.
    onSessionStarted?: (sessionId: string) => void;
}

// Route one parsed StreamEvent frame to the handlers. Unknown kinds are ignored
// (additive), so a new backend event kind never routes to onError. Shared by the
// POST turn stream (streamPost) and the reattach event stream (subscribeEvents),
// so both render a turn identically.
export function dispatchStreamEvent(
    event: StreamEvent,
    handlers: StreamHandlers,
): void {
    if (event.kind === "tool") handlers.onTool(event.tool);
    else if (event.kind === "done") handlers.onDone(event.reply);
    else if (event.kind === "error") handlers.onError(event.detail);
    else if (event.kind === "text_delta") handlers.onTextDelta?.(event.delta);
    else if (event.kind === "reasoning_delta")
        handlers.onReasoningDelta?.(event.delta);
    else if (event.kind === "session_started")
        handlers.onSessionStarted?.(event.session_id);
    // unknown kinds are ignored, not treated as errors
}

// Reattach to a run's live event bus over a GET SSE stream and drive the same
// handlers a POST turn uses, resolving once the turn reaches a terminal frame
// (done/error) or the stream closes for good. Uses the browser's native
// EventSource, which reconnects across a transient drop with Last-Event-ID (the
// backend replays events after that seq) - so a mid-turn network blip resumes
// rather than losing the turn. On a terminal frame we close() the stream so the
// now-closed run bus (which would reply replay-then-EOF forever) never triggers
// the auto-reconnect loop. A no-op resolve when EventSource is unavailable (jsdom
// tests, which drive the injected reattach directly instead).
export function subscribeEvents(
    url: string,
    handlers: StreamHandlers,
): Promise<void> {
    return new Promise<void>((resolve) => {
        if (typeof EventSource === "undefined") {
            resolve();
            return;
        }
        const source = new EventSource(url);
        let settled = false;
        const finish = (): void => {
            if (settled) return;
            settled = true;
            source.close();
            resolve();
        };
        source.onmessage = (ev: MessageEvent<string>): void => {
            let event: StreamEvent;
            try {
                event = JSON.parse(ev.data) as StreamEvent;
            } catch {
                return; // ignore a malformed frame
            }
            dispatchStreamEvent(event, handlers);
            if (event.kind === "done" || event.kind === "error") finish();
        };
        source.onerror = (): void => {
            // A 404 (no active run) or a permanently-closed stream leaves the
            // EventSource in CLOSED state; a transient drop leaves it CONNECTING
            // (native auto-reconnect with Last-Event-ID). Only give up on CLOSED.
            if (source.readyState === EventSource.CLOSED) finish();
        };
    });
}

// POST `body` to `url` and consume the SSE turn-progress stream, dispatching each
// frame to the handlers. Unknown event kinds are ignored (additive), so a new
// backend event kind never routes to onError. The body shape varies by endpoint
// (a chat turn sends `{message}`, a revert-fork sends `{message_index, text}`),
// so the raw body is the parameter - see `streamChatTurn` for the chat wrapper.
// `signal` (when given) aborts the in-flight fetch/read the instant the user hits
// the stop button, so the browser stops consuming the stream immediately - the
// backend run itself is cancelled via a separate cancel POST. An AbortError is a
// clean, user-initiated stop: it is swallowed here (not routed to onError), so the
// caller can settle the partial as "cancelled" rather than paint an error bubble.
export async function streamPost(
    url: string,
    body: unknown,
    handlers: StreamHandlers,
    signal?: AbortSignal,
): Promise<void> {
    let resp: Response;
    try {
        resp = await apiFetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal,
        });
    } catch (err) {
        if (isAbort(err)) return;
        throw err;
    }
    if (!resp.ok || !resp.body) {
        handlers.onError(`chat failed (${String(resp.status)})`);
        return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
        let chunk: ReadableStreamReadResult<Uint8Array>;
        try {
            chunk = await reader.read();
        } catch (err) {
            if (isAbort(err)) return;
            throw err;
        }
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        const parsed = parseSseFrames(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) dispatchStreamEvent(event, handlers);
    }
}

// An AbortController-triggered abort (fetch reject or reader.read reject). jsdom
// and browsers name it "AbortError"; guard by name so a real network error still
// propagates.
function isAbort(err: unknown): boolean {
    return err instanceof DOMException
        ? err.name === "AbortError"
        : err instanceof Error && err.name === "AbortError";
}

// POST a chat message (optionally with an attached image) and stream the reply.
// A thin wrapper over `streamPost` with the chat-turn body shape.
export async function streamChatTurn(
    url: string,
    message: string,
    handlers: StreamHandlers,
    image?: ImageAttachment,
    signal?: AbortSignal,
): Promise<void> {
    return streamPost(
        url,
        image ? { message, image } : { message },
        handlers,
        signal,
    );
}
