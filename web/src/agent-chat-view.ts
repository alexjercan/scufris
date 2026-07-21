// The per-agent chat on the detail page. A self-contained, multi-turn chat with
// its OWN local state (no module globals - so it never collides with the landing
// chat in agent-view). It reuses the shared streaming helper (`streamChatTurn`),
// the safe markdown renderer, and the no-yank scroll idea, but the wiring is its
// own. Mounted in its own `#agent-chat` root so the detail page's status poll
// (which replaceChildren's `#agent-detail`) never wipes the conversation.
//
// `renderChatLog` is PURE and `createAgentChat` takes INJECTED deps, so jsdom
// tests drive it without fetch. `startAgentChat` reads the agent id from the URL
// and wires the real endpoints.

import { el, fetchJson } from "./common";
import type { ChatReply, ToolCall, TranscriptMessage } from "./common";
import { renderMarkdown } from "./markdown";
import { streamChatTurn } from "./chat-stream";
import type { StreamHandlers } from "./chat-stream";
import { agentIdFromPath } from "./agent-detail-view";

// One message in the conversation. `tools` are the tools an assistant turn ran
// (chips); `streaming` marks the in-flight assistant turn.
export interface ChatMsg {
    role: "user" | "assistant";
    text: string;
    tools: ToolCall[];
    streaming?: boolean;
}

// Injected seam: how a turn is streamed and how history is loaded. `startAgentChat`
// wires these to the per-agent endpoints; tests pass fakes.
export interface AgentChatDeps {
    streamTurn(message: string, handlers: StreamHandlers): Promise<void>;
    loadTranscript(): Promise<ChatMsg[]>;
}

function toolChips(tools: ToolCall[]): HTMLElement | null {
    if (tools.length === 0) return null;
    const wrap = el("div", "chat__tools");
    for (const t of tools) {
        const chip = el("span", "chat__tool");
        chip.textContent = t.tool;
        wrap.appendChild(chip);
    }
    return wrap;
}

// Render the whole log (pure). Assistant text is untrusted model output, built
// safely via renderMarkdown (no innerHTML); user text goes in via textContent.
export function renderChatLog(log: HTMLElement, msgs: ChatMsg[]): void {
    log.replaceChildren();
    if (msgs.length === 0) {
        log.appendChild(
            el(
                "div",
                "chat__empty settings__empty",
                "no messages yet - say something to start.",
            ),
        );
        return;
    }
    for (const msg of msgs) {
        const node = el("div", `chat__msg chat__msg--${msg.role}`);
        if (msg.role === "assistant") {
            node.classList.add("chat__msg--md");
            if (msg.text) {
                node.appendChild(renderMarkdown(msg.text));
            } else if (msg.streaming) {
                node.appendChild(el("span", "chat__typing", "..."));
            }
            const chips = toolChips(msg.tools);
            if (chips) node.appendChild(chips);
        } else {
            node.textContent = msg.text;
        }
        log.appendChild(node);
    }
}

// Build the chat UI into `root` and wire it to `deps`. Returns nothing; the DOM
// is the interface. Loads the transcript, then renders + focuses the composer.
export function createAgentChat(root: HTMLElement, deps: AgentChatDeps): void {
    root.replaceChildren();
    const msgs: ChatMsg[] = [];
    let streaming = false;
    let rendering = false;
    let stickToBottom = true;

    root.appendChild(el("h2", "settings__title", "Chat"));

    const log = el("div", "chat__log");
    log.addEventListener("scroll", () => {
        if (!rendering) {
            stickToBottom =
                log.scrollHeight - log.scrollTop - log.clientHeight < 48;
        }
    });
    root.appendChild(log);

    const form = document.createElement("form");
    form.className = "chat__composer";
    const input = document.createElement("textarea");
    input.className = "chat__input";
    input.setAttribute("aria-label", "chat message");
    input.placeholder = "message the agent...";
    input.rows = 2;
    const send = document.createElement("button");
    send.type = "submit";
    send.className = "chat__send";
    send.textContent = "send";
    form.appendChild(input);
    form.appendChild(send);
    root.appendChild(form);

    const render = (): void => {
        rendering = true;
        const prevTop = log.scrollTop;
        renderChatLog(log, msgs);
        rendering = false;
        // No-yank: follow the bottom only when the reader is already there.
        log.scrollTop = stickToBottom ? log.scrollHeight : prevTop;
    };

    const setEnabled = (enabled: boolean): void => {
        input.disabled = !enabled;
        send.disabled = !enabled;
    };

    const submit = (): void => {
        const text = input.value.trim();
        if (streaming || !text) return;
        input.value = "";
        msgs.push({ role: "user", text, tools: [] });
        const assistant: ChatMsg = {
            role: "assistant",
            text: "",
            tools: [],
            streaming: true,
        };
        msgs.push(assistant);
        streaming = true;
        stickToBottom = true;
        setEnabled(false);
        render();

        const handlers: StreamHandlers = {
            onTextDelta: (delta) => {
                assistant.text += delta;
                render();
            },
            onTool: (tool) => {
                assistant.tools.push(tool);
                render();
            },
            onDone: (reply: ChatReply) => {
                // Authoritative reply wins; else keep what streamed; else a
                // placeholder for a genuinely empty turn (never a blank bubble).
                assistant.text = reply.text || assistant.text || "(no reply)";
                assistant.streaming = false;
                render();
            },
            onError: (detail) => {
                assistant.text = assistant.text || `chat failed: ${detail}`;
                assistant.streaming = false;
                render();
            },
        };
        void deps.streamTurn(text, handlers).finally(() => {
            assistant.streaming = false;
            streaming = false;
            setEnabled(true);
            render();
            input.focus();
        });
    };

    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        submit();
    });
    // Enter sends; Shift+Enter inserts a newline.
    input.addEventListener("keydown", (ev: KeyboardEvent) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            submit();
        }
    });

    render();
    void deps
        .loadTranscript()
        .then((history) => {
            msgs.splice(0, msgs.length, ...history);
            render();
        })
        .catch(() => {
            /* leave the empty log; a send will surface any error */
        });
}

interface TranscriptResponse {
    messages: TranscriptMessage[];
}

export function startAgentChat(): void {
    const root = document.getElementById("agent-chat");
    if (!root) return;
    const id = agentIdFromPath(window.location.pathname);
    if (!id) return;
    const enc = encodeURIComponent(id);

    createAgentChat(root, {
        streamTurn: (message, handlers) =>
            streamChatTurn(`/api/agents/${enc}/chat`, message, handlers),
        loadTranscript: async () => {
            const resp = await fetchJson<TranscriptResponse>(
                `/api/agents/${enc}/transcript`,
            );
            return resp.messages.map((m) => ({
                role: m.role === "assistant" ? "assistant" : "user",
                text: m.text,
                tools: m.tool_calls,
            }));
        },
    });
}
