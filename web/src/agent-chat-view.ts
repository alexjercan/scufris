// The ONE chat component, shared by the orchestrator landing (agent-view) and the
// per-agent detail page (startAgentChat below). It builds its own DOM into a mount
// `root` and keeps its OWN local state (no module globals), so two instances never
// collide and jsdom tests drive it without fetch. Capabilities are opt-in through
// `AgentChatConfig`: image attach, a slash-command palette, and edit-to-fork all
// render only when their config is present, and the fork endpoint (new-session vs
// revert) is injected - so the landing and a project agent share this code and
// differ only in wiring.
//
// The sidebar (session switcher + context/usage boxes) is ORCHESTRATOR-only and
// lives OUTSIDE this component (see chat-sidebar + the agent-view entry); this
// component drives it back via `config.onAfterTurn` after each settled turn.
//
// The pieces that own state of their own live beside this file: the streaming
// turn in `agent-chat-turn.ts`, the palette and image attach in
// `agent-chat-composer.ts`, the pure log render in `agent-chat-log.ts`.

import { apiFetch, el, fetchJson } from "./common";
import type { AgentRunStatus, TranscriptMessage } from "./agent-types";
import { streamChatTurn, streamPost, subscribeEvents } from "./chat-stream";
import { parseIso } from "./chat-format";
import { downloadChatMarkdown } from "./chat-commands";
import { agentIdFromPath } from "./agent-detail-view";
import type {
    AgentChatConfig,
    ChatControl,
    ChatMsg,
    RenderChatOpts,
} from "./agent-chat-types";
import { renderChatLog, transcriptReply } from "./agent-chat-log";
import { createTurnRunner } from "./agent-chat-turn";
import {
    autosize,
    createImageAttach,
    createSlashPalette,
} from "./agent-chat-composer";

// Build the chat UI into `root` and wire it to `config`. Returns an imperative
// handle for the entry (setMessages/reset/slash commands). The DOM is otherwise
// the interface: user text, streaming reply, meta, footers, and the pill.
export function createAgentChat(
    root: HTMLElement,
    config: AgentChatConfig,
): ChatControl {
    root.replaceChildren();

    let msgs: ChatMsg[] = [];
    let editingIndex: number | null = null;
    let rendering = false;
    let stickToBottom = true;
    let unreadCount = 0;
    let prevMsgCount = 0;
    const enabled = !config.disabledReason;

    const header = el("div", "chat__topbar");
    if (config.title)
        header.appendChild(el("h2", "settings__title", config.title));
    else header.appendChild(el("div", "chat__topbar-spacer"));
    const exportBtn = document.createElement("button");
    exportBtn.type = "button";
    exportBtn.className = "chat__export";
    exportBtn.textContent = "Export";
    exportBtn.title = "download this chat as markdown";
    exportBtn.setAttribute("aria-label", "download this chat as markdown");
    header.appendChild(exportBtn);
    root.appendChild(header);

    // --- Log + jump pill ---
    const logWrap = el("div", "chat__log-wrap");
    const log = el("div", "chat__log");
    log.setAttribute("role", "log");
    log.setAttribute("aria-live", "polite");
    log.setAttribute("aria-relevant", "additions");
    log.setAttribute("aria-label", "conversation");
    const pill = el("button", "chat__jump", "new messages");
    pill.setAttribute("type", "button");
    pill.hidden = true;
    logWrap.append(log, pill);
    root.appendChild(logWrap);

    // --- Composer ---
    const form = document.createElement("form");
    form.className = "chat__form";
    const palette = el("div", "chat__palette");
    palette.setAttribute("role", "listbox");
    palette.setAttribute("aria-label", "slash commands");
    palette.hidden = true;
    const attach = el("div", "chat__attach");
    attach.hidden = true;
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.className = "chat__file";
    fileInput.hidden = true;
    const attachBtn = document.createElement("button");
    attachBtn.type = "button";
    attachBtn.className = "chat__attach-btn";
    attachBtn.title = "attach an image";
    attachBtn.setAttribute("aria-label", "attach an image");
    attachBtn.textContent = "📎";
    const input = document.createElement("textarea");
    input.className = "chat__input";
    input.setAttribute("aria-label", "chat message");
    input.rows = 1;
    input.placeholder =
        "ask the agent, or type / for commands... (Enter to send, Shift+Enter for newline)";
    const send = document.createElement("button");
    send.type = "submit";
    send.className = "chat__send";
    send.textContent = "send";

    form.appendChild(palette);
    if (config.enableImage) {
        form.append(attach, fileInput, attachBtn);
    }
    form.append(input, send);
    root.appendChild(form);

    const isNearBottom = (): boolean =>
        log.scrollHeight - log.scrollTop - log.clientHeight < 48;

    const refreshPill = (): void => {
        pill.hidden = stickToBottom;
        pill.textContent =
            unreadCount > 0
                ? `${unreadCount} new message${unreadCount === 1 ? "" : "s"}`
                : "jump to latest";
    };

    const maybeScroll = (restoreTop?: number): void => {
        if (stickToBottom) {
            log.scrollTop = log.scrollHeight;
            unreadCount = 0;
        } else if (restoreTop !== undefined) {
            log.scrollTop = restoreTop;
        }
        refreshPill();
    };

    // The onboarding empty state (orchestrator): a welcome + example prompts a
    // click fills into the composer, plus an optional fork tip.
    const buildWelcome = (): HTMLElement => {
        const wrap = el("div", "chat__welcome");
        wrap.appendChild(
            el("div", "chat__welcome-title", "Ask your scuffed Jarvis"),
        );
        wrap.appendChild(
            el(
                "div",
                "chat__welcome-sub",
                "It can inspect this host and run tools. Try one of these:",
            ),
        );
        const chips = el("div", "chat__examples");
        for (const prompt of config.welcome?.examples ?? []) {
            const chip = el("button", "chat__example");
            chip.setAttribute("type", "button");
            chip.textContent = prompt;
            chip.addEventListener("click", () => fillComposer(prompt));
            chips.appendChild(chip);
        }
        wrap.appendChild(chips);
        if (config.welcome?.forkHint) {
            wrap.appendChild(
                el(
                    "div",
                    "chat__welcome-hint",
                    "Tip: edit one of your messages to branch the conversation from that point.",
                ),
            );
        }
        return wrap;
    };

    const renderOpts = (): RenderChatOpts => ({
        emptyState: enabled && config.welcome ? buildWelcome() : undefined,
        editingIndex,
        buildEditor: config.forkTurn ? buildEditor : undefined,
        onEdit: config.forkTurn ? beginEdit : undefined,
    });

    const render = (): void => {
        // Count messages that arrived while scrolled up (for the pill).
        const grew = msgs.length - prevMsgCount;
        if (grew > 0 && !stickToBottom) unreadCount += grew;
        prevMsgCount = msgs.length;
        const prevTop = log.scrollTop;
        rendering = true;
        renderChatLog(log, msgs, renderOpts());
        rendering = false;
        maybeScroll(prevTop);
    };

    const setComposerEnabled = (on: boolean): void => {
        input.disabled = !on;
        send.disabled = !on;
        attachBtn.disabled = !on;
    };

    // Swap the composer button between "send" and the square STOP control. In stop
    // mode the label is cleared (CSS draws the square) and aria-label flips, so the
    // button reads as "stop this run". Assert the mode via the class/aria-label, not
    // textContent (which is empty in stop mode). Stop mode also re-enables the
    // button: the rest of the composer is disabled while a turn streams, but the
    // stop control has to stay live.
    const setStopMode = (stopping: boolean): void => {
        send.classList.toggle("is-stopping", stopping);
        send.setAttribute("aria-label", stopping ? "stop" : "send message");
        send.textContent = stopping ? "" : "send";
        if (stopping) send.disabled = false;
    };

    const slash = createSlashPalette(palette, input);
    const image = config.enableImage
        ? createImageAttach({ attach, fileInput, attachBtn, input })
        : null;

    const turn = createTurnRunner({
        log,
        config,
        appendMessage: (msg) => {
            msgs.push(msg);
        },
        lastMessage: () => msgs[msgs.length - 1],
        render: () => render(),
        maybeScroll: () => maybeScroll(),
        setComposerEnabled,
        setStopMode,
        onSettled: () => {
            autosize(input);
            input.focus();
        },
    });

    const submit = (): void => {
        if (turn.isStreaming() || input.disabled) return;
        const text = input.value.trim();
        if (!text) return;
        slash.close();
        const attached = image?.pending() ?? null; // captured before clearing
        editingIndex = null;
        msgs.push({
            role: "user",
            text,
            ts: Date.now(),
            imageUrl: attached?.dataUrl,
        });
        stickToBottom = true;
        input.value = "";
        image?.clear();
        autosize(input);
        render();
        turn.run((h, signal) =>
            config.streamTurn(text, h, attached?.attachment, signal),
        );
    };

    // --- Edit-to-fork ---
    function beginEdit(index: number): void {
        editingIndex = index;
        render();
    }

    function buildEditor(index: number, text: string): HTMLElement {
        const box = el("div", "chat__editor");
        if (config.forkHint) {
            box.appendChild(el("div", "chat__editor-hint", config.forkHint));
        }
        const area = document.createElement("textarea");
        area.className = "chat__editor-input";
        area.value = text;
        const actions = el("div", "chat__editor-actions");
        const save = el("button", "chat__send", config.forkVerb ?? "fork");
        save.setAttribute("type", "button");
        save.addEventListener("click", () => forkFrom(index, area.value));
        const cancel = el("button", "chat__reset", "cancel");
        cancel.setAttribute("type", "button");
        cancel.addEventListener("click", () => {
            editingIndex = null;
            render();
        });
        actions.append(save, cancel);
        box.append(area, actions);
        return box;
    }

    const forkFrom = (index: number, text: string): void => {
        const trimmed = text.trim();
        editingIndex = null;
        if (!trimmed || !config.forkTurn || turn.isStreaming()) {
            render();
            return;
        }
        // Keep the turns BEFORE the edit, replace the edited message, and run that
        // as the fork's first turn (semantics - new session vs revert - are the
        // injected forkTurn's business).
        msgs = msgs
            .slice(0, index)
            .concat([{ role: "user", text: trimmed, ts: Date.now() }]);
        stickToBottom = true;
        render();
        const fork = config.forkTurn;
        turn.run((h, signal) => fork(index, trimmed, h, signal));
    };

    // --- Composer wiring ---
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        // While a turn streams the button is the STOP control - submit cancels the
        // run instead of sending. Otherwise it sends normally.
        if (turn.isStreaming()) {
            turn.requestCancel();
            return;
        }
        submit();
    });
    input.addEventListener("keydown", (event: KeyboardEvent) => {
        if (slash.handleKey(event)) return;
        // Enter sends; Shift+Enter inserts a newline. Guard isComposing so
        // committing an IME candidate with Enter does not fire a half-typed send.
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            submit();
        }
    });
    input.addEventListener("input", () => {
        autosize(input);
        slash.refresh();
    });
    input.addEventListener("blur", () => slash.close());

    log.addEventListener("scroll", () => {
        if (rendering) return;
        stickToBottom = isNearBottom();
        if (stickToBottom) unreadCount = 0;
        refreshPill();
    });
    pill.addEventListener("click", () => {
        stickToBottom = true;
        unreadCount = 0;
        log.scrollTop = log.scrollHeight;
        refreshPill();
    });

    // --- Control handle ---
    function fillComposer(text: string): void {
        if (input.disabled) return;
        input.value = text;
        autosize(input);
        input.focus();
    }
    const control: ChatControl = {
        setMessages: (history) => {
            msgs = history.slice();
            editingIndex = null;
            stickToBottom = true;
            render();
        },
        reset: () => {
            msgs = [];
            editingIndex = null;
            stickToBottom = true;
            unreadCount = 0;
            image?.clear();
            render();
            input.focus();
        },
        focus: () => input.focus(),
        fillComposer,
        exportChat: () =>
            downloadChatMarkdown(msgs, {
                title: config.exportTitle ?? config.title,
                filename: config.exportFilename,
            }),
        setSlashCommands: (commands) => {
            slash.setCommands(commands);
        },
        note: (text) => {
            const bubble = el("div", "chat__msg chat__msg--system");
            bubble.textContent = text;
            log.appendChild(bubble);
            maybeScroll();
        },
    };
    exportBtn.addEventListener("click", () => control.exportChat());

    // Inert when the agent is disabled: show the notice, keep the composer off.
    if (!enabled) {
        log.appendChild(
            el("div", "chat__msg chat__msg--system", config.disabledReason),
        );
        setComposerEnabled(false);
        return control;
    }

    render();
    input.focus();
    void config
        .loadTranscript()
        .then((history) => {
            if (history.length === 0) return;
            msgs = history;
            render();
        })
        .catch(() => {
            /* leave the empty log; a send will surface any error */
        })
        .finally(() => {
            // With the settled transcript in place, reattach to any in-flight run
            // so a reload/reselect mid-turn keeps streaming. Guarded on !streaming
            // in case the user already fired a local turn during the async load.
            if (config.reattach && !turn.isStreaming()) {
                turn.run((h) => config.reattach!(h), { reattach: true });
            }
        });
    return control;
}

interface TranscriptResponse {
    messages: TranscriptMessage[];
}

// Per-agent (project) chat entry: mounts the shared component on the detail page's
// `#agent-chat`, wired to the agent's own endpoints. Fork here is a REVERT (the
// single session rewinds to the edit) via the per-agent fork endpoint. Image
// attach, slash commands (export/help) and export are all on; no sidebar.
export function startAgentChat(): void {
    const root = document.getElementById("agent-chat");
    if (!root) return;
    const id = agentIdFromPath(window.location.pathname);
    if (!id) return;
    const enc = encodeURIComponent(id);

    const control = createAgentChat(root, {
        title: "Chat",
        exportTitle: `Agent ${id} chat`,
        exportFilename: `scufris-agent-${id}-chat.md`,
        enableImage: true,
        forkVerb: "revert",
        forkHint:
            "Editing rewinds this conversation to here and continues from your edit (the later messages are dropped).",
        streamTurn: (message, handlers, image, signal) =>
            streamChatTurn(
                `/api/agents/${enc}/chat`,
                message,
                handlers,
                image,
                signal,
            ),
        // Stop this agent's in-flight run (the manual "go to its chat and cancel"
        // path). Truly aborts the backend turn; ignore a 404 (nothing running).
        cancelTurn: async () => {
            try {
                await apiFetch(`/api/agents/${enc}/cancel`, { method: "POST" });
            } catch {
                // best-effort: the local settle already kept the partial
            }
        },
        forkTurn: (index, text, handlers, signal) =>
            streamPost(
                `/api/agents/${enc}/fork`,
                { message_index: index, text },
                handlers,
                signal,
            ),
        loadTranscript: async () => {
            const resp = await fetchJson<TranscriptResponse>(
                `/api/agents/${enc}/transcript`,
            );
            return resp.messages.map((m) => ({
                role: m.role === "assistant" ? "assistant" : "user",
                text: m.text,
                reply: transcriptReply(m),
                ts: parseIso(m.ts),
                // Re-hydrate the reloaded turn's "thinking" spoiler; renderChatLog
                // renders it collapsed, identical to the live/settled turn.
                reasoning: m.reasoning ?? undefined,
            }));
        },
        reattach: async (handlers) => {
            // Follow the live run bus so a reload/reselect mid-turn keeps
            // streaming (the turn may be one the orchestrator drives against this
            // agent). Gate on active status: a finished run's bus replays its last
            // turn then closes, which must NOT render as a phantom live bubble and
            // would otherwise loop the EventSource reconnect. 404/idle -> no-op.
            let status: AgentRunStatus;
            try {
                status = await fetchJson<AgentRunStatus>(
                    `/api/agents/${enc}/status`,
                );
            } catch {
                return;
            }
            if (status.state !== "running" && status.state !== "queued") return;
            // Render the driving turn's prompt (e.g. one the orchestrator sent)
            // as a user bubble before streaming the reply, so the conversation
            // reads from the start instead of showing only the answer until a
            // later reload re-reads the transcript.
            if (status.prompt) handlers.onUserPrompt?.(status.prompt);
            await subscribeEvents(`/api/agents/${enc}/events`, handlers);
        },
    });

    control.setSlashCommands([
        {
            name: "export",
            description: "download this chat as markdown",
            run: () => control.exportChat(),
        },
    ]);
}
