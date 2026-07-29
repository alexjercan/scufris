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

import { apiFetch, el, escapeHtml, fetchJson } from "./common";
import type {
    AgentRunStatus,
    ChatReply,
    ImageAttachment,
    ToolCall,
    TranscriptMessage,
} from "./common";
import { renderMarkdown } from "./markdown";
import { streamChatTurn, streamPost, subscribeEvents } from "./chat-stream";
import type { StreamHandlers } from "./chat-stream";
import { formatTimestamp, parseIso } from "./chat-format";
import {
    downloadChatMarkdown,
    matchSlashCommands,
    type SlashCommand,
} from "./chat-commands";
import { readImageFile, type PendingImage } from "./chat-image";
import { agentIdFromPath } from "./agent-detail-view";

// One settled message in the conversation. `reply` carries an assistant turn's
// tools/tokens (the meta line) so they survive a re-render or a transcript reload.
// `ts` is epoch ms for the timestamp; `imageUrl` is a user's attached image.
export interface ChatMsg {
    role: "user" | "assistant";
    text: string;
    reply?: ChatReply;
    ts?: number;
    imageUrl?: string;
    // Codex "thinking" (reasoning) that streamed live during the turn. Carried
    // onto the settled message so it survives the settle re-render as a
    // collapsed spoiler. Ephemeral: not recoverable from the transcript on a
    // hard reload (reasoning is persisted only as an encrypted blob).
    reasoning?: string;
    // The user stopped this turn mid-stream: the partial answer is kept (so the
    // conversation can continue with it in mind) and tagged as interrupted.
    cancelled?: boolean;
}

// The injected wiring. `streamTurn`/`loadTranscript` are required; the rest are
// opt-in capabilities (present -> the affordance renders).
export interface AgentChatConfig {
    // Stream one chat turn (optionally with an attached image). `signal` aborts
    // the in-flight fetch when the user hits stop (the backend run is cancelled
    // separately via `cancelTurn`).
    streamTurn(
        message: string,
        handlers: StreamHandlers,
        image?: ImageAttachment,
        signal?: AbortSignal,
    ): Promise<void>;
    // Present -> the composer shows a stop button while a turn streams; hitting it
    // calls this to cancel the agent's in-flight run on the backend (truly aborts
    // it, not just detaches the SSE relay). The local fetch is aborted separately.
    // Absent -> no stop affordance (the turn can only run to completion).
    cancelTurn?: () => Promise<void>;
    // Load the initial conversation (empty for the orchestrator, which starts on
    // the welcome state and loads a session only when one is picked).
    loadTranscript(): Promise<ChatMsg[]>;
    // Present -> after the transcript loads on mount, attach to any IN-FLIGHT run
    // for this agent and stream it into the log via the given handlers, resolving
    // when the turn ends (or immediately when no run is active). Lets a reload or
    // reselect mid-turn - including a turn the orchestrator drives against this
    // sub-agent - keep streaming instead of freezing on the settled transcript.
    // Injected (not built here) so the pure component stays testable without a
    // real EventSource; the real wiring lives in startAgentChat.
    reattach?(handlers: StreamHandlers): Promise<void>;
    // Present -> user turns get an edit-to-fork affordance. Forks the conversation
    // at `index`, replacing that message with `text`, and streams the reply.
    forkTurn?: (
        index: number,
        text: string,
        handlers: StreamHandlers,
        signal?: AbortSignal,
    ) => Promise<void>;
    // Copy for the fork editor's confirm button + its hint (new-session vs revert).
    forkVerb?: string;
    forkHint?: string;
    // Called after each settled turn/fork so the orchestrator can refresh its
    // sidebar (the authoritative token/session source).
    onAfterTurn?(): void;
    // Opt-in capabilities.
    enableImage?: boolean;
    title?: string;
    exportTitle?: string;
    exportFilename?: string;
    // The onboarding empty state (example prompts + a fork tip). Orchestrator only.
    welcome?: { examples: string[]; forkHint?: boolean };
    // When set, the chat is inert: the notice shows in the log and the composer is
    // disabled (the agent is turned off).
    disabledReason?: string;
}

// An imperative handle the entry drives: load a session's messages, reset to the
// empty state, focus/fill the composer, export, and install slash commands (which
// close over this handle, so they are set after creation - see the entries).
export interface ChatControl {
    setMessages(history: ChatMsg[]): void;
    reset(): void;
    focus(): void;
    fillComposer(text: string): void;
    exportChat(): void;
    setSlashCommands(commands: SlashCommand[]): void;
    // Append a transient system note to the log (e.g. /help output). Not tracked
    // in the conversation; wiped by the next render.
    note(text: string): void;
}

// Distinct tool names in first-occurrence order. A polling turn calls the same
// tool many times (e.g. agent_status while waiting); the meta line lists WHICH
// tools ran, not how often, so collapse the repeats to one name each.
export function distinctTools(names: readonly string[]): string[] {
    return [...new Set(names)];
}

// The assistant meta line (tool chips + token count) built from a reply. Returns
// null when there is nothing to show (a plain reply), so no empty line renders.
export function messageMeta(reply: ChatReply): HTMLElement | null {
    const bits: string[] = [];
    if (reply.tool_calls.length > 0) {
        // A clear "ran" label in front of prominent tool chips - tool execution is
        // the point of the agent, so surface it rather than a faint badge.
        bits.push(`<span class="chat__ran">ran</span>`);
        for (const tool of distinctTools(reply.tool_calls.map((c) => c.tool))) {
            bits.push(`<span class="chat__chip">${escapeHtml(tool)}</span>`);
        }
    }
    if (reply.usage) {
        bits.push(
            `<span class="chat__tok">${reply.usage.output_tokens} tok</span>`,
        );
    }
    if (bits.length === 0) return null;
    const meta = el("div", "chat__meta");
    meta.innerHTML = bits.join("");
    return meta;
}

// Rebuild an assistant message's reply meta from a reloaded transcript, so a past
// session shows what it ran. Undefined for a turn with no tools/usage (user turns
// or a plain assistant answer) so no meta line renders.
export function transcriptReply(m: TranscriptMessage): ChatReply | undefined {
    if (m.tool_calls.length === 0 && !m.usage) return undefined;
    return { text: m.text, tool_calls: m.tool_calls, usage: m.usage };
}

// A clipboard-copy button that flips its label to "copied" briefly. Guarded:
// `navigator.clipboard` is absent in jsdom and on insecure origins, so it no-ops
// there rather than throwing. `getText` is read lazily at click time.
function copyButton(getText: () => string): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chat__copy";
    btn.textContent = "copy";
    btn.title = "copy to clipboard";
    btn.addEventListener("click", () => {
        const clip = navigator.clipboard;
        if (!clip) return;
        void clip.writeText(getText()).then(
            () => {
                btn.textContent = "copied";
                setTimeout(() => (btn.textContent = "copy"), 1200);
            },
            () => undefined,
        );
    });
    return btn;
}

// Options that turn the pure log render into the interactive one: the empty state
// to show, the message being edited (with its editor builder), and the edit-to-
// fork callback (present -> user turns get an "edit" button).
export interface RenderChatOpts {
    emptyState?: HTMLElement;
    editingIndex?: number | null;
    buildEditor?: (index: number, text: string) => HTMLElement;
    onEdit?: (index: number) => void;
}

function messageFoot(
    entry: ChatMsg,
    index: number,
    opts: RenderChatOpts,
): HTMLElement {
    const foot = el("div", `chat__foot chat__foot--${entry.role}`);
    const label = formatTimestamp(entry.ts);
    if (label && entry.ts !== undefined) {
        const time = document.createElement("time");
        time.className = "chat__time";
        time.textContent = label;
        time.dateTime = new Date(entry.ts).toISOString();
        time.title = new Date(entry.ts).toLocaleString();
        foot.appendChild(time);
    }
    if (entry.role === "assistant") {
        foot.appendChild(copyButton(() => entry.text));
    } else if (entry.role === "user" && opts.onEdit) {
        const edit = el("button", "chat__edit");
        edit.setAttribute("type", "button");
        edit.textContent = "edit";
        edit.title = "edit this message and branch the conversation";
        const onEdit = opts.onEdit;
        edit.addEventListener("click", () => onEdit(index));
        foot.appendChild(edit);
    }
    return foot;
}

// Render the whole settled log (PURE - no component state). Assistant text is
// untrusted model output, built safely via renderMarkdown (no innerHTML); user
// text goes in via textContent; a user's own attached image renders inline.
export function renderChatLog(
    log: HTMLElement,
    msgs: ChatMsg[],
    opts: RenderChatOpts = {},
): void {
    log.replaceChildren();
    if (msgs.length === 0 && opts.editingIndex == null) {
        log.appendChild(
            opts.emptyState ??
                el(
                    "div",
                    "chat__empty settings__empty",
                    "no messages yet - say something to start.",
                ),
        );
        return;
    }
    msgs.forEach((entry, index) => {
        if (
            entry.role === "user" &&
            index === opts.editingIndex &&
            opts.buildEditor
        ) {
            log.appendChild(opts.buildEditor(index, entry.text));
            return;
        }
        // A settled assistant turn re-shows the reasoning that streamed live as
        // a collapsed spoiler above its answer, mirroring the live bubble's
        // layout (status, thinking, body) and reusing the same styling. Closed
        // by default: `<details>` with no `open` attribute.
        if (entry.role === "assistant" && entry.reasoning) {
            const thinking = el("details", "chat__thinking");
            const thinkingBody = el("div", "chat__thinking-body");
            thinkingBody.textContent = entry.reasoning;
            thinking.append(el("summary", "", "thinking"), thinkingBody);
            log.appendChild(thinking);
        }
        const node = el("div", `chat__msg chat__msg--${entry.role}`);
        if (entry.role === "assistant") {
            node.classList.add("chat__msg--md");
            node.appendChild(renderMarkdown(entry.text));
        } else {
            node.textContent = entry.text;
        }
        if (entry.imageUrl) {
            const img = document.createElement("img");
            img.className = "chat__attach-img";
            img.src = entry.imageUrl; // the user's own image, safe to display
            img.alt = "attached image";
            node.appendChild(img);
        }
        if (entry.role === "assistant" && entry.cancelled) {
            // A muted inline tag on the kept partial, so the reader sees the turn
            // was stopped rather than completed - without polluting entry.text.
            node.appendChild(el("span", "chat__cancelled", "(cancelled)"));
        }
        log.appendChild(node);
        if (entry.role === "assistant" && entry.reply) {
            const meta = messageMeta(entry.reply);
            if (meta) log.appendChild(meta);
        }
        log.appendChild(messageFoot(entry, index, opts));
    });
}

// Grow the composer textarea to fit its content up to a max, then let it scroll.
// jsdom has no layout (scrollHeight is 0), so this is a no-op under tests.
const COMPOSER_MAX_HEIGHT = 200;
function autosize(input: HTMLTextAreaElement): void {
    input.style.height = "auto";
    const next = Math.min(input.scrollHeight, COMPOSER_MAX_HEIGHT);
    input.style.height = `${next}px`;
    input.style.overflowY =
        input.scrollHeight > COMPOSER_MAX_HEIGHT ? "auto" : "hidden";
}

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
    let streaming = false;
    // The current turn's stop handle: abort the local fetch + POST the backend
    // cancel. Set while a turn streams (only when config.cancelTurn is wired),
    // cleared on settle. requestCancel() drives it from the stop button.
    let cancelCurrent: (() => void) | null = null;
    let rendering = false;
    let stickToBottom = true;
    let unreadCount = 0;
    let prevMsgCount = 0;
    let pendingImage: PendingImage | null = null;
    let slashCommands: SlashCommand[] = [];
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
    // textContent (which is empty in stop mode).
    const setStopMode = (stopping: boolean): void => {
        send.classList.toggle("is-stopping", stopping);
        send.setAttribute("aria-label", stopping ? "stop" : "send message");
        send.textContent = stopping ? "" : "send";
    };

    // The stop button (and Enter-while-streaming is deliberately inert): cancel the
    // current turn if one is streaming and a cancel path is wired.
    const requestCancel = (): void => {
        if (!streaming) return;
        cancelCurrent?.();
    };

    // --- Image attach ---
    const renderAttachPreview = (): void => {
        if (!pendingImage) {
            attach.hidden = true;
            attach.replaceChildren();
            return;
        }
        attach.replaceChildren();
        const img = document.createElement("img");
        img.className = "chat__attach-thumb";
        img.src = pendingImage.dataUrl; // the user's own image, safe to display
        img.alt = "attached image";
        const remove = el("button", "chat__attach-remove", "×");
        remove.setAttribute("type", "button");
        remove.setAttribute("aria-label", "remove attachment");
        remove.addEventListener("click", () => {
            pendingImage = null;
            renderAttachPreview();
        });
        attach.append(img, remove);
        attach.hidden = false;
    };

    const acceptImage = (file: File): void => {
        void readImageFile(file).then((img) => {
            if (!img) return;
            pendingImage = img;
            renderAttachPreview();
        });
    };

    // --- Streaming turn (shared by send, fork, and reattach) ---
    // Append a live pending bubble and drive it from a turn runner. exec backends
    // have no deltas -> a "working... Ns" indicator + tool line; app_server streams
    // text token-by-token with a collapsible "thinking" section.
    //
    // `mode.reattach` follows a turn that was started ELSEWHERE (an orchestrator
    // message_agent/run_agent turn, or a turn already in flight when this page
    // (re)loaded), streamed off the run's event bus. The one difference from a
    // local turn is that the bubble appears only once a frame actually arrives: an
    // idle run yields no frame, so the composer is never needlessly disabled and
    // no phantom bubble shows. It settles the same way a local turn does (push the
    // terminal reply); the reattached turn's user/prompt side comes from the
    // mount-time transcript load.
    const runTurn = (
        runner: (h: StreamHandlers, signal: AbortSignal) => Promise<void>,
        mode: { reattach?: boolean } = {},
    ): void => {
        // Aborts the local fetch the instant the user hits stop; the backend run is
        // cancelled separately via config.cancelTurn. One controller per turn.
        const controller = new AbortController();
        // A single-settle guard: cancel, done and error all try to finalize the
        // turn once - whichever fires first wins, the rest no-op.
        let done = false;
        const pending = el("div", "chat__msg chat__msg--pending");
        const status = el("div", "chat__status");
        const spinner = el("span", "chat__spinner");
        const statusLabel = el("span", "chat__pending-label");
        status.append(spinner, statusLabel);
        const thinking = el("details", "chat__thinking");
        const thinkingBody = el("div", "chat__thinking-body");
        thinking.append(el("summary", "", "thinking"), thinkingBody);
        thinking.hidden = true;
        const body = el("div", "chat__stream-body");
        pending.append(status, thinking, body);

        const started = Date.now();
        const tools: string[] = [];
        let streamed = "";
        let reasoning = "";
        let lastRender = 0;
        let flushTimer = 0;
        let timer = 0;
        let attached = false;

        const paintStatus = (): void => {
            const secs = Math.floor((Date.now() - started) / 1000);
            const ran = tools.length
                ? ` · ran ${distinctTools(tools).join(", ")}`
                : "";
            const what = streamed ? "streaming" : "working";
            statusLabel.textContent = `${what}... ${secs}s${ran}`;
        };
        // Paint the growing answer eagerly (first token shows at once), throttled
        // to ~50ms so a fast stream does not thrash the DOM. Deliberately NOT rAF:
        // a queued frame can be clobbered by the settle re-render before it paints.
        const renderNow = (): void => {
            window.clearTimeout(flushTimer);
            flushTimer = 0;
            lastRender = Date.now();
            body.replaceChildren(renderMarkdown(streamed));
            maybeScroll();
        };
        const scheduleRender = (): void => {
            const since = Date.now() - lastRender;
            if (since >= 50) renderNow();
            else if (!flushTimer)
                flushTimer = window.setTimeout(renderNow, 50 - since);
        };
        // Show the live bubble and take over the composer. Deferred until the first
        // frame in reattach mode, so following an idle run is a no-op.
        const ensureBubble = (): void => {
            if (attached) return;
            attached = true;
            streaming = true;
            setComposerEnabled(false);
            // Keep the button live as a STOP control while streaming (only when a
            // cancel path is wired). Without cancelTurn the turn runs to completion
            // and the button stays disabled, as before.
            if (config.cancelTurn) {
                send.disabled = false;
                setStopMode(true);
                cancelCurrent = (): void => {
                    finishCancelled();
                    controller.abort();
                    void config.cancelTurn?.();
                };
            }
            log.appendChild(pending);
            paintStatus();
            timer = window.setInterval(paintStatus, 500);
            maybeScroll();
        };
        const stop = (): void => {
            window.clearInterval(timer);
            window.clearTimeout(flushTimer);
            streaming = false;
            cancelCurrent = null;
            setComposerEnabled(true);
            setStopMode(false);
            autosize(input);
            input.focus();
        };
        // Settle the partial as a KEPT, interrupted assistant message: the tokens
        // streamed so far stay in the transcript tagged "(cancelled)", so the
        // conversation can continue with them in mind. No reply meta (the turn did
        // not finish), just the partial text + the cancelled tag.
        const finishCancelled = (): void => {
            if (done) return;
            done = true;
            msgs.push({
                role: "assistant",
                text: streamed,
                ts: Date.now(),
                reasoning: reasoning || undefined,
                cancelled: true,
            });
            render();
            config.onAfterTurn?.();
            stop();
        };
        // Settle identically whether the turn was local or reattached: push the
        // terminal reply the bus/POST gave us (it carries text + tool_calls +
        // usage, so chips and the token count survive). A reattached turn's
        // user/prompt side is already in the log from the mount-time transcript
        // load; we deliberately do NOT re-fetch the transcript here, because the
        // backend persists the (possibly new) session id in a post-turn callback
        // that races the `done` frame - a reload could read an empty/stale
        // transcript and drop the very turn we just streamed.
        const settle = (reply: ChatReply): void => {
            if (done) return;
            done = true;
            msgs.push({
                role: "assistant",
                text: reply.text || streamed || "(no reply)",
                reply,
                ts: Date.now(),
                // Keep the live "thinking" so the next render still shows it as
                // a collapsed spoiler instead of dropping it on settle.
                reasoning: reasoning || undefined,
            });
            render();
            // Fires on every settled turn, reattached ones included. Only the
            // orchestrator landing sets it (to refresh its sidebar) and that entry
            // does not use reattach, so it is a no-op on the per-agent page today.
            config.onAfterTurn?.();
            stop();
        };
        const fail = (detail: string): void => {
            if (done) return;
            done = true;
            ensureBubble();
            pending.classList.remove("chat__msg--pending");
            pending.classList.add("chat__msg--error");
            pending.replaceChildren();
            pending.textContent = streamed
                ? streamed
                : `chat failed: ${detail}`;
            stop();
        };

        if (!mode.reattach) ensureBubble();

        void runner(
            {
                onTextDelta: (delta) => {
                    ensureBubble();
                    streamed += delta;
                    pending.classList.add("chat__msg--md");
                    scheduleRender();
                },
                onReasoningDelta: (delta) => {
                    ensureBubble();
                    reasoning += delta;
                    thinking.hidden = false;
                    thinkingBody.textContent = reasoning;
                },
                onTool: (tool: ToolCall) => {
                    ensureBubble();
                    tools.push(tool.tool);
                    paintStatus();
                },
                // Reattach-only: the in-flight turn's prompt, injected as a user
                // bubble the mount-time transcript did not yet carry (the backend has
                // not flushed the turn to its durable log). Runs BEFORE the pending
                // bubble attaches (ensureBubble is deferred to the first frame in
                // reattach mode), so this render is not clobbered. Skip if the log
                // already ends with the same prompt, so a transcript that DID catch up
                // is not duplicated.
                onUserPrompt: (text: string) => {
                    const last = msgs[msgs.length - 1];
                    if (last && last.role === "user" && last.text === text)
                        return;
                    msgs.push({ role: "user", text, ts: Date.now() });
                    render();
                },
                onDone: settle,
                onError: fail,
            },
            controller.signal,
        )
            .then(() => {
                // The runner resolved without a terminal frame. Already finalized
                // (settled / failed / cancelled) -> nothing to do. A normal turn has
                // already settled; a user-cancelled turn already settled its partial
                // (the aborted fetch resolves here). A reattach to an idle run never
                // attached, so this is a clean no-op. If a stream did show frames but
                // closed with no terminal (defensive - the backend always sends
                // done/error), drop the dangling bubble.
                if (done || !streaming) return;
                stop();
                if (attached) render();
            })
            .catch((err: unknown) => {
                fail(err instanceof Error ? err.message : "error");
            });
    };

    const submit = (): void => {
        if (streaming || input.disabled) return;
        const text = input.value.trim();
        if (!text) return;
        closePalette();
        const image = pendingImage; // captured before clearing the composer
        editingIndex = null;
        msgs.push({
            role: "user",
            text,
            ts: Date.now(),
            imageUrl: image?.dataUrl,
        });
        stickToBottom = true;
        input.value = "";
        pendingImage = null;
        renderAttachPreview();
        autosize(input);
        render();
        runTurn((h, signal) =>
            config.streamTurn(text, h, image?.attachment, signal),
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
        if (!trimmed || !config.forkTurn || streaming) {
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
        runTurn((h, signal) => fork(index, trimmed, h, signal));
    };

    // --- Slash-command palette ---
    let paletteItems: SlashCommand[] = [];
    let paletteIdx = 0;
    const paletteOpen = (): boolean => !palette.hidden;
    function closePalette(): void {
        paletteIdx = 0;
        palette.hidden = true;
        palette.replaceChildren();
    }
    const renderPalette = (): void => {
        paletteItems = matchSlashCommands(input.value, slashCommands);
        if (paletteItems.length === 0) {
            closePalette();
            return;
        }
        if (paletteIdx >= paletteItems.length) paletteIdx = 0;
        palette.replaceChildren();
        paletteItems.forEach((cmd, i) => {
            const item = el(
                "div",
                `chat__palette-item${i === paletteIdx ? " is-active" : ""}`,
                `<span class="chat__palette-name">/${escapeHtml(cmd.name)}</span>` +
                    `<span class="chat__palette-desc">${escapeHtml(cmd.description)}</span>`,
            );
            item.setAttribute("role", "option");
            item.setAttribute(
                "aria-selected",
                i === paletteIdx ? "true" : "false",
            );
            // mousedown (not click) so it fires before the textarea blurs.
            item.addEventListener("mousedown", (event) => {
                event.preventDefault();
                runCommand(cmd);
            });
            palette.appendChild(item);
        });
        palette.hidden = false;
    };
    const runCommand = (cmd: SlashCommand): void => {
        input.value = "";
        autosize(input);
        closePalette();
        cmd.run();
    };

    // --- Composer wiring ---
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        // While a turn streams the button is the STOP control - submit cancels the
        // run instead of sending. Otherwise it sends normally.
        if (streaming) {
            requestCancel();
            return;
        }
        submit();
    });
    input.addEventListener("keydown", (event: KeyboardEvent) => {
        if (paletteOpen()) {
            if (event.key === "ArrowDown") {
                event.preventDefault();
                paletteIdx = (paletteIdx + 1) % paletteItems.length;
                renderPalette();
                return;
            }
            if (event.key === "ArrowUp") {
                event.preventDefault();
                paletteIdx =
                    (paletteIdx - 1 + paletteItems.length) %
                    paletteItems.length;
                renderPalette();
                return;
            }
            if (event.key === "Enter" || event.key === "Tab") {
                event.preventDefault();
                runCommand(paletteItems[paletteIdx]);
                return;
            }
            if (event.key === "Escape") {
                event.preventDefault();
                closePalette();
                return;
            }
        }
        // Enter sends; Shift+Enter inserts a newline. Guard isComposing so
        // committing an IME candidate with Enter does not fire a half-typed send.
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            submit();
        }
    });
    input.addEventListener("input", () => {
        autosize(input);
        renderPalette();
    });
    input.addEventListener("blur", () => closePalette());

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

    if (config.enableImage) {
        attachBtn.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", () => {
            const file = fileInput.files?.[0];
            if (file) acceptImage(file);
            fileInput.value = ""; // allow re-picking the same file
        });
        input.addEventListener("paste", (event) => {
            const items = event.clipboardData?.items;
            if (!items) return;
            for (const item of items) {
                if (item.type.startsWith("image/")) {
                    const file = item.getAsFile();
                    if (file) {
                        event.preventDefault();
                        acceptImage(file);
                    }
                    break;
                }
            }
        });
    }

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
            pendingImage = null;
            stickToBottom = true;
            unreadCount = 0;
            renderAttachPreview();
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
            slashCommands = commands;
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
            if (config.reattach && !streaming) {
                runTurn((h) => config.reattach!(h), { reattach: true });
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
