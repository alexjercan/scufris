// The streaming turn, shared by send, fork and reattach. It owns the turn's own
// state - `streaming` and the current stop handle - which the composer, the fork
// editor and the mount-time reattach guard all read back through the returned
// handle.
//
// It takes explicit deps rather than the component: `appendMessage`/`lastMessage`
// are callbacks because `createAgentChat` REASSIGNS its `msgs` array (setMessages,
// reset, forkFrom), so a captured array reference would go stale.

import { el } from "./common";
import type { ChatReply, ToolCall } from "./agent-types";
import { renderMarkdown } from "./markdown";
import type { StreamHandlers } from "./chat-stream";
import { distinctTools } from "./agent-chat-log";
import type { AgentChatConfig, ChatMsg } from "./agent-chat-types";

export interface TurnRunnerDeps {
    log: HTMLElement;
    config: AgentChatConfig;
    appendMessage: (msg: ChatMsg) => void;
    lastMessage: () => ChatMsg | undefined;
    render: () => void;
    maybeScroll: () => void;
    setComposerEnabled: (on: boolean) => void;
    setStopMode: (stopping: boolean) => void;
    // Run after a turn settles, however it ended: re-fit and refocus the composer.
    onSettled: () => void;
}

export interface TurnRunner {
    // Append a live pending bubble and drive it from `runner`. exec backends have
    // no deltas -> a "working... Ns" indicator + tool line; app_server streams text
    // token-by-token with a collapsible "thinking" section.
    //
    // `mode.reattach` follows a turn that was started ELSEWHERE (an orchestrator
    // message_agent/run_agent turn, or a turn already in flight when this page
    // (re)loaded), streamed off the run's event bus. The one difference from a
    // local turn is that the bubble appears only once a frame actually arrives: an
    // idle run yields no frame, so the composer is never needlessly disabled and
    // no phantom bubble shows. It settles the same way a local turn does (push the
    // terminal reply); the reattached turn's user/prompt side comes from the
    // mount-time transcript load.
    run: (
        runner: (h: StreamHandlers, signal: AbortSignal) => Promise<void>,
        mode?: { reattach?: boolean },
    ) => void;
    isStreaming: () => boolean;
    // The stop button (and Enter-while-streaming is deliberately inert): cancel the
    // current turn if one is streaming and a cancel path is wired.
    requestCancel: () => void;
}

export function createTurnRunner(deps: TurnRunnerDeps): TurnRunner {
    const { log, config } = deps;
    let streaming = false;
    // The current turn's stop handle: abort the local fetch + POST the backend
    // cancel. Set while a turn streams (only when config.cancelTurn is wired),
    // cleared on settle. requestCancel() drives it from the stop button.
    let cancelCurrent: (() => void) | null = null;

    const run = (
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
            deps.maybeScroll();
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
            deps.setComposerEnabled(false);
            // Keep the button live as a STOP control while streaming (only when a
            // cancel path is wired). Without cancelTurn the turn runs to completion
            // and the button stays disabled, as before.
            if (config.cancelTurn) {
                deps.setStopMode(true);
                cancelCurrent = (): void => {
                    finishCancelled();
                    controller.abort();
                    void config.cancelTurn?.();
                };
            }
            log.appendChild(pending);
            paintStatus();
            timer = window.setInterval(paintStatus, 500);
            deps.maybeScroll();
        };
        const stop = (): void => {
            window.clearInterval(timer);
            window.clearTimeout(flushTimer);
            streaming = false;
            cancelCurrent = null;
            deps.setComposerEnabled(true);
            deps.setStopMode(false);
            deps.onSettled();
        };
        // Settle the partial as a KEPT, interrupted assistant message: the tokens
        // streamed so far stay in the transcript tagged "(cancelled)", so the
        // conversation can continue with them in mind. No reply meta (the turn did
        // not finish), just the partial text + the cancelled tag.
        const finishCancelled = (): void => {
            if (done) return;
            done = true;
            deps.appendMessage({
                role: "assistant",
                text: streamed,
                ts: Date.now(),
                reasoning: reasoning || undefined,
                cancelled: true,
            });
            deps.render();
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
            deps.appendMessage({
                role: "assistant",
                text: reply.text || streamed || "(no reply)",
                reply,
                ts: Date.now(),
                // Keep the live "thinking" so the next render still shows it as
                // a collapsed spoiler instead of dropping it on settle.
                reasoning: reasoning || undefined,
            });
            deps.render();
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
                    const last = deps.lastMessage();
                    if (last && last.role === "user" && last.text === text)
                        return;
                    deps.appendMessage({
                        role: "user",
                        text,
                        ts: Date.now(),
                    });
                    deps.render();
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
                if (attached) deps.render();
            })
            .catch((err: unknown) => {
                fail(err instanceof Error ? err.message : "error");
            });
    };

    return {
        run,
        isStreaming: () => streaming,
        requestCancel: () => {
            if (!streaming) return;
            cancelCurrent?.();
        },
    };
}
