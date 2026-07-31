// The chat component's own contracts: what a message is, what the entry injects
// (`AgentChatConfig`), what it gets back (`ChatControl`), and the render options
// that turn the pure log render into the interactive one. Kept apart from
// `agent-chat-view.ts` so the log, turn and composer modules can name `ChatMsg`
// without importing the module that imports them.

import type { ChatReply, ImageAttachment } from "./agent-types";
import type { StreamHandlers } from "./chat-stream";
import type { SlashCommand } from "./chat-commands";

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

// Options that turn the pure log render into the interactive one: the empty state
// to show, the message being edited (with its editor builder), and the edit-to-
// fork callback (present -> user turns get an "edit" button).
export interface RenderChatOpts {
    emptyState?: HTMLElement;
    editingIndex?: number | null;
    buildEditor?: (index: number, text: string) => HTMLElement;
    onEdit?: (index: number) => void;
}
