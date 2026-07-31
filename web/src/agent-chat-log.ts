// The settled conversation log: PURE rendering with no component state, so the
// jsdom tests drive it directly. Assistant text is untrusted model output and
// goes through renderMarkdown (never innerHTML); user text goes in via
// textContent.

import { el, escapeHtml } from "./common";
import type { ChatReply, TranscriptMessage } from "./agent-types";
import { renderMarkdown } from "./markdown";
import { formatTimestamp } from "./chat-format";
import type { ChatMsg, RenderChatOpts } from "./agent-chat-types";

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
