// Agent page: the chat panel. No import-time side effects (the `agent.ts` entry
// calls `startAgent`).

import { el, loadConfig, type AppConfig, type ChatReply } from "./common";

function appendMessage(
    log: HTMLElement,
    role: string,
    text: string,
): HTMLElement {
    const msg = el("div", `chat__msg chat__msg--${role}`);
    msg.textContent = text;
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
    return msg;
}

async function sendChat(message: string): Promise<ChatReply> {
    const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
    });
    if (!resp.ok) {
        const detail = (await resp.json().catch(() => null)) as {
            detail?: string;
        } | null;
        throw new Error(
            detail?.detail || `chat failed (${String(resp.status)})`,
        );
    }
    return (await resp.json()) as ChatReply;
}

export function initChat(config: AppConfig): void {
    const form = document.getElementById("chat-form") as HTMLFormElement | null;
    const input = document.getElementById(
        "chat-input",
    ) as HTMLInputElement | null;
    const log = document.getElementById("chat-log");
    const reset = document.getElementById("chat-reset");
    if (!form || !input || !log || !reset) return;

    if (!config.agent_enabled) {
        appendMessage(
            log,
            "system",
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login`.",
        );
        input.disabled = true;
        return;
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const message = input.value.trim();
        if (!message) return;
        appendMessage(log, "user", message);
        input.value = "";
        input.disabled = true;
        const pending = appendMessage(log, "assistant", "...");
        sendChat(message)
            .then((reply) => {
                pending.textContent = reply.text || "(no reply)";
            })
            .catch((err: unknown) => {
                pending.classList.add("chat__msg--error");
                pending.textContent =
                    err instanceof Error ? err.message : "error";
            })
            .finally(() => {
                input.disabled = false;
                input.focus();
                log.scrollTop = log.scrollHeight;
            });
    });

    reset.addEventListener("click", () => {
        void fetch("/api/chat/reset", { method: "POST" }).finally(() => {
            log.replaceChildren();
        });
    });
}

export async function startAgent(): Promise<void> {
    const config = await loadConfig();
    initChat(config);
}
