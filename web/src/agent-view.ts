// Agent page: the chat panel plus the agent info (model), tools panel, per-turn
// tool-call chips and a running token/context indicator. No import-time side
// effects (the `agent.ts` entry calls `startAgent`); the render helpers are
// exported so the jsdom tests can drive them without fetch.

import {
    el,
    escapeHtml,
    fetchJson,
    loadConfig,
    type AgentInfo,
    type AgentTool,
    type AppConfig,
    type ChatReply,
    type TokenUsage,
} from "./common";

// Session usage, persisted across turns; reset on "new chat".
let _cumulativeOutput = 0;
let _lastContext = 0;

function fmtTokens(n: number): string {
    return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

export function applyUsage(usage: TokenUsage | null): void {
    if (usage) {
        _cumulativeOutput += usage.output_tokens;
        _lastContext = usage.input_tokens;
    }
    const usageEl = document.getElementById("agent-usage");
    if (!usageEl) return;
    usageEl.textContent =
        _cumulativeOutput > 0 || _lastContext > 0
            ? `ctx ${fmtTokens(_lastContext)} · ${fmtTokens(_cumulativeOutput)} out`
            : "";
}

export function _resetAgentState(): void {
    _cumulativeOutput = 0;
    _lastContext = 0;
    applyUsage(null);
}

export function renderAgentPanel(
    info: AgentInfo | null,
    tools: AgentTool[],
): void {
    const modelEl = document.getElementById("agent-model");
    if (modelEl) modelEl.textContent = info ? `model ${info.model}` : "";

    const toggle = document.getElementById(
        "agent-tools-toggle",
    ) as HTMLButtonElement | null;
    const panel = document.getElementById("agent-tools");
    if (!toggle || !panel) return;

    if (tools.length === 0) {
        toggle.hidden = true;
        return;
    }
    toggle.hidden = false;
    toggle.textContent = `tools (${tools.length})`;
    panel.replaceChildren();
    for (const tool of tools) {
        const item = el("div", "agent-tools__item");
        item.innerHTML =
            `<span class="agent-tools__name">${escapeHtml(tool.name)}</span>` +
            `<span class="agent-tools__desc">${escapeHtml(tool.description)}</span>`;
        panel.appendChild(item);
    }
    toggle.onclick = () => {
        panel.hidden = !panel.hidden;
    };
}

export function messageMeta(reply: ChatReply): HTMLElement | null {
    const bits: string[] = [];
    for (const call of reply.tool_calls) {
        bits.push(`<span class="chat__chip">${escapeHtml(call.tool)}</span>`);
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
                const meta = messageMeta(reply);
                if (meta) pending.after(meta);
                applyUsage(reply.usage);
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
        _resetAgentState();
        void fetch("/api/chat/reset", { method: "POST" }).finally(() => {
            log.replaceChildren();
        });
    });
}

export async function startAgent(): Promise<void> {
    const config = await loadConfig();
    initChat(config);
    if (!config.agent_enabled) return;
    try {
        const [info, tools] = await Promise.all([
            fetchJson<AgentInfo>("/api/agent/info"),
            fetchJson<AgentTool[]>("/api/agent/tools"),
        ]);
        renderAgentPanel(info, tools);
    } catch (err: unknown) {
        console.error(err);
    }
}
