import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatReply, ImageAttachment } from "./agent-types";
import type { StreamHandlers } from "./chat-stream";
import { createAgentChat } from "./agent-chat-view";
import type { AgentChatConfig } from "./agent-chat-types";

function reply(over: Partial<ChatReply> = {}): ChatReply {
    return { text: "hi", tool_calls: [], usage: null, ...over };
}

function config(over: Partial<AgentChatConfig> = {}): AgentChatConfig {
    return {
        streamTurn: () => Promise.resolve(),
        loadTranscript: () => Promise.resolve([]),
        ...over,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

function mount(over: Partial<AgentChatConfig> = {}) {
    const root = document.createElement("div");
    document.body.appendChild(root);
    const control = createAgentChat(root, config(over));
    return { root, control };
}

function composer(root: HTMLElement) {
    return {
        input: root.querySelector<HTMLTextAreaElement>(
            'textarea[aria-label="chat message"]',
        )!,
        form: root.querySelector<HTMLFormElement>(".chat__form")!,
    };
}

describe("sending from the composer (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    it("does not send an empty message", async () => {
        const streamTurn = vi.fn(() => Promise.resolve());
        const { root } = mount({ streamTurn });
        await flush();
        const { input, form } = composer(root);
        input.value = "   ";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(streamTurn).not.toHaveBeenCalled();
    });

    it("sends on Enter but not on Shift+Enter", async () => {
        const streamTurn = vi.fn((_m: string, h: StreamHandlers) => {
            h.onDone(reply({ text: "ok" }));
            return Promise.resolve();
        });
        const { root } = mount({ streamTurn });
        await flush();
        const { input } = composer(root);
        input.value = "with shift";
        input.dispatchEvent(
            new KeyboardEvent("keydown", { key: "Enter", shiftKey: true }),
        );
        await flush();
        expect(streamTurn).not.toHaveBeenCalled();
        input.value = "plain enter";
        input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
        await flush();
        expect(streamTurn).toHaveBeenCalledWith(
            "plain enter",
            expect.anything(),
            undefined,
            expect.any(AbortSignal),
        );
    });
});

describe("slash-command palette (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    function type(input: HTMLTextAreaElement, value: string): void {
        input.value = value;
        input.dispatchEvent(new Event("input"));
    }

    it("opens and filters the palette from installed commands", async () => {
        const { root, control } = mount();
        control.setSlashCommands([
            { name: "new", description: "new chat", run: () => undefined },
            { name: "host", description: "host", run: () => undefined },
        ]);
        await flush();
        const { input } = composer(root);
        const palette = root.querySelector<HTMLElement>(".chat__palette")!;
        expect(palette.hidden).toBe(true);
        type(input, "/");
        expect(palette.hidden).toBe(false);
        expect(palette.querySelectorAll(".chat__palette-item").length).toBe(2);
        type(input, "/ho");
        const names = palette.querySelectorAll(".chat__palette-name");
        expect(names.length).toBe(1);
        expect(names[0].textContent).toBe("/host");
    });

    it("Enter runs the highlighted command instead of sending", async () => {
        const run = vi.fn();
        const streamTurn = vi.fn(() => Promise.resolve());
        const { root, control } = mount({ streamTurn });
        control.setSlashCommands([
            { name: "tasks", description: "tasks", run },
        ]);
        await flush();
        const { input } = composer(root);
        type(input, "/tasks");
        input.dispatchEvent(
            new KeyboardEvent("keydown", { key: "Enter", cancelable: true }),
        );
        expect(run).toHaveBeenCalled();
        expect(streamTurn).not.toHaveBeenCalled();
    });
});

describe("image attachments (createAgentChat)", () => {
    beforeEach(() => document.body.replaceChildren());

    it("attaches a picked image, previews it, and sends it in the bubble", async () => {
        const streamTurn = vi.fn(
            (_m: string, h: StreamHandlers, _img?: ImageAttachment) => {
                h.onDone(reply({ text: "ok" }));
                return Promise.resolve();
            },
        );
        const { root } = mount({ enableImage: true, streamTurn });
        await flush();
        const file = new File([new Uint8Array([1, 2, 3])], "x.png", {
            type: "image/png",
        });
        const fileInput = root.querySelector<HTMLInputElement>(".chat__file")!;
        Object.defineProperty(fileInput, "files", {
            value: [file],
            configurable: true,
        });
        fileInput.dispatchEvent(new Event("change"));
        await new Promise((r) => setTimeout(r, 30)); // FileReader is async
        const attach = root.querySelector<HTMLElement>(".chat__attach")!;
        expect(attach.hidden).toBe(false);
        expect(attach.querySelector(".chat__attach-thumb")).not.toBeNull();

        const { input, form } = composer(root);
        input.value = "what is this?";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(
            root.querySelector(".chat__msg--user .chat__attach-img"),
        ).not.toBeNull();
        // The image payload reaches streamTurn and the preview clears.
        const arg = streamTurn.mock.calls[0][2];
        expect(arg?.mime).toBe("image/png");
        expect(attach.hidden).toBe(true);
    });

    it("has no attach button when image is disabled", async () => {
        const { root } = mount();
        await flush();
        expect(root.querySelector(".chat__attach-btn")).toBeNull();
    });
});
