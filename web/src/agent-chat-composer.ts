// The two composer affordances that own state of their own - the slash-command
// palette (`paletteItems`/`paletteIdx`/the installed commands) and image attach
// (`pendingImage`) - plus the textarea autosize they share. Each takes the
// elements it drives and hands back a small handle, so `createAgentChat` wires
// them without holding their state.

import { el, escapeHtml } from "./common";
import { matchSlashCommands, type SlashCommand } from "./chat-commands";
import { readImageFile, type PendingImage } from "./chat-image";

// Grow the composer textarea to fit its content up to a max, then let it scroll.
// jsdom has no layout (scrollHeight is 0), so this is a no-op under tests.
const COMPOSER_MAX_HEIGHT = 200;
export function autosize(input: HTMLTextAreaElement): void {
    input.style.height = "auto";
    const next = Math.min(input.scrollHeight, COMPOSER_MAX_HEIGHT);
    input.style.height = `${next}px`;
    input.style.overflowY =
        input.scrollHeight > COMPOSER_MAX_HEIGHT ? "auto" : "hidden";
}

export interface SlashPalette {
    isOpen: () => boolean;
    close: () => void;
    // Re-match the composer's current text against the installed commands and
    // show/hide the list accordingly.
    refresh: () => void;
    // Handle a composer keydown while the palette is open. Returns true when the
    // event was consumed, so the caller skips its own Enter-to-send handling.
    handleKey: (event: KeyboardEvent) => boolean;
    setCommands: (commands: SlashCommand[]) => void;
}

export function createSlashPalette(
    palette: HTMLElement,
    input: HTMLTextAreaElement,
): SlashPalette {
    let commands: SlashCommand[] = [];
    let paletteItems: SlashCommand[] = [];
    let paletteIdx = 0;

    const isOpen = (): boolean => !palette.hidden;
    const close = (): void => {
        paletteIdx = 0;
        palette.hidden = true;
        palette.replaceChildren();
    };
    const refresh = (): void => {
        paletteItems = matchSlashCommands(input.value, commands);
        if (paletteItems.length === 0) {
            close();
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
        close();
        cmd.run();
    };

    const handleKey = (event: KeyboardEvent): boolean => {
        if (!isOpen()) return false;
        if (event.key === "ArrowDown") {
            event.preventDefault();
            paletteIdx = (paletteIdx + 1) % paletteItems.length;
            refresh();
            return true;
        }
        if (event.key === "ArrowUp") {
            event.preventDefault();
            paletteIdx =
                (paletteIdx - 1 + paletteItems.length) % paletteItems.length;
            refresh();
            return true;
        }
        if (event.key === "Enter" || event.key === "Tab") {
            event.preventDefault();
            runCommand(paletteItems[paletteIdx]);
            return true;
        }
        if (event.key === "Escape") {
            event.preventDefault();
            close();
            return true;
        }
        return false;
    };

    return {
        isOpen,
        close,
        refresh,
        handleKey,
        setCommands: (next) => {
            commands = next;
        },
    };
}

export interface ImageAttach {
    // The image staged for the next turn, or null. Read at submit time and
    // cleared by the caller once captured.
    pending: () => PendingImage | null;
    clear: () => void;
}

// Wire the attach button, the file input and paste-to-attach onto the composer.
// Created ONLY when the config enables images: the paste listener would
// otherwise stage an attachment on a chat that cannot send one.
export function createImageAttach(deps: {
    attach: HTMLElement;
    fileInput: HTMLInputElement;
    attachBtn: HTMLButtonElement;
    input: HTMLTextAreaElement;
}): ImageAttach {
    const { attach, fileInput, attachBtn, input } = deps;
    let pendingImage: PendingImage | null = null;

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

    return {
        pending: () => pendingImage,
        clear: () => {
            pendingImage = null;
            renderAttachPreview();
        },
    };
}
