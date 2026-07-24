// The composer slash-command palette, shared by both chat entries. codex itself
// has no slash commands, so this is a client-side palette: a command either runs
// an action, navigates, or fills the composer with a prompt the user can send.
// Pure/side-effect-free (matching + markdown), so jsdom tests drive it directly.

// One command. `run` performs the action (fill the composer, navigate, export...).
export interface SlashCommand {
    name: string;
    description: string;
    run: () => void;
}

export interface ChatExportMessage {
    role: string;
    text: string;
    ts?: number;
}

export interface ChatExportOptions {
    title?: string;
    filename?: string;
    generatedAt?: Date;
}

// Commands matching what the user has typed: a lone `/token` at the very start of
// the composer (no space/newline yet - once they type an arg, it is a real prompt).
export function matchSlashCommands(
    value: string,
    commands: SlashCommand[],
): SlashCommand[] {
    if (!value.startsWith("/")) return [];
    const query = value.slice(1);
    if (/\s/.test(query)) return [];
    return commands.filter((c) => c.name.startsWith(query));
}

function roleHeading(role: string): string {
    return role ? role.charAt(0).toUpperCase() + role.slice(1) : "Message";
}

function defaultFilename(title: string): string {
    const slug = title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
    return `${slug || "scufris-chat"}.md`;
}

// Render the tracked conversation as markdown, for `/export` download.
export function chatMarkdown(
    messages: ChatExportMessage[],
    options: ChatExportOptions = {},
): string {
    const kept = messages.filter((m) => m.text.trim());
    if (kept.length === 0) return "";
    const title = options.title ?? "Scufris chat";
    const generatedAt = options.generatedAt ?? new Date();
    const parts = [`# ${title}`, `Exported: ${generatedAt.toISOString()}`];
    for (const message of kept) {
        const block = [`## ${roleHeading(message.role)}`];
        if (message.ts !== undefined) {
            block.push(`Sent: ${new Date(message.ts).toISOString()}`);
        }
        block.push(message.text.trim());
        parts.push(block.join("\n\n"));
    }
    return parts.join("\n\n---\n\n");
}

// Download the conversation as a markdown file. No-op when there is nothing to
// export or when Blob/URL are absent (e.g. jsdom).
export function downloadChatMarkdown(
    messages: ChatExportMessage[],
    options: ChatExportOptions = {},
): void {
    const title = options.title ?? "Scufris chat";
    const filename = options.filename ?? defaultFilename(title);
    const text = chatMarkdown(messages, { ...options, title });
    if (!text) return;
    try {
        const blob = new Blob([text], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename.endsWith(".md") ? filename : `${filename}.md`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch {
        // Blob/URL are absent in some environments (e.g. jsdom) - no-op there.
    }
}
