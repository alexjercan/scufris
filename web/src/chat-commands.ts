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

// Render the tracked conversation as markdown, for `/export` download.
export function chatMarkdown(
    messages: { role: string; text: string }[],
): string {
    return messages
        .map((m) => `**${m.role}**\n\n${m.text}`)
        .join("\n\n---\n\n");
}

// Download the conversation as a markdown file. No-op when there is nothing to
// export or when Blob/URL are absent (e.g. jsdom).
export function downloadChatMarkdown(
    messages: { role: string; text: string }[],
): void {
    const text = chatMarkdown(messages);
    if (!text) return;
    try {
        const blob = new Blob([text], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "scufris-chat.md";
        a.click();
        URL.revokeObjectURL(url);
    } catch {
        // Blob/URL are absent in some environments (e.g. jsdom) - no-op there.
    }
}
