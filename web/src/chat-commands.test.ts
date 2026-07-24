import { describe, expect, it, vi } from "vitest";

import {
    chatMarkdown,
    downloadChatMarkdown,
    matchSlashCommands,
    type SlashCommand,
} from "./chat-commands";

const CMDS: SlashCommand[] = [
    { name: "new", description: "start a new chat", run: () => undefined },
    { name: "host", description: "summarize this host", run: () => undefined },
    {
        name: "tasks",
        description: "list open tatr tasks",
        run: () => undefined,
    },
];

describe("matchSlashCommands", () => {
    it("filters by a lone /token and ignores real prompts", () => {
        expect(matchSlashCommands("", CMDS).length).toBe(0);
        expect(matchSlashCommands("hello", CMDS).length).toBe(0);
        // A bare slash lists everything.
        expect(matchSlashCommands("/", CMDS).length).toBe(3);
        // A prefix filters.
        expect(matchSlashCommands("/ho", CMDS).map((c) => c.name)).toEqual([
            "host",
        ]);
        // Once there is a space (an argument / real prompt), no commands match.
        expect(matchSlashCommands("/new the thing", CMDS).length).toBe(0);
        expect(matchSlashCommands("/nope", CMDS).length).toBe(0);
    });
});

describe("chatMarkdown", () => {
    it("renders titled markdown with timestamps", () => {
        const md = chatMarkdown(
            [
                { role: "user", text: "hi", ts: Date.UTC(2026, 6, 24, 9, 30) },
                { role: "assistant", text: "hello" },
                { role: "assistant", text: "   " },
            ],
            {
                title: "Builder chat",
                generatedAt: new Date(Date.UTC(2026, 6, 24, 9, 31)),
            },
        );
        expect(md).toContain("# Builder chat");
        expect(md).toContain("Exported: 2026-07-24T09:31:00.000Z");
        expect(md).toContain("## User");
        expect(md).toContain("Sent: 2026-07-24T09:30:00.000Z");
        expect(md).toContain("hi");
        expect(md).toContain("## Assistant");
        expect(md).toContain("---");
        expect(md).not.toContain("   ");
    });
});

describe("downloadChatMarkdown", () => {
    it("does not create a blob for an empty export", () => {
        const createObjectURL = vi.fn();
        vi.stubGlobal("URL", {
            createObjectURL,
            revokeObjectURL: vi.fn(),
        });
        downloadChatMarkdown([]);
        expect(createObjectURL).not.toHaveBeenCalled();
    });
});
