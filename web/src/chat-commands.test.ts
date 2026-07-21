import { describe, expect, it } from "vitest";

import {
    chatMarkdown,
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
    it("renders the conversation as markdown with turn separators", () => {
        const md = chatMarkdown([
            { role: "user", text: "hi" },
            { role: "assistant", text: "hello" },
        ]);
        expect(md).toContain("**user**");
        expect(md).toContain("hi");
        expect(md).toContain("**assistant**");
        expect(md).toContain("---");
    });
});
