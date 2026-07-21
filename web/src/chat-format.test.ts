import { describe, expect, it } from "vitest";

import {
    fmtTokens,
    formatTimestamp,
    parseIso,
    relativeTime,
} from "./chat-format";

describe("fmtTokens", () => {
    it("uses a k suffix past a thousand", () => {
        expect(fmtTokens(999)).toBe("999");
        expect(fmtTokens(1500)).toBe("1.5k");
    });
});

describe("parseIso", () => {
    it("parses an ISO stamp and returns undefined for null/garbage", () => {
        expect(parseIso("2020-01-15T09:05:00Z")).toBe(
            Date.parse("2020-01-15T09:05:00Z"),
        );
        expect(parseIso(null)).toBeUndefined();
        expect(parseIso("not a date")).toBeUndefined();
    });
});

describe("formatTimestamp", () => {
    it("shows HH:MM for a same-day time and month+day for older", () => {
        const now = new Date();
        const today = new Date(
            now.getFullYear(),
            now.getMonth(),
            now.getDate(),
            9,
            5,
        );
        expect(formatTimestamp(today.getTime())).toBe("09:05");
        const old = new Date(2020, 0, 15, 9, 5);
        expect(formatTimestamp(old.getTime())).toBe("Jan 15, 09:05");
    });

    it("is empty for a missing stamp", () => {
        expect(formatTimestamp(undefined)).toBe("");
    });
});

describe("relativeTime", () => {
    it("labels recent stamps coarsely and empty for garbage", () => {
        const now = Date.now();
        expect(relativeTime(new Date(now - 30_000).toISOString())).toBe(
            "just now",
        );
        expect(relativeTime(new Date(now - 5 * 60_000).toISOString())).toBe(
            "5m ago",
        );
        expect(relativeTime(new Date(now - 3 * 3600_000).toISOString())).toBe(
            "3h ago",
        );
        expect(relativeTime(null)).toBe("");
        expect(relativeTime("nope")).toBe("");
    });
});
