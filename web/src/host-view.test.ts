import { beforeEach, describe, expect, it, vi } from "vitest";

import type { HostDigest, HostDigestView, ScheduleState } from "./host-types";
import { isTyping, renderHost } from "./host-view";
import type { HostViewData } from "./host-view";
import { _resetHostError } from "./host-actions";
import type { HostActions } from "./host-actions";
import {
    expiryMillis,
    formatAgo,
    formatExpiry,
    formatRequester,
    staleReason,
} from "./host-format";

import {
    NOW,
    actions,
    confirmation,
    proposal,
    record,
    result,
    root,
    view,
} from "./host-fixtures";
import { auditRow, oneWayFixture } from "./host-fixtures";

beforeEach(() => {
    _resetHostError();
    document.body.replaceChildren();
    vi.restoreAllMocks();
});

describe("escaping", () => {
    // Every string on this page is attacker-influenceable: a systemd unit is named
    // by a FILE, and a preview quotes journal lines, store paths and command
    // output. This is the exact class of bug that shipped in the stats cards, so
    // the assertion is structural: the SAME
    // page rendered with a hostile string and with a harmless one must have the
    // same elements, and differ only in text.
    function pageWith(value: string): HostViewData {
        return view({
            queue: [
                record({
                    proposal: proposal({
                        summary: `restart ${value}`,
                        kind: value,
                        steps: [
                            {
                                argv: ["systemctl", "restart", "--", value],
                                label: value,
                            },
                        ],
                        preview: {
                            kind: "state",
                            headline: value,
                            label: value,
                            available: { ok: false, reason: value, caveat: "" },
                            lines: [value],
                        },
                        requester: { actor: value, agent: value, run: value },
                    }),
                    confirmation: confirmation({
                        undo: value,
                        risk_label: value,
                    }),
                }),
            ],
            audit: [
                auditRow({
                    detail: value,
                    outcome: value,
                    requester: { actor: value, agent: "", run: "" },
                    steps: [{ argv: [value], label: "" }],
                }),
            ],
            error: value,
        });
    }

    it("creates no element from host-supplied text", () => {
        const hostile = '<img src=x onerror="alert(1)">';

        const clean = root();
        renderHost(clean, pageWith("a-harmless-string"), actions());
        const cleanElements = clean.getElementsByTagName("*").length;

        const node = root();
        renderHost(node, pageWith(hostile), actions());

        // Nothing was parsed into the DOM. Counted rather than grepped from
        // innerHTML because the risk badge sets `title` as a DOM PROPERTY, which is
        // never parsed as markup - the serialised attribute legitimately contains
        // the raw string, and what must hold is that the value created nothing.
        expect(node.querySelector("img")).toBeNull();
        expect(node.getElementsByTagName("*").length).toBe(cleanElements);
        // And it is still SHOWN - escaped, not dropped - so a hostile unit name is
        // legible to the operator deciding about it.
        expect(node.textContent).toContain(hostile);
    });
});

describe("the edges", () => {
    it("refuses to offer controls for an expired proposal", () => {
        const node = root();
        const stale = record({
            proposal: proposal({ expires_at: NOW / 1000 - 1 }),
        });
        renderHost(node, view({ queue: [stale] }), actions());
        expect(node.querySelector(".host__btn-approve")).toBeNull();
        expect(node.querySelector(".host__btn-deny")).toBeNull();
        expect(node.querySelector(".host__stale")?.textContent).toContain(
            "window has closed",
        );
        expect(node.textContent).toContain("expired");
    });

    it("explains a drifted proposal in those words", () => {
        const node = root();
        const drifted = record({ proposal: proposal({ state: "drifted" }) });
        renderHost(node, view({ queue: [drifted] }), actions());
        expect(node.querySelector(".host__stale")?.textContent).toContain(
            "has changed since the preview",
        );
        expect(node.querySelector(".host__btn-approve")).toBeNull();
    });

    it("shows a failed apply's partial progress rather than 'nothing happened'", () => {
        const node = root();
        const halfway = record({
            decision: "approved",
            decided_by: "operator:abc",
            run_id: "host:act-1",
            result: result({
                ok: false,
                outcome: "failed",
                returncode: 1,
                steps_completed: 1,
                steps_total: 2,
                detail: "the profile was set; the switch failed",
            }),
        });
        renderHost(node, view({ queue: [halfway] }), actions());
        const shown = node.textContent ?? "";
        expect(shown).toContain("FAILED");
        expect(shown).toContain("1/2");
        expect(shown).toContain("the profile was set");
        expect(shown).toContain("stopped part-way through");
    });

    it("surfaces a decision the other surface just made", () => {
        const node = root();
        renderHost(
            node,
            view({
                error: "this action was already approved by operator:telegram:42",
            }),
            actions(),
        );
        expect(node.querySelector(".host__error")?.textContent).toContain(
            "already approved by operator:telegram:42",
        );
    });

    it("says the helper is not configured instead of looking broken", () => {
        const node = root();
        renderHost(node, view({ configured: false }), actions());
        expect(node.querySelector(".host__unconfigured")).not.toBeNull();
        expect(node.textContent).toContain("not configured");
        // No empty queue/audit furniture pretending there is something to see.
        expect(node.querySelector("#host-pending")).toBeNull();
        expect(node.querySelector(".host__audit")).toBeNull();
    });

    it("states an empty queue and an empty record", () => {
        const node = root();
        renderHost(node, view({ queue: [], audit: [] }), actions());
        expect(node.textContent).toContain("nothing is waiting for a decision");
        expect(node.textContent).toContain("the helper's log is empty");
    });
});

describe("decided actions", () => {
    it("offers the undo exactly where the record says one exists", async () => {
        const node = root();
        const acts = actions();
        const applied = record({
            decision: "approved",
            decided_by: "operator:abc",
            run_id: "host:act-1",
            result: result(),
        });
        renderHost(node, view({ queue: [applied] }), acts);

        const revert =
            node.querySelector<HTMLButtonElement>(".host__btn-revert");
        expect(revert?.textContent).toBe("propose the undo");
        revert?.click();
        await Promise.resolve();
        expect(acts.reverted).toEqual(["act-1"]);
    });

    it("offers no undo for an action whose result says there is none", () => {
        const node = root();
        const applied = record({
            decision: "approved",
            result: result({
                reversal: {
                    possible: false,
                    summary: "ONE-WAY.",
                    kind: null,
                    args: {},
                },
            }),
        });
        renderHost(node, view({ queue: [applied] }), actions());
        expect(node.querySelector(".host__btn-revert")).toBeNull();
    });

    it("offers no undo for an action that never applied", () => {
        const node = root();
        const denied = record({
            decision: "denied",
            decided_by: "operator:abc",
            reason: "not during the week",
        });
        renderHost(node, view({ queue: [denied] }), actions());
        expect(node.querySelector(".host__btn-revert")).toBeNull();
        expect(node.textContent).toContain("not during the week");
        expect(node.querySelector(".host__decision--denied")).not.toBeNull();
    });

    it("streams a running apply's output and offers a stop", async () => {
        const node = root();
        const acts = actions();
        vi.spyOn(window, "confirm").mockReturnValue(true);
        const running = record({
            decision: "approved",
            decided_by: "operator:abc",
            run_id: "host:act-1",
        });
        renderHost(
            node,
            view({
                queue: [running],
                output: { "act-1": "building...\nactivating...\n" },
            }),
            acts,
        );
        expect(node.querySelector(".host__output")?.textContent).toBe(
            "building...\nactivating...\n",
        );
        node.querySelector<HTMLButtonElement>(".host__btn-cancel")?.click();
        await Promise.resolve();
        expect(acts.cancelled).toEqual(["act-1"]);
    });
});

describe("the record", () => {
    it("renders the helper's audit tail with its own attribution", () => {
        const node = root();
        renderHost(
            node,
            view({
                audit: [
                    auditRow({
                        event: "requested",
                        requester: { actor: "agent", agent: "host", run: "" },
                    }),
                    auditRow({ event: "denied", outcome: "blocked" }),
                ],
            }),
            actions(),
        );
        const rows = [...node.querySelectorAll(".host__audit-row")];
        expect(rows).toHaveLength(2);
        expect(rows[0].textContent).toContain("requested");
        expect(rows[1].classList.contains("host__audit-row--denied")).toBe(
            true,
        );
        expect(node.textContent).toContain("Written by the root helper itself");
    });
});

describe("helpers", () => {
    it("formats an expiry as a countdown, and says expired past it", () => {
        // Both arguments are milliseconds (see R1.4): the wire's unix-seconds field
        // is converted once, by `expiryMillis`.
        expect(formatExpiry(NOW + 75_000, NOW)).toBe("in 1m 15s");
        expect(formatExpiry(NOW + 9_000, NOW)).toBe("in 9s");
        expect(formatExpiry(NOW, NOW)).toBe("expired");
        expect(formatExpiry(NOW - 30_000, NOW)).toBe("expired");
    });

    it("names who asked from the record's own requester", () => {
        expect(formatRequester(record())).toBe(
            "agent - agent host - run run-9",
        );
        expect(
            formatRequester(
                record({
                    proposal: proposal({
                        requester: {
                            actor: "operator:abc",
                            agent: "",
                            run: "",
                        },
                    }),
                }),
            ),
        ).toBe("operator:abc");
    });

    it("reports why a proposal is no longer decidable", () => {
        expect(staleReason(record(), NOW)).toBe("");
        expect(
            staleReason(
                record({ proposal: proposal({ state: "applied" }) }),
                NOW,
            ),
        ).toContain("already moved this proposal to applied");
    });
});

describe("startHost", () => {
    // The orchestration layer is thin, but one thing in it is not obvious and was
    // wrong first: WHICH signal means "this box has no privileged helper". Measured
    // against a running server with no helper, `/api/host/actions` answers `200 []`
    // and only `/api/host/audit` answers 503 - so reading it off the queue showed an
    // empty queue and an empty log, which reads as "nothing has happened" rather
    // than "nothing can".
    function stubApi(routes: Record<string, [number, unknown]>): void {
        vi.stubGlobal(
            "fetch",
            vi.fn((input: string) => {
                const url = new URL(input, "http://localhost").pathname;
                const [status, body] = routes[url] ?? [404, { detail: "no" }];
                return Promise.resolve(
                    new Response(JSON.stringify(body), {
                        status,
                        headers: { "Content-Type": "application/json" },
                    }),
                );
            }),
        );
    }

    it("says not configured when the helper is absent, empty queue and all", async () => {
        const node = document.createElement("main");
        node.id = "host";
        document.body.replaceChildren(node);
        stubApi({
            "/api/host/actions": [200, []],
            "/api/host/audit": [
                503,
                {
                    detail: "the privileged host helper is not configured: set X",
                },
            ],
        });

        const { startHost } = await import("./host-view");
        startHost();
        await vi.waitFor(() => {
            expect(node.querySelector(".host__unconfigured")).not.toBeNull();
        });
        // The server's own sentence is shown, since it names what to set.
        expect(node.textContent).toContain("is not configured: set X");
        // And no empty-queue furniture pretending there is something to look at.
        expect(node.textContent).not.toContain(
            "nothing is waiting for a decision",
        );
        vi.unstubAllGlobals();
    });

    it("renders the queue when the helper IS configured", async () => {
        const node = document.createElement("main");
        node.id = "host";
        document.body.replaceChildren(node);
        stubApi({
            "/api/host/actions": [200, [record()]],
            "/api/host/audit": [200, [auditRow()]],
        });

        const { startHost } = await import("./host-view");
        startHost();
        await vi.waitFor(() => {
            expect(
                node.querySelector('[data-action-id="act-1"]'),
            ).not.toBeNull();
        });
        expect(node.querySelector(".host__unconfigured")).toBeNull();
        expect(node.querySelector(".host__audit-row")).not.toBeNull();
        vi.unstubAllGlobals();
    });
});

describe("the review-round fixes", () => {
    it("never re-renders over what the operator is typing (R1.1)", () => {
        // `renderHost` rebuilds the page, so a poll landing mid-word would replace
        // the input under the operator's hands - measured: a partially typed
        // acknowledgement token and the focus both vanished. `isTyping` is what the
        // poll consults; a decision-triggered render deliberately ignores it.
        const node = root();
        renderHost(node, view({ queue: [oneWayFixture()] }), actions());
        expect(isTyping(node)).toBe(false);

        const token = node.querySelector<HTMLInputElement>(".host__ack");
        token?.focus();
        expect(isTyping(node)).toBe(true);

        // Focus outside the page (or on a button) is not typing: a poll should still
        // refresh the queue then.
        token?.blur();
        expect(isTyping(node)).toBe(false);
        node.querySelector<HTMLButtonElement>(
            ".host__btn-approve-one-way",
        )?.focus();
        expect(isTyping(node)).toBe(false);
    });

    it("clears the error banner once something succeeds (R1.2)", async () => {
        // A refused approve followed by a successful deny went on reporting the 409
        // forever, on the one page whose job is to say truthfully what happened.
        const node = root();
        const refusing: HostActions = {
            runChecks: () => Promise.resolve(),
            approve: () => Promise.reject(new Error("409 already decided")),
            deny: () => Promise.resolve(),
            cancel: () => Promise.resolve(),
            revert: () => Promise.resolve(),
            reload: () => Promise.resolve(),
        };
        renderHost(node, view(), refusing);
        node.querySelector<HTMLButtonElement>(".host__btn-approve")?.click();
        await new Promise((resolve) => setTimeout(resolve, 5));
        renderHost(node, view(), refusing);
        expect(node.querySelector(".host__error")?.textContent).toContain(
            "409 already decided",
        );

        node.querySelector<HTMLButtonElement>(".host__btn-deny")?.click();
        await new Promise((resolve) => setTimeout(resolve, 5));
        renderHost(node, view(), refusing);
        expect(node.querySelector(".host__error")).toBeNull();
    });

    it("says the log could not be READ rather than that it is empty (R1.3)", () => {
        const node = root();
        renderHost(
            node,
            view({
                audit: [],
                auditFailed: "the helper's log could not be read (HTTP 500)",
            }),
            actions(),
        );
        expect(node.querySelector(".host__audit")).toBeNull();
        expect(node.textContent).toContain("could not be read (HTTP 500)");
        expect(node.textContent).not.toContain("log is empty");
    });

    it("converts the expiry once, at the boundary (R1.4)", () => {
        // Both arguments are milliseconds now; `expiryMillis` is the one place the
        // wire's unix-seconds field is converted.
        expect(expiryMillis(record())).toBe(NOW + 600_000);
        expect(formatExpiry(NOW + 75_000, NOW)).toBe("in 1m 15s");
    });
});

// --- the scheduled checks ---------------------------------------------------

function digest(over: Partial<HostDigest> = {}): HostDigest {
    return {
        at: NOW / 1000 - 300,
        schedule: "daily",
        verdict: "ok",
        text: "08:00 - all clear on 5 check(s)",
        delivered: true,
        delivery_error: "",
        states: { disk: "ok" },
        ...over,
    };
}

function schedule(over: Partial<ScheduleState> = {}): ScheduleState {
    return {
        name: "watch",
        next_due: NOW / 1000 + 600,
        last_run: NOW / 1000 - 600,
        last_result: "ran: nothing to report",
        missed: 0,
        runs: 12,
        ...over,
    };
}

function checks(over: Partial<HostDigestView> = {}): HostDigestView {
    return {
        schedules: [schedule(), schedule({ name: "daily", runs: 3 })],
        digests: [digest()],
        muted_until: 0,
        enabled: true,
        ...over,
    };
}

describe("renderHostDigests", () => {
    it("answers 'did it fire' when the answer was silence", () => {
        // `watch` says nothing when there is nothing to say, so the last RESULT of
        // each schedule has to be visible somewhere - here.
        const node = root();
        renderHost(node, view({ checks: checks() }), actions());
        const section = node.querySelector("#host-checks");
        expect(section).not.toBeNull();
        const shown = section?.textContent ?? "";
        expect(shown).toContain("watch");
        expect(shown).toContain("ran: nothing to report");
        expect(shown).toContain("10m ago");
        // And the digest itself is readable without asking Telegram.
        expect(node.querySelector(".host__digest-body")?.textContent).toBe(
            "08:00 - all clear on 5 check(s)",
        );
    });

    it("says out loud when a digest never reached the operator", () => {
        const node = root();
        renderHost(
            node,
            view({
                checks: checks({
                    digests: [
                        digest({
                            delivered: false,
                            delivery_error: "telegram is down",
                        }),
                    ],
                }),
            }),
            actions(),
        );
        expect(node.textContent).toContain("not delivered: telegram is down");
    });

    it("distinguishes a mute from a failure", () => {
        const node = root();
        renderHost(
            node,
            view({
                checks: checks({
                    muted_until: NOW / 1000 + 3600,
                    digests: [
                        digest({ delivered: false, delivery_error: "muted" }),
                    ],
                }),
            }),
            actions(),
        );
        expect(node.textContent).toContain("not sent: delivery is muted");
        expect(node.textContent).toContain("still run and are still recorded");
    });

    it("says when the checks are switched off, rather than looking idle", () => {
        const node = root();
        renderHost(
            node,
            view({ checks: checks({ enabled: false, digests: [] }) }),
            actions(),
        );
        expect(
            node.querySelector(".host__unconfigured")?.textContent,
        ).toContain("switched off");
        expect(node.textContent).toContain("no digest yet");
    });

    it("runs a schedule on demand", async () => {
        const node = root();
        const acts = actions();
        renderHost(node, view({ checks: checks() }), acts);
        const buttons = [
            ...node.querySelectorAll<HTMLButtonElement>(".host__btn-run"),
        ];
        expect(buttons.map((b) => b.textContent)).toEqual([
            "run watch now",
            "run daily now",
        ]);
        buttons[1].click();
        await Promise.resolve();
        expect(acts.ran).toEqual(["daily"]);
    });

    it("shows a placeholder until the first read lands", () => {
        const node = root();
        renderHost(node, view({ checks: undefined }), actions());
        expect(node.querySelector("#host-checks")?.textContent).toContain(
            "reading the scheduled checks",
        );
    });

    it("formats recency in units a person reads", () => {
        expect(formatAgo(NOW - 30_000, NOW)).toBe("30s");
        expect(formatAgo(NOW - 600_000, NOW)).toBe("10m");
        expect(formatAgo(NOW - 7_200_000, NOW)).toBe("2h");
        expect(formatAgo(NOW - 3 * 86_400_000, NOW)).toBe("3d");
        expect(formatAgo(NOW + 5_000, NOW)).toBe("0s");
    });
});
