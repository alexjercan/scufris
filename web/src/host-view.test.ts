import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
    HostActionRecord,
    HostActionResult,
    HostAuditRecord,
    HostConfirmation,
    HostDigest,
    HostDigestView,
    HostProposal,
    ScheduleState,
} from "./host-types";
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

// A fixed "now" so expiry rendering is deterministic; the proposal below expires
// ten minutes after it.
const NOW = 1_800_000_000_000;

function confirmation(over: Partial<HostConfirmation> = {}): HostConfirmation {
    return {
        style: "ordinary",
        risk: "r1",
        risk_label: "service control - changes a unit's runtime state",
        undo: "stop nginx.service, restoring it to inactive",
        no_undo: false,
        acknowledge: "",
        ...over,
    };
}

function proposal(over: Partial<HostProposal> = {}): HostProposal {
    return {
        id: "act-1",
        kind: "unit_restart",
        risk: "r1",
        args: { unit: "nginx" },
        steps: [
            {
                argv: ["systemctl", "restart", "--", "nginx.service"],
                label: "",
            },
        ],
        summary: "restart the nginx.service unit",
        preview: {
            kind: "state",
            headline: "restart nginx.service",
            label: "systemd cannot simulate this; these lines are current state",
            available: { ok: true, reason: "", caveat: "" },
            lines: ["unit: nginx.service (Nginx)", "now:  active (running)"],
        },
        reversal: {
            possible: true,
            summary: "stop nginx.service, restoring it to inactive",
            kind: "unit_stop",
            args: { unit: "nginx" },
        },
        requester: { actor: "agent", agent: "host", run: "run-9" },
        created_at: NOW / 1000,
        expires_at: NOW / 1000 + 600,
        state: "pending",
        ...over,
    };
}

function record(over: Partial<HostActionRecord> = {}): HostActionRecord {
    return {
        proposal: proposal(),
        decision: "pending",
        decided_by: "",
        decided_at: null,
        reason: "",
        run_id: null,
        result: null,
        error: "",
        confirmation: confirmation(),
        ...over,
    };
}

function result(over: Partial<HostActionResult> = {}): HostActionResult {
    return {
        ok: true,
        outcome: "ok",
        returncode: 0,
        duration_seconds: 0.4,
        steps_completed: 1,
        steps_total: 1,
        reversal: {
            possible: true,
            summary: "stop nginx.service, restoring it to inactive",
            kind: "unit_stop",
            args: { unit: "nginx" },
        },
        detail: "",
        ...over,
    };
}

function auditRow(over: Partial<HostAuditRecord> = {}): HostAuditRecord {
    return {
        ts: NOW / 1000,
        at: "2026-07-30T08:00:00+00:00",
        event: "applied",
        action_id: "act-1",
        kind: "unit_restart",
        risk: "r1",
        steps: [
            {
                argv: ["systemctl", "restart", "--", "nginx.service"],
                label: "",
            },
        ],
        requester: { actor: "operator:abc", agent: "host", run: "" },
        outcome: "ok",
        returncode: 0,
        duration_seconds: 0.4,
        steps_completed: 1,
        reversal: "stop nginx.service",
        restore_point: "active",
        detail: "",
        ...over,
    };
}

function view(over: Partial<HostViewData> = {}): HostViewData {
    return {
        queue: [record()],
        audit: [auditRow()],
        configured: true,
        now: NOW,
        ...over,
    };
}

// A recording stand-in for the page's actions, so a test can assert WHAT the
// control sent rather than that something happened.
interface RecordedActions extends HostActions {
    ran: string[];
    approved: [string, string][];
    denied: [string, string][];
    cancelled: string[];
    reverted: string[];
    reloads: () => number;
}

function actions(): RecordedActions {
    const ran: string[] = [];
    const approved: [string, string][] = [];
    const denied: [string, string][] = [];
    const cancelled: string[] = [];
    const reverted: string[] = [];
    let reloads = 0;
    return {
        ran,
        approved,
        denied,
        cancelled,
        reverted,
        reloads: () => reloads,
        runChecks(schedule: string) {
            ran.push(schedule);
            return Promise.resolve();
        },
        approve(id: string, acknowledge: string) {
            approved.push([id, acknowledge]);
            return Promise.resolve();
        },
        deny(id: string, reason: string) {
            denied.push([id, reason]);
            return Promise.resolve();
        },
        cancel(id: string) {
            cancelled.push(id);
            return Promise.resolve();
        },
        revert(id: string) {
            reverted.push(id);
            return Promise.resolve();
        },
        reload() {
            reloads += 1;
            return Promise.resolve();
        },
    };
}

function root(): HTMLElement {
    const node = document.createElement("main");
    document.body.replaceChildren(node);
    return node;
}

beforeEach(() => {
    _resetHostError();
    document.body.replaceChildren();
    vi.restoreAllMocks();
});

describe("the pending queue", () => {
    it("renders what the operator is deciding against", () => {
        const node = root();
        renderHost(node, view(), actions());

        const card = node.querySelector<HTMLElement>(
            '[data-action-id="act-1"]',
        );
        expect(card).not.toBeNull();
        const shown = card?.textContent ?? "";
        // The summary, the risk class (letter AND word, so R1 and R3 do not read
        // alike), the kind, who asked - with its run - and the expiry.
        expect(shown).toContain("restart the nginx.service unit");
        expect(shown).toContain("R1 service");
        expect(shown).toContain("unit_restart");
        expect(shown).toContain("agent host");
        expect(shown).toContain("run run-9");
        expect(shown).toContain("in 10m 0s");
        // The command, verbatim.
        expect(card?.querySelector(".host__argv")?.textContent).toBe(
            "systemctl restart -- nginx.service",
        );
        // The preview: its label and every line, unreflowed.
        expect(shown).toContain("systemd cannot simulate this");
        expect(card?.querySelector(".host__preview-body")?.textContent).toBe(
            "unit: nginx.service (Nginx)\nnow:  active (running)",
        );
        // And the undo sentence.
        expect(shown).toContain("UNDO:");
        expect(shown).toContain("restoring it to inactive");
    });

    it("lists EVERY command of a multi-step action, in order", () => {
        // A half-applied activation means this boot and the next boot disagree, so
        // the operator must see the sequence rather than a summary of it.
        const node = root();
        const multi = record({
            proposal: proposal({
                kind: "activate",
                risk: "r3",
                summary: "activate the built configuration",
                steps: [
                    {
                        argv: [
                            "nix-env",
                            "-p",
                            "/nix/var/nix/profiles/system",
                            "--set",
                            "/nix/store/aaa",
                        ],
                        label: "point the system profile at the new toplevel",
                    },
                    {
                        argv: [
                            "/nix/store/aaa/bin/switch-to-configuration",
                            "switch",
                        ],
                        label: "switch to it",
                    },
                ],
            }),
            confirmation: confirmation({
                risk: "r3",
                risk_label: "system configuration - switches the whole system",
                undo: "activate generation 190 again",
            }),
        });
        renderHost(node, view({ queue: [multi] }), actions());

        const argvs = [...node.querySelectorAll(".host__argv")].map(
            (n) => n.textContent,
        );
        expect(argvs).toEqual([
            "nix-env -p /nix/var/nix/profiles/system --set /nix/store/aaa",
            "/nix/store/aaa/bin/switch-to-configuration switch",
        ]);
        expect(node.textContent).toContain("2 commands, in order");
        expect(node.textContent).toContain("point the system profile");
        // R3 is visually its own class, not R1's.
        expect(node.querySelector(".host__risk--r3")).not.toBeNull();
        expect(node.querySelector(".host__risk--r1")).toBeNull();
    });

    it("says when a preview could not be produced", () => {
        const node = root();
        const blind = record({
            proposal: proposal({
                preview: {
                    kind: "none",
                    headline: "restart nginx.service",
                    label: "the unit's current state could not be read",
                    available: {
                        ok: false,
                        reason: "systemctl exited 1",
                        caveat: "",
                    },
                    lines: [],
                },
            }),
        });
        renderHost(node, view({ queue: [blind] }), actions());
        expect(node.querySelector(".host__unavailable")?.textContent).toContain(
            "systemctl exited 1",
        );
        // An empty preview is stated, never rendered as a blank that reads as fine.
        expect(node.textContent).toContain("no preview lines");
    });

    it("approves a reversible action with one control and no token", async () => {
        const node = root();
        const acts = actions();
        renderHost(node, view(), acts);

        const approve =
            node.querySelector<HTMLButtonElement>(".host__btn-approve");
        expect(approve).not.toBeNull();
        expect(approve?.disabled).toBe(false);
        // No acknowledgement field on a reversible action: the undo sentence above
        // the button is what makes an ordinary confirmation enough.
        expect(node.querySelector(".host__ack")).toBeNull();
        approve?.click();
        await Promise.resolve();
        expect(acts.approved).toEqual([["act-1", ""]]);
    });

    it("denies with a reason that reaches the agent", async () => {
        const node = root();
        const acts = actions();
        renderHost(node, view(), acts);

        const reason = node.querySelector<HTMLInputElement>(".host__reason");
        expect(reason).not.toBeNull();
        if (reason) reason.value = "  nginx is serving the demo  ";
        node.querySelector<HTMLButtonElement>(".host__btn-deny")?.click();
        await Promise.resolve();
        expect(acts.denied).toEqual([["act-1", "nginx is serving the demo"]]);
    });
});

// A one-way (irreversible) proposal: the case whose approve control is gated.
function oneWayFixture(): HostActionRecord {
    return record({
        proposal: proposal({
            id: "gc-1",
            kind: "gc_store",
            risk: "r2",
            summary: "delete every unreachable store path",
            steps: [{ argv: ["nix-store", "--gc"], label: "" }],
            reversal: {
                possible: false,
                summary: "ONE-WAY. Deleted store paths cannot be restored.",
                kind: null,
                args: {},
            },
        }),
        confirmation: confirmation({
            style: "one_way",
            risk: "r2",
            risk_label: "disposable cleanup - ONE-WAY",
            undo: "ONE-WAY. Deleted store paths cannot be restored.",
            no_undo: true,
            acknowledge: "gc_store",
        }),
    });
}

describe("the one-way gate", () => {
    const oneWay = oneWayFixture;

    it("offers NO ordinary approve control at all", () => {
        // The point of the gate: the plain path cannot approve a one-way action
        // even by mistake, because the control that would do it does not exist.
        const node = root();
        renderHost(node, view({ queue: [oneWay()] }), actions());
        expect(node.querySelector(".host__btn-approve")).toBeNull();
        expect(node.querySelector(".host__btn-approve-one-way")).not.toBeNull();
        expect(node.querySelector(".host__no-undo")?.textContent).toContain(
            "NO UNDO:",
        );
        expect(node.querySelector(".host__risk--r2")).not.toBeNull();
    });

    it("stays disabled until the acknowledgement matches, then sends it", async () => {
        const node = root();
        const acts = actions();
        renderHost(node, view({ queue: [oneWay()] }), acts);

        const approve = node.querySelector<HTMLButtonElement>(
            ".host__btn-approve-one-way",
        );
        const token = node.querySelector<HTMLInputElement>(".host__ack");
        expect(approve?.disabled).toBe(true);

        // A wrong token does not enable it, and clicking anyway sends nothing.
        if (token) {
            token.value = "yes";
            token.dispatchEvent(new Event("input"));
        }
        expect(approve?.disabled).toBe(true);
        approve?.click();
        await Promise.resolve();
        expect(acts.approved).toEqual([]);

        // The real token enables it, and the click carries that token.
        if (token) {
            token.value = "gc_store";
            token.dispatchEvent(new Event("input"));
        }
        expect(approve?.disabled).toBe(false);
        approve?.click();
        await Promise.resolve();
        expect(acts.approved).toEqual([["gc-1", "gc_store"]]);
    });
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
