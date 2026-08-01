import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderHost } from "./host-view";
import { _resetHostError } from "./host-actions";
import {
    actions,
    confirmation,
    proposal,
    record,
    root,
    view,
} from "./host-fixtures";
import { oneWayFixture } from "./host-fixtures";

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
