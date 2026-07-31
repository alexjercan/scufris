// What the page can DO, and the one piece of state that doing it writes.
//
// Every mutating control funnels through `dispatch`, which records the last
// failure and reloads; `renderHost` reads it back through `hostError()`. The write
// and its only read live together here rather than a module apart.

// `startHost` wires these to the API; the jsdom tests pass fakes. Every mutating
// one resolves after the server applied it, and the caller reloads.
export interface HostActions {
    // Run one schedule's checks now. The server answers 202 and the run happens in
    // the background (a pass walks the nix store), so this resolves before the
    // digest exists - the poll is what shows it.
    runChecks(schedule: string): Promise<void>;
    approve(id: string, acknowledge: string): Promise<void>;
    deny(id: string, reason: string): Promise<void>;
    cancel(id: string): Promise<void>;
    revert(id: string): Promise<void>;
    reload(): Promise<void>;
}

export async function dispatch(
    actions: HostActions,
    run: () => Promise<void>,
): Promise<void> {
    try {
        await run();
        // Cleared on success, or the banner outlives the failure it describes: a
        // refused approve followed by a successful deny would go on reporting "409
        // already decided" forever, on the one page whose job is to say truthfully
        // what happened to this machine.
        lastError = "";
    } catch (err: unknown) {
        // Surfaced through the reload below (the page shows `error`), so a failed
        // decision is never silent - and never a dead end either: the reload
        // re-reads the real state, which for a 409 is the state the other surface
        // just created.
        lastError = err instanceof Error ? err.message : String(err);
    }
    await actions.reload();
}

// The last failed action's message. Module state so the render can show it after
// a reload; `_resetHostError` is the test-reset hook that keeps it from leaking
// between jsdom cases.
let lastError = "";

export function _resetHostError(): void {
    lastError = "";
}

export function hostError(): string {
    return lastError;
}
