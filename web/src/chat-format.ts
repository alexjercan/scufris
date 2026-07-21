// Small pure formatting helpers shared by the chat component (message timestamps)
// and the orchestrator sidebar (token counts, session "2h ago" labels). Kept
// side-effect-free so jsdom tests drive them directly.

// A compact token count: "1.2k" past a thousand, the plain number below.
export function fmtTokens(n: number): string {
    return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

// Parse an ISO timestamp (from the transcript API) to epoch ms, or undefined.
export function parseIso(iso: string | null): number | undefined {
    if (!iso) return undefined;
    const ms = Date.parse(iso);
    return Number.isNaN(ms) ? undefined : ms;
}

// A short clock label for a message ("14:39" same day, "Jul 19, 14:39" older).
// Empty for a missing/unparseable stamp. Also used as the element's title (full).
export function formatTimestamp(ms: number | undefined): string {
    if (ms === undefined || Number.isNaN(ms)) return "";
    const d = new Date(ms);
    const hh = `${d.getHours()}`.padStart(2, "0");
    const mm = `${d.getMinutes()}`.padStart(2, "0");
    const now = new Date();
    const sameDay =
        d.getFullYear() === now.getFullYear() &&
        d.getMonth() === now.getMonth() &&
        d.getDate() === now.getDate();
    if (sameDay) return `${hh}:${mm}`;
    const month = d.toLocaleString(undefined, { month: "short" });
    return `${month} ${d.getDate()}, ${hh}:${mm}`;
}

// A coarse "2h ago" label for the session list; empty for an unparseable stamp.
export function relativeTime(iso: string | null): string {
    if (!iso) return "";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return "";
    const secs = Math.max(0, (Date.now() - then) / 1000);
    if (secs < 60) return "just now";
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}
