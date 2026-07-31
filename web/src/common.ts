// DOM, fetch and auth helpers shared by every page, plus the display labels and
// poll defaults that go with them. The wire shapes live in `stats-types.ts`,
// `host-types.ts` and `agent-types.ts`.

export interface AppConfig {
    poll_seconds: number;
    agent_enabled: boolean;
    // The host overview shells out (systemctl, nixos-rebuild), so it polls on
    // its own much slower clock rather than riding the 2s stats poll.
    host_overview_seconds: number;
}

// A human label for an auth mode wire value (the subscription login differs per
// backend). Null/empty -> a dash so a panel never shows a raw blank.
export function authLabel(mode: string | null): string {
    switch (mode) {
        case "chatgpt":
            return "ChatGPT";
        case "claude_ai":
            return "claude.ai";
        case "api_key":
            return "API key";
        case "local":
            return "Local";
        default:
            return mode || "-";
    }
}

// The write postures an agent can run in (Claude-style), default "manual".
export const PERMISSION_MODES = ["manual", "edit", "auto"];

// Friendly display names for the backend ids, shown on the cards. The list of
// SELECTABLE backends (and each one's default model) now comes from the server
// (GET /api/agents/backends, see BackendOption); these labels are the display
// fallback used when only a backend id is in hand (e.g. an agent card).
// "mock" is included so a dev-flag agent still reads cleanly.
export const BACKEND_LABELS: Record<string, string> = {
    codex: "Codex",
    claude: "Claude",
    opencode: "Opencode",
    mock: "Mock",
};

// The human label for a backend id, falling back to the raw id for anything
// unexpected (a legacy value that slipped through, say).
export function backendLabel(backend: string): string {
    return BACKEND_LABELS[backend] ?? backend;
}

// The reserved orchestrator agent's id: it is undeletable and configured from
// settings, so the UI hides its delete + settings affordances.
export const ORCHESTRATOR_ID = "orchestrator";

export const DEFAULT_POLL_SECONDS = 2;
// The host overview's own, much slower clock (it runs subprocesses server-side).
export const DEFAULT_HOST_OVERVIEW_SECONDS = 30;

// Escape a host-derived string before it goes into innerHTML. Numbers (percent,
// bytes) are formatted via toFixed and cannot inject markup, but strings such as
// a disk mountpoint or hostname could, so every interpolated string is escaped.
export function escapeHtml(value: string): string {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

export function el(
    tag: string,
    className?: string,
    html?: string,
): HTMLElement {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (html !== undefined) node.innerHTML = html;
    return node;
}

// --- authentication ---------------------------------------------------------
//
// One seam for every call to the API, mirroring the backend's single enforcement
// middleware: it attaches the CSRF header the server requires on state-changing
// requests, and turns a 401 into a trip to the login page. Call `apiFetch`, never
// bare `fetch`, so a new call site cannot silently miss either.

export const CSRF_COOKIE = "scufris_csrf";
export const CSRF_HEADER = "X-Scufris-CSRF";

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

// The CSRF cookie is deliberately readable here (the session cookie is not):
// echoing it back in a header is what a cross-site attacker cannot do.
export function csrfToken(): string {
    const match = new RegExp(`(?:^|;\\s*)${CSRF_COOKIE}=([^;]*)`).exec(
        document.cookie,
    );
    return match ? decodeURIComponent(match[1]) : "";
}

function goToLogin(): void {
    // Never bounce a login page to itself: a 401 answered while ON /login/ means
    // "not logged in yet", which is exactly where the operator already is.
    if (window.location.pathname.startsWith("/login")) return;
    const next = window.location.pathname + window.location.search;
    window.location.assign(`/login/?next=${encodeURIComponent(next)}`);
}

export async function apiFetch(
    url: string,
    init: RequestInit = {},
): Promise<Response> {
    const method = (init.method ?? "GET").toUpperCase();
    const headers = new Headers(init.headers);
    if (!SAFE_METHODS.has(method)) {
        const token = csrfToken();
        if (token) headers.set(CSRF_HEADER, token);
    }
    const resp = await fetch(url, {
        ...init,
        method: init.method,
        headers,
        credentials: "same-origin",
    });
    if (resp.status === 401) goToLogin();
    return resp;
}

export async function logout(): Promise<void> {
    await apiFetch("/api/auth/logout", { method: "POST" });
    window.location.assign("/login/");
}

export async function fetchJson<T>(url: string): Promise<T> {
    const resp = await apiFetch(url);
    if (!resp.ok) throw new Error(`${url} -> ${String(resp.status)}`);
    return (await resp.json()) as T;
}

// Send a JSON body with a method (PATCH/POST/DELETE) and parse the JSON reply.
// On a non-2xx it throws an Error carrying the server's `detail` when present, so
// a caller can surface a clear message (e.g. a 422 for a bad MCP id).
export async function sendJson<T>(
    url: string,
    method: string,
    body?: unknown,
): Promise<T> {
    const resp = await apiFetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!resp.ok) {
        let detail = `${url} -> ${String(resp.status)}`;
        try {
            const data = (await resp.json()) as { detail?: string };
            if (data.detail) detail = data.detail;
        } catch {
            // non-JSON error body; keep the status-based message
        }
        throw new Error(detail);
    }
    return (await resp.json()) as T;
}

export function formatBytes(bytes: number): string {
    if (bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB", "PB"];
    const exp = Math.min(
        units.length - 1,
        Math.floor(Math.log(bytes) / Math.log(1024)),
    );
    const value = bytes / Math.pow(1024, exp);
    return `${value.toFixed(exp === 0 ? 0 : 1)} ${units[exp]}`;
}

export async function loadConfig(): Promise<AppConfig> {
    try {
        return await fetchJson<AppConfig>("/api/config");
    } catch {
        return {
            poll_seconds: DEFAULT_POLL_SECONDS,
            agent_enabled: false,
            host_overview_seconds: DEFAULT_HOST_OVERVIEW_SECONDS,
        };
    }
}
