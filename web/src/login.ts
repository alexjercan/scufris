import "./style.css";
import { initLogin } from "./login-view";

// The login POST goes out with bare `fetch`, not `apiFetch`: there is no session
// and no CSRF cookie yet (the server issues both in this response), and
// apiFetch's 401 handler would bounce the page to itself.
initLogin({
    post: (password: string) =>
        fetch("/api/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "same-origin",
            body: JSON.stringify({ password }),
        }),
    navigate: (url: string) => {
        window.location.assign(url);
    },
});
