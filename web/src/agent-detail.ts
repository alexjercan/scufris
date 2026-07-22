import "./style.css";
import { initNav } from "./nav";
import { startAgentDetail } from "./agent-detail-view";
import { startAgentChat } from "./agent-chat-view";
import { startAgentSettings } from "./agent-settings-view";

initNav();

// The backend serves this same shell for /agents/<id> AND /agents/<id>/settings.
// The sub-path picks the surface: the settings PAGE, or the chat-first detail.
// Only the chosen main is shown; the other stays hidden (present in the shell).
if (/^\/agents\/[^/]+\/settings\/?$/.test(window.location.pathname)) {
    const detail = document.getElementById("agent-detail-main");
    const settings = document.getElementById("agent-settings");
    if (detail) detail.hidden = true;
    if (settings) settings.hidden = false;
    startAgentSettings();
} else {
    void startAgentDetail();
    startAgentChat();
}
