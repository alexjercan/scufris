import "./style.css";
import { initNav } from "./nav";
import { startSettings } from "./agent-settings-view";

initNav();
// The /settings page IS the orchestrator's settings, rendered by the SAME unified
// component as /agents/<id>/settings (agent-settings-view).
startSettings();
