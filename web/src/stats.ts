import "./style.css";
import { initNav } from "./nav";
import { startProcesses } from "./processes-view";
import { startStats } from "./stats-view";

initNav();
void startStats();
void startProcesses();
