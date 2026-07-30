# B5c: orchestrator multi-session in the agent model (switch/fork/list/delete)

- STATUS: CLOSED
- PRIORITY: 32
- TAGS: agents,backend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Closed: merged into B5bc (20260721-180208)

Reading the code during B5b's start showed B5b and B5c are architecturally
inseparable: `CodexCliAgent.current_session_id()` is the single session state
shared by BOTH the landing chat endpoints AND the session endpoints, so retiring
the Agent protocol (B5b) forces moving the session state (B5c). Splitting would
require a throwaway session-holder shim.

This task is CLOSED as merged; the multi-session work lives in
tasks/20260721-180208 (B5bc: retire the Agent protocol + move orchestrator
sessions to the unified model). No code shipped under this id.
