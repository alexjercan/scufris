# Review: B3 agent description + retire the required goal

- TASK: 20260721-112432
- BRANCH: feature/agent-description

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Suites (reviewer ran both): backend ruff+mypy clean, all passed; frontend
`npm run ci` green, 135 passed. Verified in-session (256 backend).

Reviewer verified: `description` wires through every layer (store create/update
stripped, both API models + endpoints, frontend interface + create field +
detail row) and round-trips; `goal` kept optional so old records load and
run_agent still resolves `req.goal or agent.goal` (422s clearly on empty); the
frontend is consistent (create sends description not goal; detail shows it; the
XSS test targets the rendered description); tests meaningful; close-out honest.

- [ ] R1.1 (NIT) agent_store.py module docstring / app.py OPENAPI agents tag
  still describe agents as "launch a goal". Accurate today (run takes a goal),
  but front-lines goal after it was retired from the create UX.
  - Response: Left for B4 - the run/chat surface is reworked when the per-agent
    chat endpoint replaces goal-as-run-input; the text is accurate today and
    would be premature to rephrase around "chat" before B4 exists. Noted for B4.
