# Review: Stand up opencode serve + prove one turn (gemma-4-26B-A4B-it)

- TASK: 20260722-135520
- BRANCH: spike/opencode-serve-llamacpp

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Out-of-context reviewer independently re-ran every DoD proof against the live
daemon (:14096, model warm): `check_health.py` -> healthy True / v1.17.9;
`prove_turn.py` -> `hello from gemma` (coherent, manual DoD met);
`ruff check examples/`, `mypy examples/opencode/`, `pytest -q` all pass;
`grep 11433` and `grep permission` hit; `opencode models` lists
`llamacpp/gemma-4-26B-A4B-it`. Honesty check: NOTES.md accurate against
re-verified shapes. In-session supplement re-verified the health-probe claim.

- [x] R1.1 (MINOR) examples/opencode/check_health.py:55-60 - the probe checks
  HTTP 200 but never asserts `body["healthy"] is True`, so a 200 +
  `{"healthy": false}` would still print OK and exit 0; automation keying on the
  exit code would miss it. Suggest gating the return on the healthy flag.
  - Response: fixed - the probe now returns 1 and prints "NOT HEALTHY" when the
    flag is not True (check_health.py). Re-ran against the live daemon: still
    prints healthy True and exits 0.
- [ ] R1.2 (NIT) examples/opencode/prove_turn.py:78 - `tool_parts` uses a loose
  `"tool" in type` substring match rather than the known part types;
  informational-only (gate is on `text`), fine to leave for a spike.
  - Response: acknowledged; left as-is (informational counter only, does not
    affect the pass/fail gate).
- [ ] R1.3 (NIT) NOTES.md permission section is proposed/unverified - the `tools`
  boolean map -> manual|edit|auto was recorded but not exercised (only the
  default all-tools turn ran). Honestly disclosed; the task only required
  recording the mechanism. The next task (135525) must not treat the map as
  verified.
  - Response: acknowledged; NOTES already flags this as "verify live ... deferred
    to 135525". 135525's plan includes a live permission-verification step.

Pending user (manual) checks: none open - the sole manual DoD item (the proven
turn's reply is coherent) was confirmed by the reviewer (`hello from gemma`).
