# Review: F3 /agents/<id> detail page + per-agent settings-edit

- TASK: 20260721-112435
- BRANCH: feature/agent-settings-edit

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

CI ran green in the worktree (prettier + eslint + vitest 150/150 across 9 files
incl. agent-fields.test.ts + webpack build, agent-detail.js emitted). Zero
findings.

Verified by the reviewer:
- The settings form sends exactly `{name, backend, description,
  permission_mode}`, all accepted by `AgentUpdate` (extra=forbid) - no field
  would 422. The create form still sends its historic shape.
- `editingSettings()` focus-guard skips the poll render so an in-progress edit
  is never wiped.
- `id == null` is safe: the "no such agent" fallback renders before the form,
  so the injected `save` (empty-id PATCH) is unreachable dead code.
- `saves edited settings on submit` is a genuine proof (edits diverge from the
  prefills and reach `save`); the blank-name no-op and XSS (`.value` on the
  textarea, `escapeHtml` on the title) assertions are sound for jsdom.
- The `agentFields` extraction preserves the create form's defaults/trimming;
  side-effect-free module, consistent with the pure-render + injected-actions
  seam.

No BLOCKER/MAJOR/MINOR/NIT issues. APPROVE.
