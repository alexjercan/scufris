# Retro: Feature: export chats to markdown; for all agents

- TASK: 20260724-012212
- BRANCH: feature/export-chats-markdown
- REVIEW ROUNDS: 1

## What went well

- The feature stayed on the shared chat component, so one export control covered
  both orchestrator and per-agent surfaces.
- The frontend tests were written at the user-facing boundary: markdown content,
  no-op empty export, visible button download, and per-agent title/filename
  wiring.
- Running `npm run format` before `npm run ci` kept the check gate clean and
  matched the existing frontend lesson.

## What went wrong

- `sprout new feature/export-chats-markdown` failed because the branch and
  worktree already existed. Root cause: I assumed the named task had no existing
  sprout state before checking `sprout ls` or inspecting the worktree.

## What to improve next time

- When `sprout new` reports an existing branch, inspect that worktree as task
  state first. If it belongs to the same task, preserve it and continue from
  there instead of asking for cleanup or recreating work.

## Action items

- [x] Added `resume-existing-sprout-state` to `LESSONS.md`.
