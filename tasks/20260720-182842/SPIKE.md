# Spike: define the "project" concept for the agent (cwd model + sesh)

- DATE: 20260720-182842
- STATUS: RECOMMENDED
- TAGS: spike, agent, ui

## Question

The user asked for "projects" - group the agent's sessions by project, give
each a saved context/cwd, and integrate with `sesh`. Before building anything:
what IS a project in scufris? Is it just a working-directory group (which codex
already records), or a first-class saved object with pinned context/files/env?
How does it relate to `sesh` and to the existing sessions sidebar? A good
answer names the model concretely enough that a planner can expand it without
re-litigating the concept, and it says what NOT to build.

Trigger: task 20260720-122518, which the user deferred with "a spike is a good
idea here". This spike supersedes that task.

## Context

Grounding gathered from the code and live tools (codex 0.142.2):

- **codex tags every session with its cwd.** Each rollout's meta carries
  `cwd`; `scufris/sessions.py:SessionInfo` already surfaces it.
- **scufris already scopes the session list to ONE cwd.**
  `app.py:get_sessions` calls `list_sessions(home, os.getcwd())`, and
  `list_sessions` hard-filters to sessions whose `cwd == os.getcwd()` (plus
  the `_SCUFRIS_ORIGINATORS` scope). The server runs with a single fixed cwd
  (the systemd `WorkingDirectory`), so today every session shares one cwd and
  "which sessions you see" is already a cwd decision - just a frozen one.
- **codex can run a turn in an arbitrary working root.** `codex exec -C/--cd
  <DIR>` sets the agent's working root. scufris does NOT pass it today
  (`agent.py:_exec_args`), so codex inherits the process cwd. So per-project
  cwd is a supported, unused capability.
- **`sesh` is an interactive tmux-sessionizer, not a data source.** It is a
  custom `tmux-sessionizer` script: `-o <dir>` opens a dir in a tmux session,
  `-c` creates a project dir, and with no args it fzf-picks a directory under
  `~/personal`/`~/work`. It needs a tty (fails "inappropriate ioctl for
  device" headless). Its only non-interactive value to scufris is its
  CONVENTION: "a project is a directory under ~/personal or ~/work". A browser
  app cannot attach to a tty tmux session, so scufris will not drive tmux.
- **The sidebar already has a session list.** `agent-view.ts:renderSessions`
  fills `#session-list`; adding a project switcher above it is an additive nav
  change, not a redesign.
- **Two relevant lessons.** `prefer-one-authoritative-render-over-a-parallel-
  client-counter` (do not build a second source of truth for something already
  recorded authoritatively) and `codex-tool-choice-only-steers-via-the-turn-
  prompt` (AGENTS.md via `-C` is ignored for steering; the turn-prompt preamble
  is the proven instruction channel).

## Options considered

- **A - Project = a working directory (ride codex's cwd tagging).** A project
  is a cwd. Stop hard-filtering to `os.getcwd()`; list all scufris sessions and
  group by their recorded `cwd`. Opening a project sets an active cwd; new
  turns pass `-C <cwd>`; the session list filters to that cwd. Pinned context
  rides the existing steering preamble.
  - Pros: rides codex's native model (cwd IS codex's working root and is
    already recorded); membership needs zero new persistence - projects emerge
    from existing session metadata; matches sesh's "dir under ~/personal/
    ~/work" notion; reuses the proven steering channel for context.
  - Cons: an empty project (no sessions yet) has nowhere to live, and a project
    name/pinned-context cannot be derived from a cwd - both need SOME small
    store; dropping the single-cwd filter widens what is listed, so grouping/
    filtering must be deliberate and keep the originator scope; `-C` lets codex
    read arbitrary dirs (sandbox stays read-only, but cwd wants validation).

- **B - Project = a first-class saved object.** New persistence:
  `Project{id, name, cwd, context_md, pinned_files[], env, created}`; sessions
  carry a `project_id`; CRUD endpoints; a switcher.
  - Pros: richest - empty projects, rename, pinned files/env independent of
    cwd, clean model for future features.
  - Cons: most build, and a SECOND source of truth for session->project
    membership when codex already records cwd authoritatively (the parallel-
    counter lesson at the architecture level - the two will drift). Migration,
    CRUD, a store to keep consistent. Heavy for a single-user homelab tool
    before we know the daily use.

- **C - Project = a live sesh directory, no scufris store.** Projects ARE
  exactly the dirs sesh discovers under ~/personal/~/work; scufris scans them,
  opening one sets the cwd and filters sessions. No scufris-side store.
  - Pros: zero new persistence; one source of truth (the filesystem); matches
    the user's mental model ("my projects are my repo dirs").
  - Cons: cannot pin a per-project name/context or hold an empty project
    without some store; excludes sessions whose cwd is outside ~/personal/
    ~/work; sesh is interactive, so scufris scans dirs itself rather than
    driving sesh anyway.

- **D - Do nothing / defer further.** The session list already works.
  - Cost: the user asked for this and the deferral is already in effect; having
    done the spike, we can now give a concrete direction instead of deferring
    blind.

## Recommendation

**Option A, plus a thin slice of B for the two things a cwd cannot carry
(a name, a pinned context, and the existence of an empty project).**

Concretely, a **project** is:

> a working directory, optionally decorated with a small saved record
> `{cwd, name, context_md}`. A session belongs to a project by its
> codex-recorded `cwd` - there is NO separate session<->project link; codex's
> cwd tagging stays the single authority for membership.

Design:

- **Membership** comes from codex: group scufris-originated sessions by their
  recorded `cwd`. No new per-session field.
- **The project list** shown = the union of (distinct cwds across sessions) and
  (saved project records) and (optionally, `~/personal`/`~/work` dirs as
  "create a project" candidates). This yields three natural kinds: a saved
  project with sessions, an auto-project (a cwd with sessions but no record,
  labelled by basename), and an empty saved project (a record with no sessions
  yet).
- **Opening a project** sets an active cwd on the agent; `get_sessions`
  filters to that cwd instead of the hardcoded `os.getcwd()`; new turns pass
  `-C <cwd>` to `codex exec` (and the app-server equivalent).
- **Persistence** is one tiny JSON file under scufris state
  (`projects.json`: `[{cwd, name, context_md}]`) - not sqlite, matching the
  repo's file-based ethos. It holds ONLY what cwd-grouping cannot.
- **Pinned context** rides the existing steering preamble for the active
  project (the proven channel; AGENTS.md-via-`-C` is unreliable per the
  steering lesson). Start there; writing an AGENTS.md into the dir is a later
  option to probe, not the default.
- **sesh integration** is honest and minimal: reuse sesh's directory
  CONVENTION (scan `~/personal`, `~/work`, configurable) to offer a "create
  project from a directory" picker. scufris does not drive tmux.
- **UI**: a project switcher at the top of the sidebar; the session list
  becomes "sessions in this project". Additive, not a redesign.

Why A-hybrid over B: B builds a second authority for membership when codex
already records cwd; the parallel-counter lesson says delete the parallel
store, not sync it. Why over C: C cannot hold a name/context or an empty
project, and drops sessions run outside the sesh dirs. The hybrid keeps codex's
cwd as the membership authority and adds the minimal store only for the
metadata cwd cannot carry.

## Open questions

- **cwd confinement (security).** Should the active project cwd be confined to
  an allowlist root (`~/personal`, `~/work`, configurable) or free-form? `-C`
  lets codex read the dir; sandbox stays `read-only`, but an unvalidated cwd
  could point codex at `~/.ssh`. Lean allowlist-by-default; resolve at build.
- **Pinned context channel.** Steering preamble (invisible, sandbox-safe,
  proven) vs an AGENTS.md written into the dir (native to codex via `-C` but
  unreliable for steering). Lean preamble; probe AGENTS.md-read separately.
- **app-server working root.** The `codex exec` path has `-C`; does the
  `codex app-server` `thread/start` accept a working root? Needs a probe before
  the app-server backend honours projects (the exec path can ship first).

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps:

- tatr 20260720-182938: Projects backend - cwd-scoped sessions, projects.json
  store, per-turn working-root (`-C`), cwd validation.
- tatr 20260720-182953: Projects UI - sidebar project switcher + "sessions in
  this project" + create-from-directory picker (sesh dirs).
- tatr 20260720-182959: Per-project pinned context via the steering preamble.

## Fix record

(Appended by each implementing task as it lands.)
