# B6: sesh.py directory discovery + Projects discovery/create (no tmux)

- STATUS: OPEN
- PRIORITY: 25
- TAGS: agents,backend,projects


## Goal

`scufris/sesh.py`: `discover()` scans configurable base dirs one level deep ->
candidate {path, name, language?} (language inferred from marker files:
pyproject.toml->python, package.json->node, Cargo.toml->rust, ...); `create(name,
base)` -> mkdir (NO tmux) and returns the path. Projects page surfaces DISCOVERED
dirs UNION registered projects (marking which are registered); create registers +
mkdirs. Base dirs default to the sesh set (~/personal, ~/personal/_tests, ~/work,
~/third-party), configurable in settings.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 4; recommendation B6/F5). NO tmux - directory only.
- Independent; can slot anywhere.
