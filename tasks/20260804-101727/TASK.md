# Fix collection when pytest is given tests/ as a path

- PRIORITY: 20
- TAGS: tests,dx
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

`uv run pytest -q tests/` - a path argument instead of the configured
`testpaths` - fails collection with `ModuleNotFoundError: No module named
'test_host_actions'` raised from `tests/conftest.py`. The configured run
(`uv run pytest`, which uses `testpaths = ["tests", "packages/*/tests"]`) is
green, so CI and `nix flake check` never see it.

It bites an agent or a contributor who reaches for the obvious narrowing
command and gets a collection error that looks like their change. Found while
reviewing 20260804-053002; it predates that branch and is not caused by it.

The conftest imports a sibling test module by bare name, which only resolves
under the rootdir-relative `sys.path` entry the configured invocation produces.
Either the import becomes robust to how pytest was invoked, or the command is
documented as unsupported - the first is worth trying before the second.
