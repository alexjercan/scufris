"""The nix store, the packages in it, flake status, and closure-diff parsing.

Every parser is driven by output CAPTURED FROM THE REAL HOST
(``tests/fixtures/host/``): a parser written against imagined output is a parser
written against the wrong thing. Covers store-path parsing, what-provides,
flake inputs and a missing lock, reclaimable space - which only ever enumerates,
never deletes - and the closure-diff installable forms.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_host_inspection import ok

from scufris.host import (
    FakeRunner,
    closure_diff,
    flake_status,
    reclaimable_space,
    render,
    what_provides,
)


def test_flake_status_without_a_lock_points_at_the_setting(tmp_path: Path) -> None:
    report = flake_status(tmp_path)
    assert not report.ok
    assert "SCUFRIS_HOST_CONFIG_REPO" in report.available.reason


def test_no_dead_paths_is_stated_rather_than_shown_as_a_missing_number() -> None:
    report = reclaimable_space(
        FakeRunner(results={"nix-store": ok("0 store paths would be deleted")})
    )
    assert report.ok and report.dead_paths == 0
    assert "nothing would be collected" in render.render_reclaimable(report)


def test_reclaimable_never_presents_the_path_count_as_a_size() -> None:
    """nix reports a COUNT and no byte total; saying otherwise would be a lie."""
    report = reclaimable_space(
        FakeRunner(results={"nix-store": ok("7974 store paths would be deleted")})
    )
    assert report.dead_paths == 7974
    assert report.bytes_reclaimable is None
    text = render.render_reclaimable(report)
    assert "7974 store paths" in text
    assert "not a size" in text


def test_reclaimable_space_has_no_deleting_argv_at_all() -> None:
    """The read-only guarantee, asserted at the argv rather than in prose."""
    runner = FakeRunner(results={"nix-store": ok("1 store paths would be deleted")})
    reclaimable_space(runner)
    assert runner.calls
    for argv in runner.calls:
        joined = " ".join(argv)
        assert "--print-dead" in argv, argv
        for deleting in ("--delete", "-d", "--gc-keep-outputs", "nix-collect-garbage"):
            assert deleting not in argv, f"a deleting argument reached nix: {joined}"


# --- parsing details worth pinning ------------------------------------------


def test_store_path_parse_does_not_swallow_the_binary_path() -> None:
    """The package name is the store DIRECTORY, not the path to the binary.

    Regression pin for a bug caught by running the example against the real host:
    ``what_provides`` matches its regex against every ancestor of the resolved
    binary starting with the binary itself, so a permissive pattern matched the
    full path first and reported systemctl's package as
    "systemd-261/bin/systemctl".

    The load-bearing assertion is the NEGATIVE one: the pattern must REFUSE a
    path that descends below the store directory. Asserting only that the store
    directory parses would pass with the buggy pattern too.
    """
    from scufris.host.packages import _STORE_PATH, _split_name_version

    store_dir = f"/nix/store/{'a' * 32}-systemd-261"

    # The bug: these must NOT match, or the first ancestor tried wins with a
    # package name containing a path.
    assert _STORE_PATH.match(f"{store_dir}/bin/systemctl") is None
    assert _STORE_PATH.match(f"{store_dir}/bin") is None

    match = _STORE_PATH.match(store_dir)
    assert match is not None
    name, version = _split_name_version(match.group("name"))
    assert (name, version) == ("systemd", "261")


def test_what_provides_names_the_package_for_a_real_binary() -> None:
    """End-to-end over the real PATH: the package name carries no path segment.

    The other half of the regression above, driven the way the tool is: whatever
    provides `sh` on this host, the answer must be a package name, not a path.
    """
    report = what_provides("sh")
    assert report.ok
    assert report.path
    if report.package:
        assert "/" not in report.package, f"package name is a path: {report.package}"
        assert not report.store_path.endswith("/sh")


def test_what_provides_reports_a_binary_that_is_not_on_path() -> None:
    report = what_provides("definitely-not-a-real-binary-xyz")
    assert not report.ok
    assert "not on PATH" in report.available.reason


def test_flake_status_reports_only_the_root_flakes_direct_inputs(
    tmp_path: Path,
) -> None:
    """Transitive nodes (flake-parts_2, nixpkgs-lib_3) are not what was pinned."""
    (tmp_path / "flake.lock").write_text(
        json.dumps(
            {
                "version": 7,
                "root": "root",
                "nodes": {
                    "root": {"inputs": {"nixpkgs": "nixpkgs", "hm": "hm"}},
                    "nixpkgs": {
                        "locked": {
                            "lastModified": 1782949081,
                            "rev": "abc123",
                            "owner": "NixOS",
                            "repo": "nixpkgs",
                        }
                    },
                    "hm": {
                        "locked": {
                            "lastModified": 1782949081,
                            "rev": "def456",
                            "owner": "nix-community",
                            "repo": "home-manager",
                        }
                    },
                    "flake-parts_2": {"locked": {"lastModified": 1, "rev": "old"}},
                },
            }
        )
    )
    report = flake_status(tmp_path)
    assert report.ok
    assert [i.name for i in report.inputs] == ["hm", "nixpkgs"]
    assert report.inputs[1].source == "NixOS/nixpkgs"
    assert report.inputs[1].age_days() is not None
    # It must not claim anything is out of date without a network fetch.
    assert "network fetch" in report.available.caveat


def test_closure_diff_refuses_a_flake_installable() -> None:
    """`nix store diff-closures` takes installables, so an unvalidated string
    would let a "read-only" inspection fetch and BUILD a derivation."""
    runner = FakeRunner(results={"nix": ok("")})
    report = closure_diff(runner, "nixpkgs#firefox", 191)
    assert not report.ok
    assert "will not evaluate a flake installable" in report.available.reason
    assert not runner.calls, "nix ran with a flake installable"


def test_reclaimable_space_only_enumerates() -> None:
    """The read-only guarantee comes from the COMMAND, not from nix honouring a
    --dry-run flag: `--delete-older-than` also trims profile generations, so it
    has no place in an inspection tool even in dry-run form."""
    runner = FakeRunner(
        results={"nix-store": ok("/nix/store/aaa-foo\n/nix/store/bbb-bar\n")}
    )
    report = reclaimable_space(runner)
    assert report.ok
    assert report.dead_paths == 2
    assert report.bytes_reclaimable is None
    assert runner.calls == [["nix-store", "--gc", "--print-dead"]]
    text = render.render_reclaimable(report)
    assert "not a size" in text


def test_reclaimable_space_reads_a_summary_count_when_nix_prints_one() -> None:
    report = reclaimable_space(
        FakeRunner(results={"nix-store": ok("7974 store paths would be deleted")})
    )
    assert report.dead_paths == 7974


def test_an_empty_dead_set_reports_zero_not_unknown() -> None:
    """A freshly collected store prints NOTHING, and that means zero.

    Round 2 regression: `nix-store --gc --print-dead` was measured on this host
    to emit bare store paths and no summary line at all, so an empty listing is
    the healthiest possible answer. Treating "no output" as unrecognised would
    report "unknown" for a perfectly good store - the empty-vs-broken confusion
    this package exists to prevent, reintroduced by the fix that chose this
    command.
    """
    report = reclaimable_space(FakeRunner(results={"nix-store": ok("")}))
    assert report.ok
    assert report.dead_paths == 0
    assert not report.available.caveat
    text = render.render_reclaimable(report)
    assert "nothing would be collected" in text
    assert "no dead-path count was reported" not in text


def test_a_listing_of_dead_paths_is_counted() -> None:
    """The real output shape on this host: bare store paths, one per line."""
    listing = "\n".join(f"/nix/store/{'a' * 32}-pkg{i}" for i in range(2048))
    report = reclaimable_space(FakeRunner(results={"nix-store": ok(listing)}))
    assert report.ok
    assert report.dead_paths == 2048
    assert not report.available.caveat


def test_unclassifiable_gc_output_is_unknown_rather_than_miscounted() -> None:
    """Output this build cannot read must not yield a confident number."""
    report = reclaimable_space(
        FakeRunner(results={"nix-store": ok("/nix/store/aaa-x\nsomething else\n")})
    )
    assert report.ok
    assert report.dead_paths is None
    assert "unknown rather than zero" in report.available.caveat


def test_closure_diff_accepts_the_profile_link_form_it_produces() -> None:
    """resolve() must accept a value it itself returns for an int."""
    from scufris.host.packages import SYSTEM_PROFILES_DIR

    link = SYSTEM_PROFILES_DIR / "system-190-link"
    if not link.exists():  # pragma: no cover - depends on the host
        pytest.skip("this host has no system-190-link to resolve")
    runner = FakeRunner(results={"nix": ok("linux: 1 -> 2")})
    report = closure_diff(runner, str(link), 191)
    assert report.ok
    assert runner.calls, "the diff did not run"
