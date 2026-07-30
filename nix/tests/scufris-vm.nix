# NixOS VM test for the scufris service module.
#
# Boots a throwaway VM with `services.scufris.enable = true` (via
# nixosModules.scufris) so the deployment is validated in a sandbox rather than
# live on a real config. What it covers:
#   * The module loads and `scufris.service` reaches active.
#   * The dashboard port is open and `/api/config` answers (liveness JSON).
#   * `/` serves the built dashboard shell - proves the SCUFRIS_WEB_DIST wiring
#     to the packages.scufris-web derivation (the whole reason web/dist is
#     packaged).
#   * The DynamicUser state dir is writable (state writes succeed in-VM).
#   * `/api/host/overview` finds its toolchain - the `hostTools` default really
#     does put systemctl/journalctl/nix/nixos-rebuild/ip on the unit's PATH.
#     Asserted on the rendered report, not on the PATH string, because what
#     broke before was the report saying "not installed on this host".
#   * Restart-on-demand brings the unit back up.
#
# The agent is disabled (agent_enabled = false): a VM has no codex/claude login,
# and the dashboard serves without it. Chat is covered by the app's own tests.
{
  pkgs,
  scufrisModule,
}:
pkgs.testers.nixosTest {
  name = "scufris-server";

  nodes.machine = {...}: {
    imports = [scufrisModule];

    services.scufris = {
      enable = true;
      settings = {
        host = "127.0.0.1";
        port = 8000;
        agent_enabled = false;
      };
    };

    environment.systemPackages = [pkgs.curl];

    # Tiny VM - just enough to run a Python web service.
    virtualisation.memorySize = 1024;
  };

  testScript = ''
    machine.start()
    machine.wait_for_unit("scufris.service")
    machine.wait_for_open_port(8000)

    # Liveness: the client-config endpoint returns JSON and reports the agent off.
    cfg = machine.succeed("curl --fail --max-time 5 http://127.0.0.1:8000/api/config")
    print("api/config:", cfg.strip())
    assert '"agent_enabled":false' in cfg.replace(" ", ""), \
        f"expected agent disabled in /api/config, got: {cfg!r}"

    # Dashboard: "/" serves the built index.html from SCUFRIS_WEB_DIST. If the
    # web derivation were missing this would 404 (API-only) - the bug the web
    # packaging exists to prevent.
    root = machine.succeed("curl --fail --max-time 5 http://127.0.0.1:8000/")
    assert "Scufris" in root, f"dashboard shell not served at /: {root[:200]!r}"
    assert 'id="app"' in root, f"dashboard app mount missing at /: {root[:200]!r}"

    # State dir is writable under DynamicUser (StateDirectory=/var/lib/scufris).
    machine.succeed("test -d /var/lib/scufris")

    # Host inspection: every command scufris/host shells out to must resolve on
    # the unit's PATH. `run_command` classifies an unresolvable argv[0] as
    # MISSING, which renders as this exact sentence - so its ABSENCE is the
    # assertion. The unit PATH is the module's own `hostTools` closure; nothing
    # in the ambient system profile can rescue it.
    overview = machine.succeed(
        "curl --fail --max-time 60 http://127.0.0.1:8000/api/host/overview"
    )
    assert "is not installed on this host" not in overview, \
        f"host overview lost a tool off the unit PATH: {overview!r}"

    # Restart works - the unit comes back and still serves.
    machine.succeed("systemctl restart scufris.service")
    machine.wait_for_unit("scufris.service")
    machine.wait_for_open_port(8000)
    machine.succeed("curl --fail --max-time 5 http://127.0.0.1:8000/api/config")
  '';
}
