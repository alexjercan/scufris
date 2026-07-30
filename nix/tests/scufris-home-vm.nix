# NixOS VM test for the scufris HOME-MANAGER service module.
#
# The companion to scufris-vm.nix, and not a redundant one: the two branches of
# nix/scufris-service.nix build their PATH by different mechanisms
# (`systemd.services.<n>.path` vs an explicit `Environment=PATH=`), so proving
# the system service works proves nothing about the user service. This test
# boots a real user session, starts the real `systemd --user` unit, and asks the
# running server what it can see.
#
# What it covers, over and above the system-service test:
#   * The home-manager module evaluates and produces a user unit that reaches
#     active under a normal (non-root) login.
#   * `/api/host/overview` finds its toolchain from the unit's OWN PATH. This is
#     the regression that motivated the test: the user unit's PATH tail was
#     ~/.nix-profile/bin, which on NixOS contains no system tool at all, so
#     every host page reported "not installed on this host" while the same
#     server run by hand from a login shell worked perfectly.
#   * `hostTools = false` still yields a working server - the escape hatch is an
#     escape hatch, not a way to break the unit.
#
# The agent is disabled: a VM has no codex/claude login. Same rationale as
# scufris-vm.nix.
{
  pkgs,
  homeManagerModule,
  homeManagerNixosModule,
}: let
  user = "alice";
  uid = 1000;

  # Both nodes run the same server; only `hostTools` differs.
  scufrisHome = hostTools: {
    home.stateVersion = "24.05";
    programs.scufris = {
      inherit hostTools;
      enable = true;
      settings = {
        host = "127.0.0.1";
        port = 8000;
        agent_enabled = false;
      };
    };
  };

  machineWith = hostTools: {...}: {
    imports = [homeManagerNixosModule];

    users.users.${user} = {
      inherit uid;
      isNormalUser = true;
      # A user unit needs a live `systemd --user` instance. Without lingering
      # it would only exist for the duration of a login session, and the test
      # drives the machine over `su` rather than a real login.
      linger = true;
    };

    home-manager.users.${user} = {...}: {
      imports = [homeManagerModule];
      config = scufrisHome hostTools;
    };

    environment.systemPackages = [pkgs.curl];
    virtualisation.memorySize = 1024;
  };
in
  pkgs.testers.nixosTest {
    name = "scufris-home";

    nodes = {
      machine = machineWith true;
      notools = machineWith false;
    };

    testScript = ''
      # Every user-manager call goes through the login user with its runtime
      # dir set; `systemctl --user` as root would talk to root's own manager and
      # cheerfully report the unit missing.
      USER_CTL = (
          "su -l ${user} -c "
          "'XDG_RUNTIME_DIR=/run/user/${toString uid} systemctl --user {}'"
      )

      def user_ctl(node, args):
          return node.succeed(USER_CTL.format(args))

      start_all()

      machine.wait_for_unit("multi-user.target")
      # The user manager first - the scufris user unit cannot exist before it.
      machine.wait_for_unit("user@${toString uid}.service")
      machine.wait_until_succeeds(USER_CTL.format("is-active scufris.service"), timeout=60)
      machine.wait_for_open_port(8000)

      # Liveness, same shape as the system-service test.
      cfg = machine.succeed("curl --fail --max-time 5 http://127.0.0.1:8000/api/config")
      assert '"agent_enabled":false' in cfg.replace(" ", ""), \
          f"expected agent disabled in /api/config, got: {cfg!r}"

      # THE REGRESSION. `run_command` renders an unresolvable argv[0] as
      # "<name> is not installed on this host"; before `hostTools` the user
      # unit produced that sentence for systemctl, journalctl, nixos-rebuild,
      # nix-store and ip all at once. Assert on the report rather than on the
      # PATH string: a PATH assertion would pass on a directory that merely
      # exists, which is exactly how the bug survived.
      overview = machine.succeed(
          "curl --fail --max-time 60 http://127.0.0.1:8000/api/host/overview"
      )
      assert "is not installed on this host" not in overview, \
          f"user unit is missing host tools on PATH: {overview!r}"

      # And prove WHERE they come from. The report above would also be clean if
      # some ambient directory happened to supply the tools; the module's claim
      # is stronger than that - they are in the unit's own closure. Resolve each
      # one against the unit's literal PATH in an otherwise empty environment,
      # and require a /nix/store answer.
      unit_env = user_ctl(machine, "show scufris.service -p Environment")
      unit_path = next(
          field[len("PATH=") :]
          for field in unit_env.strip().split()
          if field.startswith("PATH=")
      )
      assert "/run/current-system/sw/bin" not in unit_path, \
          f"unit PATH leans on the system profile: {unit_path!r}"

      for tool in ("systemctl", "journalctl", "nix", "nix-store", "nixos-rebuild", "ip"):
          resolved = machine.succeed(
              f"env -i PATH={unit_path} /bin/sh -c 'command -v {tool}'"
          ).strip()
          assert resolved.startswith("/nix/store/"), \
              f"{tool} resolved to {resolved!r}, which is not a pinned closure entry"

      # The escape hatch: hostTools = false must still give a working server.
      # An operator turning it off is saying "I supply these myself", not
      # "break my dashboard".
      notools.wait_for_unit("multi-user.target")
      notools.wait_for_unit("user@${toString uid}.service")
      notools.wait_until_succeeds(USER_CTL.format("is-active scufris.service"), timeout=60)
      notools.wait_for_open_port(8000)
      notools.succeed("curl --fail --max-time 5 http://127.0.0.1:8000/api/config")
    '';
  }
