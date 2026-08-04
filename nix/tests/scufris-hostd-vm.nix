# NixOS VM test for the privileged host-action helper.
#
# The Python suite drives the engine with an injected executor, which proves the
# refusals and the bookkeeping but deliberately never runs a command. This test
# is the other half: a real root unit, on a real socket, actually restarting a
# real service - the part that cannot be faked and that a mistake in would only
# show up on the operator's machine.
#
# What it covers:
#   * The module loads and `scufris-hostd.service` reaches active.
#   * The socket appears with mode 0660 and the configured group.
#   * A caller with no valid secret is refused (and it is recorded).
#   * An unknown verb is refused.
#   * A deny-listed unit is refused before any argv is built.
#   * propose does NOT execute: the target unit is untouched afterwards.
#   * list_pending reports the real held proposal (what the app rebuilds its
#     approval queue from after a restart) and drops it once it is used.
#   * apply on the issued id DOES execute, as root.
#   * The same id cannot be applied twice.
#   * The audit log exists, is root-owned, and holds the record.
#   * R3, the half no Python test can reach: a REAL activation of a real second
#     toplevel (a specialisation), as root, on the real system profile - and a
#     REAL rollback to the generation it replaced.
{
  pkgs,
  hostdModule,
}: let
  secret = "vm-shared-secret";

  # A client that speaks the protocol: one JSON line out, frames back. Kept
  # tiny on purpose - it is standing in for scufris_hostctl's client, and the point
  # is to exercise the SERVER.
  client = pkgs.writeText "hostd-client.py" ''
    import json, socket, sys

    def call(request, read_all=False):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect("/run/scufris-hostd/hostd.sock")
        s.sendall((json.dumps(request) + "\n").encode())
        data = s.makefile("rb")
        frames = []
        while True:
            line = data.readline()
            if not line:
                break
            frames.append(json.loads(line))
            if not read_all or frames[-1]["type"] in ("result", "error"):
                break
        s.close()
        return frames

    if __name__ == "__main__":
        request = json.loads(sys.argv[1])
        print(json.dumps(call(request, read_all=True)))
  '';
in
  pkgs.testers.nixosTest {
    name = "scufris-hostd";

    nodes.machine = {...}: {
      imports = [hostdModule];

      users.groups.scufris = {};
      environment.etc."scufris-hostd-secret".text = secret;

      services.scufris-hostd = {
        enable = true;
        group = "scufris";
        secretFile = "/etc/scufris-hostd-secret";
      };

      # A harmless unit for the helper to actually act on. Nothing on the
      # deny-list can be used here, by design.
      systemd.services.demo = {
        wantedBy = ["multi-user.target"];
        serviceConfig = {
          ExecStart = "${pkgs.coreutils}/bin/sleep infinity";
          Restart = "always";
        };
      };

      # A test VM boots through qemu's -kernel rather than a bootloader, and
      # `switch-to-configuration` updates the boot entries: measured, the
      # rollback's switch reached install-grub.pl and failed with "will not
      # proceed with blocklists" on the test image's ext2 root. That failure is
      # the test environment, not the code - and the code handled it correctly,
      # reporting "step 2 of 2 failed after 1 succeeded" with the split-state
      # explanation. Disabling grub here removes the environment's own
      # limitation so the test can assert the successful path.
      boot.loader.grub.enable = false;

      # A REAL second system to activate. A specialisation is a complete
      # toplevel in the store, built from this same configuration - which is how
      # NixOS's own switch tests get a second system without a network or a
      # second nixpkgs. `/etc/r3-marker` is what makes the activation
      # observable: it exists only while the specialisation is the running
      # system.
      specialisation.r3.configuration = {
        environment.etc."r3-marker".text = "the r3 configuration is running\n";
      };

      environment.systemPackages = [pkgs.python3];
      # A switch builds an /etc closure and restarts units, with both systems in
      # the store.
      virtualisation.memorySize = 2048;
    };

    testScript = ''
      import json

      machine.start()
      machine.wait_for_unit("scufris-hostd.service")
      machine.wait_for_unit("demo.service")
      machine.wait_for_file("/run/scufris-hostd/hostd.sock")

      SECRET = "${secret}"
      CLIENT = "${client}"

      def call(request):
          out = machine.succeed(
              "python3 %s %s" % (CLIENT, "'" + json.dumps(request) + "'")
          )
          return json.loads(out)

      # The socket is group-readable and no wider.
      mode = machine.succeed("stat -c %a /run/scufris-hostd/hostd.sock").strip()
      assert mode == "660", f"socket mode is {mode}, expected 660"
      group = machine.succeed("stat -c %G /run/scufris-hostd/hostd.sock").strip()
      assert group == "scufris", f"socket group is {group}, expected scufris"

      # No valid secret: refused. This is the check that keeps another process
      # of the same user - notably an agent CLI subprocess - off the socket.
      frames = call({"verb": "hello", "secret": "wrong"})
      assert frames[0]["type"] == "error", frames
      assert frames[0]["code"] == "unauthorized", frames

      # An unknown verb never reaches the code that builds commands.
      frames = call({"verb": "run_shell", "secret": SECRET})
      assert frames[0]["type"] == "error", frames
      assert frames[0]["code"] == "bad_request", frames

      # The helper announces the verbs it has, and a shell is not among them.
      frames = call({"verb": "hello", "secret": SECRET})
      assert frames[0]["type"] == "hello", frames
      assert "shell" not in json.dumps(frames[0]["verbs"]), frames[0]["verbs"]

      # A deny-listed unit is refused outright.
      frames = call({
          "verb": "propose", "secret": SECRET,
          "kind": "unit_restart", "args": {"unit": "sshd"},
      })
      assert frames[0]["type"] == "error", frames
      assert frames[0]["code"] == "refused", frames

      # Propose against the real unit. This must NOT restart it.
      before = machine.succeed(
          "systemctl show -p MainPID --value demo.service"
      ).strip()
      frames = call({
          "verb": "propose", "secret": SECRET,
          "kind": "unit_restart", "args": {"unit": "demo"},
      })
      assert frames[0]["type"] == "proposal", frames
      proposal = frames[0]["proposal"]
      steps = [step["argv"] for step in proposal["steps"]]
      assert steps == [["systemctl", "restart", "--", "demo.service"]], proposal
      assert proposal["preview"]["lines"], proposal["preview"]
      still = machine.succeed("systemctl show -p MainPID --value demo.service").strip()
      assert still == before, "proposing restarted the unit; it must change nothing"

      # The queue the app recovers after a restart, read from the REAL helper: its
      # own registry is in-memory by design, so this verb is what stops a restart
      # inside a proposal's window from stranding a live approval.
      frames = call({"verb": "list_pending", "secret": SECRET})
      assert frames[0]["type"] == "pending", frames
      pending = frames[0]["proposals"]
      assert [p["id"] for p in pending] == [proposal["id"]], pending
      # The whole proposal, not a stub - the app renders the preview from this.
      assert pending[0]["preview"]["lines"], pending[0]
      assert [s["argv"] for s in pending[0]["steps"]] == steps, pending[0]

      # Apply the id the helper issued. NOW it runs, as root.
      frames = call({
          "verb": "apply", "secret": SECRET,
          "proposal_id": proposal["id"], "approved_by": "vm-operator",
      })
      result = frames[-1]
      assert result["type"] == "result", frames
      assert result["ok"], result
      machine.wait_until_succeeds(
          "test \"$(systemctl show -p MainPID --value demo.service)\" != %s" % before
      )

      # The same approval does not replay.
      frames = call({
          "verb": "apply", "secret": SECRET,
          "proposal_id": proposal["id"], "approved_by": "vm-operator",
      })
      assert frames[0]["type"] == "error", frames
      assert frames[0]["code"] == "already_used", frames

      # A used proposal is no longer pending: the queue and the decision agree.
      frames = call({"verb": "list_pending", "secret": SECRET})
      assert frames[0]["type"] == "pending", frames
      assert frames[0]["proposals"] == [], frames[0]

      # Reading the queue needs the secret like everything else.
      frames = call({"verb": "list_pending", "secret": "wrong"})
      assert frames[0]["type"] == "error", frames
      assert frames[0]["code"] == "unauthorized", frames

      # The record is root's, and it holds what happened.
      machine.succeed("test -f /var/log/scufris-hostd/audit.jsonl")
      owner = machine.succeed(
          "stat -c %U /var/log/scufris-hostd/audit.jsonl"
      ).strip()
      assert owner == "root", f"audit log is owned by {owner}, expected root"
      audit_text = machine.succeed("cat /var/log/scufris-hostd/audit.jsonl")
      assert '"event":"requested"' in audit_text.replace(" ", ""), audit_text
      assert '"event":"applied"' in audit_text.replace(" ", ""), audit_text
      assert '"event":"refused"' in audit_text.replace(" ", ""), audit_text
      assert SECRET not in audit_text, "the audit log quoted the shared secret"

      # --- R3: a real activation, and a real rollback ---------------------
      #
      # The Python suite proves which argv WOULD run; only here does one actually
      # run, as root, against the real system profile. The second system is a
      # specialisation - a complete toplevel in this store.
      target = machine.succeed(
          "readlink -f /run/current-system/specialisation/r3"
      ).strip()
      before_system = machine.succeed("readlink -f /run/current-system").strip()
      assert target != before_system, target
      machine.fail("test -e /etc/r3-marker")

      def generations():
          out = machine.succeed(
              "nix-env -p /nix/var/nix/profiles/system --list-generations"
          )
          return [int(line.split()[0]) for line in out.strip().splitlines()]

      # A test VM boots its toplevel DIRECTLY and has no system profile, so
      # `--list-generations` is empty here where an installed machine always has
      # at least one. Give it the generation a real host already has; without it
      # there would be nothing to roll back to, and the test would be describing
      # a machine nobody runs.
      machine.succeed(
          "nix-env -p /nix/var/nix/profiles/system --set " + before_system
      )
      before_generations = generations()
      assert before_generations, "the system profile still has no generation"
      current_generation = max(before_generations)

      # The helper validates the path even though the app is what builds it: a
      # subpath is not a system, and is refused by shape.
      frames = call({
          "verb": "propose", "secret": SECRET,
          "kind": "activate", "args": {"toplevel": target + "/bin"},
      })
      assert frames[0]["type"] == "error", frames
      assert frames[0]["code"] == "refused", frames

      frames = call({
          "verb": "propose", "secret": SECRET,
          "kind": "activate", "args": {"toplevel": target},
      })
      assert frames[0]["type"] == "proposal", frames
      change = frames[0]["proposal"]
      assert change["risk"] == "r3", change
      steps = [step["argv"] for step in change["steps"]]
      assert steps[0] == [
          "nix-env", "--profile", "/nix/var/nix/profiles/system", "--set", target,
      ], steps
      assert steps[1][-2:] == [target + "/bin/switch-to-configuration", "switch"], steps
      # The preview is nix's own closure comparison, and the undo names the
      # generation that is running right now.
      assert change["preview"]["available"]["ok"], change["preview"]
      assert change["reversal"]["possible"], change["reversal"]
      assert change["reversal"]["args"] == {"generation": current_generation}, change
      # Proposing changed nothing: same system, same generations, no marker.
      assert machine.succeed("readlink -f /run/current-system").strip() == before_system
      assert generations() == before_generations
      machine.fail("test -e /etc/r3-marker")

      # Approve it. This is a real switch-to-configuration, as root, in a
      # transient unit.
      frames = call({
          "verb": "apply", "secret": SECRET,
          "proposal_id": change["id"], "approved_by": "vm-operator",
      })
      result = frames[-1]
      assert result["type"] == "result", frames
      assert result["ok"], frames
      assert result["steps_completed"] == 2, result
      assert result["steps_total"] == 2, result

      machine.succeed("test -e /etc/r3-marker")
      after_system = machine.succeed("readlink -f /run/current-system").strip()
      assert after_system == target, after_system
      after_generations = generations()
      assert len(after_generations) == len(before_generations) + 1, after_generations
      assert max(after_generations) > current_generation, after_generations

      # Roll back by NUMBER. The helper resolves which store path that is.
      frames = call({
          "verb": "propose", "secret": SECRET,
          "kind": "rollback", "args": {"generation": current_generation},
      })
      assert frames[0]["type"] == "proposal", frames
      rollback = frames[0]["proposal"]
      assert rollback["args"]["toplevel"] == before_system, rollback
      frames = call({
          "verb": "apply", "secret": SECRET,
          "proposal_id": rollback["id"], "approved_by": "vm-operator",
      })
      result = frames[-1]
      assert result["type"] == "result", frames
      assert result["ok"], frames

      machine.fail("test -e /etc/r3-marker")
      assert machine.succeed("readlink -f /run/current-system").strip() == before_system

      # Rolling back to the generation that is now running is refused.
      frames = call({
          "verb": "propose", "secret": SECRET,
          "kind": "rollback", "args": {"generation": current_generation},
      })
      assert frames[0]["type"] == "error", frames
      assert frames[0]["code"] == "refused", frames

      # And the root-written record carries both privileged acts, with the exact
      # store path each one activated.
      audit_text = machine.succeed("cat /var/log/scufris-hostd/audit.jsonl")
      compact = audit_text.replace(" ", "")
      assert '"kind":"activate"' in compact, audit_text
      assert '"kind":"rollback"' in compact, audit_text
      assert target in audit_text, "the audit did not record what was activated"
    '';
  }
