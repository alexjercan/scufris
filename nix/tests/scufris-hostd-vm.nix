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
#   * apply on the issued id DOES execute, as root.
#   * The same id cannot be applied twice.
#   * The audit log exists, is root-owned, and holds the record.
{
  pkgs,
  hostdModule,
}: let
  secret = "vm-shared-secret";

  # A client that speaks the protocol: one JSON line out, frames back. Kept
  # tiny on purpose - it is standing in for scufris.hostclient, and the point
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

      environment.systemPackages = [pkgs.python3];
      virtualisation.memorySize = 1024;
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
      assert proposal["argv"] == ["systemctl", "restart", "--", "demo.service"], proposal
      assert proposal["preview"]["lines"], proposal["preview"]
      still = machine.succeed("systemctl show -p MainPID --value demo.service").strip()
      assert still == before, "proposing restarted the unit; it must change nothing"

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
    '';
  }
