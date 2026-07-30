# The privileged host-action helper: a NixOS SYSTEM unit running as root.
#
# Deliberately a SEPARATE module from `nix/scufris-service.nix`, and separately
# exported as `nixosModules.scufris-hostd`. Enabling host agency is then its own
# diffable act in the operator's configuration - upgrading scufris can never
# silently acquire the ability to act on the machine.
#
# The privilege model is `tasks/20260729-125020/DECISION.md`. In short: a closed
# set of typed verbs over a unix socket, no sudo rules, no shell verb at any
# privilege, and an append-only audit log this unit writes as root so it
# survives the app being the thing that misbehaved.
#
# Two preconditions, both fail-closed:
#
#   1. `secretFile` must exist and be non-empty. It holds the shared secret the
#      app presents on every frame; without it the helper refuses to start,
#      because a root socket with no credential in front of it is reachable by
#      anything running as the operator - including the agent CLI subprocesses
#      scufris itself spawns, which run arbitrary shell.
#   2. The same secret must reach the app as SCUFRIS_HOSTD_SECRET (the sops
#      dotenv it already takes as an EnvironmentFile).
#
# Stated plainly, because it must not be oversold: the secret raises the bar
# against the model acting unasked. It is NOT a boundary against a compromised
# operator account, which is already root-equivalent through the docker group.
{self}: {
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) types mkOption mkEnableOption mkIf;

  cfg = config.services.scufris-hostd;
in {
  options.services.scufris-hostd = {
    enable =
      mkEnableOption ''
        the scufris privileged host-action helper. This grants the scufris
        dashboard the ability to restart units and collect nix garbage on this
        machine, through an operator-approved proposal for each action. Read
        tasks/20260729-125020/DECISION.md before turning it on
      '';

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.system}.scufris;
      defaultText = "scufris.packages.\${system}.scufris";
      description = "The package providing the `scufris-hostd` binary.";
    };

    socketPath = mkOption {
      type = types.path;
      default = "/run/scufris-hostd/hostd.sock";
      description = ''
        Where the helper listens. Must match SCUFRIS_HOSTD_SOCKET in the app's
        configuration. Lives under the unit's own RuntimeDirectory.
      '';
    };

    group = mkOption {
      type = types.str;
      example = "scufris";
      description = ''
        Group given read/write on the socket - the group the scufris USER
        service runs as. Filesystem permissions are the outer gate; the shared
        secret is the one that distinguishes scufris from another process of the
        same user.

        Use a DEDICATED group. A shared one like `users` is the default primary
        group of every normal account on NixOS, which would expose the root
        helper's socket to every human user on the box - and reaching the socket
        is the first step of every attack the secret is there to slow down.
      '';
    };

    secretFile = mkOption {
      # `str`, not `path`: a nix PATH literal (`./secret`) would be COPIED into
      # the world-readable store, which is the one thing this option's
      # description forbids. A string makes that a configuration error rather
      # than a silent leak (review round 1, R1.16).
      type = types.str;
      example = "/run/secrets/scufris-hostd-secret";
      description = ''
        File holding the shared secret callers must present, e.g. a sops-nix
        secret. Read as root at start; never in the nix store. The helper
        refuses to start if it is missing or empty.
      '';
    };

    auditLog = mkOption {
      type = types.path;
      default = "/var/log/scufris-hostd/audit.jsonl";
      description = ''
        The append-only record of every requested, refused, approved, applied
        and cancelled action. Root-owned and root-written; nothing outside the
        helper may delete an entry, and no protocol verb can.
      '';
    };

    auditMaxBytes = mkOption {
      type = types.ints.between 65536 (16 * 1024 * 1024 * 1024);
      default = 16 * 1024 * 1024;
      description = ''
        Rotate the audit log once it would exceed this size. Rotation is by
        SIZE rather than on a timer so a burst cannot outrun the policy.
      '';
    };

    auditKeep = mkOption {
      # At least 2: one rotated sibling is the minimum that makes rotation a
      # rotation rather than a deletion (review round 1, R1.8).
      type = types.ints.between 2 100;
      default = 5;
      description = ''
        How many audit files to keep IN TOTAL (the active one plus its rotated
        siblings). The oldest is pruned first. With the defaults the log is
        bounded at 80 MiB.
      '';
    };

    ttlSeconds = mkOption {
      type = types.int;
      default = 600;
      description = ''
        How long an operator has to approve a proposal before it goes stale. A
        proposal describes the system as it was when previewed; approving an
        old one is refused rather than applied to a system that has moved.
      '';
    };
  };

  config = mkIf cfg.enable {
    systemd.services.scufris-hostd = {
      description = "Scufris privileged host-action helper";
      wantedBy = ["multi-user.target"];
      after = ["network.target"];
      # The verbs shell out to systemctl and nix; both come from the system
      # profile rather than being pinned here, so the helper acts with the
      # same tools the operator would use by hand.
      path = [pkgs.nix pkgs.systemd pkgs.nixos-rebuild];
      serviceConfig = {
        ExecStart = lib.escapeShellArgs [
          "${cfg.package}/bin/scufris-hostd"
          "--socket"
          (toString cfg.socketPath)
          "--secret-file"
          (toString cfg.secretFile)
          "--audit-log"
          (toString cfg.auditLog)
          "--audit-max-bytes"
          (toString cfg.auditMaxBytes)
          "--audit-keep"
          (toString cfg.auditKeep)
          "--group"
          cfg.group
          "--ttl-seconds"
          (toString cfg.ttlSeconds)
        ];
        # Root, and that is the whole point of this unit. What keeps it honest
        # is the absence of a shell verb, not a sandbox: the verbs it does have
        # must be able to restart system units and collect the store.
        User = "root";
        # The process's primary group, so RuntimeDirectory is created
        # root:<group> and the operator's scufris service can traverse it to
        # reach the socket. uid 0 is unaffected.
        Group = cfg.group;
        RuntimeDirectory = "scufris-hostd";
        RuntimeDirectoryMode = "0750";
        # /var/log/scufris-hostd, which outlives the process - a restart must
        # not lose the audit.
        LogsDirectory = "scufris-hostd";
        LogsDirectoryMode = "0750";
        Restart = "on-failure";
        RestartSec = 5;
      };
    };
  };
}
