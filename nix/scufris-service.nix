# Shared scufris service module, specialised for home-manager (user service)
# and NixOS (system service) by the `isNixos` flag. Both drive the single
# `scufris serve` web server, configured through SCUFRIS_ env vars (the
# pydantic-settings surface in scufris/config.py). The built dashboard is
# served from a separate nix derivation (packages.scufris-web) via
# SCUFRIS_WEB_DIST - the Python wheel excludes web/dist (lesson
# web_dist-via-__file__-is-dev-only).
#
# Agent backends (codex / claude) are operator-installed binaries the server
# shells out to; they are NOT Python deps (lessons codex-binary-breaks-uv2nix-venv,
# codex-exec-is-the-nixos-path). Put them on the service PATH via `path`.
#
# The host INSPECTION toolchain is different: those binaries are not an
# operator's choice, they are what scufris/host is written against, so the
# module ships them itself (`hostTools`) rather than hoping an ambient profile
# supplies them. See `hostToolPackages` below.
{self}: {isNixos}: {
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) types mkOption mkEnableOption mkIf mkPackageOption mapAttrsToList mapAttrs' nameValuePair optional optionals optionalAttrs makeBinPath toUpper;

  cfg =
    if isNixos
    then config.services.scufris
    else config.programs.scufris;

  defaults = self.packages.${pkgs.system};

  # Render a nix scalar as a SCUFRIS_ env value. pydantic-settings parses
  # 1/0 for bools; ints/floats/paths/strings stringify directly.
  toEnv = v:
    if builtins.isBool v
    then
      (
        if v
        then "1"
        else "0"
      )
    else builtins.toString v;

  # Env pairs keyed by the SCUFRIS_ suffix (e.g. "WEB_DIST"). The web dist is
  # always injected (overridable via settings.web_dist); state_dir only when set.
  envAttrs =
    {WEB_DIST = "${cfg.webPackage}";}
    // optionalAttrs (cfg.stateDir != null) {STATE_DIR = toString cfg.stateDir;}
    // mapAttrs' (n: v: nameValuePair (toUpper n) (toEnv v)) cfg.settings;

  # Two shapes: a "K=V" list for the home-manager unit schema, a prefixed
  # attrset for the NixOS systemd.services.<n>.environment option.
  envList = mapAttrsToList (n: v: "SCUFRIS_${n}=${v}") envAttrs;
  envPrefixed = mapAttrs' (n: v: nameValuePair "SCUFRIS_${n}" v) envAttrs;

  options = {
    enable = mkEnableOption "the Scufris dashboard web server";

    package = mkOption {
      type = types.package;
      default = defaults.scufris;
      defaultText = "scufris.packages.\${system}.scufris";
      description = "The scufris package providing the `scufris` binary.";
    };

    webPackage = mkOption {
      type = types.package;
      default = defaults.scufris-web;
      defaultText = "scufris.packages.\${system}.scufris-web";
      description = ''
        The built dashboard assets served at "/". Wired to SCUFRIS_WEB_DIST so
        a packaged server finds the frontend (the Python wheel omits web/dist).
      '';
    };

    stateDir = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = ''
        Where scufris persists runtime state (SCUFRIS_STATE_DIR). Null leaves
        the app default (~/.local/state/scufris for a user service).
      '';
    };

    settings = mkOption {
      type = types.attrsOf (types.oneOf [types.str types.int types.float types.bool types.path]);
      default = {};
      example = {
        host = "127.0.0.1";
        port = 8000;
        agent_backend = "app_server";
      };
      description = ''
        Scufris config as a flat attrset mirroring scufris/config.py fields
        (lowercase). Each key `foo` becomes `SCUFRIS_FOO`. Use this for any
        knob: host, port, log_level, poll_seconds, agent_enabled,
        agent_backend, agent_model, claude_model, agent_auth_mode, etc. For
        list/JSON fields (mcp_servers, disabled_tools) pass a JSON string.
        Secrets (e.g. openai_api_key) belong in `environmentFile`, not here.
      '';
    };

    environmentFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      example = "/home/alex/.config/scufris/env";
      description = ''
        Path to an EnvironmentFile with secret SCUFRIS_ vars (e.g.
        SCUFRIS_OPENAI_API_KEY). Read at service start; not in the nix store.
      '';
    };

    path = mkOption {
      type = types.listOf types.package;
      default = [];
      example = "[ pkgs.codex pkgs.claude-code pkgs.git ]";
      description = ''
        Extra packages on the service PATH - the agent shells out to the
        codex/claude binaries (and git). Operator-installed, never Python deps.
      '';
    };

    hostTools = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Put the host-inspection toolchain (systemd, nix, nixos-rebuild,
        iproute2) on the service PATH. On by default: the host pages are core
        dashboard surface, not an opt-in, and without these every one of them
        reports "not installed on this host".

        Turn it off only to supply the same commands yourself through `path` -
        e.g. a non-NixOS host that has no `nixos-rebuild` to offer.
      '';
    };
  };

  # The binaries scufris/host shells out to, pinned as a closure rather than
  # inherited from whatever profile happens to be on PATH:
  #
  #   systemd        systemctl, journalctl   (host/units.py, host/journal.py)
  #   nix            nix, nix-store          (host/packages.py, host/storage.py)
  #   nixos-rebuild  nixos-rebuild           (host/packages.py)
  #   iproute2       ip                      (host/network.py)
  #
  # (Thermals and disk usage read /sys and Python APIs directly - no `sensors`
  # or `smartctl` involved, so nothing else belongs here.)
  #
  # This mirrors scufris-hostd, which has always pinned its own verbs' tools.
  # The app module used to lean on an ambient profile instead, which made the
  # user-service case silently toolless: a home-manager unit's profile is
  # ~/.nix-profile/bin, and on NixOS no system tool is ever installed there.
  hostToolPackages = optionals cfg.hostTools [
    pkgs.systemd
    pkgs.nix
    pkgs.nixos-rebuild
    pkgs.iproute2
  ];

  # Operator packages FIRST: an explicit `path` entry outranks our default, so
  # pinning e.g. a specific nix stays possible without disabling `hostTools`.
  servicePath = cfg.path ++ hostToolPackages;

  # The PATH the user service runs with: the packages above, then the user's
  # own profile so operator-installed extras (git and friends) still resolve.
  # Always emitted - making it conditional on `path` being non-empty would mean
  # adding one package silently swapped the whole ambient PATH for this one.
  pathValue = "${makeBinPath servicePath}:${config.home.profileDirectory}/bin";
in
  if isNixos
  then {
    options.services.scufris = options;
    config = mkIf cfg.enable {
      systemd.services.scufris = {
        description = "Scufris dashboard web server";
        wantedBy = ["multi-user.target"];
        after = ["network.target"];
        # DynamicUser has no real home, so the app default state_dir
        # (Path.home()/.local/state) is unwritable. Point state (and HOME, for
        # any Path.home() lookups) at the StateDirectory unless the operator set
        # an explicit stateDir.
        environment =
          envPrefixed
          // optionalAttrs (cfg.stateDir == null) {
            SCUFRIS_STATE_DIR = "/var/lib/scufris";
            HOME = "/var/lib/scufris";
          };
        path = servicePath;
        serviceConfig = {
          ExecStart = "${cfg.package}/bin/scufris serve";
          EnvironmentFile = optional (cfg.environmentFile != null) (toString cfg.environmentFile);
          Restart = "on-failure";
          RestartSec = 5;
          DynamicUser = true;
          StateDirectory = "scufris";
        };
      };
    };
  }
  else {
    options.programs.scufris = options;
    config = mkIf cfg.enable {
      home.packages = [cfg.package];
      systemd.user.services.scufris = {
        Unit = {
          Description = "Scufris dashboard web server";
          After = ["network.target"];
        };
        Install.WantedBy = ["default.target"];
        Service =
          {
            ExecStart = "${cfg.package}/bin/scufris serve";
            Environment = envList ++ ["PATH=${pathValue}"];
            Restart = "on-failure";
            RestartSec = 5;
          }
          // optionalAttrs (cfg.environmentFile != null) {
            EnvironmentFile = toString cfg.environmentFile;
          };
      };
    };
  }
