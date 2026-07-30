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
{self}: {isNixos}: {
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) types mkOption mkEnableOption mkIf mkPackageOption mapAttrsToList mapAttrs' nameValuePair optional optionalAttrs makeBinPath toUpper;

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
  };

  # The PATH the service runs with: `path` packages, then the ambient profile
  # so git and friends resolve. Only emitted when `path` is non-empty.
  profileBin =
    if isNixos
    then "/run/current-system/sw/bin"
    else "${config.home.profileDirectory}/bin";
  pathValue = "${makeBinPath cfg.path}:${profileBin}";
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
        path = cfg.path;
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
            Environment = envList ++ optional (cfg.path != []) "PATH=${pathValue}";
            Restart = "on-failure";
            RestartSec = 5;
          }
          // optionalAttrs (cfg.environmentFile != null) {
            EnvironmentFile = toString cfg.environmentFile;
          };
      };
    };
  }
